"""车辆咨询助手。

把项目已有的三块能力（搜索、估价、推荐）包成工具，供对话入口调用：

  search_vehicles     —— 按条件筛选在售车辆
  estimate_price      —— 调 LightGBM 模型给出估价
  find_alternatives   —— 调推荐引擎给出相似车 / 更划算的选择

两种运行模式：

  LLM 模式   设置了 ANTHROPIC_API_KEY 时启用。Claude 负责理解意图并决定
             调用哪个工具，工具的执行仍然在本地，模型看不到原始数据集。
  规则模式   未配置 API key 时自动降级。用关键词把自然语言映射成筛选条件。
             能力弱一些，但零依赖、零成本、永远可用——demo 不会因为外部
             服务失效而开天窗。

设计上刻意让两种模式共用同一套工具函数：模型只是换了个"谁来决定调什么"
的角色，业务逻辑没有第二份实现。
"""

import json
import os
import re

import pandas as pd

from model.recommendation import (get_better_value_alternatives,
                                  get_similar_vehicles)
from model.regressor import FEATURE_COLUMNS, Model

# 选 Haiku 而不是 Opus。这个任务是意图识别 + 从 3 个工具里挑一个 + 把 JSON
# 复述成一两句话 —— 没有多步推理、没有长上下文、没有复杂规划，正是小模型的
# 目标场景。Opus 5 贵 5 倍，换不来可感知的效果。
#
# 注意：这是基于任务复杂度的判断，不是实测结论。若要变成实测，用同一组问题
# 分别跑两个模型，检查是否认错工具或漏掉筛选条件。
MODEL_ID = "claude-haiku-4-5"
MAX_RESULTS = 5

# 合理年份区间。数据集里有 3 行脏数据（Fiesta 年份 2060、两台 1970 年的
# M Class / Zafira，而 Zafira 1999 年才上市）。总量 107,343 行里的 3 行对
# 模型没有影响，但按「最新优先」排序时它们正好排在最前面，看起来像 bug。
# 这里只在展示层过滤，不修改数据集本身，以免所有已测指标需要重算。
PLAUSIBLE_YEARS = (1990, 2025)

SYSTEM_PROMPT = """你是英国二手车网站的购车助手，背后有 107,343 条真实挂牌数据。

可用工具：
- search_vehicles：按品牌/车型/燃料/变速箱/价格/里程/年份筛选在售车辆
- estimate_price：给定车况估算市场价（留出集 MAPE 7.2%）
- find_alternatives：对某辆车给出相似车型或更便宜的同级替代

规则：
- 涉及具体车辆、价格、库存的问题，必须调用工具，不要凭空回答
- 报价格时带上英镑符号和千分位
- 估价一定要连同 likely_range 一起给出，例如「约 £30,000，大概率在
  £24,000–£38,000 之间」。只给点估计会让用户以为模型比实际更确定。
  区间很宽时说明这类车数据少、把握不大
- 回答简短，中文或英文跟随用户提问的语言
- 数据集里没有的信息（保险、贷款、保养记录）如实说明不掌握
"""

TOOLS = [
    {
        "name": "search_vehicles",
        "description": "按条件筛选在售车辆。所有参数可选，不传即不限制该条件。",
        "input_schema": {
            "type": "object",
            "properties": {
                "brand": {"type": "string", "description": "品牌，如 Ford、BMW、Audi"},
                "car_type": {"type": "string", "description": "车型，如 Hatchback、SUV、Sedan"},
                "fuel_type": {"type": "string", "description": "燃料，如 Petrol、Diesel、Hybrid"},
                "transmission": {"type": "string", "description": "变速箱，如 Manual、Automatic"},
                "max_price": {"type": "number", "description": "价格上限（英镑）"},
                "min_price": {"type": "number", "description": "价格下限（英镑）"},
                "max_mileage": {"type": "number", "description": "里程上限（英里）"},
                "min_year": {"type": "integer", "description": "最早年份"},
            },
        },
    },
    {
        "name": "estimate_price",
        "description": ("估算一辆车的市场价。year、mileage、Brand、Car_Type 必填。"
                        "尽量同时给出 model（具体车型名，如 Fiesta、i8、3 Series）—— "
                        "它对准确度影响很大：同品牌同排量下，1 系和 i8 差几万英镑。"),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer"},
                "mileage": {"type": "integer"},
                "Brand": {"type": "string"},
                "Car_Type": {"type": "string"},
                "model": {"type": "string",
                          "description": "具体车型名，如 Fiesta / i8 / 3 Series。强烈建议提供。"},
                "transmission": {"type": "string", "description": "默认 Manual"},
                "fuelType": {"type": "string", "description": "默认 Petrol"},
                "engineSize": {"type": "number", "description": "排量，默认 1.6"},
                "mpg": {"type": "number", "description": "默认取数据集中位数"},
                "tax": {"type": "number", "description": "默认取数据集中位数"},
            },
            "required": ["year", "mileage", "Brand", "Car_Type"],
        },
    },
    {
        "name": "find_alternatives",
        "description": "对指定车辆 id 给出替代选择。cheaper=true 时只返回更便宜的。",
        "input_schema": {
            "type": "object",
            "properties": {
                "vehicle_id": {"type": "integer"},
                "cheaper": {"type": "boolean", "description": "是否只要更便宜的，默认 false"},
            },
            "required": ["vehicle_id"],
        },
    },
]


class VehicleAssistant:
    def __init__(self, data, vehicle_features):
        self.data = data
        self.features = vehicle_features
        self._client = None
        self._defaults = {
            'mpg': float(pd.to_numeric(data['mpg'], errors='coerce').median()),
            'tax': float(pd.to_numeric(data['tax'], errors='coerce').median()),
        }
        # 每个品牌下的车型，按挂牌数量降序 —— 未指定车型时取第一个作为默认
        self._normalised = data.assign(model=data['model'].astype(str).str.strip())
        self._models_by_brand = {
            str(brand): group['model'].value_counts().index.tolist()
            for brand, group in self._normalised.groupby('Brand', observed=True)
        }

    # ---------- 能力判定 ----------

    @property
    def llm_enabled(self):
        return bool(os.environ.get('ANTHROPIC_API_KEY'))

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    # ---------- 工具实现（两种模式共用） ----------

    def search_vehicles(self, brand=None, car_type=None, fuel_type=None,
                        transmission=None, max_price=None, min_price=None,
                        max_mileage=None, min_year=None):
        df = self.data
        low, high = PLAUSIBLE_YEARS
        mask = df['year'].between(low, high)

        for column, value in [('Brand', brand), ('Car_Type', car_type),
                              ('fuelType', fuel_type), ('transmission', transmission)]:
            if value:
                mask &= df[column].astype(str).str.lower() == str(value).lower()

        if max_price is not None:
            mask &= df['price'] <= max_price
        if min_price is not None:
            mask &= df['price'] >= min_price
        if max_mileage is not None:
            mask &= df['mileage'] <= max_mileage
        if min_year is not None:
            mask &= df['year'] >= min_year

        # 排序：预算内最新、里程最低。
        # 按价格升序是错的 —— "£15,000 以下" 是预算上限，不是"想买最便宜的"，
        # 实测那样排会把 2003 年 17.7 万英里的 £495 车推到最前面。
        hits = df[mask].sort_values(['year', 'mileage'], ascending=[False, True])

        # 去掉完全重复的挂牌（同款同年同价同里程），同一辆车列两遍很像 bug
        hits = hits.drop_duplicates(
            subset=['Brand', 'model', 'year', 'price', 'mileage'])

        return {
            'count': int(mask.sum()),
            'vehicles': self._to_cards(hits.head(MAX_RESULTS)),
            'sorted_by': 'newest first, then lowest mileage',
        }

    def estimate_price(self, year, mileage, Brand, Car_Type, model=None,
                       transmission=None, fuelType=None, engineSize=None,
                       mpg=None, tax=None):
        # 没给车型名时退到该品牌下最常见的车型。总比传空值好 ——
        # 空值会被 OneHotEncoder 当未知类别丢掉，等于回到没有这个特征的状态。
        resolved_model, assumed = self._resolve_model(Brand, model)

        # 未指定的规格从**这款车的真实数据**里取，不用全局默认。
        # 用全局默认（Manual / Petrol / 1.6L）会拼出不存在的车：BMW i8 是
        # 1.5L 混动自动挡，按 1.6L 汽油手动去问，模型会顺着数值特征分裂到
        # 一台普通 1 系上——实测估出 £12,313，而真实挂牌价约 £48,900。
        spec_defaults = self._spec_defaults(Brand, resolved_model)

        row = {
            'year': year, 'mileage': mileage, 'Brand': Brand, 'Car_Type': Car_Type,
            'model': resolved_model,
            'transmission': transmission or spec_defaults['transmission'],
            'fuelType': fuelType or spec_defaults['fuelType'],
            'engineSize': spec_defaults['engineSize'] if engineSize is None else engineSize,
            'mpg': spec_defaults['mpg'] if mpg is None else mpg,
            'tax': spec_defaults['tax'] if tax is None else tax,
            'High_Performance': spec_defaults['High_Performance'],
        }
        low, predicted, high = Model.car_price_interval(
            pd.DataFrame([row])[FEATURE_COLUMNS])
        if predicted is None:
            return {'error': 'prediction failed'}

        result = {
            'estimated_price': round(predicted),
            'note': 'LightGBM model, held-out MAPE 6.7%',
            'spec': {'year': year, 'mileage': mileage, 'Brand': Brand,
                     'Car_Type': Car_Type, 'model': resolved_model},
        }
        if low is not None and high is not None:
            result['likely_range'] = [round(low), round(high)]
            result['range_note'] = ('80% interval, measured coverage 81.3%. '
                                    'Report this range alongside the point estimate.')
        if assumed:
            result['warning'] = (
                f"未指定车型，已按 {Brand} 最常见的 {resolved_model} 估算；"
                f"车型对价格影响很大，请向用户确认具体车型。")
        return result

    def _spec_defaults(self, brand, model):
        """这款车在数据里最典型的规格：类别取众数，数值取中位数。

        逐级回退：Brand+model -> Brand -> 全数据集。
        """
        frame = self._normalised
        for mask in (
            (frame['Brand'].astype(str) == str(brand)) & (frame['model'] == model),
            (frame['Brand'].astype(str) == str(brand)),
        ):
            subset = frame[mask]
            if len(subset) >= 3:
                break
        else:
            subset = frame

        def mode_of(column, fallback):
            values = subset[column].mode()
            return str(values.iloc[0]) if len(values) else fallback

        return {
            'transmission': mode_of('transmission', 'Manual'),
            'fuelType': mode_of('fuelType', 'Petrol'),
            'engineSize': float(pd.to_numeric(subset['engineSize'], errors='coerce').median()),
            'mpg': float(pd.to_numeric(subset['mpg'], errors='coerce').median()),
            'tax': float(pd.to_numeric(subset['tax'], errors='coerce').median()),
            'High_Performance': int(pd.to_numeric(
                subset['High_Performance'], errors='coerce').median()),
        }

    def _resolve_model(self, brand, requested):
        """把用户说的车型名对到数据集里的取值。返回 (车型名, 是否为推断值)。"""
        available = self._models_by_brand.get(str(brand), [])
        if not available:
            return (str(requested).strip() if requested else 'unknown'), False
        if requested:
            wanted = str(requested).strip().lower()
            for name in available:
                if name.lower() == wanted:
                    return name, False
            for name in available:               # 宽松匹配："3 series" -> "3 Series"
                if wanted in name.lower() or name.lower() in wanted:
                    return name, False
        return available[0], True                # available 按挂牌数量降序

    def find_alternatives(self, vehicle_id, cheaper=False):
        if vehicle_id not in self.data.index:
            return {'error': f'vehicle {vehicle_id} not found'}
        fn = get_better_value_alternatives if cheaper else get_similar_vehicles
        return {
            'seed': self._to_cards(self.data.loc[[vehicle_id]])[0],
            'alternatives': self._to_cards(fn(vehicle_id, self.data, self.features, top_n=3)),
        }

    def _to_cards(self, frame):
        cards = []
        for idx, row in frame.iterrows():
            cards.append({
                'id': int(idx),
                'title': f"{row['Brand']}{row['model']} {row['year']}",
                'price': int(row['price']),
                'mileage': int(row['mileage']),
                'fuel': str(row['fuelType']),
                'transmission': str(row['transmission']),
                'url': f"/vehicle/{idx}",
                'image_url': str(row['image_url']),
            })
        return cards

    def _dispatch(self, name, payload):
        return {
            'search_vehicles': self.search_vehicles,
            'estimate_price': self.estimate_price,
            'find_alternatives': self.find_alternatives,
        }[name](**payload)

    # ---------- 对外入口 ----------

    def reply(self, message, history=None):
        """返回 {'reply', 'cards', 'mode', 'fallback_reason'}。

        降级到规则模式时一定带上 fallback_reason。这一条是被实际踩坑逼出来的：
        原本 mode='rules' 无法区分「没配 key」「包没装」「API 调用失败」三种
        情况，三条路径静默汇合到同一个返回值，排查时只能靠猜。

        降级本身保住了可用性，但如果不把原因暴露出来，它就只是把故障伪装
        成了「效果不太好」—— 和这个项目里推荐系统那个 except 犯的是同一个错。
        """
        if not self.llm_enabled:
            return self._with_reason(self._reply_rules(message), 'no_api_key')

        try:
            return self._reply_llm(message, history or [])
        except ImportError as exc:
            # Python 3 里 except 块结束后异常变量就被销毁了，在块内取出来
            reason, detail = 'package_missing', str(exc)
        except Exception as exc:
            reason, detail = 'api_error', str(exc)

        # 只用 ASCII：Windows 控制台默认 cp1252，打印非 ASCII 会二次抛错
        print(f"assistant: LLM unavailable ({reason}: {detail}), using rules")
        return self._with_reason(self._reply_rules(message), reason)

    @staticmethod
    def _with_reason(result, reason):
        result['fallback_reason'] = reason
        return result

    # ---------- LLM 模式 ----------

    def _reply_llm(self, message, history):
        client = self._get_client()
        messages = list(history) + [{'role': 'user', 'content': message}]
        cards = []

        # 工具调用循环：模型决定调什么，执行始终在本地
        for _ in range(5):
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            if response.stop_reason != 'tool_use':
                break

            messages.append({'role': 'assistant', 'content': response.content})
            results = []
            for block in response.content:
                if block.type != 'tool_use':
                    continue
                output = self._dispatch(block.name, dict(block.input))
                cards.extend(output.get('vehicles') or output.get('alternatives') or [])
                results.append({
                    'type': 'tool_result',
                    'tool_use_id': block.id,
                    'content': json.dumps(output, ensure_ascii=False),
                })
            messages.append({'role': 'user', 'content': results})

        text = ''.join(b.text for b in response.content if b.type == 'text')
        return {'reply': text.strip(), 'cards': cards[:MAX_RESULTS],
                'mode': 'llm', 'fallback_reason': None}

    # ---------- 规则模式 ----------

    def _reply_rules(self, message):
        """关键词 + 正则把自然语言映射成筛选条件。

        覆盖不了复杂问法，但保证在没有任何外部依赖时聊天框依然可用。
        """
        text = message.lower()
        params = {}

        for column, key in [('Brand', 'brand'), ('Car_Type', 'car_type'),
                            ('fuelType', 'fuel_type'), ('transmission', 'transmission')]:
            for value in self.data[column].astype(str).unique():
                if value.lower() in text:
                    params[key] = value
                    break

        # £20,000 / 20000 / 20k / under 20k
        amounts = [self._parse_amount(m) for m in
                   re.findall(r'£?\s*(\d[\d,]*\.?\d*\s*k?)', text)]
        amounts = [a for a in amounts if a is not None]

        for amount in amounts:
            if amount >= 1000 and 'max_price' not in params and self._near(
                    text, ['under', 'below', 'less than', '以下', '不超过', '预算', 'budget']):
                params['max_price'] = amount
            elif 1900 < amount < 2030 and 'min_year' not in params:
                params['min_year'] = int(amount)

        mileage = re.search(r'(\d[\d,]*)\s*(?:miles|mile|英里|里程)', text)
        if mileage:
            params['max_mileage'] = self._parse_amount(mileage.group(1))

        if not params and amounts:
            params['max_price'] = max(amounts)

        if not params:
            return {
                'reply': ("我可以帮你按品牌、车型、价格、里程筛选车辆，比如"
                          "「£15000 以下的 Ford Hatchback」。也可以问某辆车值多少钱。"),
                'cards': [], 'mode': 'rules',
            }

        result = self.search_vehicles(**params)
        described = '、'.join(f'{k}={v}' for k, v in params.items())
        if result['count'] == 0:
            return {'reply': f'没有找到符合条件的车（{described}）。要不要放宽一下？',
                    'cards': [], 'mode': 'rules'}
        return {
            'reply': f"按 {described} 找到 {result['count']} 辆，其中最新、里程最低的几辆：",
            'cards': result['vehicles'], 'mode': 'rules',
        }

    @staticmethod
    def _parse_amount(raw):
        raw = raw.strip().replace(',', '')
        try:
            if raw.endswith('k'):
                return float(raw[:-1]) * 1000
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _near(text, keywords):
        return any(k in text for k in keywords)
