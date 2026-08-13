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

MODEL_ID = "claude-opus-5"
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
- 提到估价时说明这是模型预测，平均误差约 7%
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
        "description": "估算一辆车的市场价。year、mileage、Brand、Car_Type 必填。",
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer"},
                "mileage": {"type": "integer"},
                "Brand": {"type": "string"},
                "Car_Type": {"type": "string"},
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

    def estimate_price(self, year, mileage, Brand, Car_Type, transmission='Manual',
                       fuelType='Petrol', engineSize=1.6, mpg=None, tax=None):
        row = {
            'year': year, 'mileage': mileage, 'Brand': Brand, 'Car_Type': Car_Type,
            'transmission': transmission, 'fuelType': fuelType,
            'engineSize': engineSize,
            'mpg': self._defaults['mpg'] if mpg is None else mpg,
            'tax': self._defaults['tax'] if tax is None else tax,
            'High_Performance': 0,
        }
        predicted = Model.car_price(pd.DataFrame([row])[FEATURE_COLUMNS])
        if predicted is None:
            return {'error': 'prediction failed'}
        return {
            'estimated_price': round(predicted),
            'note': 'LightGBM model, held-out MAPE 7.2%',
            'spec': {'year': year, 'mileage': mileage, 'Brand': Brand, 'Car_Type': Car_Type},
        }

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
        """返回 {'reply': str, 'cards': [...], 'mode': 'llm'|'rules'}。"""
        if self.llm_enabled:
            try:
                return self._reply_llm(message, history or [])
            except Exception as e:
                # LLM 不可用时不能让整个聊天挂掉，退回规则模式
                print(f"LLM reply failed, falling back to rules: {e}")
        return self._reply_rules(message)

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
        return {'reply': text.strip(), 'cards': cards[:MAX_RESULTS], 'mode': 'llm'}

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
