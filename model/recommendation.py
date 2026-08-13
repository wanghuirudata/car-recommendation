"""基于内容的相似车辆推荐。

设计要点（面试常问的几处取舍都在这里）：

1. price 不参与相似度计算。价格是「结果」不是「属性」，把它放进特征里会让
   推荐退化成「找同价位挂牌」，用户看到的等于同一辆车列了三遍。价格只用于
   第二路召回（更划算的选择）的筛选条件。

2. 用加权欧氏距离，不用余弦。数值特征已做 z-score 标准化（有正有负），
   未中心化的余弦在这种空间里几何意义很弱；欧氏距离配上显式权重更可解释：
   - 数值特征的距离单位就是「几个标准差」
   - 一个类别特征取值不同，贡献固定的 sqrt(2) * weight

3. 权重是人工先验，不是学出来的。有了点击/询价埋点后，应该用
   learning-to-rank 或 triplet loss 来学，这里的常数只是冷启动方案。

4. 结果按 (Brand, model) 去重，保证 3 个推荐位是 3 款不同的车。
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 参与相似度计算的特征。注意 price 不在其中。
NUMERIC_FEATURES = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
CATEGORICAL_FEATURES = ['transmission', 'fuelType', 'Brand', 'Car_Type']
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['price']

# 特征权重（人工先验）。数值特征以「标准差」为单位；
# 类别特征取值不同时贡献 sqrt(2) * weight 的距离。
FEATURE_WEIGHTS = {
    'year': 1.0,          # 车龄，用户最在意
    'mileage': 1.0,       # 里程，同上
    'engineSize': 0.8,
    'mpg': 0.5,
    'tax': 0.3,           # 税档基本由排量决定，避免重复计权
    'Brand': 2.0,         # 换品牌是很大的跨越
    'Car_Type': 2.0,      # 换车型（两厢/SUV）同理
    'fuelType': 1.0,
    'transmission': 0.8,
}


def load_and_prepare_data(file_path):
    """读取数据并构造加权特征矩阵。

    返回 (data, vehicle_features)：
      data             —— 原始 DataFrame，用于展示
      vehicle_features —— float32 特征矩阵，权重已在建表时乘好，
                          单次查询零额外开销
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Failed to read {file_path}: {e}")

    if data.empty:
        raise ValueError("Data file is empty")

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 数值特征：转数值 -> 中位数填充 -> 标准化 -> 加权
    numeric_data = data[NUMERIC_FEATURES].apply(pd.to_numeric, errors='coerce').values
    numeric_data = SimpleImputer(strategy='median').fit_transform(numeric_data)
    numeric_data = StandardScaler().fit_transform(numeric_data)
    numeric_data = numeric_data * np.array(
        [FEATURE_WEIGHTS[c] for c in NUMERIC_FEATURES]
    )

    blocks = [pd.DataFrame(numeric_data, columns=NUMERIC_FEATURES, index=data.index)]

    # 类别特征：独热编码后整块乘权重
    # 注意这里从头构造特征矩阵，而不是在 data 的副本上删删改改 ——
    # 原实现保留了 CSV 里的 model / image_url 等字符串列，导致 .values
    # 退化成 object dtype，np.dot 直接 TypeError。
    for feature in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(data[feature].astype(str), prefix=feature)
        blocks.append(dummies.astype(np.float32) * FEATURE_WEIGHTS[feature])

    vehicle_features = pd.concat(blocks, axis=1).astype(np.float32)
    return data, vehicle_features


def _distances_to(position, vehicle_features):
    """目标车对全表的加权欧氏距离。107k x 30 的规模下约几毫秒。"""
    features_array = vehicle_features.values
    target = features_array[position]
    return np.linalg.norm(features_array - target, axis=1)


#: 去重候选池大小。见 _rank_and_dedupe 的取值说明。
POOL_SIZE = 10000


def _rank_and_dedupe(distances, data, candidate_mask, exclude_position, top_n,
                     pool_size=POOL_SIZE):
    """在最近的 pool_size 辆车里按 (Brand, model) 去重后取 top_n。

    两个设计决定：

    - 先截断候选池再去重。如果直接对全表去重，第三个「不同款」可能已经离得
      非常远（实测：2015 款 Fiesta 会被推一辆 2019 款 £23000 的 Puma）。
    - 池内凑不满 top_n 个不同款时，用池内最近的补齐。宁可推同款的另一条挂牌，
      也不推一辆完全不相干的车。

    pool_size 由 200 个随机种子车实测选定（多样性 / 相似度 / 延迟的折中）：

        pool     不同车型数(满分3)   p50 价差   p95 车龄差   ms/查询
        100          2.04            £1,166       1年        14.0
        300          2.23            £1,222       1年        14.3
        3,000        2.62            £1,540       1年        15.6
        10,000       2.95            £1,920       1年        17.1   <-- 采用
        30,000       3.00            £1,995       1年        22.9
        全表         3.00            £1,993       1年        40.9

    10000 基本能凑满 3 个不同车型，价差和延迟的增量都还很小；再往上收益递减。
    """
    masked = np.where(candidate_mask, distances, np.inf)
    masked[exclude_position] = np.inf

    n_valid = int(np.isfinite(masked).sum())
    if n_valid == 0:
        return data.iloc[0:0]

    # argpartition 取最近 k 个是 O(N)，比全量 argsort 的 O(N log N) 快得多
    k = min(pool_size, n_valid)
    pool = np.argpartition(masked, k - 1)[:k]
    pool = pool[np.argsort(masked[pool], kind='stable')]

    # 向量化取出去重键，避免在循环里逐行 data.iloc[pos]（pandas 行访问极慢）
    brands = data['Brand'].to_numpy()[pool]
    models = pd.Series(data['model'].to_numpy()[pool]).astype(str).str.strip().to_numpy()

    seen_models = set()
    picked = []
    for i, pos in enumerate(pool):
        key = (brands[i], models[i])
        if key in seen_models:
            continue
        seen_models.add(key)
        picked.append(pos)
        if len(picked) == top_n:
            break

    if len(picked) < top_n:
        chosen = set(picked)
        for pos in pool:
            if pos not in chosen:
                picked.append(pos)
                if len(picked) == top_n:
                    break

    return data.iloc[picked]


def get_similar_vehicles(vehicle_id, data, vehicle_features, top_n=3,
                         price_band=0.4):
    """同类车推荐：配置相近、款式不同。

    price_band 是候选**筛选**条件，不是相似度特征 —— 两者的区别很关键：
    价格不参与排序（否则推荐退化成「找同价位挂牌」），但价格区间差太远的车
    根本不是替代品。看 £10,000 车的用户不会是 £23,000 车的买家；不加这条带
    时，实测会出现「2018 款 £10,000 的 Fiesta 推 2019 款 £23,000 的 Puma」。
    ±40% 对应实测 p95 价差从 £9,600 收敛到合理范围。
    """
    try:
        position = data.index.get_loc(vehicle_id)
        target_price = float(data.iloc[position]['price'])
        distances = _distances_to(position, vehicle_features)

        prices = pd.to_numeric(data['price'], errors='coerce').to_numpy()
        candidate_mask = ((prices >= target_price * (1 - price_band)) &
                          (prices <= target_price * (1 + price_band)))
        if not candidate_mask.any():
            candidate_mask = np.ones(len(data), dtype=bool)

        return _rank_and_dedupe(distances, data, candidate_mask, position, top_n)
    except Exception as e:
        # 只用 ASCII：Windows 控制台默认 cp1252，打印中文会再抛
        # UnicodeEncodeError，把兜底逻辑本身冲掉，最终变成 500
        print(f"get_similar_vehicles failed, falling back to random: {e}")
        return data.iloc[np.random.choice(len(data), size=top_n, replace=False)]


def get_better_value_alternatives(vehicle_id, data, vehicle_features,
                                  top_n=3, min_discount=0.05):
    """更划算的选择：配置相近但明显更便宜的车。

    min_discount=0.05 表示至少便宜 5%，低于这个幅度对用户没有决策价值。
    没有满足条件的车时返回空 DataFrame（调用方需要判空）。
    """
    try:
        position = data.index.get_loc(vehicle_id)
        target_price = float(data.iloc[position]['price'])

        prices = pd.to_numeric(data['price'], errors='coerce').values
        candidate_mask = prices <= target_price * (1.0 - min_discount)
        if not candidate_mask.any():
            return data.iloc[0:0]

        distances = _distances_to(position, vehicle_features)
        return _rank_and_dedupe(distances, data, candidate_mask, position, top_n)
    except Exception as e:
        print(f"get_better_value_alternatives failed: {e}")
        return data.iloc[0:0]


def collaborative_filtering_recommendation(user_id, user_vehicle_data, num_recommendations=3):
    """协同过滤占位。

    目前站点没有用户行为埋点（浏览/收藏/询价），没有交互矩阵可用，
    因此暂不实现。有了埋点后的演进路线见 README。
    """
    return []
