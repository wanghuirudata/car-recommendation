"""推荐引擎的不变量。

这里锁住的都是曾经出过问题、或者容易被后续改动破坏的性质：
特征矩阵必须是纯数值、price 不能进相似度、结果不能包含自身。
"""

import numpy as np
import pandas as pd
import pytest

from model.recommendation import (CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                                  get_better_value_alternatives,
                                  get_similar_vehicles)

SEEDS = [5, 1000, 50000, 90000]


def test_feature_matrix_is_pure_numeric(features):
    """曾因 model / image_url 字符串列混入而退化成 object dtype，
    导致 np.dot 抛 TypeError。pandas 2.0 的 bool 型 dummies 也会触发同样问题。"""
    assert features.values.dtype == np.float32
    assert not any(features.dtypes == object)


def test_price_is_not_a_similarity_feature():
    """price 是车况的结果而非属性；计入相似度会让推荐退化成'找同价位挂牌'。"""
    assert "price" not in NUMERIC_FEATURES
    assert "price" not in CATEGORICAL_FEATURES


@pytest.mark.parametrize("vehicle_id", SEEDS)
def test_similar_excludes_seed_and_returns_requested_count(data, features, vehicle_id):
    recs = get_similar_vehicles(vehicle_id, data, features, top_n=3)
    assert len(recs) == 3
    assert vehicle_id not in recs.index
    assert len(set(recs.index)) == 3           # 不重复返回同一条挂牌


@pytest.mark.parametrize("vehicle_id", SEEDS)
def test_similar_respects_price_band(data, features, vehicle_id):
    """价格不参与排序，但作为候选筛选：看 £10k 车的人不是 £23k 车的买家。

    这里刻意**不传** price_band，走默认值——app.py 调用时也不传。
    早期版本显式传了 0.4，结果把默认值改坏了测试也照样通过（变异测试发现）。
    """
    seed_price = float(data.loc[vehicle_id, "price"])
    recs = get_similar_vehicles(vehicle_id, data, features, top_n=3)
    assert recs["price"].between(seed_price * 0.6, seed_price * 1.4).all()


def test_price_band_default_is_forty_percent():
    """锁住默认值本身：它是生产路径上真正生效的那个数。"""
    import inspect
    default = inspect.signature(get_similar_vehicles).parameters["price_band"].default
    assert default == 0.4


def test_price_band_parameter_is_honoured(data, features):
    """显式收窄价格带应当真的收窄结果。"""
    seed_price = float(data.loc[50000, "price"])
    recs = get_similar_vehicles(50000, data, features, top_n=3, price_band=0.05)
    assert recs["price"].between(seed_price * 0.95, seed_price * 1.05).all()


@pytest.mark.parametrize("vehicle_id", SEEDS)
def test_better_value_alternatives_are_actually_cheaper(data, features, vehicle_id):
    seed_price = float(data.loc[vehicle_id, "price"])
    recs = get_better_value_alternatives(vehicle_id, data, features, top_n=3,
                                         min_discount=0.05)
    assert not recs.empty
    assert (recs["price"] <= seed_price * 0.95).all()


def test_dedupe_gives_more_than_one_distinct_model(data, features):
    """去重前，三个推荐位常被同一款车的三条挂牌占满。"""
    distinct = []
    for vehicle_id in SEEDS:
        recs = get_similar_vehicles(vehicle_id, data, features, top_n=3)
        keys = {(r.Brand, str(r.model).strip()) for _, r in recs.iterrows()}
        distinct.append(len(keys))
    assert sum(distinct) / len(distinct) >= 2.0


def test_unknown_id_falls_back_instead_of_raising(data, features):
    """兜底路径本身不能抛异常——它曾因打印非 ASCII 而在 cp1252 上二次崩溃。"""
    recs = get_similar_vehicles(-1, data, features, top_n=3)
    assert len(recs) == 3


def test_shrink_keeps_values_intact():
    """category 化和数值降位只为省内存，不能改变数据。"""
    from model.recommendation import _shrink
    raw = pd.read_csv("vehicle.csv.gz")
    shrunk = _shrink(raw.copy())
    assert len(shrunk) == len(raw)
    assert (shrunk["price"].astype("int64") == raw["price"].astype("int64")).all()
    assert (shrunk["Brand"].astype(str) == raw["Brand"].astype(str)).all()
