"""价格模型：契约与合理性。

train / serve 用同一份 FEATURE_COLUMNS 是这里最关键的一条 ——
两者不一致正是当初 vehicle_id / model / image_url 混进特征的原因。
"""

import pandas as pd
import pytest

from model.regressor import (CATEGORICAL_FEATURES, FEATURE_COLUMNS,
                             NUMERIC_FEATURES, PASSTHROUGH_FEATURES, Model)

SPEC = {
    "year": 2017, "mileage": 15944, "tax": 150.0, "mpg": 57.7,
    "engineSize": 1.0, "transmission": "Automatic", "fuelType": "Petrol",
    "Brand": "Ford", "Car_Type": "Hatchback", "High_Performance": 0,
}


def test_feature_columns_are_explicit():
    """显式列表是防线：CSV 后来新增的展示列不能悄悄流进模型。"""
    assert FEATURE_COLUMNS == NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
    for leaked in ("price", "vehicle_id", "model", "image_url"):
        assert leaked not in FEATURE_COLUMNS


def test_prediction_is_plausible():
    price = Model.car_price(pd.DataFrame([SPEC]))
    assert price is not None
    assert 3000 < price < 40000


def test_prediction_is_deterministic():
    frame = pd.DataFrame([SPEC])
    assert Model.car_price(frame) == Model.car_price(frame)


def test_extra_columns_are_ignored():
    """purchase 表单之外多传字段不应影响结果——car_price 按列名取子集。"""
    noisy = dict(SPEC, vehicle_id=123, model=" Fiesta", image_url="/x.jpg")
    assert Model.car_price(pd.DataFrame([noisy])) == Model.car_price(pd.DataFrame([SPEC]))


def test_missing_column_is_reported_not_crashed():
    incomplete = {k: v for k, v in SPEC.items() if k != "mileage"}
    assert Model.car_price(pd.DataFrame([incomplete])) is None


def test_mileage_lowers_price():
    """基本单调性：同款车里程更高，估价应更低。"""
    low = Model.car_price(pd.DataFrame([dict(SPEC, mileage=10000)]))
    high = Model.car_price(pd.DataFrame([dict(SPEC, mileage=120000)]))
    assert high < low


def test_newer_year_raises_price():
    old = Model.car_price(pd.DataFrame([dict(SPEC, year=2012)]))
    new = Model.car_price(pd.DataFrame([dict(SPEC, year=2020)]))
    assert new > old
