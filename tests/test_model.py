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
    "Brand": "Ford", "Car_Type": "Hatchback", "model": "Fiesta",
    "High_Performance": 0,
}


def test_feature_columns_are_explicit():
    """显式列表是防线：CSV 后来新增的展示列不能悄悄流进模型。

    注意 model 是特征、image_url 不是 —— 两列同一天为了页面显示被加进 CSV，
    但只有前者对价格有解释力。当初的白名单把两者一起挡在了外面。
    """
    assert FEATURE_COLUMNS == NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
    assert "model" in FEATURE_COLUMNS
    for leaked in ("price", "vehicle_id", "image_url"):
        assert leaked not in FEATURE_COLUMNS


def test_model_name_whitespace_is_normalised():
    """CSV 里是 ' Fiesta'（带前导空格），表单传的是 'Fiesta'。

    两侧必须用同一套清洗，否则服务端的值会被 OneHotEncoder 当作未知类别
    静默丢弃——不报错，只是悄悄退回到没有这个特征的水平。
    """
    from model.regressor import normalise
    padded = pd.DataFrame([dict(SPEC, model=" Fiesta ")])
    assert normalise(padded)["model"].iloc[0] == "Fiesta"
    assert Model.car_price(padded) == Model.car_price(pd.DataFrame([SPEC]))


def test_model_name_carries_real_weight():
    """model 若没被模型用上，加这一列就没有意义。

    这里断言特征重要性而不是'换个车型名预测就该变'：把 engineSize、
    fuelType、Car_Type 固定住只换名字，等于在问'1.0L 汽油两厢的 BMW i8
    值多少钱'——这车不存在，树会先按数值特征分裂，路径上根本走不到车型。
    第一版测试就是这么写的，失败的是测试不是模型。
    """
    from model.regressor import get_model
    pipeline = get_model()
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pipeline.named_steps["regressor"].regressor_.feature_importances_
    share = sum(v for n, v in zip(names, importance) if "__model_" in n) / importance.sum()
    assert share > 0.05, f"model 特征组只占 {share:.1%}，可能没有真正生效"


def test_distinguishes_trim_levels_on_real_specs():
    """同品牌下，用数据里真实存在的两台车对比：贵的那台应当预测更贵。"""
    frame = pd.read_csv("vehicle.csv.gz")
    frame["model"] = frame["model"].astype(str).str.strip()
    cheap = frame[(frame.Brand == "BMW") & (frame.model == "1 Series")].iloc[[0]]
    dear = frame[(frame.Brand == "BMW") & (frame.model == "i8")].iloc[[0]]
    assert Model.car_price(dear) > Model.car_price(cheap) * 1.5


def test_prediction_is_plausible():
    price = Model.car_price(pd.DataFrame([SPEC]))
    assert price is not None
    assert 3000 < price < 40000


def test_prediction_is_deterministic():
    frame = pd.DataFrame([SPEC])
    assert Model.car_price(frame) == Model.car_price(frame)


def test_extra_columns_are_ignored():
    """多传非特征字段不应影响结果——car_price 按 FEATURE_COLUMNS 取子集。"""
    noisy = dict(SPEC, vehicle_id=123, image_url="/x.jpg", price=99999)
    assert Model.car_price(pd.DataFrame([noisy])) == Model.car_price(pd.DataFrame([SPEC]))


def test_missing_column_is_reported_not_crashed():
    incomplete = {k: v for k, v in SPEC.items() if k != "mileage"}
    assert Model.car_price(pd.DataFrame([incomplete])) is None


def test_interval_brackets_the_point_estimate():
    low, point, high = Model.car_price_interval(pd.DataFrame([SPEC]))
    assert low is not None and high is not None
    assert low <= point <= high


def test_interval_is_wider_for_rare_cars():
    """区间宽度要携带信息：数据少的车型应当明显更宽，否则区间只是装饰。"""
    frame = pd.read_csv("vehicle.csv.gz")
    frame["model"] = frame["model"].astype(str).str.strip()

    def relative_width(brand, name):
        row = frame[(frame.Brand == brand) & (frame.model == name)].iloc[[0]]
        low, point, high = Model.car_price_interval(row)
        return (high - low) / point

    common = relative_width("Ford", "Fiesta")      # 数据集中数量最多的车型之一
    rare = relative_width("BMW", "i8")             # 全数据集只有 6 台
    assert rare > common


def test_interval_degrades_gracefully(monkeypatch):
    """区间模型缺失时只丢区间，点估计必须照常返回。"""
    import model.regressor as regressor
    monkeypatch.setattr(regressor, "_interval_models", None)
    monkeypatch.setattr(regressor, "INTERVAL_MODEL_PATH", "/nonexistent.pkl")
    low, point, high = Model.car_price_interval(pd.DataFrame([SPEC]))
    assert low is None and high is None
    assert point is not None


def test_mileage_lowers_price():
    """基本单调性：同款车里程更高，估价应更低。"""
    low = Model.car_price(pd.DataFrame([dict(SPEC, mileage=10000)]))
    high = Model.car_price(pd.DataFrame([dict(SPEC, mileage=120000)]))
    assert high < low


def test_newer_year_raises_price():
    old = Model.car_price(pd.DataFrame([dict(SPEC, year=2012)]))
    new = Model.car_price(pd.DataFrame([dict(SPEC, year=2020)]))
    assert new > old
