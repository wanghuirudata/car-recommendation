"""二手车价格回归模型。

设计要点：

1. 目标做 log1p 变换。price 右偏（skew=1.15，£450 ~ £50,000），直接对原始
   价格做平方误差回归会让贵车主导损失函数，便宜车被系统性高估。log 空间里
   优化的近似是「相对误差」，这正是业务关心的口径（「估价差 8%」比
   「估价差 £1,800」有意义）。

2. 主指标用 MAE 和 MAPE，不用 RMSE。用户感知的是「差了百分之几」，
   而 RMSE 会被少数豪车的绝对误差放大。

3. LightGBM 替代 RandomForest。原来的 RF 不限深度，在 107k 行上完全生长，
   产出 874MB 的 pkl；LightGBM 精度更好、体积小三个数量级，才有可能部署到
   免费平台。对比数据见 train_model.py 的输出和 README。
"""

import os
import threading

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "price_model.pkl")
INTERVAL_MODEL_PATH = os.path.join(_HERE, "price_interval_models.pkl")

# 用名义 85% 的分位数来给出 80% 的区间。
# 分位数回归不保证校准，实测每档都欠覆盖约 4 个百分点：
#   名义 80% -> 实测 76.3%      名义 90% -> 实测 86.4%
#   名义 85% -> 实测 81.3%      名义 95% -> 实测 92.1%
# 所以标称 80% 的区间取 0.075/0.925，实测覆盖 81.3%。
# 复现见 analysis/prediction_intervals.py。
QUANTILE_LEVELS = (0.075, 0.925)
NOMINAL_COVERAGE = 0.80

# 建模用的列。显式列出，避免 CSV 里的 vehicle_id / model / image_url
# 被 remainder='passthrough' 混进特征里。
NUMERIC_FEATURES = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
# 'model' 是具体车型名（192 个取值）。它长期不在特征里 —— 该列 2024/11 才为了
# 在页面上显示车名加进 CSV，而当初把特征改成显式白名单，正是为了把同批加进来的
# image_url 挡在外面（字符串列会让训练直接报错）。那个修复是对的，但顺手也把
# model 锁在了门外。
#
# 误差分析发现最大的几个误差都源于此：BMW i8（1.5L 混动超跑，£48,898）被预测成
# £24,380，因为 Brand+Car_Type+engineSize 无法区分它和 1 系 —— 同一组合内价格
# 最大能差 £48,645。加上这一列后 MAE 降 8.2%，>30% 的离谱误差少了三分之一。
# 复现见 analysis/feature_experiment.py。
CATEGORICAL_FEATURES = ['transmission', 'fuelType', 'Brand', 'Car_Type', 'model']
PASSTHROUGH_FEATURES = ['High_Performance']
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
TARGET = 'price'


def normalise(frame):
    """训练和推理共用的清洗。

    CSV 里的 model 值带前导空格（' Fiesta'）。两侧必须用同一套处理，
    否则服务端传 'Fiesta' 会匹配不上训练时的 ' Fiesta'，被 OneHotEncoder
    当成未知类别静默丢弃 —— 不报错，只是悄悄变差。
    """
    frame = frame.copy()
    if 'model' in frame.columns:
        frame['model'] = frame['model'].astype(str).str.strip()
    return frame

_model = None
_interval_models = None
_model_lock = threading.Lock()


def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False,
                                 handle_unknown='ignore')),
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder='passthrough',
    )


def build_model():
    """LightGBM + log1p 目标变换。

    超参是在 20% 留出集上手工调的，见 train_model.py --tune 的输出：
    再往上加 n_estimators 收益已经进入千分位。
    """
    regressor = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        # log1p / expm1：在对数空间拟合，预测时自动变换回英镑
        ('regressor', TransformedTargetRegressor(
            regressor=regressor, func=np.log1p, inverse_func=np.expm1)),
    ])


def build_quantile_model(quantile):
    """分位数回归：同样的预处理，把损失换成 pinball loss。

    目标依然做 log1p 变换 —— 分位数在单调变换下不变，所以
    expm1(log 空间的 q90) 严格等于价格空间的 q90，不是近似。
    """
    regressor = LGBMRegressor(
        objective='quantile', alpha=quantile,
        n_estimators=800, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
    )
    return Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('regressor', TransformedTargetRegressor(
            regressor=regressor, func=np.log1p, inverse_func=np.expm1)),
    ])


def get_model():
    """进程内只加载一次模型。

    原实现每次预测都 joblib.load 一遍 874MB 的 pkl，单次请求要 0.8 秒并且
    反复申请释放大块内存。这里做惰性加载 + 双重检查锁。
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(
                        f"Model file not found: {MODEL_PATH}. "
                        f"Run `python train_model.py` first."
                    )
                _model = joblib.load(MODEL_PATH)
    return _model


def get_interval_models():
    """区间模型（低/高分位）。缺失时返回 None —— 只降级区间，点估计照常。"""
    global _interval_models
    if _interval_models is None:
        with _model_lock:
            if _interval_models is None:
                if not os.path.exists(INTERVAL_MODEL_PATH):
                    return None
                _interval_models = joblib.load(INTERVAL_MODEL_PATH)
    return _interval_models


class Model:
    @staticmethod
    def train_and_save_model(data, model=None, path=MODEL_PATH):
        missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in data.columns]
        if missing:
            raise ValueError(f"Training data is missing columns: {missing}")

        model = model if model is not None else build_model()
        data = normalise(data)
        model.fit(data[FEATURE_COLUMNS], data[TARGET])
        joblib.dump(model, path, compress=3)

        global _model
        _model = model
        return model

    @staticmethod
    def train_interval_models(data, path=INTERVAL_MODEL_PATH):
        """训练并保存两个分位数模型，用于给出预测区间。"""
        data = normalise(data)
        X, y = data[FEATURE_COLUMNS], data[TARGET]
        models = {}
        for quantile in QUANTILE_LEVELS:
            model = build_quantile_model(quantile)
            model.fit(X, y)
            models[quantile] = model
        joblib.dump(models, path, compress=3)

        global _interval_models
        _interval_models = models
        return models

    @staticmethod
    def car_price(input_data):
        try:
            missing = [c for c in FEATURE_COLUMNS if c not in input_data.columns]
            if missing:
                raise ValueError(f"Input is missing columns: {missing}")

            prediction = get_model().predict(normalise(input_data)[FEATURE_COLUMNS])
            return float(prediction[0])
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    @staticmethod
    def car_price_interval(input_data):
        """返回 (low, point, high)。区间模型不可用时返回 (None, point, None)。

        一个点估计暗示了它没有的确定性。区间宽度本身也是信息：老车、
        稀有车型的区间明显更宽，等于模型在说「这台我没什么把握」。
        """
        point = Model.car_price(input_data)
        if point is None:
            return None, None, None

        models = get_interval_models()
        if not models:
            return None, point, None

        try:
            frame = normalise(input_data)[FEATURE_COLUMNS]
            low_q, high_q = QUANTILE_LEVELS
            low = float(models[low_q].predict(frame)[0])
            high = float(models[high_q].predict(frame)[0])
            # 三个模型互相独立，理论上分位数可能交叉；排序保证区间有效
            low, high = min(low, high), max(low, high)
            # 点估计落在区间外时把区间撑开，避免出现自相矛盾的展示
            return min(low, point), point, max(high, point)
        except Exception as e:
            print(f"Interval prediction failed: {e}")
            return None, point, None
