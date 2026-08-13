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

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_model.pkl")

# 建模用的列。显式列出，避免 CSV 里的 vehicle_id / model / image_url
# 被 remainder='passthrough' 混进特征里。
NUMERIC_FEATURES = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
CATEGORICAL_FEATURES = ['transmission', 'fuelType', 'Brand', 'Car_Type']
PASSTHROUGH_FEATURES = ['High_Performance']
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
TARGET = 'price'

_model = None
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


class Model:
    @staticmethod
    def train_and_save_model(data, model=None, path=MODEL_PATH):
        missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in data.columns]
        if missing:
            raise ValueError(f"Training data is missing columns: {missing}")

        model = model if model is not None else build_model()
        model.fit(data[FEATURE_COLUMNS], data[TARGET])
        joblib.dump(model, path, compress=3)

        global _model
        _model = model
        return model

    @staticmethod
    def car_price(input_data):
        try:
            missing = [c for c in FEATURE_COLUMNS if c not in input_data.columns]
            if missing:
                raise ValueError(f"Input is missing columns: {missing}")

            prediction = get_model().predict(input_data[FEATURE_COLUMNS])
            return float(prediction[0])
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
