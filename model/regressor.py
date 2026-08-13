import os
import threading

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf_model_pipeline.pkl")

# 建模用的列。显式列出，避免 CSV 里新增的 vehicle_id / model / image_url
# 被 remainder='passthrough' 混进特征里（训练会直接报错，服务端会静默错位）
NUMERIC_FEATURES = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
CATEGORICAL_FEATURES = ['transmission', 'fuelType', 'Brand', 'Car_Type']
PASSTHROUGH_FEATURES = ['High_Performance']
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
TARGET = 'price'

_model = None
_model_lock = threading.Lock()


def get_model():
    """进程内只加载一次模型。

    rf_model_pipeline.pkl 有 874MB，原来每次预测都 joblib.load 一遍，
    单次请求要十几秒并且可能 OOM。这里做惰性加载 + 双重检查锁。
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(
                        f"Model file not found: {MODEL_PATH}. Run `python train_model.py` first."
                    )
                _model = joblib.load(MODEL_PATH)
    return _model


class Model:
    @staticmethod
    def train_and_save_model(data):
        missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in data.columns]
        if missing:
            raise ValueError(f"Training data is missing columns: {missing}")

        # Create preprocessing pipelines for numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUMERIC_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)
            ],
            remainder='passthrough'
        )

        # Create a pipeline with preprocessor and random forest
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # 只取建模列，列顺序固定，保证训练与推理一致
        X = data[FEATURE_COLUMNS]
        y = data[TARGET]

        model.fit(X, y)

        joblib.dump(model, MODEL_PATH)

        # 训练完直接换掉进程内缓存，省一次磁盘加载
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
            print(f"Error in prediction: {str(e)}")
            return None
