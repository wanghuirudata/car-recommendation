"""训练并评估价格模型。

用法：
    python train_model.py              # 只训练并保存最终模型
    python train_model.py --benchmark  # 对比多个模型，打印指标表后保存最优的

评估口径：20% 留出集，MAE / MAPE / RMSE / R2。主指标看 MAPE
（用户感知的是「估价差了百分之几」，不是「差了多少英镑」）。
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from model.regressor import (FEATURE_COLUMNS, MODEL_PATH, TARGET, Model,
                             build_model, build_preprocessor)

DATA_PATH = 'vehicle.csv'
RANDOM_STATE = 42


def evaluate(name, model, X_train, X_test, y_train, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    pred = np.clip(model.predict(X_test), 1, None)
    return {
        'model': name,
        'MAE': mean_absolute_error(y_test, pred),
        'MAPE': np.mean(np.abs((y_test - pred) / y_test)) * 100,
        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        'R2': r2_score(y_test, pred),
        'fit_s': fit_seconds,
    }


def log_target(regressor):
    """把任意回归器包成「在 log 空间拟合」的版本。"""
    return TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def candidates():
    """返回 (名称, pipeline) 列表，从最朴素的基线到最终模型。"""
    from lightgbm import LGBMRegressor

    def pipe(reg):
        return Pipeline([('preprocessor', build_preprocessor()), ('regressor', reg)])

    lgbm_kwargs = dict(n_estimators=800, learning_rate=0.05, num_leaves=63,
                       min_child_samples=20, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, random_state=RANDOM_STATE,
                       n_jobs=-1, verbose=-1)

    return [
        ('Baseline (predict median)', pipe(DummyRegressor(strategy='median'))),
        ('Ridge regression', pipe(Ridge(alpha=1.0))),
        ('RandomForest (original)',
         pipe(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))),
        ('LightGBM (raw target)', pipe(LGBMRegressor(**lgbm_kwargs))),
        ('LightGBM + log1p target', pipe(log_target(LGBMRegressor(**lgbm_kwargs)))),
    ]


def print_table(rows):
    header = f"{'model':<28}{'MAE':>10}{'MAPE':>9}{'RMSE':>10}{'R2':>8}{'fit(s)':>9}"
    print('\n' + header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['model']:<28}{r['MAE']:>10,.0f}{r['MAPE']:>8.1f}%"
              f"{r['RMSE']:>10,.0f}{r['R2']:>8.3f}{r['fit_s']:>9.1f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true',
                        help='对比所有候选模型并打印指标表')
    args = parser.parse_args()

    data = pd.read_csv(DATA_PATH)
    X, y = data[FEATURE_COLUMNS], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"train={len(X_train):,}  test={len(X_test):,}  features={len(FEATURE_COLUMNS)}")

    if args.benchmark:
        rows = []
        for name, model in candidates():
            print(f"  fitting {name} ...")
            rows.append(evaluate(name, model, X_train, X_test, y_train, y_test))
        print_table(rows)

    # 最终模型：先在留出集上报指标，再用全量数据重训后保存
    final = evaluate('LightGBM + log1p (final)', build_model(),
                     X_train, X_test, y_train, y_test)
    print(f"held-out: MAE GBP{final['MAE']:,.0f}  MAPE {final['MAPE']:.1f}%  "
          f"R2 {final['R2']:.3f}")

    Model.train_and_save_model(data)
    size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print(f"saved {MODEL_PATH} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
