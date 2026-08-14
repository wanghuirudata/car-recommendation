"""实验：把 model（具体车型名）加进特征会怎样。

误差分析显示最差的预测集中在同一类情况：BMW i8（1.5L 混动超跑，£48,898）
被当成 1 系，Audi R8 被当成普通 Audi，VW Caravelle 被当成 Golf。

原因是特征里只有 Brand + Car_Type + engineSize，没有具体车型。同一个
(Brand, Car_Type, engineSize) 组合内价格最大能差 £48,645 —— 模型没有任何
信息可以区分。

model 列有 192 个取值，是挂牌时就已知的合法属性，不构成泄漏。它当初没被
用上，只是因为它是后来为了页面显示才加进 CSV 的。

用法：
    python analysis/feature_experiment.py
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model.regressor import (CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                             PASSTHROUGH_FEATURES, TARGET)

RANDOM_STATE = 42
LGBM_KWARGS = dict(n_estimators=800, learning_rate=0.05, num_leaves=63,
                   min_child_samples=20, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, random_state=RANDOM_STATE,
                   n_jobs=-1, verbose=-1)


def build(categoricals):
    from lightgbm import LGBMRegressor
    pre = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), NUMERIC_FEATURES),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('onehot', OneHotEncoder(drop='first', sparse_output=False,
                                                   handle_unknown='ignore'))]), categoricals),
    ], remainder='passthrough')
    return Pipeline([('preprocessor', pre),
                     ('regressor', TransformedTargetRegressor(
                         regressor=LGBMRegressor(**LGBM_KWARGS),
                         func=np.log1p, inverse_func=np.expm1))])


def evaluate(name, columns, categoricals, data):
    X, y = data[columns], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = build(categoricals)
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 1, None)
    pct = np.abs((pred - y_test) / y_test)
    return {
        'variant': name,
        'MAE': mean_absolute_error(y_test, pred),
        'MAPE%': pct.mean() * 100,
        'R2': r2_score(y_test, pred),
        'within10%': (pct <= 0.10).mean() * 100,
        'off_by_30%+': (pct > 0.30).mean() * 100,
    }


def main():
    data = pd.read_csv('vehicle.csv.gz')
    data['model'] = data['model'].astype(str).str.strip()

    base_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
    rows = [
        evaluate('current (no model name)', base_cols, CATEGORICAL_FEATURES, data),
        evaluate('+ model name', base_cols + ['model'],
                 CATEGORICAL_FEATURES + ['model'], data),
    ]

    header = f"{'variant':<26}{'MAE':>9}{'MAPE%':>8}{'R2':>8}{'within10%':>11}{'off>30%':>9}"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['variant']:<26}{r['MAE']:>9,.0f}{r['MAPE%']:>7.1f}%"
              f"{r['R2']:>8.3f}{r['within10%']:>10.1f}%{r['off_by_30%+']:>8.1f}%")

    a, b = rows
    print(f"\nMAE   {a['MAE']:,.0f} -> {b['MAE']:,.0f}  "
          f"({(b['MAE']-a['MAE'])/a['MAE']*100:+.1f}%)")
    print(f"MAPE  {a['MAPE%']:.1f}% -> {b['MAPE%']:.1f}%")
    print(f"tail  {a['off_by_30%+']:.1f}% -> {b['off_by_30%+']:.1f}% of predictions off by >30%")


if __name__ == '__main__':
    main()
