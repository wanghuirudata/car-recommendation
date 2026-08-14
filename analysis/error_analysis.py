"""价格模型的误差分析。

一个全局 MAPE 回答不了「模型什么时候会失败」。这个脚本在同一个留出集上
按价格段、车龄、品牌、里程、燃料类型拆开误差，并检查系统性偏差
（bias，即预测是否整体偏高或偏低）。

用法：
    python analysis/error_analysis.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model.regressor import FEATURE_COLUMNS, TARGET, build_model

DATA_PATH = 'vehicle.csv.gz'
RANDOM_STATE = 42
REFERENCE_YEAR = 2020          # 数据集中最新的年份


def segment_report(frame, by, label, min_count=200):
    """按 by 分组统计误差。样本太少的组不报，避免读出噪声。"""
    rows = []
    for name, group in frame.groupby(by, observed=True):
        if len(group) < min_count:
            continue
        rows.append({
            label: name,
            'n': len(group),
            'MAE': group['abs_err'].mean(),
            'MAPE%': group['pct_err'].abs().mean() * 100,
            'bias%': group['pct_err'].median() * 100,      # 负=系统性低估
            'within10%': (group['pct_err'].abs() <= 0.10).mean() * 100,
        })
    return pd.DataFrame(rows).sort_values('MAPE%', ascending=False)


def show(title, frame, fmt=None):
    print(f"\n{title}")
    print('-' * 78)
    print(frame.to_string(index=False, float_format=lambda v: f'{v:,.1f}'))


def main():
    data = pd.read_csv(DATA_PATH)
    X, y = data[FEATURE_COLUMNS], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 必须在这里重新训练，不能直接加载 model/price_model.pkl ——
    # 那个模型是用全量数据训的（服务端要用上所有数据），测试集的行它见过。
    # 拿它做误差分析会得到偏乐观的结果（实测 6.7% vs 真实 7.2%）。
    model = build_model()
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 1, None)

    df = X_test.copy()
    df['actual'] = y_test
    df['pred'] = pred
    df['abs_err'] = (df['pred'] - df['actual']).abs()
    df['pct_err'] = (df['pred'] - df['actual']) / df['actual']
    df['age'] = REFERENCE_YEAR - df['year']

    print(f"held-out rows: {len(df):,}")
    print(f"overall  MAE GBP{df['abs_err'].mean():,.0f}   "
          f"MAPE {df['pct_err'].abs().mean()*100:.1f}%   "
          f"bias {df['pct_err'].median()*100:+.1f}%   "
          f"within 10% {(df['pct_err'].abs()<=0.10).mean()*100:.0f}%")

    df['price_band'] = pd.cut(
        df['actual'], [0, 7500, 12500, 20000, 30000, np.inf],
        labels=['<7.5k', '7.5-12.5k', '12.5-20k', '20-30k', '>30k'])
    df['age_band'] = pd.cut(
        df['age'], [-1, 1, 3, 6, 10, np.inf],
        labels=['0-1y', '2-3y', '4-6y', '7-10y', '>10y'])
    df['mileage_band'] = pd.cut(
        df['mileage'], [-1, 10000, 30000, 60000, 100000, np.inf],
        labels=['<10k', '10-30k', '30-60k', '60-100k', '>100k'])

    show('BY PRICE BAND', segment_report(df, 'price_band', 'price_band'))
    show('BY VEHICLE AGE', segment_report(df, 'age_band', 'age'))
    show('BY MILEAGE', segment_report(df, 'mileage_band', 'mileage'))
    show('BY BRAND', segment_report(df, 'Brand', 'brand'))
    show('BY FUEL TYPE', segment_report(df, 'fuelType', 'fuel'))

    worst = df.nlargest(8, 'abs_err')[
        ['Brand', 'year', 'mileage', 'engineSize', 'actual', 'pred', 'pct_err']]
    worst['pct_err'] = worst['pct_err'] * 100
    show('WORST 8 ABSOLUTE ERRORS', worst)

    tail = (df['pct_err'].abs() > 0.30).mean() * 100
    print(f"\npredictions off by more than 30%: {tail:.1f}% of the held-out set")


if __name__ == '__main__':
    main()
