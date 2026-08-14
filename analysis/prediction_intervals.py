"""预测区间：分位数回归，并检验它是否真的校准。

一个点估计 £30,210 暗示了它没有的确定性。对二手车来说，
「£30,210，大概率 £24,000–£38,000」才是能拿来做决策的输出。

做法是训练三个 LightGBM：q=0.1 / 0.5 / 0.9（pinball loss）。目标仍在
log 空间——分位数在单调变换下不变，所以 expm1(log 空间的 q90) 就是
价格空间的 q90，这一步是严格成立的，不是近似。

关键检验是**覆盖率**：声称 80% 的区间，实际要盖住约 80% 的真实价格。
不校准的区间比没有区间更糟，因为它给了虚假的安全感。

用法：
    python analysis/prediction_intervals.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model.regressor import FEATURE_COLUMNS, TARGET, build_quantile_model, normalise

RANDOM_STATE = 42
REFERENCE_YEAR = 2020
QUANTILES = (0.1, 0.5, 0.9)          # 名义覆盖率 80%


def main():
    data = normalise(pd.read_csv('vehicle.csv.gz'))
    X, y = data[FEATURE_COLUMNS], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    preds = {}
    for q in QUANTILES:
        model = build_quantile_model(q)
        model.fit(X_train, y_train)
        preds[q] = np.clip(model.predict(X_test), 1, None)

    low, mid, high = preds[0.1], preds[0.5], preds[0.9]
    # 分位数回归是三个独立模型，理论上可能交叉；排序保证区间有效
    low, high = np.minimum(low, high), np.maximum(low, high)

    covered = (y_test >= low) & (y_test <= high)
    width = high - low

    print(f"held-out rows : {len(y_test):,}")
    print(f"nominal cover : {int((QUANTILES[-1]-QUANTILES[0])*100)}%")
    print(f"actual cover  : {covered.mean()*100:.1f}%")
    print(f"median width  : GBP{np.median(width):,.0f} "
          f"({np.median(width / mid)*100:.0f}% of the point estimate)")
    print(f"median |err|  : GBP{np.abs(mid - y_test).median():,.0f}")

    df = pd.DataFrame({
        'actual': y_test.values, 'low': low, 'mid': mid, 'high': high,
        'covered': covered.values, 'width': width,
        'age': REFERENCE_YEAR - X_test['year'].values,
        'Brand': X_test['Brand'].values,
    })
    df['age_band'] = pd.cut(df['age'], [-1, 1, 3, 6, 10, np.inf],
                            labels=['0-1y', '2-3y', '4-6y', '7-10y', '>10y'])
    df['price_band'] = pd.cut(df['actual'], [0, 7500, 12500, 20000, 30000, np.inf],
                              labels=['<7.5k', '7.5-12.5k', '12.5-20k', '20-30k', '>30k'])

    for by in ('age_band', 'price_band', 'Brand'):
        rows = []
        for name, group in df.groupby(by, observed=True):
            if len(group) < 200:
                continue
            rows.append({by: name, 'n': len(group),
                         'cover%': group['covered'].mean() * 100,
                         'width': group['width'].median(),
                         'width%': (group['width'] / group['mid']).median() * 100})
        table = pd.DataFrame(rows).sort_values('width%', ascending=False)
        print(f"\nBY {by.upper()}")
        print('-' * 56)
        print(table.to_string(index=False, float_format=lambda v: f'{v:,.1f}'))

    print("\n区间宽度是否随不确定性变化（越不确定应当越宽）：")
    q = df['width'] / df['mid']
    print(f"  最窄的 10%: {q.quantile(0.1)*100:.0f}% of estimate")
    print(f"  最宽的 10%: {q.quantile(0.9)*100:.0f}% of estimate")

    # 名义覆盖率和实测覆盖率不一致 —— 分位数回归本身不保证校准。
    # 扫一遍档位，看哪一组能真正达到 80%。
    print("\n=== 校准：名义档位 vs 实测覆盖率 ===")
    print(f"{'nominal':>9}{'actual':>9}{'median width':>15}{'width%':>9}")
    print('-' * 42)
    for lo in (0.10, 0.075, 0.05, 0.025):
        hi = 1 - lo
        bounds = {}
        for qq in (lo, hi):
            m = build_quantile_model(qq)
            m.fit(X_train, y_train)
            bounds[qq] = np.clip(m.predict(X_test), 1, None)
        a, b = np.minimum(bounds[lo], bounds[hi]), np.maximum(bounds[lo], bounds[hi])
        cov = ((y_test >= a) & (y_test <= b)).mean() * 100
        w = np.median(b - a)
        print(f"{(hi-lo)*100:>8.0f}%{cov:>8.1f}%{w:>14,.0f}"
              f"{np.median((b-a)/mid)*100:>8.0f}%")


if __name__ == '__main__':
    main()
