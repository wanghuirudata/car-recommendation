"""现有特征能达到的误差下限。

「模型不够准，是因为特征太少」——这个判断可以被验证。

如果两辆车在**所有现有特征**上取值相同，模型必然给出同一个预测；它们真实
价格的差异，任何模型都消不掉。把这些同特征组内的价格离散度汇总起来，就得到
当前特征集下的误差下限（近似的贝叶斯误差）。

    模型实际误差 = 建模还能改进的部分 + 现有特征无法解释的部分

下限接近实际误差，说明模型已经把现有特征用尽，剩下的只能靠采集新特征；
下限远低于实际误差，说明建模本身还有空间。

用法：
    python analysis/noise_floor.py
"""

import numpy as np
import pandas as pd

from model.regressor import CATEGORICAL_FEATURES, PASSTHROUGH_FEATURES

MILEAGE_BIN = 5000          # 里程分箱：连续值上精确重复几乎不存在
ACHIEVED_MAPE = 6.7         # 当前留出集表现，见 README


def main():
    data = pd.read_csv('vehicle.csv.gz')
    data['model'] = data['model'].astype(str).str.strip()
    data['mileage_bin'] = (data['mileage'] // MILEAGE_BIN) * MILEAGE_BIN

    # 「模型眼中相同」= 所有类别特征 + 年份 + 排量 + 里程分箱一致
    keys = (CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
            + ['year', 'engineSize', 'mileage_bin'])
    groups = data.groupby(keys, observed=True)['price']

    sizes = groups.size()
    twins = sizes[sizes >= 2]
    covered = int(sizes[sizes >= 2].sum())

    print(f"listings                 : {len(data):,}")
    print(f"groups of identical cars : {len(twins):,}")
    print(f"listings inside a group  : {covered:,} ({covered/len(data)*100:.0f}%)")
    print(f"(identical = same {', '.join(keys[:5])}, … and mileage within "
          f"{MILEAGE_BIN:,} miles)\n")

    # 组内每条挂牌 vs 组中位数的相对偏差 —— 这正是模型无法区分的部分
    medians = groups.transform('median')
    counts = groups.transform('size')
    inside = data[counts >= 2].copy()
    inside['rel_dev'] = (inside['price'] - medians[counts >= 2]).abs() / medians[counts >= 2]

    floor = inside['rel_dev'].mean() * 100
    print(f"noise floor (MAPE)       : {floor:.1f}%")
    print(f"model achieves (MAPE)    : {ACHIEVED_MAPE:.1f}%")
    print(f"headroom from modelling  : {max(ACHIEVED_MAPE - floor, 0):.1f} points")
    print(f"unexplainable by current features: {floor:.1f} points "
          f"({floor/ACHIEVED_MAPE*100:.0f}% of current error)\n")

    print("同特征组内价差最大的几组（记录到的特征完全一致，价格却差很多）：")
    spread = (groups.max() - groups.min()).sort_values(ascending=False)
    field = {name: i for i, name in enumerate(keys)}
    for key, gap in spread.head(5).items():
        brand = key[field['Brand']]
        model_name = key[field['model']]
        year = key[field['year']]
        print(f"  {brand} {model_name} {year}: 价差 GBP{gap:,.0f}")

    print("\n按价格段看下限：")
    inside['band'] = pd.cut(inside['price'], [0, 7500, 12500, 20000, 30000, np.inf],
                            labels=['<7.5k', '7.5-12.5k', '12.5-20k', '20-30k', '>30k'])
    table = (inside.groupby('band', observed=True)['rel_dev']
             .agg(['size', 'mean']).rename(columns={'size': 'n', 'mean': 'floor'}))
    table['floor'] = table['floor'] * 100
    print(table.to_string(float_format=lambda v: f'{v:,.1f}'))


if __name__ == '__main__':
    main()
