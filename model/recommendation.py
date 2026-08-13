# recommendation.py
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(file_path):
    # 1. 首先读取CSV文件
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Failed to read {file_path}: {e}")
    
    # 2. 检查数据是否为空
    if data.empty:
        raise ValueError("Data file is empty")
    
    # 3. 定义特征列
    numeric_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize', 'price']
    categorical_features = ['transmission', 'fuelType', 'Brand', 'Car_Type']
    
    # 4. 验证所有必需的列是否存在
    missing_columns = [col for col in numeric_features + categorical_features if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 5. 数值特征：转数值 -> 中位数填充 -> 标准化
    numeric_data = data[numeric_features].apply(pd.to_numeric, errors='coerce').values
    numeric_data = SimpleImputer(strategy='median').fit_transform(numeric_data)
    numeric_data = StandardScaler().fit_transform(numeric_data)

    vehicle_features = pd.DataFrame(
        numeric_data, columns=numeric_features, index=data.index
    )

    # 6. 分类特征：独热编码后拼接
    # 注意：这里从头构造特征矩阵，而不是在 data 的副本上删删改改。
    # 原实现保留了 CSV 里的 model / image_url / vehicle_id 等非特征列，
    # 导致 .values 退化成 object dtype，np.dot 直接 TypeError。
    dummy_frames = [
        pd.get_dummies(data[feature].astype(str), prefix=feature)
        for feature in categorical_features
    ]
    vehicle_features = pd.concat([vehicle_features] + dummy_frames, axis=1)

    # 7. 统一成 float32，保证下游可以做纯数值运算
    vehicle_features = vehicle_features.astype(np.float32)

    return data, vehicle_features


# 说明：这里原本有一个 content_based_recommendation()，用 dask 计算全量 N×N
# 相似度矩阵。它有两个致命问题，且从未被调用，已删除：
#   1. 107343² × 4 字节 ≈ 46GB，必然 OOM；
#   2. pairwise_distances(metric='cosine') 返回的是「距离」，代码却按降序取
#      top-k，实际选出的是最不相似的车。
# 单条查询只需要一个向量对全表，见下面的 get_similar_vehicles()。


def get_similar_vehicles(vehicle_id, data, vehicle_features, top_n=3):
    """
    基于特征相似度推荐车辆
    """
    try:
        # 确保数据是numpy数组格式
        features_array = vehicle_features.values
        target_features = features_array[vehicle_id]
        
        # 计算余弦相似度
        dot_product = np.dot(features_array, target_features)
        norm_target = np.linalg.norm(target_features)
        norm_all = np.linalg.norm(features_array, axis=1)
        
        # 避免除以零
        similarity = np.zeros_like(norm_all)
        valid_indices = (norm_all != 0) & (norm_target != 0)
        similarity[valid_indices] = dot_product[valid_indices] / (norm_all[valid_indices] * norm_target)
        
        # 获取最相似的车辆（排除自身）
        similar_indices = np.argsort(similarity)[::-1]
        similar_indices = similar_indices[similar_indices != vehicle_id][:top_n]
        
        # 返回推荐车辆的详细信息
        recommendations = data.iloc[similar_indices]
        
        return recommendations
        
    except Exception as e:
        # 这里刻意只用 ASCII：Windows 控制台默认 cp1252，打印中文会再抛
        # UnicodeEncodeError，把兜底逻辑本身也冲掉，最终变成 500
        print(f"Recommendation failed, falling back to random: {e}")
        random_indices = np.random.choice(len(data), size=top_n, replace=False)
        return data.iloc[random_indices]

def collaborative_filtering_recommendation(user_id, user_vehicle_data, num_recommendations=3):
    # 假设我们有用户-车辆的交互数据，可以根据这个来生成推荐
    # 这是一个简化的占位函数
    return []
