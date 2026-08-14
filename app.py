import hashlib
import os
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
from flask import (Flask, abort, flash, jsonify, redirect, render_template,
                   request, session, url_for)

from dashboard import create_dash_app
from model.assistant import VehicleAssistant
from model.recommendation import (get_better_value_alternatives,
                                  get_similar_vehicles, load_and_prepare_data)
from model.regressor import Model

# 加载和处理数据
data, vehicle_features = load_and_prepare_data("vehicle.csv.gz")
data['vehicle_id'] = data.index

app = Flask(__name__)
# 生产环境请通过环境变量注入；未设置时用随机值，重启后 session 失效但不会泄露密钥
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24).hex()


# 购车助手。原来这里接的是 Chatbase 的浏览器端 embed，免费层 bot 闲置 14 天
# 会被删除（实测那个 bot 已返回 404），且一个 script 标签体现不出任何设计。
# 改成自建：工具调用直接复用本项目的搜索 / 估价 / 推荐三块能力。
assistant = VehicleAssistant(data, vehicle_features)


@app.route('/api/chat', methods=['POST'])
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get('message') or '').strip()
    if not message:
        return jsonify({'error': 'message is required'}), 400
    if len(message) > 1000:
        return jsonify({'error': 'message too long'}), 400

    history = session.get('chat_history', [])
    result = assistant.reply(message, history)

    # 只在 session 里保留纯文本往返，且限制长度：
    # cookie session 有 4KB 上限，塞工具调用块会直接溢出
    history = (history + [
        {'role': 'user', 'content': message},
        {'role': 'assistant', 'content': result['reply'] or '(no text)'},
    ])[-6:]
    session['chat_history'] = history

    return jsonify(result)


@app.route('/api/chat/reset', methods=['POST'])
def chat_reset():
    session.pop('chat_history', None)
    return jsonify({'status': 'ok'})


# 添加登录要求装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('login required')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    images = [
        "static/logo.png",
        "static/img2.jpg",
        "static/img3.jpg"
    ]
    return render_template("index.html" , images=images)


# 演示用的单账号登录。凭据从环境变量读取，默认值仅供本地演示。
# 注意：SHA256 无盐不适合真实用户系统，真实项目应换成 werkzeug 的
# generate_password_hash / check_password_hash（bcrypt/scrypt）。
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')
ADMIN_PASSWORD_HASH = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if username == ADMIN_USERNAME and hashed_password == ADMIN_PASSWORD_HASH:
            session['username'] = username
            session['login_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return redirect(url_for('dashboard'))
        else:
            flash('username or password error')
    return render_template('login.html')



@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session['username'])


@app.route('/logout')
def logout():
    session.pop('username', None)  
    return redirect(url_for('login'))  


# @app.route('/purchase')
# def purchase():
#     return render_template('purchase.html')


@app.route('/sales', methods=['GET'])
def sales():
    # Get search parameters with default values
    query = request.args.get('query', '').strip()
    price_min = request.args.get('price_min', type=float, default=0)
    price_max = request.args.get('price_max', type=float, default=float('inf'))
    brand = request.args.get('brand', default=None)
    year_min = request.args.get('year_min', type=int, default=0)
    sort_by = request.args.get('sort_by', default='price')
    
    # 添加分页参数
    page = request.args.get('page', type=int, default=1)
    per_page = 9  # 每页显示 9 辆车
    
    # Start with full dataset
    filtered_data = data.copy()
    
    # Apply filters
    if query:
        # regex=False：用户输入 "(" / "*" 等正则元字符时不会抛异常
        filtered_data = filtered_data[
            filtered_data['Brand'].str.contains(query, case=False, regex=False, na=False) |
            filtered_data['Car_Type'].str.contains(query, case=False, regex=False, na=False)
        ]
    if price_min:
        filtered_data = filtered_data[filtered_data['price'] >= price_min]
    if price_max != float('inf'):
        filtered_data = filtered_data[filtered_data['price'] <= price_max]
    if brand:
        filtered_data = filtered_data[filtered_data['Brand'] == brand]
    if year_min:
        filtered_data = filtered_data[filtered_data['year'] >= year_min]
    
    # Sort the results
    if sort_by == 'price_asc':
        filtered_data = filtered_data.sort_values('price')
    elif sort_by == 'price_desc':
        filtered_data = filtered_data.sort_values('price', ascending=False)
    elif sort_by == 'year':
        filtered_data = filtered_data.sort_values('year', ascending=False)
    
    # 计算分页
    total_items = len(filtered_data)
    total_pages = (total_items + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # 获取当前页的数据
    paginated_data = filtered_data.iloc[start_idx:end_idx]
    
    # Get recommendations
    # 筛选结果为空时 np.random.choice 会抛 ValueError，这里直接给一个空的推荐区
    recommendations = data.iloc[0:0]
    if not filtered_data.empty:
        vehicle_id = session.get('last_viewed_vehicle_id')
        if vehicle_id is None or vehicle_id not in data.index:
            vehicle_id = int(np.random.choice(filtered_data.index))
        recommendations = get_similar_vehicles(vehicle_id, data, vehicle_features, top_n=3)

    # Get statistics for the filtered results
    stats = {
        'count': total_items,
        'avg_price': filtered_data['price'].mean(),
        'min_price': filtered_data['price'].min(),
        'max_price': filtered_data['price'].max(),
    }
    
    # Get unique values for filters
    available_brands = sorted(data['Brand'].unique())
    available_car_types = sorted(data['Car_Type'].unique())
    
    return render_template('sales.html', 
                         search_results=paginated_data,
                         recommendations=recommendations,
                         available_brands=available_brands,
                         available_car_types=available_car_types,
                         stats=stats,
                         current_page=page,
                         total_pages=total_pages,
                         search_params={
                             'query': query,
                             'price_min': price_min if price_min else '',
                             'price_max': price_max if price_max != float('inf') else '',
                             'brand': brand,
                             'year_min': year_min if year_min else '',
                             'sort_by': sort_by
                         })


@app.route('/vehicle/<int:vehicle_id>')
def vehicle_detail(vehicle_id):
    # 不存在的 id 直接 404，避免 data.loc[] 抛 KeyError 变成 500
    if vehicle_id not in data.index:
        abort(404)

    # 获取用户当前查看的车辆信息
    current_vehicle = data.loc[vehicle_id]

    # 两路召回：同类车 + 更划算的选择
    recommendations = get_similar_vehicles(vehicle_id, data, vehicle_features, top_n=3)
    better_value = get_better_value_alternatives(vehicle_id, data, vehicle_features, top_n=3)

    # 将当前车辆 ID 存入会话
    session['last_viewed_vehicle_id'] = vehicle_id

    return render_template('vehicle_detail.html',
                           vehicle=current_vehicle,
                           recommendations=recommendations,
                           better_value=better_value)




# @app.route('/sales')
# def sales():
#     # Get the last viewed vehicle ID
#     vehicle_id = session.get('last_viewed_vehicle_id', None)

#     if vehicle_id is not None:
#         # recommendations = content_based_recommendation(vehicle_id, data, similarity_matrix, num_recommendations=5)
#         recommendations = get_similar_items(vehicle_id, num_recommendations=5)
    
#     else:
#         # Randomly select a vehicle if no last-viewed vehicle is found
#         random_vehicle_id = np.random.randint(0, len(data))
#         # recommendations = content_based_recommendation(random_vehicle_id, data, similarity_matrix, num_recommendations=5)
#         recommendations = get_similar_items(vehicle_id, num_recommendations=5)

#     return render_template('sales.html', recommendations=recommendations)


 

# @app.route('/vehicle/<int:vehicle_id>')
# def vehicle_detail(vehicle_id):
#     # 获取用户当前查看的车辆信息
#     current_vehicle = data.loc[vehicle_id]

#     # 使用推荐函数生成推荐列表
#     recommendations = content_based_recommendation(vehicle_id, data, vehicle_features, num_recommendations=5)

#     # 将当前车辆 ID 存入会话
#     session['last_viewed_vehicle_id'] = vehicle_id

#     return render_template('vehicle_detail.html', vehicle=current_vehicle, recommendations=recommendations)




def _purchase_form_options():
    """表单选项由数据本身决定，不写死。

    原来的下拉是硬编码的，已经和数据脱节：列了 Nissan（数据集里没有），
    Car_Type 只列了 3 种（实际有 7 种）。选到数据里不存在的组合，
    OneHotEncoder 会当成未知类别静默忽略，估价照样出数但没有依据。
    """
    models_by_brand = (
        data.assign(model=data['model'].astype(str).str.strip())
            .groupby('Brand', observed=True)['model']
            .apply(lambda s: sorted(s.unique()))
            .to_dict()
    )
    return {
        'brands': sorted(models_by_brand),
        'models_by_brand': {str(k): v for k, v in models_by_brand.items()},
        'car_types': sorted(data['Car_Type'].astype(str).unique()),
        'fuel_types': sorted(data['fuelType'].astype(str).unique()),
        'transmissions': sorted(data['transmission'].astype(str).unique()),
    }


@app.route("/purchase", methods=["GET", "POST"])
def purchase():
    options = _purchase_form_options()

    if request.method == "POST":
        # 变量名沿用表单字段名（含 Brand / Car_Type 的大写），
        # 这样和 FEATURE_COLUMNS 一一对应，少一层心智映射
        spec = {
            'year': int(request.form["year"]),
            'transmission': request.form["transmission"],
            'mileage': int(request.form["mileage"]),
            'fuelType': request.form["fuelType"],
            'tax': float(request.form["tax"]),
            'mpg': float(request.form["mpg"]),
            'engineSize': float(request.form["engineSize"]),
            'Brand': request.form["Brand"],
            'Car_Type': request.form["Car_Type"],
            'model': request.form.get("model", "").strip(),
            'High_Performance': int(request.form["High_Performance"]),
        }

        predicted_price = Model.car_price(pd.DataFrame(spec, index=[0]))
        return render_template("purchase.html", predicted_price=predicted_price,
                               submitted=spec, **options)

    return render_template("purchase.html", **options)



# 说明：原来这里有一个 /search 路由，渲染的 search_results.html 并不存在
# （必然 TemplateNotFound → 500），且功能与 /sales 完全重复、没有任何页面链接到它，
# 已移除。搜索统一走 /sales。


# # 添加用户收藏功能
# @app.route('/favorite/<int:vehicle_id>', methods=['POST'])
# @login_required
# def add_favorite(vehicle_id):
#     if 'favorites' not in session:
#         session['favorites'] = []
#     if vehicle_id not in session['favorites']:
#         session['favorites'].append(vehicle_id)
#     return jsonify({'status': 'success'})

# 保留初始化部分
def init_dashboard(app):
    # 传入已加载的 data，避免 Dash 再读一遍 CSV
    dash_app = create_dash_app(app, data)
    return dash_app

# 初始化 dashboard
init_dashboard(app)

# def update_brand_name():
#     with engine.connect() as conn:
#         conn.execute(text("UPDATE vehicles SET Brand = 'Vauxhall-Opel' WHERE Brand = 'Vauxhall/Opel'"))
#         conn.commit()

if __name__ == "__main__":
    # debug 默认关闭；本地调试用 FLASK_DEBUG=1 python app.py
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1',
            port=int(os.environ.get('PORT', 5000)))
