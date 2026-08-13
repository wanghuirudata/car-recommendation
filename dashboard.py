from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

def create_dash_app(server, df=None):
    # 创建Dash应用
    dash_app = Dash(
        server=server,
        routes_pathname_prefix='/dashapp/',
        external_stylesheets=[
            '/static/css/style.css'
        ]
    )

    # 复用 app.py 已经加载好的 DataFrame。原来这里独立 read_csv 一次，
    # 同一份 107k 行数据在进程里存了两份。
    if df is None:
        df = pd.read_csv('vehicle.csv.gz')

    # 创建多个图表
    brand_count = df['Brand'].value_counts().head(10)
    brand_fig = px.bar(
        x=brand_count.index, 
        y=brand_count.values,
        title='Popular Brand Distribution'
    )

    price_hist = px.histogram(
        df, 
        x='price',
        title='Vehicle Price Distribution'
    )

    avg_price_by_year = df.groupby('year')['price'].mean().reset_index()
    year_price_fig = px.line(
        avg_price_by_year, 
        x='year', 
        y='price',
        title='Average Price Trend by Year'
    )

    # 定义布局
    dash_app.layout = html.Div([
        html.H1('Vehicle Data Analysis Dashboard', style={'textAlign': 'center'}),
        
        # 过滤器部分
        html.Div([
            html.Label('Select Brand:'),
            dcc.Dropdown(
                id='brand-dropdown',
                options=[{'label': brand, 'value': brand} for brand in sorted(df['Brand'].unique())],
                value=None,
                multi=True
            ),
            
            html.Label('Year Range:'),
            dcc.RangeSlider(
                id='year-slider',
                min=df['year'].min(),
                max=2024, #df['year'].max(),
                value=[df['year'].min(), 2024], #df['year'].max()],
                marks={str(year): str(year) for year in range(df['year'].min(), 2024 + 1, 2)}
            ),
        ], style={'padding': '20px'}),
        
        # 图表展示区
        html.Div([
            dcc.Graph(id='brand-distribution', figure=brand_fig),
            dcc.Graph(id='price-distribution', figure=price_hist),
            dcc.Graph(id='year-price-trend', figure=year_price_fig)
        ]),
        
        # 统计信息区
        html.Div([
            html.H3('Statistics'),
            html.Div(id='statistics-panel')
        ])
    ])

    # 添加回调函数
    @dash_app.callback(
        [Output('statistics-panel', 'children'),
         Output('price-distribution', 'figure')],
        [Input('brand-dropdown', 'value'),
         Input('year-slider', 'value')]
    )
    def update_statistics(selected_brands, year_range):
        filtered_df = df.copy()
        
        if selected_brands:
            filtered_df = filtered_df[filtered_df['Brand'].isin(selected_brands)]
        
        filtered_df = filtered_df[
            (filtered_df['year'] >= year_range[0]) & 
            (filtered_df['year'] <= year_range[1])
        ]
        
        # 更新统计信息
        stats = html.Div([
            html.P(f'Average Price: £{filtered_df["price"].mean():,.2f}'),
            html.P(f'Highest Price: £{filtered_df["price"].max():,.2f}'),
            html.P(f'Lowest Price: £{filtered_df["price"].min():,.2f}'),
            html.P(f'Number of Vehicles: {len(filtered_df)}')
        ])
        
        # 更新价格分布图
        new_hist = px.histogram(
            filtered_df,
            x='price',
            title='Filtered Price Distribution'
        )
        
        return stats, new_hist

    return dash_app 