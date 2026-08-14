"""路由与边界输入。

每个用例都对应一个真实修过的 bug，不是为了凑覆盖率：
/sales 的两个查询串曾让服务 500，/search 曾渲染不存在的模板，
越界的 vehicle id 曾抛 KeyError。
"""

import pytest


@pytest.mark.parametrize("path", [
    "/", "/login", "/sales", "/purchase", "/dashapp/",
])
def test_public_pages_render(client, path):
    assert client.get(path).status_code == 200


def test_search_accepts_regex_metacharacters(client):
    """'(' 曾让 str.contains 抛异常（默认按正则解析）。"""
    assert client.get("/sales?query=(((").status_code == 200
    assert client.get("/sales?query=a[b").status_code == 200


def test_empty_search_results_do_not_crash(client):
    """空结果曾让 np.random.choice 抛 ValueError。"""
    response = client.get("/sales?query=zzzznotacar")
    assert response.status_code == 200


def test_sales_filters_combine(client):
    response = client.get("/sales?query=Audi&price_min=10000&price_max=20000")
    assert response.status_code == 200


def test_vehicle_detail(client):
    assert client.get("/vehicle/5").status_code == 200


def test_unknown_vehicle_is_404_not_500(client):
    """data.loc[缺失 id] 会抛 KeyError；应转成 404。"""
    assert client.get("/vehicle/99999999").status_code == 404


def test_removed_search_route_is_gone(client):
    """/search 渲染的模板不存在，已删除；不应复活。"""
    assert client.get("/search").status_code == 404


def test_price_prediction_form(client):
    response = client.post("/purchase", data={
        "year": 2017, "transmission": "Automatic", "mileage": 15944,
        "fuelType": "Petrol", "tax": 150, "mpg": 57.7, "engineSize": 1.0,
        "Brand": "Ford", "Car_Type": "Hatchback", "High_Performance": 0,
    })
    assert response.status_code == 200
    assert b"\xc2\xa3" in response.data  # 页面上出现了 £


def test_login_rejects_bad_credentials(client):
    response = client.post("/login", data={"username": "admin", "password": "wrong"})
    assert response.status_code == 200          # 留在登录页
    assert "/dashboard" not in response.headers.get("Location", "")


def test_dashboard_requires_login(client):
    response = client.get("/dashboard")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
