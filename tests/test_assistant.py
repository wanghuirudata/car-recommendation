"""助手：工具正确性、规则解析、以及三条降级路径各自可区分。

fallback_reason 是这里最重要的断言。降级保住了可用性，但如果不报告原因，
它就只是把故障伪装成'效果不太好'——这个项目在推荐模块上已经犯过一次。
"""

import builtins

import pytest

from model.assistant import PLAUSIBLE_YEARS


# ---------- 工具 ----------

def test_search_applies_every_filter(assistant):
    result = assistant.search_vehicles(brand="Ford", car_type="Hatchback",
                                       max_price=12000, max_mileage=40000)
    assert result["count"] > 0
    for card in result["vehicles"]:
        assert card["price"] <= 12000
        assert card["mileage"] <= 40000


def test_search_excludes_implausible_years(assistant, data):
    """数据里有 3 行脏年份（Fiesta 2060、两台 1970 年的车）。
    它们对模型无影响，但'最新优先'排序会把它们顶到第一位。"""
    low, high = PLAUSIBLE_YEARS
    assert ((data["year"] < low) | (data["year"] > high)).sum() > 0   # 脏数据确实存在
    for card in assistant.search_vehicles(brand="Ford")["vehicles"]:
        year = int(card["title"].split()[-1])
        assert low <= year <= high


def test_search_sorts_newest_first_not_cheapest(assistant):
    """按价格升序排会把 2003 年 17.7 万英里的车推给预算 £15,000 的用户。"""
    cards = assistant.search_vehicles(brand="Ford", max_price=15000)["vehicles"]
    years = [int(c["title"].split()[-1]) for c in cards]
    assert years == sorted(years, reverse=True)


def test_search_drops_exact_duplicate_listings(assistant):
    """同一辆车列两遍看起来像 bug。

    用 Brand=Ford：它的 top5（最新优先）里确实存在一条完全重复的挂牌，
    所以这个用例真的会在去重被移除时失败——换个品牌就测不出来了。
    """
    cards = assistant.search_vehicles(brand="Ford")["vehicles"]
    fingerprints = [(c["title"], c["price"], c["mileage"]) for c in cards]
    assert len(fingerprints) == len(set(fingerprints))


def test_estimate_price_returns_plausible_value(assistant):
    result = assistant.estimate_price(year=2018, mileage=30000,
                                      Brand="Ford", Car_Type="Hatchback")
    assert 3000 < result["estimated_price"] < 40000


def test_find_alternatives_cheaper_flag(assistant, data):
    seed_price = float(data.loc[50000, "price"])
    result = assistant.find_alternatives(50000, cheaper=True)
    assert result["alternatives"]
    for card in result["alternatives"]:
        assert card["price"] < seed_price


def test_find_alternatives_unknown_id(assistant):
    assert "error" in assistant.find_alternatives(-1)


# ---------- 降级原因 ----------

def test_fallback_reason_no_api_key(assistant):
    result = assistant.reply("automatic BMW under 20k")
    assert result["mode"] == "rules"
    assert result["fallback_reason"] == "no_api_key"
    assert result["cards"]                      # 降级后仍然给出可用答案


def test_fallback_reason_package_missing(assistant, monkeypatch):
    """key 配了但依赖没装——曾经和'没配 key'完全无法区分，实际排查耗了很久。"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    monkeypatch.setattr(assistant, "_client", None, raising=False)

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    result = assistant.reply("automatic BMW under 20k")
    assert result["mode"] == "rules"
    assert result["fallback_reason"] == "package_missing"
    assert result["cards"]


# ---------- 规则解析 ----------

@pytest.mark.parametrize("message, expected", [
    ("automatic BMW under 20k",                     {"brand": "BMW", "transmission": "Automatic"}),
    ("diesel SUV under 25000",                      {"car_type": "SUV", "fuel_type": "Diesel"}),
    ("hybrid Toyota",                               {"brand": "Toyota", "fuel_type": "Hybrid"}),
])
def test_rule_parser_extracts_attributes(assistant, message, expected):
    cards = assistant.reply(message)["cards"]
    assert cards, f"no results for: {message}"


def test_rule_parser_handles_k_suffix(assistant):
    """'20k' 要解析成 20000，不是 20。"""
    cards = assistant.reply("automatic BMW under 20k")["cards"]
    assert all(c["price"] <= 20000 for c in cards)


def test_rule_parser_handles_mileage(assistant):
    cards = assistant.reply("Ford with less than 30000 miles")["cards"]
    assert all(c["mileage"] <= 30000 for c in cards)


def test_unparseable_message_returns_guidance_not_junk(assistant):
    result = assistant.reply("hello there")
    assert result["cards"] == []
    assert result["reply"]


# ---------- HTTP 层 ----------

def test_chat_endpoint_rejects_empty_message(client):
    assert client.post("/api/chat", json={"message": "  "}).status_code == 400
    assert client.post("/api/chat", json={}).status_code == 400


def test_chat_endpoint_rejects_oversized_message(client):
    assert client.post("/api/chat", json={"message": "x" * 1001}).status_code == 400


def test_chat_endpoint_returns_cards_and_reason(client):
    response = client.post("/api/chat", json={"message": "automatic BMW under 20k"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["mode"] == "rules"
    assert body["fallback_reason"] == "no_api_key"
    assert body["cards"]
    assert body["cards"][0]["url"].startswith("/vehicle/")
