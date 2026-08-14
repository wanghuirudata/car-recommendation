"""共享 fixtures。

数据集有 107k 行，加载一次约 5 秒，所以做成 session 级 fixture，
整个测试会话只付一次代价。

注意：这里强制清掉 ANTHROPIC_API_KEY。测试绝不能打真实 API ——
既要花钱，又会让结果依赖网络。助手的测试全部针对规则模式。
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


@pytest.fixture(scope="session", autouse=True)
def _repo_root_cwd():
    """app.py 用相对路径读 vehicle.csv.gz，保证工作目录正确。"""
    old = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(old)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """每个测试都跑在规则模式下，不触碰真实 API。"""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(scope="session")
def dataset():
    from model.recommendation import load_and_prepare_data
    return load_and_prepare_data("vehicle.csv.gz")


@pytest.fixture(scope="session")
def data(dataset):
    return dataset[0]


@pytest.fixture(scope="session")
def features(dataset):
    return dataset[1]


@pytest.fixture(scope="session")
def flask_app():
    import app as app_module
    app_module.app.config.update(TESTING=True)
    return app_module.app


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


@pytest.fixture(scope="session")
def assistant(data, features):
    from model.assistant import VehicleAssistant
    return VehicleAssistant(data, features)
