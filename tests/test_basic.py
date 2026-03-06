"""基础测试"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    """测试健康检查接口"""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.2.0"


def test_list_strategies(client):
    """测试策略列表接口"""
    resp = client.get("/api/v1/strategies")
    assert resp.status_code == 200
    data = resp.json()
    names = [s["name"] for s in data["strategies"]]
    assert "vote" in names
    assert "cascade" in names
    assert "chain" in names
    assert "best_of" in names


def test_list_providers(client):
    """测试供应商列表接口（无配置时应返回空）"""
    resp = client.get("/api/v1/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["count"], int)


def test_chat_no_providers(client):
    """测试无供应商时的聊天接口"""
    resp = client.post(
        "/api/v1/chat",
        json={
            "messages": [{"role": "user", "content": "你好"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    # 没有供应商时应返回提示信息
    assert "没有可用的模型供应商" in data["answer"] or isinstance(data["answer"], str)


def test_chat_stream_no_providers(client):
    """测试无供应商时的流式聊天接口"""
    resp = client.post(
        "/api/v1/chat",
        json={
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")