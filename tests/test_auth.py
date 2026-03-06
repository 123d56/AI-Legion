"""认证中间件测试"""

import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


def _make_app(api_token: str = ""):
    """创建带指定 token 的测试应用"""
    with patch.dict(os.environ, {"API_TOKEN": api_token}, clear=False):
        # 重新加载模块以应用新的环境变量
        import importlib
        import config.settings
        importlib.reload(config.settings)

        import app.main
        importlib.reload(app.main)

        return app.main.app


class TestAuthDisabled:
    """API_TOKEN 为空时，认证应被跳过"""

    @pytest.fixture
    def client(self):
        test_app = _make_app("")
        return TestClient(test_app)

    def test_health_no_auth(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_chat_no_auth(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "你好"}]},
        )
        assert resp.status_code == 200

    def test_providers_no_auth(self, client):
        resp = client.get("/api/v1/providers")
        assert resp.status_code == 200


class TestAuthEnabled:
    """API_TOKEN 有值时，应验证 Bearer Token"""

    TEST_TOKEN = "test-secret-token-12345"

    @pytest.fixture
    def client(self):
        test_app = _make_app(self.TEST_TOKEN)
        return TestClient(test_app)

    def test_health_always_open(self, client):
        """健康检查不需要认证"""
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_no_auth_header_401(self, client):
        """无认证头应返回 401"""
        resp = client.get("/api/v1/providers")
        assert resp.status_code == 401
        assert "未提供认证信息" in resp.json()["error"]

    def test_wrong_format_401(self, client):
        """错误格式应返回 401"""
        resp = client.get(
            "/api/v1/providers",
            headers={"Authorization": "Basic abc123"},
        )
        assert resp.status_code == 401

    def test_wrong_token_403(self, client):
        """错误 Token 应返回 403"""
        resp = client.get(
            "/api/v1/providers",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 403
        assert "认证失败" in resp.json()["error"]

    def test_correct_token_passes(self, client):
        """正确 Token 应放行"""
        resp = client.get(
            "/api/v1/providers",
            headers={"Authorization": f"Bearer {self.TEST_TOKEN}"},
        )
        assert resp.status_code == 200

    def test_chat_with_correct_token(self, client):
        """使用正确 Token 调用聊天接口"""
        resp = client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "你好"}]},
            headers={"Authorization": f"Bearer {self.TEST_TOKEN}"},
        )
        assert resp.status_code == 200