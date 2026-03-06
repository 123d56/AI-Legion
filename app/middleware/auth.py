"""
API 认证中间件
支持 Bearer Token 认证，保护 /api/v1/* 端点
"""

from __future__ import annotations

import secrets
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config.settings import settings


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """
    Bearer Token 认证中间件

    规则：
    - 如果 settings.api_token 为空，跳过认证（开发模式）
    - /api/health 和 /docs, /redoc, /openapi.json 不需要认证
    - 其他 /api/* 路径需要 Authorization: Bearer <token>
    """

    # 白名单路径 —— 不需要认证
    WHITELIST_PATHS = {
        "/api/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/",
    }

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 如果没有配置 Token，跳过认证（开发模式）
        if not settings.api_token:
            return await call_next(request)

        # 白名单路径放行
        path = request.url.path
        if path in self.WHITELIST_PATHS:
            return await call_next(request)

        # 非 API 路径放行
        if not path.startswith("/api"):
            return await call_next(request)

        # 提取并验证 Token
        auth_header = request.headers.get("Authorization", "")

        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "未提供认证信息",
                    "detail": "请在请求头中添加 Authorization: Bearer <your-token>",
                },
            )

        # 解析 Bearer Token
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return JSONResponse(
                status_code=401,
                content={
                    "error": "认证格式错误",
                    "detail": "格式应为: Authorization: Bearer <your-token>",
                },
            )

        token = parts[1].strip()

        # 使用 constant-time 比较防止时序攻击
        if not secrets.compare_digest(token, settings.api_token):
            return JSONResponse(
                status_code=403,
                content={
                    "error": "认证失败",
                    "detail": "Token 无效或已过期",
                },
            )

        # 认证通过
        return await call_next(request)