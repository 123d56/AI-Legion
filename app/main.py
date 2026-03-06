"""
AI Legion - 应用入口
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.settings import settings
from app.providers.registry import ProviderRegistry
from app.engine.orchestrator import Orchestrator
from app.api.routes import router, init_orchestrator
from app.middleware.auth import BearerTokenMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # ── 启动 ──
    print("=" * 50)
    print("🤖 AI 军团 (AI Legion) 正在集结...")
    print("=" * 50)

    # 初始化供应商注册表
    registry = ProviderRegistry()
    configs = settings.get_provider_configs()
    registry.register_from_configs(configs)

    if registry.count() == 0:
        print("⚠️  警告: 未发现任何可用的模型供应商！")
        print("   请在 .env 文件中配置至少一个 API 密钥。")
    else:
        print(f"\n🎯 共 {registry.count()} 个供应商就绪")
        print(f"📋 聚合策略: {settings.strategy}")

    # 认证状态
    if settings.api_token:
        print(f"🔒 API 认证: 已启用")
    else:
        print(f"🔓 API 认证: 未启用（开发模式）")

    # 初始化编排引擎
    orchestrator = Orchestrator(registry)
    init_orchestrator(orchestrator)

    print(f"\n🚀 服务启动于 http://{settings.host}:{settings.port}")
    print(f"📖 API 文档: http://{settings.host}:{settings.port}/docs")
    print("=" * 50)

    yield

    # ── 关闭 ──
    print("\n👋 AI 军团已解散")


app = FastAPI(
    title="🤖 AI 军团 (AI Legion)",
    description="打造属于您自己的超级 AI 大脑 —— 多模型聚合智能体 API 服务",
    version="0.2.0",
    lifespan=lifespan,
)

# 挂载认证中间件
app.add_middleware(BearerTokenMiddleware)

app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )