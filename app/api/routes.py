"""
FastAPI 路由定义
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse
from app.engine.orchestrator import Orchestrator
from app.providers.registry import ProviderRegistry
from config.settings import settings

router = APIRouter()

# 全局实例（在 main.py 中初始化后注入）
_orchestrator: Orchestrator | None = None


def init_orchestrator(orchestrator: Orchestrator) -> None:
    global _orchestrator
    _orchestrator = orchestrator


def _get_orchestrator() -> Orchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="引擎未初始化")
    return _orchestrator


@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查接口"""
    orch = _get_orchestrator()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        active_providers=orch.registry.list_names(),
        strategy=settings.strategy,
    )


@router.post("/v1/chat", response_model=ChatResponse, tags=["对话"])
async def chat(request: ChatRequest):
    """
    核心对话接口

    发送消息到 AI 军团，所有已注册的模型将协同工作，
    通过指定的聚合策略产出最终回答。
    """
    orch = _get_orchestrator()
    return await orch.chat(request)


@router.get("/v1/providers", tags=["管理"])
async def list_providers():
    """列出所有已激活的模型供应商"""
    orch = _get_orchestrator()
    providers = orch.registry.get_all()
    return {
        "count": len(providers),
        "providers": [
            {"name": p.name, "model": p.model} for p in providers
        ],
    }


@router.get("/v1/strategies", tags=["管理"])
async def list_strategies():
    """列出所有可用的聚合策略"""
    from app.engine.strategies import STRATEGIES

    return {
        "strategies": [
            {"name": name, "description": cls.__doc__.strip() if cls.__doc__ else ""}
            for name, cls in STRATEGIES.items()
        ]
    }