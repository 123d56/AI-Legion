"""
请求 / 响应的数据模型定义
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    """单条对话消息"""
    role: str = Field(..., description="角色: system / user / assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """聊天请求"""
    messages: list[Message] = Field(..., description="对话历史")
    strategy: str | None = Field(None, description="聚合策略覆盖 (vote/best_of/chain/cascade)")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(4096, ge=1, le=128000, description="最大生成 token 数")


class ProviderResponse(BaseModel):
    """单个供应商的响应"""
    provider: str = Field(..., description="供应商名称")
    model: str = Field(..., description="模型名称")
    content: str = Field(..., description="响应内容")
    usage: dict | None = Field(None, description="token 用量")
    latency_ms: float = Field(..., description="响应耗时(毫秒)")
    error: str | None = Field(None, description="错误信息")


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str = Field(..., description="最终聚合后的回答")
    strategy_used: str = Field(..., description="使用的聚合策略")
    provider_responses: list[ProviderResponse] = Field(
        default_factory=list, description="各供应商的原始响应"
    )
    total_latency_ms: float = Field(..., description="总耗时(毫秒)")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "0.1.0"
    active_providers: list[str] = Field(default_factory=list)
    strategy: str = ""