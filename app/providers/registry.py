"""
供应商注册表
根据配置自动实例化所有已启用的供应商
"""

from __future__ import annotations

from config.settings import ProviderConfig
from app.providers.base import BaseProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.google_provider import GoogleProvider
from app.providers.deepseek_provider import DeepSeekProvider


# 供应商名称 -> 类的映射
_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "deepseek": DeepSeekProvider,
}


class ProviderRegistry:
    """供应商注册表 —— 管理所有已激活的供应商实例"""

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register_from_configs(self, configs: list[ProviderConfig]) -> None:
        """根据配置列表批量注册供应商"""
        for cfg in configs:
            if not cfg.enabled:
                continue
            cls = _PROVIDER_CLASSES.get(cfg.name)
            if cls is None:
                print(f"⚠️  未知供应商: {cfg.name}，已跳过")
                continue
            self._providers[cfg.name] = cls(
                api_key=cfg.api_key,
                model=cfg.model,
                base_url=cfg.base_url,
            )
            print(f"✅ 已注册供应商: {cfg.name} (模型: {cfg.model})")

    def get_all(self) -> list[BaseProvider]:
        """获取所有已注册的供应商"""
        return list(self._providers.values())

    def get(self, name: str) -> BaseProvider | None:
        """按名称获取供应商"""
        return self._providers.get(name)

    def list_names(self) -> list[str]:
        """列出所有已注册的供应商名称"""
        return list(self._providers.keys())

    def count(self) -> int:
        return len(self._providers)