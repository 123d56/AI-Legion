"""
AI Legion 全局配置
从环境变量 / .env 文件中加载所有配置项
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ProviderConfig:
    """单个模型供应商的配置"""
    name: str
    api_key: str
    model: str
    base_url: str | None = None
    enabled: bool = True


@dataclass
class Settings:
    """全局设置"""

    # 服务
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # 聚合策略
    strategy: str = field(default_factory=lambda: os.getenv("STRATEGY", "vote"))

    # API 认证 Token（为空则跳过认证，开发模式友好）
    api_token: str = field(default_factory=lambda: os.getenv("API_TOKEN", ""))

    def get_provider_configs(self) -> list[ProviderConfig]:
        """
        自动扫描环境变量，发现所有已配置的供应商。
        规则：存在 {NAME}_API_KEY 且非空，即视为已启用。
        """
        providers: list[ProviderConfig] = []

        provider_map = {
            "OPENAI": {
                "name": "openai",
                "default_model": "gpt-4o",
            },
            "ANTHROPIC": {
                "name": "anthropic",
                "default_model": "claude-sonnet-4-20250514",
            },
            "GOOGLE": {
                "name": "google",
                "default_model": "gemini-2.0-flash",
            },
            "DEEPSEEK": {
                "name": "deepseek",
                "default_model": "deepseek-chat",
            },
        }

        for prefix, info in provider_map.items():
            api_key = os.getenv(f"{prefix}_API_KEY", "")
            if api_key and not api_key.startswith("sk-xxx") and api_key != "xxx":
                providers.append(
                    ProviderConfig(
                        name=info["name"],
                        api_key=api_key,
                        model=os.getenv(f"{prefix}_MODEL", info["default_model"]),
                        base_url=os.getenv(f"{prefix}_BASE_URL"),
                    )
                )

        return providers


# 全局单例
settings = Settings()