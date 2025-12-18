"""Configuration for ARES."""

import os

import pydantic
import pydantic_settings


class _Config(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        frozen=True,
        env_file=".env",
        extra="ignore",
    )

    # Defaults to USER environment variable, if available.
    # Otherwise, falls back to getlogin().
    user: str = os.getlogin()

    # Configuration for LLM requests.
    chat_completion_api_base_url: str = "https://api.withmartian.com/v1"
    chat_completion_api_key: str = pydantic.Field(...)

    # Daytona configuration.
    daytona_auto_stop_interval: int = 30  # Minutes.
    daytona_delete_on_stop: bool = True


CONFIG = _Config()  # type: ignore
