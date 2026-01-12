"""Configuration for ARES."""

import os
import getpass

import pydantic
import pydantic_settings


def _default_user() -> str:
    """Best-effort default username for headless/container environments."""
    for k in ("USER", "LOGNAME"):
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    try:
        v = (getpass.getuser() or "").strip()
        if v:
            return v
    except Exception:
        pass
    return "unknown"


class _Config(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        frozen=True,
        env_file=".env",
        extra="ignore",
    )

    # Defaults to USER/LOGNAME environment variable, if available.
    # Otherwise, falls back to getpass.getuser().
    # As a last resort, uses "unknown".
    user: str = _default_user()

    # Configuration for LLM requests.
    chat_completion_api_base_url: str = "https://api.withmartian.com/v1"
    chat_completion_api_key: str = pydantic.Field(...)

    # Daytona configuration.
    daytona_auto_stop_interval: int = 30  # Minutes.
    daytona_delete_on_stop: bool = True


CONFIG = _Config()  # type: ignore
