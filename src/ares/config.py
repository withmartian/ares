"""Configuration for ARES."""

import getpass
import os

import pydantic_settings


def _default_user() -> str:
    """Best-effort default username for headless/container environments.

    We avoid os.getlogin() because it raises OSError in containers/CI
    environments that lack a controlling terminal.
    """
    for env_var in ("USER", "LOGNAME"):
        env_val = (os.environ.get(env_var) or "").strip()
        if env_val:
            return env_val
    try:
        env_val = (getpass.getuser() or "").strip()
        if env_val:
            return env_val
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
    chat_completion_api_key: str = ""

    # Daytona configuration.
    daytona_auto_stop_interval: int = 30  # Minutes.
    daytona_delete_on_stop: bool = True


CONFIG = _Config()  # type: ignore
