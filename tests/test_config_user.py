import getpass
import importlib
import os
import sys


def _fresh_import_config():
    # Ensure a clean import so CONFIG is rebuilt using our monkeypatched env.
    sys.modules.pop("ares.config", None)
    import ares.config as cfg
    return importlib.reload(cfg)


def test_config_user_from_getpass(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)

    # Required for CONFIG instantiation.
    # In this project, BaseSettings reads from CHAT_COMPLETION_API_KEY.
    monkeypatch.setenv("CHAT_COMPLETION_API_KEY", "test-key")

    monkeypatch.setattr(getpass, "getuser", lambda: "jane")
    cfg = _fresh_import_config()
    assert cfg.CONFIG.user == "jane"


def test_config_user_unknown_when_getpass_fails(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)
    monkeypatch.setenv("CHAT_COMPLETION_API_KEY", "test-key")

    def _boom():
        raise Exception("no user")

    monkeypatch.setattr(getpass, "getuser", _boom)
    cfg = _fresh_import_config()
    assert cfg.CONFIG.user == "unknown"
