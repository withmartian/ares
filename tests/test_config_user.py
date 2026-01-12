import importlib
import getpass


def _reload_config(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)

    # Required for config validation (pydantic-settings uses field name -> env var)
    monkeypatch.setenv("CHAT_COMPLETION_API_KEY", "test-key")

    import ares.config as cfg

    return importlib.reload(cfg)


def test_config_user_from_getpass(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: "jane")
    cfg = _reload_config(monkeypatch)
    assert cfg.CONFIG.user == "jane"


def test_config_user_unknown_when_getpass_fails(monkeypatch):
    def _boom():
        raise Exception("no user")

    monkeypatch.setattr(getpass, "getuser", _boom)
    cfg = _reload_config(monkeypatch)
    assert cfg.CONFIG.user == "unknown"
