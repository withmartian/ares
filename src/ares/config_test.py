import getpass
import importlib
import sys


def _fresh_import_config():
    sys.modules.pop("ares.config", None)
    import ares.config as cfg
    return importlib.reload(cfg)


def test_config_user_from_getpass(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)
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
