import getpass

from ares.config import _default_user


def test_default_user_from_env_user(monkeypatch):
    """Test that USER environment variable takes precedence."""
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("LOGNAME", "bob")
    assert _default_user() == "alice"


def test_default_user_from_env_logname(monkeypatch):
    """Test fallback to LOGNAME when USER is not set."""
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.setenv("LOGNAME", "bob")
    assert _default_user() == "bob"


def test_default_user_from_getpass(monkeypatch):
    """Test fallback to getpass.getuser() when env vars are not set."""
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)
    monkeypatch.setattr(getpass, "getuser", lambda: "charlie")
    assert _default_user() == "charlie"


def test_default_user_unknown_when_all_fail(monkeypatch):
    """Test fallback to 'unknown' when all methods fail."""
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)

    def _raise_exception():
        raise Exception("no user available")

    monkeypatch.setattr(getpass, "getuser", _raise_exception)
    assert _default_user() == "unknown"


def test_default_user_strips_whitespace(monkeypatch):
    """Test that whitespace is stripped from environment variables."""
    monkeypatch.setenv("USER", "  alice  ")
    assert _default_user() == "alice"


def test_default_user_skips_empty_env_vars(monkeypatch):
    """Test that empty/whitespace-only env vars are skipped."""
    monkeypatch.setenv("USER", "   ")
    monkeypatch.setenv("LOGNAME", "bob")
    assert _default_user() == "bob"
