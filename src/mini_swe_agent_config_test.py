import pathlib
import re
from unittest import mock

from harbor.registry.client import factory as harbor_client_factory
import pytest

_harbor_client = mock.Mock()
_harbor_client.get_datasets.return_value = ()

# Keep this test outside the ares package so Harbor can be patched before ares.__init__ registers presets.
with mock.patch.object(harbor_client_factory.RegistryClientFactory, "create", return_value=_harbor_client):
    from ares.code_agents import mini_swe_agent
    from ares.testing.mock_container import MockContainer
    from ares.testing.mock_llm import MockLLMClient


def _write_swebench_config(config_path: pathlib.Path, system_template: str = "test system") -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
agent:
  system_template: "{system_template}"
environment:
  timeout: 1
  env:
    FOO: BAR
""".lstrip(),
        encoding="utf-8",
    )


def test_resolve_swebench_config_path_prefers_benchmarks_layout(tmp_path: pathlib.Path) -> None:
    legacy_config_path = tmp_path / "extra" / "swebench.yaml"
    benchmarks_config_path = tmp_path / "benchmarks" / "swebench.yaml"
    _write_swebench_config(legacy_config_path, system_template="legacy")
    _write_swebench_config(benchmarks_config_path, system_template="benchmarks")

    assert mini_swe_agent._resolve_swebench_config_path(tmp_path) == benchmarks_config_path


def test_resolve_swebench_config_path_falls_back_to_extra_layout(tmp_path: pathlib.Path) -> None:
    legacy_config_path = tmp_path / "extra" / "swebench.yaml"
    _write_swebench_config(legacy_config_path)

    assert mini_swe_agent._resolve_swebench_config_path(tmp_path) == legacy_config_path


def test_resolve_swebench_config_path_raises_when_config_is_missing(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape("benchmarks/swebench.yaml")):
        mini_swe_agent._resolve_swebench_config_path(tmp_path)


def test_mini_swe_code_agent_initializes_with_benchmarks_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    config_path = tmp_path / "benchmarks" / "swebench.yaml"
    _write_swebench_config(config_path)
    monkeypatch.setattr(mini_swe_agent.minisweagent.config, "builtin_config_dir", str(tmp_path))

    agent = mini_swe_agent.MiniSWECodeAgent(container=MockContainer(), llm_client=MockLLMClient())

    assert agent._system_prompt == "test system"
    assert agent._env_timeout == 1
    assert agent._environment_env_vars == {"FOO": "BAR"}
