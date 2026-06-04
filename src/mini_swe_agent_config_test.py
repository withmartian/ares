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
    from ares.containers import containers
    from ares.llms import response
    from ares.testing.mock_container import MockContainer
    from ares.testing.mock_llm import MockLLMClient


def _write_swebench_config(config_path: pathlib.Path, system_template: str = "test system") -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
agent:
  system_template: "{system_template}"
  instance_template: "test task: {{{{ task }}}}"
  step_limit: 250
  cost_limit: 3.0
environment:
  timeout: 1
  env:
    FOO: BAR
model:
  observation_template: |
    rc={{{{ output.returncode }}}}
    exception={{{{ output.exception_info }}}}
    output={{{{ output.output }}}}
  format_error_template: "format error: {{{{ error }}}}"
""".lstrip(),
        encoding="utf-8",
    )


def test_resolve_swebench_config_path_uses_benchmarks_layout(tmp_path: pathlib.Path) -> None:
    legacy_config_path = tmp_path / "extra" / "swebench.yaml"
    benchmarks_config_path = tmp_path / "benchmarks" / "swebench.yaml"
    _write_swebench_config(legacy_config_path, system_template="legacy")
    _write_swebench_config(benchmarks_config_path, system_template="benchmarks")

    assert mini_swe_agent._resolve_swebench_config_path(tmp_path) == benchmarks_config_path


def test_resolve_swebench_config_path_rejects_extra_layout(tmp_path: pathlib.Path) -> None:
    legacy_config_path = tmp_path / "extra" / "swebench.yaml"
    _write_swebench_config(legacy_config_path)

    with pytest.raises(FileNotFoundError, match=re.escape("benchmarks/swebench.yaml")):
        mini_swe_agent._resolve_swebench_config_path(tmp_path)


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


@pytest.mark.asyncio
async def test_mini_swe_code_agent_uses_model_observation_template(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    config_path = tmp_path / "benchmarks" / "swebench.yaml"
    _write_swebench_config(config_path)
    monkeypatch.setattr(mini_swe_agent.minisweagent.config, "builtin_config_dir", str(tmp_path))
    container = MockContainer(exec_responses={"echo hi": containers.ExecResult(output="hi", exit_code=0)})
    agent = mini_swe_agent.MiniSWECodeAgent(container=container, llm_client=MockLLMClient())

    await agent.execute_action(
        response.LLMResponse(
            data=[response.TextData(content="```bash\necho hi\n```")],
            cost=0.0,
            usage=response.Usage(prompt_tokens=0, generated_tokens=0),
        )
    )

    assert agent._messages[-1]["content"] == "rc=0\nexception=None\noutput=hi"


def test_mini_swe_code_agent_uses_model_format_error_template(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    config_path = tmp_path / "benchmarks" / "swebench.yaml"
    _write_swebench_config(config_path)
    monkeypatch.setattr(mini_swe_agent.minisweagent.config, "builtin_config_dir", str(tmp_path))
    agent = mini_swe_agent.MiniSWECodeAgent(container=MockContainer(), llm_client=MockLLMClient())

    with pytest.raises(mini_swe_agent._FormatError, match="format error"):
        agent.parse_action("no action")
