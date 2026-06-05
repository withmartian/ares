import pytest
import yaml

from ares.code_agents import mini_swe_agent
from ares.containers import containers
from ares.llms import response
from ares.testing.mock_container import MockContainer
from ares.testing.mock_llm import MockLLMClient


def test_swebench_config_is_repo_owned_v1_copy() -> None:
    config_text = mini_swe_agent._SWEBENCH_CONFIG_RESOURCE.read_text(encoding="utf-8")

    assert "Copied from mini-swe-agent==1.17.3" in config_text
    assert "action_observation_template" in config_text
    assert "format_error_template" in config_text


def test_agent_config_keys_match_v1_agent_section() -> None:
    config = yaml.safe_load(mini_swe_agent._SWEBENCH_CONFIG_RESOURCE.read_text(encoding="utf-8"))

    assert frozenset(config["agent"]) == mini_swe_agent._AGENT_CONFIG_KEYS


def test_mini_swe_code_agent_initializes_from_repo_config() -> None:
    agent = mini_swe_agent.MiniSWECodeAgent(container=MockContainer(), llm_client=MockLLMClient())

    assert agent._system_prompt.startswith("You are a helpful assistant")
    assert agent._step_limit == 250
    assert agent._cost_limit == 3.0
    assert agent._env_timeout == 60
    assert agent._environment_env_vars["PAGER"] == "cat"


@pytest.mark.asyncio
async def test_mini_swe_code_agent_uses_agent_observation_template() -> None:
    container = MockContainer(exec_responses={"echo hi": containers.ExecResult(output="hi", exit_code=0)})
    agent = mini_swe_agent.MiniSWECodeAgent(container=container, llm_client=MockLLMClient())

    await agent.execute_action(
        response.LLMResponse(
            data=[response.TextData(content="```bash\necho hi\n```")],
            cost=0.0,
            usage=response.Usage(prompt_tokens=0, generated_tokens=0),
        )
    )

    assert "<returncode>0</returncode>" in agent._messages[-1]["content"]
    assert "hi" in agent._messages[-1]["content"]


def test_mini_swe_code_agent_uses_agent_format_error_template() -> None:
    agent = mini_swe_agent.MiniSWECodeAgent(container=MockContainer(), llm_client=MockLLMClient())

    with pytest.raises(mini_swe_agent._FormatError, match="found 0 actions"):
        agent.parse_action("no action")
