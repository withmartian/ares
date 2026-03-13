"""Tests for MiniSWECodeAgent state serialization."""

import pytest

from ares.code_agents import code_agent_base
from ares.code_agents import mini_swe_agent
from ares.containers import containers
from ares.llms import request
from ares.testing import mock_container
from ares.testing import mock_llm


def _make_agent() -> mini_swe_agent.MiniSWECodeAgent:
    """Create a MiniSWECodeAgent with mock dependencies."""
    container = mock_container.MockContainer()
    llm_client = mock_llm.MockLLMClient()
    return mini_swe_agent.MiniSWECodeAgent(container=container, llm_client=llm_client)


def test_get_state_captures_empty_state():
    """Test get_state() on a freshly created agent."""
    agent = _make_agent()
    state = agent.get_state()

    assert state.messages == []
    assert state.n_calls == 0
    assert state.total_cost == 0.0


def test_get_state_captures_messages():
    """Test get_state() after messages have been added."""
    agent = _make_agent()
    agent._messages = [
        request.UserMessage(role="user", content="hello"),
        request.AssistantMessage(role="assistant", content="world"),
    ]
    agent._n_calls = 5
    agent._total_cost = 1.23

    state = agent.get_state()

    assert len(state.messages) == 2
    assert state.messages[0]["content"] == "hello"
    assert state.messages[1]["content"] == "world"
    assert state.n_calls == 5
    assert state.total_cost == 1.23


def test_get_state_returns_deep_copy():
    """Test that get_state() returns independent copies of messages."""
    agent = _make_agent()
    agent._messages = [request.UserMessage(role="user", content="original")]

    state = agent.get_state()

    # Mutate the agent's messages
    agent._messages.append(request.UserMessage(role="user", content="added"))

    # State should not be affected
    assert len(state.messages) == 1


def test_code_agent_state_is_frozen():
    """Test that CodeAgentState is immutable."""
    state = code_agent_base.CodeAgentState(
        messages=[],
        n_calls=0,
        total_cost=0.0,
    )
    with pytest.raises(AttributeError):
        state.n_calls = 5  # type: ignore[misc]


@pytest.mark.asyncio
async def test_restore_and_resume_sets_state():
    """Test that restore_and_resume restores messages and counters."""
    agent = _make_agent()

    saved_messages = [
        request.UserMessage(role="user", content="saved message"),
        request.AssistantMessage(role="assistant", content="saved response"),
    ]
    state = code_agent_base.CodeAgentState(
        messages=saved_messages,
        n_calls=3,
        total_cost=0.5,
    )

    # Configure mock LLM to trigger submission so the loop terminates
    agent.llm_client = mock_llm.MockLLMClient(
        responses=["```bash\necho 'MINI_SWE_AGENT_FINAL_OUTPUT'\n```"],
    )
    # Configure container to return the submission output
    agent.container = mock_container.MockContainer(
        exec_responses={
            "echo 'MINI_SWE_AGENT_FINAL_OUTPUT'": containers.ExecResult(
                output="MINI_SWE_AGENT_FINAL_OUTPUT\n",
                exit_code=0,
            )
        },
    )

    await agent.restore_and_resume(state, task="ignored")

    # State should have been restored before the loop ran
    assert agent._n_calls == 4  # 3 from state + 1 from the query in the loop
    assert len(agent._messages) >= 2  # At least the restored messages
    # The first two messages should be the restored ones
    assert agent._messages[0]["content"] == "saved message"
    assert agent._messages[1]["content"] == "saved response"
