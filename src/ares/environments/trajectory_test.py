"""Tests for episode trajectory collection."""

import json
import pathlib
import time

import pytest

from ares.environments import trajectory
from ares.llms import request
from ares.llms import response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_request(content: str = "Hello") -> request.LLMRequest:
    """Create a simple LLMRequest for testing."""
    return request.LLMRequest(messages=[{"role": "user", "content": content}])


def _make_llm_response(content: str = "Reply") -> response.LLMResponse:
    """Create a simple LLMResponse for testing."""
    return response.LLMResponse(
        data=[response.TextData(content=content)],
        cost=0.01,
        usage=response.Usage(prompt_tokens=50, generated_tokens=25),
    )


def _make_step_record(
    step_index: int = 0,
    step_type: str = "MID",
    with_observation: bool = True,
    with_action: bool = True,
    reward: float | None = 0.0,
    discount: float | None = 1.0,
) -> trajectory.StepRecord:
    """Create a StepRecord for testing."""
    return trajectory.StepRecord(
        step_index=step_index,
        step_type=step_type,
        observation=trajectory.serialize_llm_request(_make_llm_request()) if with_observation else None,
        action=trajectory.serialize_llm_response(_make_llm_response()) if with_action else None,
        reward=reward,
        discount=discount,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


class TestSerializeLLMRequest:
    def test_basic_request(self):
        req = _make_llm_request("Test message")
        result = trajectory.serialize_llm_request(req)

        assert isinstance(result, dict)
        assert result["messages"] == [{"role": "user", "content": "Test message"}]

    def test_request_with_all_fields(self):
        req = request.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_output_tokens=100,
            temperature=0.7,
            top_p=0.9,
            system_prompt="You are a helpful assistant.",
        )
        result = trajectory.serialize_llm_request(req)

        assert result["max_output_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["system_prompt"] == "You are a helpful assistant."

    def test_result_is_json_serializable(self):
        req = _make_llm_request("Test")
        result = trajectory.serialize_llm_request(req)
        # Should not raise
        json.dumps(result)


class TestSerializeLLMResponse:
    def test_basic_response(self):
        resp = _make_llm_response("Test reply")
        result = trajectory.serialize_llm_response(resp)

        assert isinstance(result, dict)
        assert result["data"] == [{"content": "Test reply"}]
        assert result["cost"] == 0.01
        assert result["usage"]["prompt_tokens"] == 50
        assert result["usage"]["generated_tokens"] == 25

    def test_result_is_json_serializable(self):
        resp = _make_llm_response("Test")
        result = trajectory.serialize_llm_response(resp)
        # Should not raise
        json.dumps(result)


# ---------------------------------------------------------------------------
# StepRecord
# ---------------------------------------------------------------------------


class TestStepRecord:
    def test_create_first_step(self):
        record = _make_step_record(step_index=0, step_type="FIRST", with_action=False, reward=None, discount=None)
        assert record.step_index == 0
        assert record.step_type == "FIRST"
        assert record.observation is not None
        assert record.action is None
        assert record.reward is None
        assert record.discount is None

    def test_create_mid_step(self):
        record = _make_step_record(step_index=3, step_type="MID", reward=0.0, discount=1.0)
        assert record.step_index == 3
        assert record.step_type == "MID"
        assert record.observation is not None
        assert record.action is not None
        assert record.reward == 0.0
        assert record.discount == 1.0

    def test_create_last_step(self):
        record = _make_step_record(
            step_index=10, step_type="LAST", with_observation=False, reward=1.0, discount=0.0
        )
        assert record.step_index == 10
        assert record.step_type == "LAST"
        assert record.observation is None
        assert record.reward == 1.0
        assert record.discount == 0.0

    def test_to_dict(self):
        record = _make_step_record(step_index=5, step_type="MID")
        d = record.to_dict()

        assert isinstance(d, dict)
        assert d["step_index"] == 5
        assert d["step_type"] == "MID"
        assert "observation" in d
        assert "action" in d
        assert "reward" in d
        assert "discount" in d
        assert "timestamp" in d

    def test_to_dict_json_serializable(self):
        record = _make_step_record()
        d = record.to_dict()
        json.dumps(d)  # Should not raise.

    def test_from_dict_roundtrip(self):
        original = _make_step_record(step_index=7, step_type="MID", reward=0.5, discount=0.99)
        d = original.to_dict()
        restored = trajectory.StepRecord.from_dict(d)

        assert restored.step_index == original.step_index
        assert restored.step_type == original.step_type
        assert restored.observation == original.observation
        assert restored.action == original.action
        assert restored.reward == original.reward
        assert restored.discount == original.discount
        assert restored.timestamp == original.timestamp

    def test_frozen(self):
        record = _make_step_record()
        with pytest.raises(AttributeError):
            record.step_index = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EpisodeTrajectory
# ---------------------------------------------------------------------------


class TestEpisodeTrajectory:
    def test_create(self):
        traj = trajectory.EpisodeTrajectory(
            episode_id="test-123",
            task_name="my-task",
            steps=[],
            start_time=time.time(),
        )
        assert traj.episode_id == "test-123"
        assert traj.task_name == "my-task"
        assert traj.steps == []
        assert traj.end_time is None
        assert traj.total_reward is None
        assert traj.num_steps == 0
        assert traj.truncated is False

    def test_to_dict(self):
        steps = [
            _make_step_record(step_index=0, step_type="FIRST", with_action=False, reward=None, discount=None),
            _make_step_record(step_index=1, step_type="MID"),
            _make_step_record(step_index=2, step_type="LAST", with_observation=False, reward=1.0, discount=0.0),
        ]
        traj = trajectory.EpisodeTrajectory(
            episode_id="ep-abc",
            task_name="swe-bench-123",
            steps=steps,
            start_time=1000.0,
            end_time=1060.0,
            total_reward=1.0,
            num_steps=3,
            truncated=False,
        )
        d = traj.to_dict()

        assert d["episode_id"] == "ep-abc"
        assert d["task_name"] == "swe-bench-123"
        assert d["start_time"] == 1000.0
        assert d["end_time"] == 1060.0
        assert d["total_reward"] == 1.0
        assert d["num_steps"] == 3
        assert d["truncated"] is False
        assert len(d["steps"]) == 3

    def test_to_dict_json_serializable(self):
        steps = [_make_step_record(step_index=0, step_type="MID")]
        traj = trajectory.EpisodeTrajectory(
            episode_id="ep-1",
            task_name="task-1",
            steps=steps,
            start_time=time.time(),
        )
        json.dumps(traj.to_dict())  # Should not raise.

    def test_from_dict_roundtrip(self):
        steps = [
            _make_step_record(step_index=0, step_type="FIRST", with_action=False, reward=None, discount=None),
            _make_step_record(step_index=1, step_type="LAST", reward=0.75, discount=0.0),
        ]
        original = trajectory.EpisodeTrajectory(
            episode_id="ep-roundtrip",
            task_name="test-task",
            steps=steps,
            start_time=1000.0,
            end_time=1010.0,
            total_reward=0.75,
            num_steps=2,
            truncated=True,
        )
        d = original.to_dict()
        restored = trajectory.EpisodeTrajectory.from_dict(d)

        assert restored.episode_id == original.episode_id
        assert restored.task_name == original.task_name
        assert restored.start_time == original.start_time
        assert restored.end_time == original.end_time
        assert restored.total_reward == original.total_reward
        assert restored.num_steps == original.num_steps
        assert restored.truncated == original.truncated
        assert len(restored.steps) == len(original.steps)

    def test_load(self, tmp_path: pathlib.Path):
        steps = [_make_step_record(step_index=0, step_type="MID", reward=0.5)]
        traj = trajectory.EpisodeTrajectory(
            episode_id="ep-load",
            task_name="task-load",
            steps=steps,
            start_time=100.0,
            end_time=200.0,
            total_reward=0.5,
            num_steps=1,
        )
        filepath = tmp_path / "test_episode.json"
        with open(filepath, "w") as f:
            json.dump(traj.to_dict(), f)

        loaded = trajectory.EpisodeTrajectory.load(filepath)
        assert loaded.episode_id == "ep-load"
        assert loaded.task_name == "task-load"
        assert loaded.total_reward == 0.5
        assert len(loaded.steps) == 1


# ---------------------------------------------------------------------------
# JsonTrajectoryCollector
# ---------------------------------------------------------------------------


class TestJsonTrajectoryCollector:
    def test_creates_output_dir(self, tmp_path: pathlib.Path):
        output_dir = tmp_path / "trajectories" / "nested"
        collector = trajectory.JsonTrajectoryCollector(output_dir=output_dir)
        assert output_dir.exists()
        assert collector.output_dir == output_dir

    def test_full_episode_saves_file(self, tmp_path: pathlib.Path):
        collector = trajectory.JsonTrajectoryCollector(output_dir=tmp_path)

        collector.begin_episode(task_name="json-task")
        collector.record_step(
            _make_step_record(step_index=0, step_type="FIRST", with_action=False, reward=None, discount=None)
        )
        collector.record_step(_make_step_record(step_index=1, step_type="MID"))
        collector.record_step(
            _make_step_record(step_index=2, step_type="LAST", with_observation=False, reward=0.8, discount=0.0)
        )
        ep = collector.end_episode(truncated=False)

        # Check file was created.
        filepath = tmp_path / f"{ep.episode_id}.json"
        assert filepath.exists()

        # Check file contents.
        with open(filepath) as f:
            data = json.load(f)

        assert data["episode_id"] == ep.episode_id
        assert data["task_name"] == "json-task"
        assert data["num_steps"] == 3
        assert data["total_reward"] == 0.8
        assert data["truncated"] is False
        assert len(data["steps"]) == 3

    def test_load_saved_episode(self, tmp_path: pathlib.Path):
        """Test that saved episodes can be loaded back."""
        collector = trajectory.JsonTrajectoryCollector(output_dir=tmp_path)

        collector.begin_episode(task_name="roundtrip-task")
        collector.record_step(
            _make_step_record(step_index=0, step_type="FIRST", with_action=False, reward=None, discount=None)
        )
        collector.record_step(
            _make_step_record(step_index=1, step_type="LAST", reward=0.5, discount=0.0)
        )
        ep = collector.end_episode()

        filepath = tmp_path / f"{ep.episode_id}.json"
        loaded = trajectory.EpisodeTrajectory.load(filepath)

        assert loaded.episode_id == ep.episode_id
        assert loaded.task_name == ep.task_name
        assert loaded.total_reward == ep.total_reward
        assert loaded.num_steps == ep.num_steps
        assert len(loaded.steps) == len(ep.steps)

    def test_multiple_episodes_create_separate_files(self, tmp_path: pathlib.Path):
        collector = trajectory.JsonTrajectoryCollector(output_dir=tmp_path)

        episode_ids = []
        for i in range(3):
            collector.begin_episode(task_name=f"task-{i}")
            collector.record_step(
                _make_step_record(step_index=0, step_type="FIRST", with_action=False, reward=None, discount=None)
            )
            collector.record_step(
                _make_step_record(step_index=1, step_type="LAST", reward=float(i), discount=0.0)
            )
            ep = collector.end_episode()
            episode_ids.append(ep.episode_id)

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 3

        # All episode IDs should be unique.
        assert len(set(episode_ids)) == 3

    def test_record_step_without_begin_raises(self, tmp_path: pathlib.Path):
        collector = trajectory.JsonTrajectoryCollector(output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="No episode in progress"):
            collector.record_step(_make_step_record())

    def test_end_episode_without_begin_raises(self, tmp_path: pathlib.Path):
        collector = trajectory.JsonTrajectoryCollector(output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="No episode in progress"):
            collector.end_episode()

    def test_implements_protocol(self, tmp_path: pathlib.Path):
        collector = trajectory.JsonTrajectoryCollector(output_dir=tmp_path)
        assert isinstance(collector, trajectory.TrajectoryCollector)

    def test_truncated_flag_saved(self, tmp_path: pathlib.Path):
        collector = trajectory.JsonTrajectoryCollector(output_dir=tmp_path)

        collector.begin_episode(task_name="trunc-task")
        collector.record_step(
            _make_step_record(step_index=0, step_type="FIRST", with_action=False, reward=None, discount=None)
        )
        collector.record_step(_make_step_record(step_index=1, step_type="LAST", reward=0.0, discount=1.0))
        ep = collector.end_episode(truncated=True)

        filepath = tmp_path / f"{ep.episode_id}.json"
        with open(filepath) as f:
            data = json.load(f)

        assert data["truncated"] is True
