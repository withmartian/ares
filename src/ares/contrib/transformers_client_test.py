"""Unit tests for TransformersLLMClient."""

import asyncio
from unittest import mock

import pytest
import torch
import transformers

from ares.contrib import transformers_client
from ares.llms import request as request_lib
from ares.llms import response as response_lib


class TestDeviceDetection:
    """Tests for device detection logic."""

    def test_detect_device_cuda_available(self):
        """Test CUDA device is selected when available."""
        with mock.patch("torch.cuda.is_available", return_value=True):
            assert transformers_client._detect_device() == "cuda"

    def test_detect_device_mps_available(self):
        """Test MPS device is selected when CUDA not available."""
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=True),
        ):
            assert transformers_client._detect_device() == "mps"

    def test_detect_device_cpu_fallback(self):
        """Test CPU device is selected as fallback."""
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert transformers_client._detect_device() == "cpu"


class TestTorchDtype:
    """Tests for torch dtype selection."""

    def test_auto_dtype_cuda_with_bf16(self):
        """Test auto dtype selects bfloat16 on CUDA with bf16 support."""
        with mock.patch("torch.cuda.is_bf16_supported", return_value=True):
            dtype = transformers_client._get_torch_dtype("auto", "cuda")
            assert dtype == torch.bfloat16

    def test_auto_dtype_mps(self):
        """Test auto dtype selects float16 on MPS."""
        dtype = transformers_client._get_torch_dtype("auto", "mps")
        assert dtype == torch.float16

    def test_auto_dtype_cpu(self):
        """Test auto dtype selects float32 on CPU."""
        dtype = transformers_client._get_torch_dtype("auto", "cpu")
        assert dtype == torch.float32

    def test_explicit_dtype(self):
        """Test explicit dtype selection."""
        assert transformers_client._get_torch_dtype("float32", "cuda") == torch.float32
        assert transformers_client._get_torch_dtype("float16", "cuda") == torch.float16
        assert transformers_client._get_torch_dtype("bfloat16", "cuda") == torch.bfloat16


class TestTransformersLLMClientInitialization:
    """Tests for client initialization and configuration."""

    def test_client_initialization(self):
        """Test client can be initialized with required parameters."""
        client = transformers_client.TransformersLLMClient(
            model_name="test-model",
        )
        assert client.model_name == "test-model"
        assert client.batch_wait_ms == 50
        assert client.max_batch_size == 8
        assert client.device == "auto"
        assert client.torch_dtype == "auto"

    def test_client_custom_parameters(self):
        """Test client accepts custom parameters."""
        client = transformers_client.TransformersLLMClient(
            model_name="test-model",
            batch_wait_ms=100,
            max_batch_size=16,
            device="cpu",
            torch_dtype="float32",
            max_new_tokens=256,
            temperature=0.5,
            output_hidden_states=True,
        )
        assert client.batch_wait_ms == 100
        assert client.max_batch_size == 16
        assert client.device == "cpu"
        assert client.torch_dtype == "float32"
        assert client.max_new_tokens == 256
        assert client.temperature == 0.5
        assert client.output_hidden_states is True

    def test_cached_properties(self):
        """Test cached properties are initialized correctly."""
        client = transformers_client.TransformersLLMClient(model_name="test-model")
        # Properties should be cached but not accessed yet
        assert "_request_queue" not in client.__dict__
        assert "_inference_task" not in client.__dict__


class TestTransformersLLMClientLifecycle:
    """Tests for client lifecycle behavior."""

    @pytest.mark.asyncio
    async def test_lazy_task_start(self):
        """Test background task starts lazily via cached_property."""
        client = transformers_client.TransformersLLMClient(model_name="test-model")

        # Task should not be cached yet
        assert "_inference_task" not in client.__dict__

        # Mock the model and tokenizer
        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "test"
        mock_tokenizer.pad_token_id = 0

        mock_batch = mock.MagicMock()
        mock_batch.__getitem__ = (
            lambda _, key: torch.tensor([[1, 2, 3]]) if key == "input_ids" else torch.tensor([[1, 1, 1]])
        )
        mock_batch.to = mock.Mock(return_value=mock_batch)
        mock_tokenizer.return_value = mock_batch
        mock_tokenizer.batch_decode.return_value = ["Response"]

        mock_output = mock.MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_output

        with (
            mock.patch.object(type(client), "_model", new_callable=mock.PropertyMock, return_value=mock_model),
            mock.patch.object(type(client), "_tokenizer", new_callable=mock.PropertyMock, return_value=mock_tokenizer),
        ):
            req = request_lib.LLMRequest(messages=[{"role": "user", "content": "test"}])

            # Make request - should start task via cached_property
            response = await client(req)

            # Task should now be cached
            assert "_inference_task" in client.__dict__
            assert isinstance(client._inference_task, asyncio.Task)
            assert isinstance(response, response_lib.LLMResponse)


class TestTransformersLLMClientBatching:
    """Tests for request batching behavior."""

    @pytest.mark.asyncio
    async def test_single_request_processing(self):
        """Test processing a single request."""
        client = transformers_client.TransformersLLMClient(
            model_name="test-model",
            batch_wait_ms=10,  # Short wait for testing
        )

        # Create mock model and tokenizer
        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()

        # Configure tokenizer mock
        mock_tokenizer.apply_chat_template.return_value = "User: test\nAssistant:"
        mock_tokenizer.pad_token_id = 0

        # Mock the tokenizer call to return BatchEncoding-like dict
        mock_batch = mock.MagicMock()
        mock_batch.__getitem__ = (
            lambda _, key: torch.tensor([[1, 2, 3]]) if key == "input_ids" else torch.tensor([[1, 1, 1]])
        )
        mock_batch.to = mock.Mock(return_value=mock_batch)
        mock_tokenizer.return_value = mock_batch
        mock_tokenizer.batch_decode.return_value = ["Response text"]

        # Configure model mock
        mock_output = mock.MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5, 6]])  # 3 input + 3 generated
        mock_model.generate.return_value = mock_output

        with (
            mock.patch.object(
                type(client),
                "_model",
                new_callable=mock.PropertyMock,
                return_value=mock_model,
            ),
            mock.patch.object(
                type(client),
                "_tokenizer",
                new_callable=mock.PropertyMock,
                return_value=mock_tokenizer,
            ),
        ):
            req = request_lib.LLMRequest(
                messages=[{"role": "user", "content": "test"}],
            )

            response = await client(req)

            assert isinstance(response, response_lib.LLMResponse)
            assert len(response.data) == 1
            assert response.data[0].content == "Response text"
            assert response.cost == 0.0
            assert response.usage.prompt_tokens > 0
            assert response.usage.generated_tokens > 0

    @pytest.mark.asyncio
    async def test_batch_multiple_requests(self):
        """Test batching multiple concurrent requests."""
        client = transformers_client.TransformersLLMClient(
            model_name="test-model",
            batch_wait_ms=50,
            max_batch_size=3,
        )

        # Create mock model and tokenizer
        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()

        # Configure tokenizer for batch
        mock_tokenizer.apply_chat_template.return_value = "User: test\nAssistant:"
        mock_tokenizer.pad_token_id = 0

        # Mock tokenizer to return batch of 3
        def tokenizer_side_effect(*args, **_):
            batch_size = len(args[0]) if args else 1
            mock_batch = mock.MagicMock()
            mock_batch.__getitem__ = lambda _, key: (
                torch.zeros((batch_size, 5), dtype=torch.long)
                if key == "input_ids"
                else torch.ones((batch_size, 5), dtype=torch.long)
            )
            mock_batch.to = mock.Mock(return_value=mock_batch)
            return mock_batch

        mock_tokenizer.side_effect = tokenizer_side_effect
        mock_tokenizer.batch_decode.return_value = ["Response 1", "Response 2", "Response 3"]

        # Configure model to return batch
        mock_output = mock.MagicMock()
        mock_output.sequences = torch.zeros((3, 8), dtype=torch.long)  # 3 requests, 8 tokens each
        mock_model.generate.return_value = mock_output

        with (
            mock.patch.object(
                type(client),
                "_model",
                new_callable=mock.PropertyMock,
                return_value=mock_model,
            ),
            mock.patch.object(
                type(client),
                "_tokenizer",
                new_callable=mock.PropertyMock,
                return_value=mock_tokenizer,
            ),
        ):
            # Submit 3 requests concurrently
            requests = [request_lib.LLMRequest(messages=[{"role": "user", "content": f"test {i}"}]) for i in range(3)]

            responses = await asyncio.gather(*[client(req) for req in requests])

            assert len(responses) == 3
            for i, resp in enumerate(responses):
                assert isinstance(resp, response_lib.LLMResponse)
                assert resp.data[0].content == f"Response {i + 1}"

            # Verify generate was called once with batch
            mock_model.generate.assert_called_once()


class TestTransformersLLMClientActivationHooks:
    """Tests for activation capture functionality."""

    @pytest.mark.asyncio
    async def test_activation_callback_called(self):
        """Test activation callback is called when enabled."""
        captured_activations = []

        def capture_callback(_activations):
            captured_activations.append(_activations)

        client = transformers_client.TransformersLLMClient(
            model_name="test-model",
            output_hidden_states=True,
            output_attentions=True,
            activation_callback=capture_callback,
        )

        # Create mock model and tokenizer
        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()

        mock_tokenizer.apply_chat_template.return_value = "User: test\nAssistant:"
        mock_tokenizer.pad_token_id = 0

        mock_batch = mock.MagicMock()
        mock_batch.__getitem__ = (
            lambda _, key: torch.tensor([[1, 2, 3]]) if key == "input_ids" else torch.tensor([[1, 1, 1]])
        )
        mock_batch.to = mock.Mock(return_value=mock_batch)
        mock_tokenizer.return_value = mock_batch
        mock_tokenizer.batch_decode.return_value = ["Response"]

        # Configure model with hidden states and attentions
        mock_output = mock.MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        mock_output.hidden_states = (torch.randn(1, 5, 768),)  # (batch, seq, hidden)
        mock_output.attentions = (torch.randn(1, 12, 5, 5),)  # (batch, heads, seq, seq)
        mock_model.generate.return_value = mock_output

        with (
            mock.patch.object(
                type(client),
                "_model",
                new_callable=mock.PropertyMock,
                return_value=mock_model,
            ),
            mock.patch.object(
                type(client),
                "_tokenizer",
                new_callable=mock.PropertyMock,
                return_value=mock_tokenizer,
            ),
        ):
            req = request_lib.LLMRequest(
                messages=[{"role": "user", "content": "test"}],
            )

            await client(req)

            # Verify callback was called
            assert len(captured_activations) == 1
            acts = captured_activations[0]

            assert "input_ids" in acts
            assert "generated_ids" in acts
            assert "hidden_states" in acts
            assert "attentions" in acts

    @pytest.mark.asyncio
    async def test_no_callback_when_disabled(self):
        """Test activation callback is not called when outputs disabled."""
        callback_called = False

        def capture_callback(_):
            nonlocal callback_called
            callback_called = True

        client = transformers_client.TransformersLLMClient(
            model_name="test-model",
            output_hidden_states=False,  # Disabled
            output_attentions=False,  # Disabled
            activation_callback=capture_callback,
        )

        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()

        mock_tokenizer.apply_chat_template.return_value = "User: test\nAssistant:"
        mock_tokenizer.pad_token_id = 0

        mock_batch = mock.MagicMock()
        mock_batch.__getitem__ = (
            lambda _, key: torch.tensor([[1, 2, 3]]) if key == "input_ids" else torch.tensor([[1, 1, 1]])
        )
        mock_batch.to = mock.Mock(return_value=mock_batch)
        mock_tokenizer.return_value = mock_batch
        mock_tokenizer.batch_decode.return_value = ["Response"]

        mock_output = mock.MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_output

        with (
            mock.patch.object(
                type(client),
                "_model",
                new_callable=mock.PropertyMock,
                return_value=mock_model,
            ),
            mock.patch.object(
                type(client),
                "_tokenizer",
                new_callable=mock.PropertyMock,
                return_value=mock_tokenizer,
            ),
        ):
            req = request_lib.LLMRequest(
                messages=[{"role": "user", "content": "test"}],
            )

            await client(req)

            # Callback should not be called
            assert callback_called is False


@pytest.mark.asyncio
async def test_integration_with_minimal_model():
    """Integration test with a minimal GPT2 model that requires no downloads.

    Creates a tiny GPT2 model from scratch for testing the full pipeline.
    """
    # Create minimal GPT2 config - vocab_size must match GPT2Tokenizer (50257)
    config = transformers.GPT2Config(
        vocab_size=50257,  # Must match GPT2Tokenizer vocab
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )

    # Create model from config (no downloads needed)
    minimal_model = transformers.GPT2LMHeadModel(config)
    minimal_model.eval()

    # Create minimal tokenizer
    minimal_tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        "gpt2",  # Uses cached tokenizer vocab
        model_max_length=32,
    )
    minimal_tokenizer.pad_token = minimal_tokenizer.eos_token
    minimal_tokenizer.padding_side = "left"
    # Add minimal chat template
    minimal_tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"

    client = transformers_client.TransformersLLMClient(
        model_name="test-model",
        device="cpu",
        torch_dtype="float32",
        max_new_tokens=5,
        batch_wait_ms=10,
    )

    # Replace with our minimal model
    with (
        mock.patch.object(
            type(client),
            "_model",
            new_callable=mock.PropertyMock,
            return_value=minimal_model,
        ),
        mock.patch.object(
            type(client),
            "_tokenizer",
            new_callable=mock.PropertyMock,
            return_value=minimal_tokenizer,
        ),
    ):
        req = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )

        response = await client(req)

        assert isinstance(response, response_lib.LLMResponse)
        assert len(response.data) == 1
        assert isinstance(response.data[0].content, str)
        assert response.cost == 0.0
        assert response.usage.prompt_tokens > 0
        assert response.usage.generated_tokens > 0
