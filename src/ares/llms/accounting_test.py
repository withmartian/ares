"""Unit tests for the accounting module."""

import decimal
from unittest import mock

import frozendict
import httpx
import pytest
from openai.types.chat import chat_completion as chat_completion_types

from ares.llms.accounting import (
    ModelCost,
    ModelPricing,
    ModelsResponse,
    get_llm_cost,
    martian_cost_list,
)


class TestMartianCostList:
    """Tests for the martian_cost_list function."""

    def test_martian_cost_list_success(self, monkeypatch):
        """Test successful fetching and parsing of model costs."""
        # Clear the cache before testing
        martian_cost_list.cache_clear()

        mock_response_data = {
            "data": [
                {
                    "id": "gpt-4",
                    "pricing": {
                        "prompt": "0.03",
                        "completion": "0.06",
                        "image": None,
                        "request": None,
                        "web_search": None,
                        "internal_reasoning": None,
                    },
                },
                {
                    "id": "gpt-3.5-turbo",
                    "pricing": {
                        "prompt": "0.0015",
                        "completion": "0.002",
                        "image": None,
                        "request": "0.0001",
                        "web_search": None,
                        "internal_reasoning": None,
                    },
                },
            ]
        }

        mock_response = mock.Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client = mock.MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        result = martian_cost_list(client=mock_client)

        # Verify the result is a frozendict
        assert isinstance(result, frozendict.frozendict)

        # Verify the models are present
        assert "gpt-4" in result
        assert "gpt-3.5-turbo" in result

        # Verify the pricing information
        gpt4_cost = result["gpt-4"]
        assert gpt4_cost.id == "gpt-4"
        assert gpt4_cost.pricing.prompt == decimal.Decimal("0.03")
        assert gpt4_cost.pricing.completion == decimal.Decimal("0.06")
        assert gpt4_cost.pricing.image is None
        assert gpt4_cost.pricing.request is None

        gpt35_cost = result["gpt-3.5-turbo"]
        assert gpt35_cost.id == "gpt-3.5-turbo"
        assert gpt35_cost.pricing.prompt == decimal.Decimal("0.0015")
        assert gpt35_cost.pricing.completion == decimal.Decimal("0.002")
        assert gpt35_cost.pricing.request == decimal.Decimal("0.0001")

    def test_martian_cost_list_caching(self, monkeypatch):
        """Test that martian_cost_list results are cached."""
        # Clear the cache before testing
        martian_cost_list.cache_clear()

        mock_response_data = {
            "data": [
                {
                    "id": "test-model",
                    "pricing": {
                        "prompt": "0.01",
                        "completion": "0.02",
                        "image": None,
                        "request": None,
                        "web_search": None,
                        "internal_reasoning": None,
                    },
                }
            ]
        }

        mock_response = mock.Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client = mock.MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        # First call
        result1 = martian_cost_list(client=mock_client)

        # Second call should use cache
        result2 = martian_cost_list(client=mock_client)

        # Verify client.get was only called once (cached)
        assert mock_client.get.call_count == 1

        # Verify results are the same
        assert result1 is result2

    def test_martian_cost_list_http_error(self, monkeypatch):
        """Test handling of HTTP errors."""
        # Clear the cache before testing
        martian_cost_list.cache_clear()

        mock_response = mock.Mock(spec=httpx.Response)
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=mock.Mock(),
            response=mock.Mock(status_code=404),
        )

        mock_client = mock.MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        with pytest.raises(httpx.HTTPStatusError):
            martian_cost_list(client=mock_client)

    def test_martian_cost_list_network_error(self, monkeypatch):
        """Test handling of network errors."""
        # Clear the cache before testing
        martian_cost_list.cache_clear()

        mock_client = mock.MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.NetworkError("Connection failed")
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        with pytest.raises(httpx.NetworkError):
            martian_cost_list(client=mock_client)

    def test_martian_cost_list_invalid_json(self, monkeypatch):
        """Test handling of invalid JSON responses."""
        # Clear the cache before testing
        martian_cost_list.cache_clear()

        mock_response = mock.Mock(spec=httpx.Response)
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status.return_value = None

        mock_client = mock.MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        with pytest.raises(ValueError):
            martian_cost_list(client=mock_client)

    def test_martian_cost_list_creates_default_client(self, monkeypatch):
        """Test that martian_cost_list creates a default client when none provided."""
        # Clear the cache before testing
        martian_cost_list.cache_clear()

        mock_response_data = {
            "data": [
                {
                    "id": "test-model",
                    "pricing": {
                        "prompt": "0.01",
                        "completion": "0.02",
                        "image": None,
                        "request": None,
                        "web_search": None,
                        "internal_reasoning": None,
                    },
                }
            ]
        }

        mock_response = mock.Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client_instance = mock.MagicMock(spec=httpx.Client)
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.__exit__.return_value = None

        with mock.patch("httpx.Client", return_value=mock_client_instance):
            result = martian_cost_list()

        assert isinstance(result, frozendict.frozendict)
        assert "test-model" in result


class TestGetLlmCost:
    """Tests for the get_llm_cost function."""

    def test_get_llm_cost_basic(self):
        """Test basic cost calculation with prompt and completion tokens."""
        cost_mapping = frozendict.frozendict(
            {
                "test-model": ModelCost(
                    id="test-model",
                    pricing=ModelPricing(
                        prompt=decimal.Decimal("0.01"),
                        completion=decimal.Decimal("0.02"),
                        image=None,
                        request=None,
                        web_search=None,
                        internal_reasoning=None,
                    ),
                )
            }
        )

        completion = chat_completion_types.ChatCompletion(
            id="test-completion",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=chat_completion_types.CompletionUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )

        cost = get_llm_cost("test-model", completion, cost_mapping=cost_mapping)

        # 100 * 0.01 + 50 * 0.02 = 1.0 + 1.0 = 2.0
        assert cost == decimal.Decimal("2.0")

    def test_get_llm_cost_with_request_charge(self):
        """Test cost calculation with request charge."""
        cost_mapping = frozendict.frozendict(
            {
                "test-model": ModelCost(
                    id="test-model",
                    pricing=ModelPricing(
                        prompt=decimal.Decimal("0.01"),
                        completion=decimal.Decimal("0.02"),
                        image=None,
                        request=decimal.Decimal("0.001"),
                        web_search=None,
                        internal_reasoning=None,
                    ),
                )
            }
        )

        completion = chat_completion_types.ChatCompletion(
            id="test-completion",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=chat_completion_types.CompletionUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )

        cost = get_llm_cost("test-model", completion, cost_mapping=cost_mapping)

        # 0.001 + 100 * 0.01 + 50 * 0.02 = 0.001 + 1.0 + 1.0 = 2.001
        assert cost == decimal.Decimal("2.001")

    def test_get_llm_cost_decimal_precision(self):
        """Test that decimal precision is maintained correctly."""
        cost_mapping = frozendict.frozendict(
            {
                "test-model": ModelCost(
                    id="test-model",
                    pricing=ModelPricing(
                        prompt=decimal.Decimal("0.000001"),
                        completion=decimal.Decimal("0.000002"),
                        image=None,
                        request=decimal.Decimal("0.0000001"),
                        web_search=None,
                        internal_reasoning=None,
                    ),
                )
            }
        )

        completion = chat_completion_types.ChatCompletion(
            id="test-completion",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=chat_completion_types.CompletionUsage(
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
            ),
        )

        cost = get_llm_cost("test-model", completion, cost_mapping=cost_mapping)

        # 0.0000001 + 1000 * 0.000001 + 500 * 0.000002
        # = 0.0000001 + 0.001 + 0.001 = 0.0020001
        assert cost == decimal.Decimal("0.0020001")

    def test_get_llm_cost_missing_model(self):
        """Test error handling when model is not in cost mapping."""
        cost_mapping = frozendict.frozendict(
            {
                "other-model": ModelCost(
                    id="other-model",
                    pricing=ModelPricing(
                        prompt=decimal.Decimal("0.01"),
                        completion=decimal.Decimal("0.02"),
                        image=None,
                        request=None,
                        web_search=None,
                        internal_reasoning=None,
                    ),
                )
            }
        )

        completion = chat_completion_types.ChatCompletion(
            id="test-completion",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=chat_completion_types.CompletionUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )

        with pytest.raises(ValueError, match="Model test-model not found in cost mapping"):
            get_llm_cost("test-model", completion, cost_mapping=cost_mapping)

    def test_get_llm_cost_missing_usage(self):
        """Test error handling when completion has no usage information."""
        cost_mapping = frozendict.frozendict(
            {
                "test-model": ModelCost(
                    id="test-model",
                    pricing=ModelPricing(
                        prompt=decimal.Decimal("0.01"),
                        completion=decimal.Decimal("0.02"),
                        image=None,
                        request=None,
                        web_search=None,
                        internal_reasoning=None,
                    ),
                )
            }
        )

        completion = chat_completion_types.ChatCompletion(
            id="test-completion",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=None,
        )

        with pytest.raises(ValueError, match="Cannot compute cost of a completion with no usage"):
            get_llm_cost("test-model", completion, cost_mapping=cost_mapping)

    def test_get_llm_cost_none_pricing_fields(self):
        """Test that None pricing fields are treated as zero."""
        cost_mapping = frozendict.frozendict(
            {
                "test-model": ModelCost(
                    id="test-model",
                    pricing=ModelPricing(
                        prompt=None,
                        completion=None,
                        image=None,
                        request=None,
                        web_search=None,
                        internal_reasoning=None,
                    ),
                )
            }
        )

        completion = chat_completion_types.ChatCompletion(
            id="test-completion",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=chat_completion_types.CompletionUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )

        cost = get_llm_cost("test-model", completion, cost_mapping=cost_mapping)

        # All pricing fields are None, so cost should be 0
        assert cost == decimal.Decimal("0")

    def test_get_llm_cost_partial_none_pricing_fields(self):
        """Test cost calculation with some None pricing fields."""
        cost_mapping = frozendict.frozendict(
            {
                "test-model": ModelCost(
                    id="test-model",
                    pricing=ModelPricing(
                        prompt=decimal.Decimal("0.01"),
                        completion=None,
                        image=None,
                        request=None,
                        web_search=None,
                        internal_reasoning=None,
                    ),
                )
            }
        )

        completion = chat_completion_types.ChatCompletion(
            id="test-completion",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=chat_completion_types.CompletionUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )

        cost = get_llm_cost("test-model", completion, cost_mapping=cost_mapping)

        # Only prompt tokens are charged: 100 * 0.01 = 1.0
        assert cost == decimal.Decimal("1.0")

    def test_get_llm_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost_mapping = frozendict.frozendict(
            {
                "test-model": ModelCost(
                    id="test-model",
                    pricing=ModelPricing(
                        prompt=decimal.Decimal("0.01"),
                        completion=decimal.Decimal("0.02"),
                        image=None,
                        request=decimal.Decimal("0.001"),
                        web_search=None,
                        internal_reasoning=None,
                    ),
                )
            }
        )

        completion = chat_completion_types.ChatCompletion(
            id="test-completion",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=chat_completion_types.CompletionUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )

        cost = get_llm_cost("test-model", completion, cost_mapping=cost_mapping)

        # Only request charge: 0.001
        assert cost == decimal.Decimal("0.001")

    def test_get_llm_cost_large_token_counts(self):
        """Test cost calculation with very large token counts."""
        cost_mapping = frozendict.frozendict(
            {
                "test-model": ModelCost(
                    id="test-model",
                    pricing=ModelPricing(
                        prompt=decimal.Decimal("0.00001"),
                        completion=decimal.Decimal("0.00002"),
                        image=None,
                        request=None,
                        web_search=None,
                        internal_reasoning=None,
                    ),
                )
            }
        )

        completion = chat_completion_types.ChatCompletion(
            id="test-completion",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=chat_completion_types.CompletionUsage(
                prompt_tokens=1000000,
                completion_tokens=500000,
                total_tokens=1500000,
            ),
        )

        cost = get_llm_cost("test-model", completion, cost_mapping=cost_mapping)

        # 1000000 * 0.00001 + 500000 * 0.00002 = 10.0 + 10.0 = 20.0
        assert cost == decimal.Decimal("20.0")
