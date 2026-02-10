"""LLM client implementation using TransformerLens HookedTransformer."""

from collections.abc import Callable, Sequence
import dataclasses
import inspect
from typing import Any

import torch
import transformer_lens

from ares.environments import base as ares_env
from ares.environments import code_env
from ares import llms


@dataclasses.dataclass
class HookedTransformerLLMClient:
    """LLM client that uses TransformerLens HookedTransformer for inference.

    This client enables mechanistic interpretability research by providing access to
    intermediate activations and allowing hook-based interventions during agent execution.

    Args:
        model: A TransformerLens HookedTransformer instance.
        max_new_tokens: Maximum number of tokens to generate per completion.
        generation_kwargs: Additional keyword arguments passed to model.generate().
        format_messages_fn: Optional function to convert chat messages to model input.
            If None, uses a simple concatenation of message contents.

    Example:
        ```python
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained("gpt2-small")
        client = HookedTransformerLLMClient(
            model=model,
            max_new_tokens=512
        )

        # Use with ARES environments
        async with env:
            ts = await env.reset()
            while not ts.last():
                response = await client(ts.observation)
                ts = await env.step(response)
        ```
    """

    model: transformer_lens.HookedTransformer
    # TODO: Identify better default max_new_tokens size
    max_new_tokens: int = 512
    generation_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    _format_messages_fn: Callable[[Sequence[Any]], str] | None = dataclasses.field(default=None, repr=False)

    @property
    def format_messages_fn(self) -> Callable[[Sequence[Any]], str]:
        """Get the message formatting function."""
        if self._format_messages_fn is None:
            return self._default_format_messages
        return self._format_messages_fn

    @staticmethod
    def _default_format_messages(messages: Sequence[Any]) -> str:
        """Default message formatter that concatenates all message contents."""
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_parts.append(f"{role.upper()}: {content}")
        return "\n\n".join(formatted_parts)

    async def __call__(
        self,
        request: llms.LLMRequest,
        max_output_tokens: int | None = None,
    ) -> llms.LLMResponse:
        """Generate a completion using the HookedTransformer.

        Args:
            request: LLM request containing messages and optional temperature.

        Returns:
            LLM response with chat completion and cost information.
        """
        max_output_tokens = max_output_tokens or request.max_output_tokens or self.max_new_tokens

        # Format messages into text
        messages_list = []
        if request.system_prompt:
            messages_list.append({"role": "system", "content": request.system_prompt})
        messages_list.extend(request.messages)

        # Tokenize input
        # TODO: Need to support various truncation methods
        #       ESPECIALLY keeping the <assistant> token to indicate a new turn
        assert self.model.tokenizer is not None
        input_ids = self.model.tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=True,
            truncation=True,
            max_length=self.model.cfg.n_ctx - max_output_tokens,
        )
        input_ids = input_ids.to(self.model.cfg.device)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        num_input_tokens = input_ids.shape[-1]

        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_output_tokens,
            **self.generation_kwargs,
        }

        # TODO: This should be more generic - why temperature specifically?
        # Add temperature if specified
        if request.temperature is not None:
            gen_kwargs["temperature"] = request.temperature

        with torch.no_grad():
            # TODO: Should we make use of the `__call__(..., return_type="logits | loss")` here instead?
            # Generate completion
            # Note: HookedTransformer.generate returns full sequence including input
            outputs = self.model.generate(
                input_ids,
                **gen_kwargs,
            )
        assert isinstance(outputs, torch.Tensor)  # typing

        # Extract only the generated tokens
        num_output_tokens = outputs.shape[-1] - num_input_tokens
        output_ids = outputs[0, num_input_tokens:]

        # Decode output
        output_text = self.model.to_string(output_ids)
        assert isinstance(output_text, str)  # typing

        return llms.LLMResponse(
            data=[llms.TextData(content=output_text)],
            cost=0.0,  # Local inference has no cost
            usage=llms.Usage(
                prompt_tokens=num_input_tokens,
                generated_tokens=num_output_tokens,
            ),
        )


def create_hooked_transformer_client_with_chat_template(
    model: transformer_lens.HookedTransformer,
    tokenizer: Any,
    max_new_tokens: int = 2048,
    **generation_kwargs: Any,
) -> HookedTransformerLLMClient:
    """Create a HookedTransformerLLMClient with proper chat template formatting.

    This factory function creates a client that uses the tokenizer's chat template
    (like Qwen2.5, Llama, etc.) to properly format messages.

    Args:
        model: TransformerLens HookedTransformer instance.
        tokenizer: HuggingFace tokenizer with apply_chat_template method.
        max_new_tokens: Maximum tokens to generate.
        **generation_kwargs: Additional arguments for generation.

    Returns:
        Configured HookedTransformerLLMClient instance.

    Example:
        ```python
        from transformer_lens import HookedTransformer
        from transformers import AutoTokenizer

        model = HookedTransformer.from_pretrained("gpt2-small")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        client = create_hooked_transformer_client_with_chat_template(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
        )
        ```
    """

    def format_with_chat_template(messages: Sequence[Any]) -> str:
        """Format messages using tokenizer's chat template."""
        # Apply chat template without tokenization (just get the text)
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return formatted

    return HookedTransformerLLMClient(
        model=model,
        max_new_tokens=max_new_tokens,
        generation_kwargs=generation_kwargs,
        _format_messages_fn=format_with_chat_template,
    )
