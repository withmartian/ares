"""LLM client implementation using TransformerLens HookedTransformer."""

from collections.abc import Callable, Sequence
import dataclasses
import inspect
from typing import Any, Protocol, Union, runtime_checkable

import torch
import transformer_lens

from ares.contrib.mech_interp import hook_utils
from ares.environments import base as ares_env
from ares.environments import code_env
from ares import llms


HookNameFn = Callable[[str], str]


@runtime_checkable
class StateIdFn(Protocol):
    """A function that returns the `name` param of `add_hook`/`run_with_hooks` if the hook should be applied given
    the current state - otherwise `None` if it should not.
    """
    async def __call__(self, state: hook_utils.FullyObservableState) -> str | Callable[[str], bool] | None:
        ...


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
    # QUESTION: Is it better to include the "state" object in the name/id fn or in the hook fn?
    fwd_hooks: list[tuple[Union[str, HookNameFn, StateIdFn], transformer_lens.hook_points.HookFunction]] | None = None
    # TODO: Identify better default max_new_tokens size
    max_new_tokens: int = 2048
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

    async def _call_with_hooks(
        self,
        input_ids: torch.Tensor,
        state: hook_utils.FullyObservableState,
        **gen_kwargs: Any,
    ) -> torch.Tensor:
        # self.model.reset_hooks(direction="fwd", including_permanent=False)

        if self.fwd_hooks is not None:
            for name_or_id_fn, hook_fn in self.fwd_hooks:
                # Check if this is a StateIdFn, or if it is the standard format to run in all states
                if inspect.iscoroutinefunction(name_or_id_fn):
                    name = await name_or_id_fn(state)
                else:
                    name = name_or_id_fn

                self.model.add_hook(name, hook_fn, is_permanent=False)  # type: ignore

        with torch.no_grad():
            # TODO: Should we make use of the `__call__(..., return_type="logits | loss")` here instead?
            # Generate completion
            # Note: HookedTransformer.generate returns full sequence including input
            outputs = self.model.generate(
                input_ids,
                **gen_kwargs,
            )
        
        # self.model.reset_hooks(direction="fwd", including_permanent=False)

        assert isinstance(outputs, torch.Tensor)  # typing
        return outputs

    async def __call__(
        self,
        request: llms.LLMRequest,
        env: code_env.CodeEnvironment | None = None,
        timestep: ares_env.TimeStep | None = None,
    ) -> llms.LLMResponse:
        """Generate a completion using the HookedTransformer.

        Args:
            request: LLM request containing messages and optional temperature.

        Returns:
            LLM response with chat completion and cost information.
        """
        # Format messages into text
        messages_list = list(request.messages)
        input_text = self.format_messages_fn(messages_list)

        # Tokenize input
        input_ids = self.model.to_tokens(input_text, prepend_bos=True)
        num_input_tokens = input_ids.shape[-1]

        # TODO: Need to support various truncation methods
        # Truncate if input + max_new_tokens would exceed model's context window
        max_position = self.model.cfg.n_ctx
        if num_input_tokens + self.max_new_tokens > max_position:
            # Leave room for generation
            max_input_tokens = max_position - self.max_new_tokens
            input_ids = input_ids[:, :max_input_tokens]
            num_input_tokens = input_ids.shape[-1]

        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            **self.generation_kwargs,
        }

        # TODO: This should be more generic - why temperature specifically?
        # Add temperature if specified
        if request.temperature is not None:
            gen_kwargs["temperature"] = request.temperature

        outputs = await self._call_with_hooks(
            input_ids,
            state=hook_utils.FullyObservableState(
                timestep=timestep,
                # TODO: Figure out typing here
                container=env._container if env is not None else None,
                step_num=0  # TODO: How to calculate?,
            ),
            **gen_kwargs,
        )

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
