from __future__ import annotations

import json
import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import Usage
from typing_extensions import ParamSpec, TypeVar

from llmgen._func_parse import ParsedFunction

_BaseModelInputT = TypeVar("_BaseModelInputT", bound=BaseModel)
_BaseModelOutputT = TypeVar("_BaseModelOutputT", bound=BaseModel)

_logger = logging.getLogger(__name__)


class OpenAiApiError(Exception):
    ...


# Alias for broader use
LlmGenError = OpenAiApiError


def _get_json_schema(model: Union[BaseModel, Type[BaseModel]]) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False)


def _build_system_prompt(
    input_type: Type[BaseModel],
    output_type: Type[BaseModel],
    intro_prompt: Optional[str],
) -> str:
    parts = []
    if intro_prompt:
        parts.append(intro_prompt)
    parts.append("I will give you input and you will give me the output.")
    parts.append(f"Input JSON Schema:\n{_get_json_schema(input_type)}")
    parts.append(f"Output JSON Schema:\n{_get_json_schema(output_type)}")
    return "\n\n".join(parts)


def _build_message_history(
    examples: List[ExamplePair],
) -> List[ModelMessage]:
    history: List[ModelMessage] = []
    for ex in examples:
        history.append(ModelRequest(parts=[UserPromptPart(content=ex.input.model_dump_json())]))
        history.append(ModelResponse(parts=[TextPart(content=ex.output.model_dump_json())]))
    return history


class ExamplePair(BaseModel, Generic[_BaseModelInputT, _BaseModelOutputT]):
    input: _BaseModelInputT
    output: _BaseModelOutputT


class OpenAiApi(Generic[_BaseModelInputT, _BaseModelOutputT]):
    def __init__(
        self,
        model: Model,
        input_type: Type[_BaseModelInputT],
        output_type: Type[_BaseModelOutputT],
        intro_prompt: Optional[str] = None,
        examples: Optional[List[ExamplePair[_BaseModelInputT, _BaseModelOutputT]]] = None,
        model_settings: Optional[ModelSettings] = None,
    ):
        self._model = model
        self._input_type = input_type
        self._output_type = output_type
        self._intro_prompt = intro_prompt
        self._examples: List[ExamplePair] = examples or []
        self._model_settings = model_settings
        system_prompt = _build_system_prompt(input_type, output_type, intro_prompt)
        self._agent: Agent[None, _BaseModelOutputT] = Agent(
            model,
            output_type=output_type,
            system_prompt=system_prompt,
            model_settings=model_settings,
            defer_model_check=True,
        )

    def _history(self) -> List[ModelMessage]:
        return _build_message_history(self._examples)

    def call(self, input: _BaseModelInputT) -> _BaseModelOutputT:
        """
        Calls the LLM with the given input and returns the structured output.

        :raises OpenAiApiError: if the LLM call fails.
        """
        try:
            result = self._agent.run_sync(
                input.model_dump_json(),
                message_history=self._history() or None,
            )
            return result.output
        except Exception as e:
            _logger.exception("Error invoking llm")
            raise OpenAiApiError(str(e)) from e

    async def async_call(self, input: _BaseModelInputT) -> _BaseModelOutputT:
        """
        Asynchronously calls the LLM with the given input and returns the structured output.

        :raises OpenAiApiError: if the LLM call fails.
        """
        try:
            result = await self._agent.run(
                input.model_dump_json(),
                message_history=self._history() or None,
            )
            return result.output
        except Exception as e:
            _logger.exception("Error invoking llm")
            raise OpenAiApiError(str(e)) from e

    def call_with_usage(self, input: _BaseModelInputT) -> Tuple[_BaseModelOutputT, Usage]:
        """Call the LLM and also return token usage information."""
        try:
            result = self._agent.run_sync(
                input.model_dump_json(),
                message_history=self._history() or None,
            )
            return result.output, result.usage()
        except Exception as e:
            _logger.exception("Error invoking llm")
            raise OpenAiApiError(str(e)) from e

    def stream(self, input: _BaseModelInputT) -> Iterator[str]:
        """Stream text chunks from the LLM (sync generator)."""
        try:
            with self._agent.run_stream_sync(
                input.model_dump_json(),
                message_history=self._history() or None,
            ) as stream_result:
                for chunk in stream_result.stream_text():
                    yield chunk
        except Exception as e:
            _logger.exception("Error streaming llm")
            raise OpenAiApiError(str(e)) from e

    async def astream(self, input: _BaseModelInputT) -> AsyncIterator[str]:
        """Stream text chunks from the LLM (async generator)."""
        try:
            async with self._agent.run_stream(
                input.model_dump_json(),
                message_history=self._history() or None,
            ) as stream_result:
                async for chunk in stream_result.stream_text():
                    yield chunk
        except Exception as e:
            _logger.exception("Error streaming llm")
            raise OpenAiApiError(str(e)) from e


_P = ParamSpec("_P")
_T = TypeVar("_T")


class _BaseLLMFunc(Generic[_P, _T]):
    """Shared base for sync and async LLM-implemented functions."""

    def __init__(self, parsed_func: ParsedFunction[_P, _T], api_factory: BaseApiFactory):
        self._parsed_func: ParsedFunction[_P, _T] = parsed_func
        self._example_pairs: List[ExamplePair] = []
        self._api_factory = api_factory

    @property
    def _api(self) -> OpenAiApi:
        return self._api_factory.make_api(
            self._parsed_func.input_model_type,
            self._parsed_func.output_model_type,
            self._example_pairs,
            intro_prompt=(
                f"You will simulate a function. "
                f"The function name: {self._parsed_func.name}, "
                f"The function description: {self._parsed_func.description}"
            ),
        )

    def add_example(
        self,
        args: Sequence[Any],
        output: _T,
        *,
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        kwargs = kwargs or {}
        input_model = self._parsed_func.parse_input_param(*args, **kwargs)
        output_model = self._parsed_func.parse_output(output)
        self._example_pairs.append(ExamplePair(input=input_model, output=output_model))


class LLMImplmentedFunc(_BaseLLMFunc[_P, _T]):
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        input_model = self._parsed_func.parse_input_param(*args, **kwargs)
        return self._parsed_func.parse_output_model(self._api.call(input_model))


class AsyncLLMImplmentedFunc(_BaseLLMFunc[_P, _T]):
    async def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        input_model = self._parsed_func.parse_input_param(*args, **kwargs)
        return self._parsed_func.parse_output_model(await self._api.async_call(input_model))


class FuncImplDecorator:
    def __init__(self, api_factory: BaseApiFactory):
        self._api_factory = api_factory

    @overload
    def __call__(self, func: Callable[_P, _T]) -> LLMImplmentedFunc[_P, _T]: ...

    @overload
    def __call__(
        self, func: Callable[_P, Coroutine[Any, Any, _T]]
    ) -> AsyncLLMImplmentedFunc[_P, _T]: ...

    def __call__(self, func):
        parsed_func = ParsedFunction(func)
        if parsed_func.is_coroutine:
            return AsyncLLMImplmentedFunc(parsed_func, self._api_factory)
        return LLMImplmentedFunc(parsed_func, self._api_factory)


class BaseApiFactory:
    """Base factory — subclass to provide a pydantic-ai Model."""

    def _make_model(self) -> Model:
        raise NotImplementedError

    def _make_model_settings(self) -> Optional[ModelSettings]:
        return None

    def make_api(
        self,
        input_type: Type[_BaseModelInputT],
        output_type: Type[_BaseModelOutputT],
        examples: Optional[List[ExamplePair[_BaseModelInputT, _BaseModelOutputT]]] = None,
        intro_prompt: Optional[str] = None,
    ) -> OpenAiApi[_BaseModelInputT, _BaseModelOutputT]:
        return OpenAiApi(
            model=self._make_model(),
            input_type=input_type,
            output_type=output_type,
            intro_prompt=intro_prompt,
            examples=examples,
            model_settings=self._make_model_settings(),
        )

    def impl(self) -> FuncImplDecorator:
        """Return a decorator that implements a function body using the LLM."""
        return FuncImplDecorator(self)


class OpenAiApiFactory(BaseApiFactory):
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ):
        """
        Initializes a new OpenAiApiFactory.

        Args:
            api_key: The API key for the OpenAI-compatible endpoint.
            base_url: Base URL of the API (default: OpenAI).
            model_name: Model to use (default: gpt-4o).
            temperature: Sampling temperature (default: 0.2).
            max_tokens: Maximum tokens to generate (default: None).
        """
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        self._model = OpenAIModel(model_name, provider=provider)
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _make_model(self) -> Model:
        return self._model

    def _make_model_settings(self) -> Optional[ModelSettings]:
        settings: dict = {"temperature": self._temperature}
        if self._max_tokens is not None:
            settings["max_tokens"] = self._max_tokens
        return ModelSettings(**settings)  # type: ignore[arg-type]


class AgentFactory(BaseApiFactory):
    """Generic factory that accepts any pydantic-ai model name or Model instance."""

    def __init__(
        self,
        model: Union[str, Model],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ):
        """
        Args:
            model: A pydantic-ai model name string (e.g. 'openai:gpt-4o') or a Model instance.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        self._model_spec = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _make_model(self) -> Union[str, Model]:  # type: ignore[override]
        return self._model_spec

    def _make_model_settings(self) -> Optional[ModelSettings]:
        settings: dict = {"temperature": self._temperature}
        if self._max_tokens is not None:
            settings["max_tokens"] = self._max_tokens
        return ModelSettings(**settings)  # type: ignore[arg-type]

    def make_api(
        self,
        input_type: Type[_BaseModelInputT],
        output_type: Type[_BaseModelOutputT],
        examples: Optional[List[ExamplePair[_BaseModelInputT, _BaseModelOutputT]]] = None,
        intro_prompt: Optional[str] = None,
    ) -> OpenAiApi[_BaseModelInputT, _BaseModelOutputT]:
        return OpenAiApi(
            model=self._model_spec,  # type: ignore[arg-type]
            input_type=input_type,
            output_type=output_type,
            intro_prompt=intro_prompt,
            examples=examples,
            model_settings=self._make_model_settings(),
        )
