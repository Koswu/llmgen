from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar

from pydantic import BaseModel as BaseModel
from pydantic import Field as Field

from llmgen._openai import AgentFactory as AgentFactory
from llmgen._openai import AsyncLLMImplmentedFunc, LLMImplmentedFunc
from llmgen._openai import ExamplePair as ExamplePair
from llmgen._openai import LlmGenError as LlmGenError
from llmgen._openai import OpenAiApiError as OpenAiApiError
from llmgen._openai import OpenAiApiFactory as OpenAiApiFactory

_DecoratedFunc = TypeVar("_DecoratedFunc", LLMImplmentedFunc, AsyncLLMImplmentedFunc)


def add_example(
    args: Sequence,
    result: Any,
    *,
    kwargs: Optional[Mapping] = None,
) -> Callable[[_DecoratedFunc], _DecoratedFunc]:
    """add example to a LLM implemented function, this is a decorator constructor

    Args:
        args (Sequence): function positional arguments
        result (Any): function return value
        kwargs (Optional[Mapping], optional): function keyword arguments.
            Defaults to None (no keyword arguments).

    Returns:
        Callable[[_DecoratedFunc], _DecoratedFunc]: a decorator

    Usage:
    ```python
    @add_example(args=["dog", 1], result=["Why did the dog sit in the shade?"])
    @ai_impl
    def tell_joke(theme: str, count: int) -> List[str]:
        ...
    ```
    """
    kwargs = kwargs or {}

    def wrapper(func: _DecoratedFunc) -> _DecoratedFunc:
        func.add_example(args, result, kwargs=kwargs)
        return func

    return wrapper

