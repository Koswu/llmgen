from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar
from llmgen._openai import (
    OpenAiApiError as OpenAiApiError,
    ExamplePair as ExamplePair,
    OpenAiApiFactory as OpenAiApiFactory,
)
from llmgen._openai import LLMImplmentedFunc, AsyncLLMImplmentedFunc
from pydantic import BaseModel as BaseModel, Field as Field

_DecoratedFunc = TypeVar(
    "_DecoratedFunc", LLMImplmentedFunc, AsyncLLMImplmentedFunc)


def add_example(args: Sequence, result: Any, *, kwargs: Optional[Mapping] = None) -> Callable[[_DecoratedFunc], _DecoratedFunc]:
    """add example to a LLM implemented function, this is a decorator constructor

    Args:
        args (Sequence): function positional arguments
        result (Any): function return value
        kwargs (Optional[Mapping], optional): function keyword arguments. Defaults to None. mean no keyword arguments

    Returns:
        Callable[[_DecoratedFunc], _DecoratedFunc]: a decorator

    Usage:
    ```python
    @add_example(args=["dog", 1], result=["Why did the dog sit in the shade? Because he didn't want to be a hot dog!"])
    @ai_impl
    def tell_joke(theme: str, count: int) -> List[str]:
        ...
    ```
    """
    kwargs = kwargs or {}

    def wrapper(func: _DecoratedFunc) -> _DecoratedFunc:
        func.add_example_pair(args, result, kwargs=kwargs)
        return func
    return wrapper
