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
    kwargs = kwargs or {}

    def wrapper(func: _DecoratedFunc) -> _DecoratedFunc:
        func.add_example_pair(args, result, kwargs=kwargs)
        return func
    return wrapper
