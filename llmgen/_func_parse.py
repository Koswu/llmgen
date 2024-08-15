
import asyncio
from collections import OrderedDict
from functools import cached_property
import inspect
import logging
from typing import Any, Callable, Dict, Generic, Type, cast
from typing_extensions import Annotated
import docstring_parser
import pydantic
from pydantic.fields import FieldInfo
import typing_extensions

_logger = logging.getLogger(__name__)

_P = typing_extensions.ParamSpec("_P")
_T = typing_extensions.TypeVar("_T")


def _has_field_annotation(type_: type) -> bool:
    args = typing_extensions.get_args(type_)
    return any(isinstance(arg, FieldInfo) for arg in args)


class ParsedFunction(Generic[_P, _T]):
    def __init__(self, func: Callable[_P, _T]):
        self._func = func

    @cached_property
    def _docstring(self) -> docstring_parser.Docstring:
        return docstring_parser.parse(self._func.__doc__ or '')

    @cached_property
    def _type_hints(self) -> Dict[str, type]:
        return typing_extensions.get_type_hints(self._func)

    @cached_property
    def _signture(self) -> inspect.Signature:
        return inspect.signature(self._func)

    @cached_property
    def _input_params(self) -> Dict[str, type]:
        parameters = self._signture.parameters
        input_params: Dict[str, type] = OrderedDict()
        for name in parameters:
            if name not in self._type_hints:
                input_params[name] = str
                _logger.info(f"Parameter {name} has no type hint, using str")
            else:
                input_params[name] = self._type_hints[name]
        for param in self._docstring.params:
            if param.arg_name not in input_params:
                _logger.info(
                    f"Parameter {param.arg_name} in docstring not found in function signature"
                )
                continue
            if param.description and not _has_field_annotation(input_params[param.arg_name]):
                _logger.info(
                    f"Adding description to parameter {param.arg_name}: {param.description}"
                )
                input_params[param.arg_name] = cast(type, Annotated[input_params[param.arg_name],
                                                                    pydantic.Field(description=param.description)])
        return input_params

    @cached_property
    def input_model_type(self) -> Type[pydantic.BaseModel]:
        input_params: Dict[str, Any] = {
            name: (param, ...)
            for name, param in self._input_params.items()
        }
        return cast(Type[pydantic.BaseModel], pydantic.create_model(
            "InputModel",
            **input_params
        ))

    @cached_property
    def _output_type(self) -> Type[_T]:
        if "return" not in self._type_hints:
            _logger.info(f"Return type not specified, using str")
            return cast(Type[_T], str)
        res = self._type_hints["return"]
        if self._docstring.returns and not _has_field_annotation(res):
            _logger.info(
                f"Adding description to return type: {self._docstring.returns.description}")
            res = cast(type, Annotated[res, pydantic.Field(
                description=self._docstring.returns.description)])
        return res

    @cached_property
    def output_model_type(self) -> Type[pydantic.BaseModel]:
        return cast(Type[pydantic.BaseModel], pydantic.create_model(
            "OutputModel", result=(self._output_type, ...)
        ))

    def parse_input_param(self, *args, **kwargs) -> pydantic.BaseModel:
        # 将位置参数转换为关键字参数
        new_kwargs = {**kwargs}
        for i, arg in enumerate(args):
            assert i < len(self._input_params), f"Too many arguments: {args}"
            param_name = list(self._input_params.keys())[i]
            new_kwargs[param_name] = arg
        return self.input_model_type(**new_kwargs)

    def parse_output(self, output: _T) -> pydantic.BaseModel:
        return self.output_model_type(result=output)

    def parse_output_model(self, output_model: pydantic.BaseModel) -> _T:
        return getattr(output_model, 'result')

    @property
    def description(self) -> str:
        return self._docstring.description or ""

    @property
    def name(self) -> str:
        return self._func.__name__

    @cached_property
    def is_coroutine(self) -> bool:
        return asyncio.iscoroutinefunction(self._func)
