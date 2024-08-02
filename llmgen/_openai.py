import json
import logging
from typing import Any, Generic, List, Optional, Tuple, Type, Union
import uuid
from typing_extensions import TypeVar
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

_BaseModelInputT = TypeVar("_BaseModelInputT", bound=BaseModel)
_BaseModelOutputT = TypeVar("_BaseModelOutputT", bound=BaseModel)

_logger = logging.getLogger(__name__)


class OpenAiApiError(Exception): ...


def _get_json_schema(model: Union[BaseModel, Type[BaseModel]]) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False)


class ExamplePair(BaseModel, Generic[_BaseModelInputT, _BaseModelOutputT]):
    input: _BaseModelInputT
    output: _BaseModelOutputT


class _ApiPromptFactory(Generic[_BaseModelInputT, _BaseModelOutputT]):
    def __init__(
        self,
        input: _BaseModelInputT,
        output_type: Type[_BaseModelOutputT],
        intro_prompt: Optional[str] = None,
        examples: Optional[
            List[ExamplePair[_BaseModelInputT, _BaseModelOutputT]]
        ] = None,
    ):
        self._messages: List[BaseMessage] = []
        self._output_type = output_type
        self._intro_prompt = intro_prompt
        self._input = input
        self._examples: List[ExamplePair] = examples or []

    def _add_input_prompt(self, input: _BaseModelInputT):
        self._messages.extend(
            [
                SystemMessage("Your input here:"),
                SystemMessage(input.model_dump_json()),
                SystemMessage("Now, please provide the output:"),
            ]
        )

    def get_message(self) -> List[BaseMessage]:
        if self._intro_prompt:
            self._messages.append(SystemMessage(self._intro_prompt))
        self._messages.extend(
            [
                SystemMessage("I will give you input and you will give me the output."),
                SystemMessage(
                    "Your input Schema will defined by the following JSON Schema:"
                ),
                SystemMessage(content=_get_json_schema(self._input)),
                SystemMessage(
                    "Please provide the output in the following JSON Schema:"
                ),
                SystemMessage(content=_get_json_schema(self._output_type)),
            ]
        )
        for example in self._examples:
            self._add_input_prompt(example.input)
            self._messages.append(AIMessage(example.output.model_dump_json()))

        self._add_input_prompt(self._input)
        return self._messages


class OpenAiApi(Generic[_BaseModelInputT, _BaseModelOutputT]):
    def __init__(
        self,
        llm: ChatOpenAI,
        input_type: Type[_BaseModelInputT],
        output_type: Type[_BaseModelOutputT],
        intro_prompt: Optional[str] = None,
        examples: Optional[
            list[ExamplePair[_BaseModelInputT, _BaseModelOutputT]]
        ] = None,
    ):
        self._llm = llm
        self._input_type = input_type
        self._output_type = output_type
        self._intro_prompt = intro_prompt
        self._output_parser = PydanticOutputParser(pydantic_object=self._output_type)
        self._examples = examples or []

    def call(self, input: _BaseModelInputT) -> _BaseModelOutputT:
        """
        Calls the OpenAI API with the given input and returns the output.

        :param input: The input to be passed to the OpenAI API.
        :type input: _BaseModelInputT
        :return: The output returned by the OpenAI API.
        :rtype: _BaseModelOutputT
        :raises: OpenAiApiError if there is an error invoking the OpenAI API.
        """
        call_id = uuid.uuid4()
        logger = _logger.getChild(call_id.hex)
        factory = _ApiPromptFactory(
            input=input,
            output_type=self._output_type,
            intro_prompt=self._intro_prompt,
            examples=self._examples,
        )
        messages = factory.get_message()
        logger.debug(f"Invoking llm with messages: {messages}")
        try:
            res = self._llm.invoke(messages)
            logger.debug(f"llm invoked, response: {res}")
            return self._output_parser.invoke(res)
        except Exception as e:
            logger.exception("Error invoking llm")
            raise OpenAiApiError(str(e)) from e

    async def async_call(self, input: _BaseModelInputT) -> _BaseModelOutputT:
        """
        Asynchronously calls the OpenAI API with the given input and returns the output.

        :param input: The input for the API call.
        :type input: _BaseModelInputT
        :return: The output of the API call.
        :rtype: _BaseModelOutputT
        :raises: OpenAiApiError if there is an error invoking the API.
        """
        call_id = uuid.uuid4()
        logger = _logger.getChild(call_id.hex)
        factory = _ApiPromptFactory(
            input=input,
            output_type=self._output_type,
            intro_prompt=self._intro_prompt,
            examples=self._examples,
        )
        messages = factory.get_message()
        logger.debug(f"Invoking llm with messages: {messages}")
        try:
            res = await self._llm.ainvoke(messages)
            logger.debug(f"llm invoked, response: {res}")
            return await self._output_parser.ainvoke(res)
        except Exception as e:
            logger.exception("Error invoking llm")
            raise OpenAiApiError(str(e)) from e


class OpenAiApiFactory:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        intro_prompt: Optional[str] = None,
    ):
        """
        Initializes a new instance of the OpenAI class.

        Args:
            api_key (str): The API key for accessing the OpenAI API.
            base_url (str, optional): The base URL for the OpenAI API. Defaults to "https://api.openai.com/v1".
            model_name (str, optional): The name of the language model to use. Defaults to "gpt-4o".
            temperature (float, optional): The temperature parameter for generating text. Defaults to 0.2.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to None.
            intro_prompt (str, optional): An introductory prompt to provide context for the generated text. Defaults to None.
        """
        from pydantic.v1.types import SecretStr

        self._llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=SecretStr(api_key),
        )
        self._intro_prompt = intro_prompt

    def make_api(
        self,
        input_type: Type[_BaseModelInputT],
        output_type: Type[_BaseModelOutputT],
        examples: Optional[
            List[ExamplePair[_BaseModelInputT, _BaseModelOutputT]]
        ] = None,
    ) -> OpenAiApi[_BaseModelInputT, _BaseModelOutputT]:
        """
        Creates an instance of the OpenAiApi class.

        Args:
            input_type (Type[_BaseModelInputT]): The type of input for the API.
            output_type (Type[_BaseModelOutputT]): The type of output for the API.
            examples (Optional[List[ExamplePair[_BaseModelInputT, _BaseModelOutputT]]], optional): A list of example input-output pairs. Defaults to None.

        Returns:
            OpenAiApi[_BaseModelInputT, _BaseModelOutputT]: An instance of the OpenAiApi class.
        """
        return OpenAiApi(
            llm=self._llm,
            input_type=input_type,
            output_type=output_type,
            intro_prompt=self._intro_prompt,
            examples=examples,
        )
