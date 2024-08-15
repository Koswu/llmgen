import logging
from typing import List
from pydantic import HttpUrl
from pydantic_settings import BaseSettings
from llmgen import (
    OpenAiApiFactory,
    add_example,
)
# logging.basicConfig(level=logging.DEBUG)


class _MyOpenApiSetting(BaseSettings):
    base_url: HttpUrl
    api_token: str


setting = _MyOpenApiSetting.model_validate({})
llm = OpenAiApiFactory(
    base_url=str(setting.base_url),
    api_key=setting.api_token,
)
ai_impl = llm.get_impl_decorator()


@add_example(args=["dog", 1], result=["Why did the dog sit in the shade? Because he didn't want to be a hot dog!"])
@ai_impl
async def tell_joke(theme: str, count: int) -> List[str]:
    """
    tell a random joke based on the theme and count
    """
    ...


def main():
    res = tell_joke("cat", 3)
    print(res)


if __name__ == "__main__":
    main()
