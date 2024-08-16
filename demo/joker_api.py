import asyncio
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
    model_name='gpt-4o-mini'
)


@add_example(args=["cat", 1], result=["Why was the cat sitting on the computer? Because it wanted to keep an eye on the mouse!"])
@llm.impl()
async def tell_joke(theme: str, count: int) -> List[str]:
    """
    tell a random joke based on the theme and count
    """
    ...


async def main():
    res = await tell_joke("cat", 3)
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
