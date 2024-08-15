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
ai_impl = llm.get_func_impl_decorator()


@add_example(args=["肥宅", 1], result=["为什么肥宅喜欢宅在家里？因为他们觉得外面的世界太瘦了。"])
@ai_impl
def tell_joke(theme: str, count: int) -> List[str]:
    """
    根据主题生成多个笑话，返回笑话列表（内容是中文）
    """
    ...


def main():
    res = tell_joke("琪露诺", 3)
    print(res)


if __name__ == "__main__":
    main()
