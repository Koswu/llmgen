"""Async streaming demo."""
import asyncio
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from llmgen import OpenAiApiFactory


class StoryRequest(BaseModel):
    protagonist: str = Field(description="Main character name")
    setting: str = Field(description="Story setting")
    genre: str = Field(description="Story genre")


class StoryOutput(BaseModel):
    title: str = Field(description="Story title")
    story: str = Field(description="The short story (2-3 paragraphs)")


class Settings(BaseSettings):
    base_url: str
    api_token: str


async def main():
    settings = Settings.model_validate({})
    factory = OpenAiApiFactory(api_key=settings.api_token, base_url=settings.base_url)
    api = factory.make_api(StoryRequest, StoryOutput)

    request = StoryRequest(protagonist="Ada", setting="a space station", genre="mystery")

    print("Streaming story...\n")
    async for chunk in api.astream(request):
        print(chunk, end="", flush=True)
    print("\n\nDone streaming.")

    # Also demonstrate call_with_usage
    output, usage = api.call_with_usage(request)
    print(f"\nTitle: {output.title}")
    print(f"Tokens used — input: {usage.input_tokens}, output: {usage.output_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
