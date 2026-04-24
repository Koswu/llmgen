"""Translation demo: translate text between languages using llmgen."""
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from llmgen import OpenAiApiFactory


class TranslateRequest(BaseModel):
    text: str = Field(description="Text to translate")
    source_language: str = Field(description="Source language (e.g. 'English')")
    target_language: str = Field(description="Target language (e.g. 'Spanish')")


class TranslateResponse(BaseModel):
    translated_text: str = Field(description="The translated text")
    notes: str = Field(description="Any translation notes or alternative phrasing", default="")


class Settings(BaseSettings):
    base_url: str
    api_token: str


def main():
    settings = Settings.model_validate({})
    factory = OpenAiApiFactory(api_key=settings.api_token, base_url=settings.base_url)
    api = factory.make_api(TranslateRequest, TranslateResponse)

    result = api.call(
        TranslateRequest(text="Hello, world!", source_language="English", target_language="French")
    )
    print(f"Translated: {result.translated_text}")
    if result.notes:
        print(f"Notes: {result.notes}")


if __name__ == "__main__":
    main()
