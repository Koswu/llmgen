"""Entity extraction demo: extract structured entities from free text."""
from typing import List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from llmgen import OpenAiApiFactory


class TextInput(BaseModel):
    text: str = Field(description="Raw text to extract entities from")


class Entity(BaseModel):
    name: str = Field(description="The entity name/value")
    entity_type: str = Field(description="Type: PERSON, ORG, LOCATION, DATE, etc.")
    context: str = Field(description="Brief context explaining why this is an entity")


class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(description="All entities found in the text")
    summary: str = Field(description="One-sentence summary of the text")


class Settings(BaseSettings):
    base_url: str
    api_token: str


def main():
    settings = Settings.model_validate({})
    factory = OpenAiApiFactory(api_key=settings.api_token, base_url=settings.base_url)
    api = factory.make_api(
        TextInput,
        ExtractionResult,
        intro_prompt="You are an expert NLP system specialising in named entity recognition.",
    )

    text = (
        "Apple Inc. announced that CEO Tim Cook will visit London next Tuesday "
        "to meet with Prime Minister Rishi Sunak."
    )
    result = api.call(TextInput(text=text))

    print(f"Summary: {result.summary}")
    print("\nEntities found:")
    for entity in result.entities:
        print(f"  [{entity.entity_type}] {entity.name} — {entity.context}")


if __name__ == "__main__":
    main()
