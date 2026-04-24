"""Sentiment classification demo with few-shot examples."""
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from llmgen import OpenAiApiFactory, ExamplePair


class ReviewInput(BaseModel):
    review: str = Field(description="Customer review text")


class SentimentOutput(BaseModel):
    label: str = Field(description="One of: positive, negative, neutral")
    score: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)
    explanation: str = Field(description="Brief explanation of the classification")


class Settings(BaseSettings):
    base_url: str
    api_token: str


def main():
    settings = Settings.model_validate({})
    factory = OpenAiApiFactory(api_key=settings.api_token, base_url=settings.base_url)

    examples = [
        ExamplePair(
            input=ReviewInput(review="Absolutely loved it! Best purchase I've ever made."),
            output=SentimentOutput(label="positive", score=0.98, explanation="Strong positive language."),
        ),
        ExamplePair(
            input=ReviewInput(review="Terrible quality, broke after one day."),
            output=SentimentOutput(label="negative", score=0.95, explanation="Clear negative experience."),
        ),
    ]

    api = factory.make_api(
        ReviewInput,
        SentimentOutput,
        examples=examples,
        intro_prompt="You are a sentiment analysis expert for e-commerce reviews.",
    )

    reviews = [
        "It's okay, nothing special but does the job.",
        "I'm so disappointed, expected much better for the price.",
        "Fantastic product, highly recommend to everyone!",
    ]

    for review in reviews:
        result = api.call(ReviewInput(review=review))
        print(f"[{result.label.upper()} {result.score:.2f}] {review[:50]}...")
        print(f"  → {result.explanation}\n")


if __name__ == "__main__":
    main()
