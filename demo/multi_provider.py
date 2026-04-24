"""Multi-provider demo showing AgentFactory with different backends."""
from pydantic import BaseModel, Field
from llmgen import AgentFactory, OpenAiApiFactory


class SummaryRequest(BaseModel):
    text: str = Field(description="Text to summarize")
    max_words: int = Field(description="Maximum words in summary", default=50)


class SummaryResponse(BaseModel):
    summary: str = Field(description="Concise summary")
    key_points: list[str] = Field(description="Up to 3 key points")


def demo_openai():
    """Use OpenAI via OpenAiApiFactory (OpenAI-compatible endpoints)."""
    import os

    factory = OpenAiApiFactory(
        api_key=os.environ.get("OPENAI_API_KEY", "sk-..."),
        model_name="gpt-4o-mini",
    )
    api = factory.make_api(SummaryRequest, SummaryResponse)
    return api


def demo_anthropic():
    """Use Anthropic Claude via AgentFactory."""
    factory = AgentFactory("anthropic:claude-3-5-haiku-latest")
    api = factory.make_api(SummaryRequest, SummaryResponse)
    return api


def demo_gemini():
    """Use Google Gemini via AgentFactory."""
    factory = AgentFactory("google-gla:gemini-1.5-flash")
    api = factory.make_api(SummaryRequest, SummaryResponse)
    return api


def demo_ollama():
    """Use a local Ollama model (OpenAI-compatible) via OpenAiApiFactory."""
    factory = OpenAiApiFactory(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        model_name="llama3",
    )
    api = factory.make_api(SummaryRequest, SummaryResponse)
    return api


TEXT = (
    "Python is a high-level, general-purpose programming language emphasizing code readability. "
    "Guido van Rossum designed it in the late 1980s and released Python 0.9.0 in 1991. "
    "It supports multiple programming paradigms including structured, object-oriented, and functional. "
    "Python is consistently ranked among the most popular programming languages."
)

if __name__ == "__main__":
    print("This demo requires valid API keys set as environment variables.")
    print("Uncomment the provider you want to test and run the script.\n")

    # api = demo_openai()
    # result = api.call(SummaryRequest(text=TEXT, max_words=30))
    # print(f"Summary: {result.summary}")
    # for point in result.key_points:
    #     print(f"  • {point}")
