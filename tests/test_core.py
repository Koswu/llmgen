"""Tests for llmgen core functionality using pydantic-ai's TestModel (no real API key needed)."""
from __future__ import annotations

from typing import List

import pytest
from pydantic import BaseModel, Field
from pydantic_ai.models.test import TestModel

from llmgen import AgentFactory, ExamplePair, OpenAiApiError, OpenAiApiFactory, add_example

# ---------------------------------------------------------------------------
# Shared test models
# ---------------------------------------------------------------------------


class JokeRequest(BaseModel):
    theme: str


class JokeResponse(BaseModel):
    setup: str
    punchline: str


class TextInput(BaseModel):
    text: str


class SentimentOutput(BaseModel):
    label: str = Field(description="positive, negative, or neutral")
    score: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_factory_with_test_model(output_args: dict) -> AgentFactory:
    """Return an AgentFactory backed by a TestModel that returns output_args."""
    return AgentFactory(TestModel(custom_output_args=output_args))


# ---------------------------------------------------------------------------
# make_api / call tests
# ---------------------------------------------------------------------------


def test_make_api_call_basic():
    factory = _agent_factory_with_test_model({"setup": "Why did the cat…", "punchline": "Because!"})
    api = factory.make_api(JokeRequest, JokeResponse)
    result = api.call(JokeRequest(theme="cats"))
    assert isinstance(result, JokeResponse)
    assert result.setup == "Why did the cat…"
    assert result.punchline == "Because!"


@pytest.mark.asyncio
async def test_make_api_async_call():
    factory = _agent_factory_with_test_model(
        {"setup": "async setup", "punchline": "async punchline"}
    )
    api = factory.make_api(JokeRequest, JokeResponse)
    result = await api.async_call(JokeRequest(theme="dogs"))
    assert isinstance(result, JokeResponse)
    assert result.setup == "async setup"


def test_make_api_with_intro_prompt():
    factory = _agent_factory_with_test_model({"label": "positive", "score": 0.9})
    api = factory.make_api(
        TextInput,
        SentimentOutput,
        intro_prompt="You are a sentiment analysis assistant.",
    )
    result = api.call(TextInput(text="I love this!"))
    assert isinstance(result, SentimentOutput)
    assert result.label == "positive"


# ---------------------------------------------------------------------------
# Few-shot ExamplePair injection
# ---------------------------------------------------------------------------


def test_make_api_with_examples():
    examples = [
        ExamplePair(
            input=JokeRequest(theme="dog"),
            output=JokeResponse(setup="Why did the dog…", punchline="Because woof!"),
        )
    ]
    factory = _agent_factory_with_test_model({"setup": "cat setup", "punchline": "cat punchline"})
    api = factory.make_api(JokeRequest, JokeResponse, examples=examples)
    result = api.call(JokeRequest(theme="cat"))
    assert isinstance(result, JokeResponse)


# ---------------------------------------------------------------------------
# call_with_usage
# ---------------------------------------------------------------------------


def test_call_with_usage():
    factory = _agent_factory_with_test_model({"setup": "s", "punchline": "p"})
    api = factory.make_api(JokeRequest, JokeResponse)
    output, usage = api.call_with_usage(JokeRequest(theme="fish"))
    assert isinstance(output, JokeResponse)
    assert usage.requests >= 1


# ---------------------------------------------------------------------------
# @llm.impl() decorator — sync function
# ---------------------------------------------------------------------------


def test_impl_decorator_sync():
    factory = _agent_factory_with_test_model({"result": "Why did the robot…"})
    llm = factory

    @llm.impl()
    def summarize(text: str) -> str:
        """Summarize the given text."""
        ...

    result = summarize("Some long text here")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_impl_decorator_async():
    factory = _agent_factory_with_test_model({"result": "async joke"})
    llm = factory

    @llm.impl()
    async def tell_joke(theme: str) -> str:
        """Tell a joke about theme."""
        ...

    result = await tell_joke("robots")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# @add_example decorator
# ---------------------------------------------------------------------------


def test_add_example_decorator():
    factory = _agent_factory_with_test_model({"result": ["another joke"]})
    llm = factory

    @add_example(args=["cat", 1], result=["A cat joke!"])
    @llm.impl()
    def tell_jokes(theme: str, count: int) -> List[str]:
        """Tell jokes about a theme."""
        ...

    # Just verify the example was registered without error
    assert len(tell_jokes._example_pairs) == 1
    result = tell_jokes("dog", 2)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_add_example_decorator_async():
    factory = _agent_factory_with_test_model({"result": ["async joke list"]})
    llm = factory

    @add_example(args=["cat", 1], result=["A cat joke!"])
    @llm.impl()
    async def tell_jokes_async(theme: str, count: int) -> List[str]:
        """Async joke teller."""
        ...

    assert len(tell_jokes_async._example_pairs) == 1
    result = await tell_jokes_async("dog", 2)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# OpenAiApiFactory backward-compatibility (uses TestModel indirectly via AgentFactory)
# ---------------------------------------------------------------------------


def test_openai_api_factory_exports():
    """Ensure OpenAiApiFactory is importable and has the expected interface."""
    assert hasattr(OpenAiApiFactory, "make_api")
    assert hasattr(OpenAiApiFactory, "impl")


# ---------------------------------------------------------------------------
# Error handling — OpenAiApiError
# ---------------------------------------------------------------------------


def test_openai_api_error_is_exception():
    err = OpenAiApiError("test error")
    assert isinstance(err, Exception)
    assert str(err) == "test error"


def test_openai_api_error_raised_on_bad_model():
    """AgentFactory with a bad model name should raise OpenAiApiError on call."""
    from llmgen._openai import AgentFactory

    factory = AgentFactory("nonexistent-model:does-not-exist")
    api = factory.make_api(JokeRequest, JokeResponse)
    with pytest.raises((OpenAiApiError, Exception)):
        api.call(JokeRequest(theme="error"))


# ---------------------------------------------------------------------------
# AgentFactory
# ---------------------------------------------------------------------------


def test_agent_factory_with_test_model_instance():
    test_model = TestModel(custom_output_args={"setup": "s", "punchline": "p"})
    factory = AgentFactory(test_model)
    api = factory.make_api(JokeRequest, JokeResponse)
    result = api.call(JokeRequest(theme="test"))
    assert result.setup == "s"
