# llmgen

[![PyPI](https://img.shields.io/pypi/v/llmgen?label=PyPI)](https://pypi.org/project/llmgen/)
[![Python Versions](https://img.shields.io/pypi/pyversions/llmgen)](https://pypi.org/project/llmgen/)
[![CI](https://github.com/Koswu/llmgen/actions/workflows/ci.yml/badge.svg)](https://github.com/Koswu/llmgen/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Effortlessly generate LLM APIs by simply defining input and output schemas.**

`llmgen` wraps [pydantic-ai](https://ai.pydantic.dev/) to let you describe what you want from an LLM using plain Pydantic models or type-annotated functions — and get back strongly-typed results with zero boilerplate.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Schema-based API](#schema-based-api)
  - [Function decorator](#function-decorator)
- [Few-shot examples](#few-shot-examples)
- [Async support](#async-support)
- [Streaming](#streaming)
- [Token usage](#token-usage)
- [Multi-provider support](#multi-provider-support)
- [Error handling](#error-handling)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Schema-driven** — define input/output as Pydantic `BaseModel`; get validated, typed output back.
- **Decorator magic** — annotate any function with `@llm.impl()` and it becomes LLM-powered.
- **Few-shot examples** — inject examples with `@add_example` or `ExamplePair`.
- **Async-first** — every API has both `call()` (sync) and `async_call()` (async) variants.
- **Streaming** — `stream()` and `astream()` generators for incremental output.
- **Usage tracking** — `call_with_usage()` returns token counts alongside results.
- **Multi-provider** — works with OpenAI, Azure, Anthropic (via Claude), and any OpenAI-compatible endpoint.
- **Testable** — ships with pydantic-ai `TestModel` support so you can unit-test without a real API key.

---

## Installation

```bash
pip install llmgen
```

For development (tests, linting):

```bash
pip install "llmgen[dev]"
# or with poetry:
poetry install
```

---

## Quick Start

### Schema-based API

Define what goes in and what comes out:

```python
from pydantic import BaseModel, Field
from llmgen import OpenAiApiFactory

class JokeRequest(BaseModel):
    theme: str = Field(description="Topic of the joke")

class JokeResponse(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline")
    funny_level: int = Field(description="Funny level 1-5", ge=1, le=5)

factory = OpenAiApiFactory(
    api_key="sk-...",
    model_name="gpt-4o-mini",
)

api = factory.make_api(JokeRequest, JokeResponse)
result = api.call(JokeRequest(theme="cats"))

print(result.setup)      # "Why was the cat sitting on the computer?"
print(result.punchline)  # "It wanted to keep an eye on the mouse!"
print(result.funny_level)  # 4
```

### Function decorator

Use `@llm.impl()` to implement a function body with the LLM:

```python
from typing import List
from llmgen import OpenAiApiFactory

factory = OpenAiApiFactory(api_key="sk-...", model_name="gpt-4o-mini")

@factory.impl()
def tell_jokes(theme: str, count: int) -> List[str]:
    """Tell `count` short jokes about `theme`."""
    ...

jokes = tell_jokes("programming", 3)
print(jokes)  # ["Why do programmers prefer dark mode?...", ...]
```

The function's type hints and docstring are used to automatically construct the prompt.

---

## Few-shot examples

Provide examples to guide the LLM:

```python
from llmgen import OpenAiApiFactory, ExamplePair, add_example

factory = OpenAiApiFactory(api_key="sk-...")

# Via make_api
examples = [
    ExamplePair(
        input=JokeRequest(theme="dog"),
        output=JokeResponse(setup="Why did the dog sit in the shade?",
                            punchline="He didn't want to be a hot dog!", funny_level=3),
    )
]
api = factory.make_api(JokeRequest, JokeResponse, examples=examples)

# Via decorator
@add_example(args=["cat", 1], result=["Why was the cat sitting on the computer?..."])
@factory.impl()
def tell_jokes(theme: str, count: int) -> List[str]:
    """Tell jokes about a theme."""
    ...
```

---

## Async support

Every API object supports both sync and async calls:

```python
import asyncio

async def main():
    result = await api.async_call(JokeRequest(theme="robots"))
    print(result.setup)

asyncio.run(main())
```

Async decorator functions also work:

```python
@factory.impl()
async def tell_joke_async(theme: str) -> str:
    """Tell an async joke."""
    ...

joke = asyncio.run(tell_joke_async("space"))
```

---

## Streaming

Stream raw text chunks as they arrive:

```python
# Sync streaming
for chunk in api.stream(JokeRequest(theme="cats")):
    print(chunk, end="", flush=True)

# Async streaming
async def stream_joke():
    async for chunk in api.astream(JokeRequest(theme="dogs")):
        print(chunk, end="", flush=True)
```

---

## Token usage

Track token consumption alongside results:

```python
result, usage = api.call_with_usage(JokeRequest(theme="fish"))
print(result.setup)
print(f"Tokens used — input: {usage.input_tokens}, output: {usage.output_tokens}")
```

---

## Multi-provider support

Use `AgentFactory` for any pydantic-ai supported provider:

```python
from llmgen import AgentFactory

# OpenAI
factory = AgentFactory("openai:gpt-4o")

# Anthropic Claude
factory = AgentFactory("anthropic:claude-3-5-sonnet-latest")

# Gemini
factory = AgentFactory("google-gla:gemini-1.5-flash")

# Azure OpenAI (via OpenAiApiFactory with custom base_url)
from llmgen import OpenAiApiFactory
factory = OpenAiApiFactory(
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com/",
    model_name="gpt-4o",
)

# Any OpenAI-compatible endpoint (e.g. Ollama)
factory = OpenAiApiFactory(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    model_name="llama3",
)
```

---

## Error handling

All LLM errors are wrapped in `OpenAiApiError` (alias: `LlmGenError`):

```python
from llmgen import OpenAiApiError

try:
    result = api.call(JokeRequest(theme="cats"))
except OpenAiApiError as e:
    print(f"LLM call failed: {e}")
```

---

## API Reference

### `OpenAiApiFactory`

```python
OpenAiApiFactory(
    api_key: str,
    *,
    base_url: str = "https://api.openai.com/v1",
    model_name: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
)
```

OpenAI / OpenAI-compatible factory.

### `AgentFactory`

```python
AgentFactory(
    model: str | Model,   # pydantic-ai model name or Model instance
    *,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
)
```

Generic factory for any pydantic-ai model.

### `factory.make_api(input_type, output_type, examples=None, intro_prompt=None) -> OpenAiApi`

Creates an `OpenAiApi` instance.

### `OpenAiApi`

| Method | Description |
|--------|-------------|
| `call(input)` | Synchronous call, returns typed output |
| `async_call(input)` | Async call |
| `call_with_usage(input)` | Returns `(output, Usage)` tuple |
| `stream(input)` | Sync generator of text chunks |
| `astream(input)` | Async generator of text chunks |

### `factory.impl() -> FuncImplDecorator`

Returns a decorator that makes a function LLM-implemented.

### `@add_example(args, result, kwargs=None)`

Adds a few-shot example to a decorated LLM function.

### `ExamplePair`

```python
ExamplePair(input=<InputModel>, output=<OutputModel>)
```

### `OpenAiApiError` / `LlmGenError`

Exception raised when an LLM call fails.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests
4. Run `pytest` and `ruff check llmgen/ tests/`
5. Submit a pull request

---

## License

MIT — see [LICENSE](LICENSE).

