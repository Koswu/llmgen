# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-01

### Changed
- **Backend rewrite**: Replaced `langchain` / `langchain-openai` with `pydantic-ai` for a lighter, more Pythonic dependency stack.
- `OpenAiApiFactory` now uses `pydantic-ai`'s `OpenAIModel` + `OpenAIProvider` internally.
- Few-shot examples are now injected via `message_history` instead of system messages.

### Added
- `AgentFactory` — a generic factory that accepts any pydantic-ai model name string or `Model` instance (supports OpenAI, Anthropic, Gemini, and more).
- `LlmGenError` — alias for `OpenAiApiError` for cleaner naming going forward.
- `OpenAiApi.call_with_usage()` — returns `(output, Usage)` tuple with token counts.
- `OpenAiApi.stream()` — synchronous streaming generator.
- `OpenAiApi.astream()` — asynchronous streaming generator.
- `_BaseLLMFunc` shared base class for `LLMImplmentedFunc` and `AsyncLLMImplmentedFunc` (eliminates code duplication).
- `tests/` directory with full test suite using `pydantic-ai`'s `TestModel` (no real API key required).
- CI workflow (`.github/workflows/ci.yml`) for Python 3.10 / 3.11 / 3.12.
- Comprehensive `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, and `LICENSE`.
- New demo scripts: `translator.py`, `extractor.py`, `classifier.py`, `streaming.py`, `multi_provider.py`.

### Removed
- `langchain`, `langchain-openai`, `langchain-core` dependencies.

## [0.1.0] - 2024-01-01

### Added
- Initial release with `OpenAiApiFactory`, `make_api`, `call`, `async_call`.
- `@llm.impl()` decorator for sync and async functions.
- `@add_example` few-shot decorator.
- `ExamplePair` data structure.
