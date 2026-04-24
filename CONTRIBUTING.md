# Contributing to llmgen

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development setup

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/your-username/llmgen.git
   cd llmgen
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

## Running tests

All tests use `pydantic-ai`'s `TestModel` — no real API key is required:

```bash
pytest
# or with verbose output:
pytest -v
```

## Linting

```bash
ruff check llmgen/ tests/
ruff format llmgen/ tests/   # auto-format
```

## Submitting changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes and add or update tests.

3. Ensure tests pass and linting is clean:
   ```bash
   pytest && ruff check llmgen/ tests/
   ```

4. Commit with a descriptive message:
   ```bash
   git commit -m "feat: add streaming support"
   ```

5. Push and open a pull request against `main`.

## Code style

- Follow PEP 8; line length ≤ 100 characters.
- Use type hints for all public functions.
- Only add comments when clarification is genuinely needed.
- Keep public API backward compatible — new features should be additive.

## Reporting issues

Please use [GitHub Issues](https://github.com/Koswu/llmgen/issues) to report bugs or request features.
Include a minimal reproducible example when reporting bugs.
