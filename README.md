# llmgen

[![PyPI](https://img.shields.io/pypi/v/llmgen?label=pypi%20package)](https://pypi.org/project/llmgen/)

Effortlessly generate LLM APIs by simply defining input and output schemas (by pydantic).


## Quick Start

Example:

```python
from llmgen import OpenAiApiFactory

factory = OpenAiApiFactory(
    api_key="your api key here",
)
ai_impl = llm.get_impl_decorator()

@ai_impl
def make_joke(theme: str) -> str
    """generate a random short joke based on theme""" # just describe your api in docstring
    ...

# call it just like you implmented it. 

res = make_joke('cat')
print(res)

# >>> "Why was the cat sitting on the computer? It wanted to keep an eye on the mouse!"

```

checkout /demo to see more example.

