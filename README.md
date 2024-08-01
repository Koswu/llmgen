# llmgen

[![PyPI](https://img.shields.io/pypi/v/llmgen?label=pypi%20package)](https://pypi.org/project/llmgen/)

Effortlessly generate LLM APIs by simply defining input and output schemas (by pydantic).


## Quick Start

Example:

```python
from pydantic import BaseModel, Field
from llmgen import OpenAiApiFactory

# describe your input
class _JokeRequest(BaseModel):
    """Request for a joke"""

    theme: str = Field(description="The theme of the joke")


# describe your output
class _JokeResponse(BaseModel):
    """A joke response"""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke (the funny part)")
    description: str = Field(description="Description of why the joke is funny")
    level: int = Field(description="Funny level of the joke (1-5)", le=5, ge=1)



factory = OpenAiApiFactory(
    api_key="your api key here",
)

# Create an API
api = factory.make_api(_JokeRequest, _JokeResponse)

# Call the API
res = api.call(_JokeRequest(theme="cat"))

print(res.model_dump())
"""
A joke response like:
{
'setup': 'Why was the cat sitting on the computer?', 
'punchline': 'Because it wanted to keep an eye on the mouse!', 
'description': "This joke is funny because it plays on the double meaning of 'mouse'—the computer accessory and the animal that cats typically chase. The image of a cat being so tech-savvy is also amusing.", 
'level': 4
}
"""

```

