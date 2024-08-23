from pydantic import BaseModel
from typing import List

class TextInput(BaseModel):
    texts: List[str]
