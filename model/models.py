from pydantic import BaseModel
import datetime

class LLM_Request(BaseModel):
    prompt: str
    max_new_tokens: int
    min_new_tokens: int
    decoding_method: str


class LLM_Response(BaseModel):
    message: str

class Ack_LLM_Request(BaseModel):
    description: str

class WrappedText(BaseModel):
    text: str

class Async_Response(BaseModel):
    output: WrappedText

class LLM_Cache_Entry(BaseModel):
    request: LLM_Request
    response: LLM_Response
    timestamp: str

class LLM_Cache_List(BaseModel):
    cache: list[LLM_Cache_Entry]