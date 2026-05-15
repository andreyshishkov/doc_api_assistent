from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    found: bool
    content: str | None = None
    message: str | None = None
