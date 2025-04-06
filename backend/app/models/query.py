from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List


class SourceDocument(BaseModel):
    content: str
    source: str
    file_name: str
    chunk_id: int
    page: int
    is_semantic_chunk: bool


class Response(BaseModel):
    human_handoff: bool
    conversation_id: str
    source_documents: List[SourceDocument]
    used_files: List[str]


class ResponseSplitted(BaseModel):
    think: str
    theme: str
    answer: str


# Responses
class ResponseFromML(Response):
    response: str


class ResponseToDB(Response):
    response: ResponseSplitted


# QueryDB - main model
class QueryToML(BaseModel):
    query: str
    conversation_id: Optional[str] = None


class QueryInput(BaseModel):
    user_id: str
    chat_id: int
    query: str = Field(..., example="Расскажи, как стать поставщиком")


class QueryDB(QueryInput):
    # id: Optional[str] = None
    # id: str = Field(..., example="65b4f0f8a7b5b1e6a8f3d5e8",
    #                 description="MongoDB document ObjectID")
    response: Optional[ResponseToDB] = None
    category: Optional[str] = None
    time: int
    # model_config = ConfigDict(
    #     from_attributes=True
    # )


class QueryDBUpdated(QueryDB):
    mark: int

# class Category(BaseModel):
#     category: str
