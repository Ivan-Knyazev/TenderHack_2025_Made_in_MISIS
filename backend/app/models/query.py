from pydantic import BaseModel, Field
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
    mark: Optional[int] = None
    query_id: Optional[str] = None
    time: int
    # model_config = ConfigDict(
    #     from_attributes=True
    # )


# For add mark


class QueryDBUpdateFromFront(BaseModel):
    query_id: str
    rate: int


# Analitycs
class Bar(BaseModel):
    x: int
    y: int
    label: str


class Percent(BaseModel):
    x: str
    y: int


class Chart1(BaseModel):
    bars: List[Bar]
    percents: List[Percent]


class QueryTable(BaseModel):
    id: str
    name: str
    date: str
    type: str
    request: str
    answer: str
    source: List[str]
    mark: str


class QueriesTable(BaseModel):
    data: List[QueryTable]

    # class QueryDBToFront(QueryDB):
    #     query_id: str

    # class Category(BaseModel):
    #     category: str
