from pydantic import BaseModel
from typing import Optional


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    # 'sub' (subject) обычно используется для идентификатора пользователя (здесь username)
    sub: Optional[str] = None
    # Можно добавить другие поля в payload:
    # user_id: Optional[str] = None
    # roles: Optional[list[str]] = None
