from pydantic import BaseModel, Field, EmailStr, ConfigDict
from typing import Optional
from datetime import datetime, timezone
from bson import ObjectId  # Используем ObjectId из pymongo

# Вспомогательный класс для работы с ObjectId в Pydantic v2


# class PyObjectId(ObjectId):
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate

#     @classmethod
#     def validate(cls, v, *args, **kwargs):  # Адаптировано для Pydantic v2
#         if not ObjectId.is_valid(v):
#             raise ValueError("Invalid ObjectId")
#         return ObjectId(v)

#     # Как представить тип в JSON Schema (для /docs)
#     @classmethod
#     def __get_pydantic_json_schema__(cls, core_schema, handler):
#         # Представляем как строку в схеме OpenAPI
#         return {"type": "string", "example": "65b4f0f8a7b5b1e6a8f3d5e8"}

#     # Как сериализовать и валидировать в Pydantic Core Schema V2
#     @classmethod
#     def __get_pydantic_core_schema__(cls, source, handler):
#         from pydantic_core import core_schema

#         # Функция валидации, принимающая любое значение и возвращающая ObjectId
#         validation = core_schema.no_info_plain_validator_function(cls.validate)

#         # Функция сериализации, принимающая ObjectId и возвращающая строку
#         serialization = core_schema.plain_serializer_function_ser_schema(
#             # Конвертируем ObjectId в строку при сериализации
#             lambda x: str(x),
#             info_arg=False,
#             return_schema=core_schema.str_schema()  # Указываем, что возвращается строка
#         )

#         return core_schema.json_or_python_schema(
#             python_schema=validation,
#             json_schema=core_schema.str_schema(),  # В JSON ожидаем строку
#             serialization=serialization
#         )

# Базовая модель пользователя


class UserBase(BaseModel):
    username: EmailStr = Field(..., example="test@example.com")
    full_name: Optional[str] = Field(None, example="John Doe")
    is_active: bool = Field(True)
    is_admin: bool = Field(True)


# class UserLogin(BaseModel):
#     username: EmailStr = Field(..., example="test@example.com")
#     hashed_password: str = Field(..., min_length=8, example="Str0ngP@ssw0rd")

# Модель для создания пользователя (данные из запроса)


class UserCreate(UserBase):
    hashed_password: str = Field(..., min_length=8, example="Str0ngP@ssw0rd")

# Модель пользователя, хранящаяся в БД (включая хеш пароля)


class UserInDB(UserBase):
    # Используем PyObjectId и alias для маппинга на _id MongoDB
    id: Optional[ObjectId] = Field(default=None)
    hashed_password: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        populate_by_name=True,  # Разрешить заполнение по alias ('_id')
        arbitrary_types_allowed=True  # Разрешить PyObjectId
    )

# Модель пользователя для ответа API (публичные данные)


class UserPublic(UserBase):
    # Возвращаем id как строку
    id: str = Field(..., example="65b4f0f8a7b5b1e6a8f3d5e8",
                    description="MongoDB document ObjectID")

    model_config = ConfigDict(
        # Разрешить создание из атрибутов объекта (как UserInDB)
        from_attributes=True
    )

    # Кастомный сериализатор или валидатор для _id -> id (если from_attributes не справляется)
    # Pydantic V2 часто справляется автоматически, если alias настроен в UserInDB
    # и сериализация PyObjectId в строку работает


class UserAuth(BaseModel):
    username: EmailStr = Field(..., example="test@example.com")
    hashed_password: str = Field(..., min_length=8, example="Str0ngP@ssw0rd")
