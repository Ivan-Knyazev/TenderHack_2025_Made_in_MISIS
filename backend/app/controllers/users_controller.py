from fastapi import APIRouter, Depends
from typing import Annotated

from app.models.user import UserPublic, UserInDB
from app.controllers.dependencies import get_current_active_user  # Импорт зависимости

router = APIRouter(prefix="/users", tags=["Users (protected)"])

# Зависимость CurrentUserDependency = Annotated[...] можно вынести в dependencies.py
CurrentUserDependency = Annotated[UserInDB, Depends(get_current_active_user)]


@router.get("/me", response_model=UserPublic)
async def read_users_me(current_user: CurrentUserDependency):
    """
    Получение информации о текущем аутентифицированном пользователе
    """
    # Доступ защищен JWT токеном через зависимость `get_current_active_user`.
    # Зависимость уже вернула объект UserInDB.
    # FastAPI автоматически сконвертирует его в UserPublic при ответе.
    current_user.id = str(current_user.id)
    return UserPublic.model_validate(current_user)

# Можно добавить другие эндпоинты для пользователей, защищенные этой же зависимостью
# Например, обновление профиля:
# @router.patch("/me", response_model=UserPublic)
# async def update_user_me(update_data: UserUpdate, current_user: CurrentUserDependency, ...):
#     # Логика обновления пользователя через сервис
#     ...
