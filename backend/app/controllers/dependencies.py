# В файле app/controllers/dependencies.py
from fastapi import Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Annotated

from app.db.database import get_database
from app.repositories.user_repository import UserRepository
from app.repositories.query_repository import QueryRepository
from app.services.auth_service import AuthService
from app.services.query_service import QueryService
from app.models.user import UserInDB

from app.core import security
from app.models.token import TokenPayload

# --- Фабрики зависимостей ---


def get_db() -> AsyncIOMotorDatabase:
    """Зависимость: возвращает объект базы данных Motor."""
    return get_database()

# Определяем зависимость репозитория через функцию


def get_user_repository(
    db: Annotated[AsyncIOMotorDatabase, Depends(get_db)]
) -> UserRepository:
    """Dependency provider for UserRepository."""
    return UserRepository(db)


def get_query_repository(
    db: Annotated[AsyncIOMotorDatabase, Depends(get_db)]
) -> QueryRepository:
    """Dependency provider for QueryRepository."""
    return QueryRepository(db)


# Используем Annotated для ссылки на функцию get_user_repository
UserRepositoryDependency = Annotated[UserRepository, Depends(
    get_user_repository)]
QueryRepositoryDependency = Annotated[QueryRepository, Depends(
    get_query_repository)]

# Определяем зависимость сервиса через функцию


def get_auth_service(
    repo: UserRepositoryDependency  # Зависим от репозитория
) -> AuthService:
    """Dependency provider for AuthService."""
    return AuthService(repo)


def get_query_service(
    repo: QueryRepositoryDependency  # Зависим от репозитория
) -> QueryService:
    """Dependency provider for QueryService."""
    return QueryService(repo)


# Используем Annotated для ссылки на функцию get_auth_service
AuthServiceDependency = Annotated[AuthService, Depends(get_auth_service)]
QueryServiceDependency = Annotated[QueryService, Depends(get_query_service)]

# --- Зависимость для безопасности (остается как было) ---
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


async def get_current_active_user(
    token: str,
    user_repo: UserRepositoryDependency  # Используем зависимость репозитория
) -> UserInDB:
    # ... (код этой функции остается без изменений) ...
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    unauthorized_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    inactive_exception = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Inactive user",
    )

    payload = security.decode_token(token)
    if payload is None:
        raise credentials_exception

    token_data = TokenPayload(**payload)
    if token_data.sub is None:
        raise credentials_exception

    user = await user_repo.get_user_by_username(token_data.sub)
    if user is None:
        raise unauthorized_exception

    if not user.is_active:
        raise inactive_exception

    return user
