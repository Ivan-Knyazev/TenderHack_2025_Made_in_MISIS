from fastapi import HTTPException, status
from typing import Optional

from app.repositories.user_repository import UserRepository
from app.models.user import UserCreate, UserInDB
from app.models.token import TokenPayload
from app.core import security


class AuthService:
    """Сервис, содержащий бизнес-логику аутентификации и регистрации."""

    def __init__(self, user_repository: UserRepository):
        self.user_repo = user_repository

    async def register_new_user(self, user_data: UserCreate) -> UserInDB:
        """Регистрирует нового пользователя."""
        # 1. Проверка, существует ли пользователь
        existing_user = await self.user_repo.get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already registered",
            )

        # 2. Хеширование пароля
        # hashed_password = security.get_password_hash(user_data.password)

        # 3. Создание объекта пользователя для БД
        user_to_create = UserInDB(
            username=user_data.username,
            hashed_password=user_data.hashed_password,
            full_name=user_data.full_name,  # Добавляем другие поля из UserCreate
            is_active=user_data.is_active
            # id, created_at, updated_at будут добавлены автоматически или в репозитории
        )

        # 4. Сохранение в репозитории
        try:
            created_user = await self.user_repo.create_user(user_to_create)
            return created_user
        except ValueError as e:  # Ловим ошибку дубликата из репозитория
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),  # Передаем сообщение об ошибке
            )
        except Exception as e:
            # Логирование и общая ошибка сервера
            print(f"Unexpected error during user registration: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during registration.",
            )

    async def authenticate_user(self, username: str, hashed_password: str) -> Optional[UserInDB]:
        """Аутентифицирует пользователя по имени и паролю."""
        user = await self.user_repo.get_user_by_username(username)
        if not user:
            return None  # Пользователь не найден
        if not security.verify_password(hashed_password, user.hashed_password):
            return None  # Неверный пароль
        if not user.is_active:
            # Можно возвращать None или кидать HTTPException
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
        return user

    def create_jwt_token(self, user: UserInDB) -> str:
        """Создает JWT токен для пользователя."""
        print(user)
        if user.id is None:
            # Этого не должно случиться, если пользователь из БД
            raise ValueError("Cannot create token for user without ID")

        payload = TokenPayload(sub=user.username).model_dump()
        # Можно добавить ID пользователя или роли в payload, если нужно:
        # payload["user_id"] = str(user.id)
        # payload["roles"] = ["user"] # Пример
        return security.create_access_token(data=payload)
