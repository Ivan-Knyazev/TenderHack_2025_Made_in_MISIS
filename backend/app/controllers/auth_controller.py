from fastapi import APIRouter, HTTPException, status
# Стандартная форма для логина
# from fastapi.security import OAuth2PasswordRequestForm
# from typing import Annotated

from app.models.user import UserCreate, UserPublic, UserAuth
from app.models.token import Token
# from app.services.auth_service import AuthService
from app.controllers.dependencies import AuthServiceDependency  # Импорт зависимости

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_in: UserCreate,
    auth_service: AuthServiceDependency  # Внедряем сервис
):
    """
    Регистрация нового пользователя
    """
    try:
        created_user = await auth_service.register_new_user(user_in)
        created_user.id = str(created_user.id)
        print(created_user)
        # Преобразуем UserInDB в UserPublic для ответа
        return UserPublic.model_validate(created_user)
    except HTTPException as http_exc:
        # Перехватываем и перевыбрасываем HTTP исключения из сервиса
        raise http_exc
    except Exception as e:
        # Логгируем и возвращаем 500 для непредвиденных ошибок
        print(f"Error in registration controller: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during registration."
        )


@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
async def login(
    form_data: UserAuth,
    auth_service: AuthServiceDependency  # Внедряем сервис
):
    """
    Аутентификация и получение JWT токена
    """
    # Использует стандартную форму `OAuth2PasswordRequestForm` (поля username, password).
    user = await auth_service.authenticate_user(
        username=form_data.username, hashed_password=form_data.hashed_password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            # Важный заголовок для OAuth2
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Если аутентификация прошла, создаем токен
    access_token = auth_service.create_jwt_token(user)
    return Token(access_token=access_token, token_type="bearer")


# # Переименовали эндпоинт в /token для соответствия OAuth2/Swagger
# @router.post("/token", response_model=Token)
# async def login_for_access_token(
#     form_data: Annotated[UserCreate, Depends()],
#     auth_service: AuthServiceDependency  # Внедряем сервис
# ):
#     """
#     Контроллер для аутентификации и получения JWT токена.
#     # Использует стандартную форму `OAuth2PasswordRequestForm` (поля username, password).
#     """
#     user = await auth_service.authenticate_user(
#         username=form_data.username, hashed_password=form_data.hashed_password
#     )
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             # Важный заголовок для OAuth2
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     # Если аутентификация прошла, создаем токен
#     access_token = auth_service.create_jwt_token(user)
#     return Token(access_token=access_token, token_type="bearer")

# Переименовали эндпоинт в /token для соответствия OAuth2/Swagger
