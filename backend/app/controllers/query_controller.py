from fastapi import APIRouter, HTTPException, status

from app.models.query import QueryDB, QueryInput, QueryDBUpdateFromFront, Chart1, QueriesTable
# Импорт зависимости
from app.controllers.dependencies import QueryServiceDependency

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/generate", response_model=QueryDB, status_code=status.HTTP_201_CREATED)
async def generate_query(query_input: QueryInput, query_service: QueryServiceDependency):
    """
    Создание нового запроса к ML
    """

    try:
        created_query = await query_service.create_query(query_input)
        # created_query.id = str(created_query.id)
        print("[INFO] [DATA] - Answer to /generate", created_query)
        # Преобразуем UserInDB в UserPublic для ответа
        return QueryDB.model_validate(created_query)
    except HTTPException as http_exc:
        # Перехватываем и перевыбрасываем HTTP исключения из сервиса
        raise http_exc
    except Exception as e:
        # Логгируем и возвращаем 500 для непредвиденных ошибок
        print(f"Error in generate controller: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during generate."
        )


@router.post("/rate", response_model=QueryDB, status_code=status.HTTP_200_OK)
async def add_query_mark(query_to_update: QueryDBUpdateFromFront, query_service: QueryServiceDependency):
    """
    Получение оценки ответа от юзера
    """

    try:
        updated_query = await query_service.add_query_mark(query_to_update)
        # created_query.id = str(created_query.id)
        print("[INFO] [DATA] - Answer to /rate", updated_query)
        # Преобразуем UserInDB в UserPublic для ответа
        return QueryDB.model_validate(updated_query)
    except HTTPException as http_exc:
        # Перехватываем и перевыбрасываем HTTP исключения из сервиса
        raise http_exc
    except Exception as e:
        # Логгируем и возвращаем 500 для непредвиденных ошибок
        print(f"Error in rate controller: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during rate."
        )


@router.get("/analitycs", response_model=Chart1, status_code=status.HTTP_200_OK)
async def analitycs(query_service: QueryServiceDependency):
    """
    Получение оценки ответа от юзера
    """

    try:
        analitycs = await query_service.analitycs()
        print("[INFO] [DATA] - Answer to /analitycs", analitycs)
        return Chart1.model_validate(analitycs)
    except HTTPException as http_exc:
        # Перехватываем и перевыбрасываем HTTP исключения из сервиса
        raise http_exc
    except Exception as e:
        # Логгируем и возвращаем 500 для непредвиденных ошибок
        print(f"Error in analitycs controller: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during analitycs."
        )


@router.get("/all", response_model=QueriesTable, status_code=status.HTTP_200_OK)
async def all_queries(query_service: QueryServiceDependency):
    """
    Получение оценки ответа от юзера
    """

    try:
        all_queries = await query_service.all_queries()
        print("[INFO] [DATA] - Answer to /all", all_queries)
        return QueriesTable.model_validate(all_queries)
    except HTTPException as http_exc:
        # Перехватываем и перевыбрасываем HTTP исключения из сервиса
        raise http_exc
    except Exception as e:
        # Логгируем и возвращаем 500 для непредвиденных ошибок
        print(f"Error in analitycs controller: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during analitycs."
        )


# @router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
# async def register_user(
#     user_in: UserCreate,
#     auth_service: AuthServiceDependency  # Внедряем сервис
# ):
#     """
#     Регистрация нового пользователя
#     """
#     try:
#         created_user = await auth_service.register_new_user(user_in)
#         created_user.id = str(created_user.id)
#         print(created_user)
#         # Преобразуем UserInDB в UserPublic для ответа
#         return UserPublic.model_validate(created_user)
#     except HTTPException as http_exc:
#         # Перехватываем и перевыбрасываем HTTP исключения из сервиса
#         raise http_exc
#     except Exception as e:
#         # Логгируем и возвращаем 500 для непредвиденных ошибок
#         print(f"Error in registration controller: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="An unexpected error occurred during registration."
#         )


# @router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
# async def login(
#     form_data: UserAuth,
#     auth_service: AuthServiceDependency  # Внедряем сервис
# ):
#     """
#     Аутентификация и получение JWT токена
#     """
#     # Использует стандартную форму `OAuth2PasswordRequestForm` (поля username, password).
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
