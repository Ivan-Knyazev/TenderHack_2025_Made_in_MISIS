from fastapi import HTTPException, status
# from typing import Optional

from app.repositories.query_repository import QueryRepository
from app.models.query import QueryDB, QueryInput, QueryDBUpdateFromFront, Chart1, QueriesTable


class QueryService:
    """Сервис, содержащий бизнес-логику работы с запросами и ответами"""

    def __init__(self, query_repository: QueryRepository):
        self.query_repo = query_repository

    async def create_query(self, query_input: QueryInput) -> QueryDB:
        """Создаёт новый запрос"""

        # Создание запроса к ML и запись в DB
        try:
            created_query = await self.query_repo.create_query(query_input)
            return created_query
        except ValueError as e:  # Ловим ошибку дубликата из репозитория
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),  # Передаем сообщение об ошибке
            )
        except Exception as e:
            # Логирование и общая ошибка сервера
            print(f"Unexpected error during query registration: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during create.",
            )

    async def add_query_mark(self, query_to_update: QueryDBUpdateFromFront) -> QueryDB:
        """Редактирует оценку ответа на запрос"""

        # Обновление запроса в DB
        try:
            updated_query = await self.query_repo.add_query_mark(query_to_update)
            return updated_query
        except ValueError as e:  # Ловим ошибку дубликата из репозитория
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),  # Передаем сообщение об ошибке
            )
        except Exception as e:
            # Логирование и общая ошибка сервера
            print(f"Unexpected error during query update: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during update.",
            )

    async def analitycs(self) -> Chart1:
        """Редактирует оценку ответа на запрос"""

        # Обновление запроса в DB
        try:
            analitycs = await self.query_repo.analitycs()
            return analitycs
        except ValueError as e:  # Ловим ошибку дубликата из репозитория
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),  # Передаем сообщение об ошибке
            )
        except Exception as e:
            # Логирование и общая ошибка сервера
            print(f"Unexpected error during query analitycs: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during analitycs.",
            )

    async def all_queries(self) -> QueriesTable:
        """Редактирует оценку ответа на запрос"""

        # Обновление запроса в DB
        try:
            all_queries = await self.query_repo.all_queries()
            return all_queries
        except ValueError as e:  # Ловим ошибку дубликата из репозитория
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),  # Передаем сообщение об ошибке
            )
        except Exception as e:
            # Логирование и общая ошибка сервера
            print(f"Unexpected error during query analitycs: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during analitycs.",
            )
