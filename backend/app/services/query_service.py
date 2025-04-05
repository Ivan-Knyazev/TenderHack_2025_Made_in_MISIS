from fastapi import HTTPException, status
# from typing import Optional

from app.repositories.query_repository import QueryRepository
from app.models.query import QueryDB, QueryInput


class QueryService:
    """Сервис, содержащий бизнес-логику работы с запросами и ответами"""

    def __init__(self, query_repository: QueryRepository):
        self.query_repo = query_repository

    async def create_query(self, query_input: QueryInput) -> QueryDB:
        """Создаёт новый запрос"""

        # 1. Создание запроса к ML и запись в DB
        try:
            created_user = await self.query_repo.create_query(query_input)
            return created_user
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
