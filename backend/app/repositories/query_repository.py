from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
# from pymongo.errors import DuplicateKeyError
# from bson import ObjectId
# from typing import Optional

from app.core.config import ML_URL
from app.models.query import QueryInput, ResponseFromML, ResponseSplitted, ResponseToDB, QueryDB, QueryToML
from bs4 import BeautifulSoup
import httpx
import time
import requests
import json


class QueryRepository:
    """Репозиторий для асинхронной работы с коллекцией queries в MongoDB."""

    def __init__(self, database: AsyncIOMotorDatabase):
        self.collection: AsyncIOMotorCollection = database.queries

    async def create_query(self, query_input: QueryInput) -> QueryDB:
        """Создает новый запрос и ответ в БД."""

        unix_time = int(time.time())

        try:
            new_query = QueryToML(
                query=query_input.query, conversation_id="123e4567-e89b-12d3-a456-426614174000")

            # Main request to ML - generate answer
            url_ml_query = ML_URL + "query/"
            # print(url_ml_query, new_query.dict())

            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                url_ml_query, data=json.dumps(new_query.dict()), headers=headers)
            if response.status_code == 200:
                json_data = response.json()
                print("Запрос на ML успешно отправлен и получен ответ")
                response_data_from_ml = ResponseFromML.parse_obj(json_data)

                # parse lxml
                html_string = response_data_from_ml.response
                soup = BeautifulSoup(html_string, 'lxml')

                # print(soup.find('think'))

                response_splitted = ResponseSplitted(
                    think=soup.find('think').text, theme=soup.find('topic').text, answer=soup.find('answer').text)

                # create result data for DB
                response = ResponseToDB(
                    human_handoff=response_data_from_ml.human_handoff,
                    conversation_id=response_data_from_ml.conversation_id,
                    source_documents=response_data_from_ml.source_documents,
                    used_files=response_data_from_ml.used_files,
                    response=response_splitted,
                )
            else:
                print("Ошибка отправки запроса на ML:",
                      response.status_code, response.text)

            # Request to ML - generate category
            url_ml_category = ML_URL + "topic/"
            new_query.conversation_id = None

            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                url_ml_category, data=json.dumps(new_query.dict()), headers=headers)
            if response.status_code == 200:
                json_data = response.json()
                print("Запрос на ML успешно отправлен:", json_data)
                category = json_data['category']

                data = QueryDB(
                    response=response,
                    category=category,
                    user_id=query_input.user_id,
                    chat_id=query_input.chat_id,
                    query=query_input.query,
                    time=unix_time,
                )
            else:
                print("Ошибка отправки запроса на ML:",
                      response.status_code, response.text)

            # For test work with DB
            data = QueryDB(user_id=query_input.user_id, chat_id=query_input.chat_id,
                           query=query_input.query, time=unix_time)

            print(data.dict())

            insert_result = await self.collection.insert_one(data.dict())
            # Получаем созданный документ, чтобы вернуть его с _id
            created_doc = await self.collection.find_one({"_id": insert_result.inserted_id})
            if created_doc:
                # Создаем модель Pydantic из документа
                # print(created_doc['_id'])
                created_doc = {
                    'id': created_doc['_id'],
                    'user_id': created_doc['user_id'],
                    'chat_id': created_doc['chat_id'],
                    'query': created_doc['query'],
                    'response': created_doc['response'],
                    'category': created_doc['category'],
                    'time': created_doc['time'],
                }
                # print(created_doc)
                return QueryDB(**created_doc)
            else:
                # Эта ситуация маловероятна, но стоит обработать
                raise RuntimeError(
                    "Failed to retrieve created query-document")
        # except DuplicateKeyError:
        #     # Обрабатываем нарушение уникального индекса (username)
        #     raise ValueError(
        #         f"Data ID'{data}' already exists.")
        except Exception as e:
            # Логирование или дальнейшая обработка ошибок БД
            print(f"Database error during user creation: {e}")
            raise  # Перевыброс исключения

    # async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
    #     """Ищет пользователя по username."""
    #     user_doc = await self.collection.find_one({"username": username})
    #     if user_doc:
    #         user_doc = {
    #             'id': user_doc['_id'],
    #             'username': user_doc['username'],
    #             'hashed_password': user_doc['hashed_password'],
    #             'is_active': user_doc['is_active'],
    #             'is_admin': user_doc['is_admin']
    #         }
    #         return UserInDB(**user_doc)
    #     return None

    # async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
    #     """Ищет пользователя по его MongoDB _id (строковое представление)."""
    #     if not ObjectId.is_valid(user_id):
    #         print(f"Invalid ObjectId format: {user_id}")
    #         return None  # Невалидный формат ID
    #     try:
    #         oid = ObjectId(user_id)
    #         user_doc = await self.collection.find_one({"_id": oid})
    #         if user_doc:
    #             user_doc = {
    #                 'id': user_doc['_id'],
    #                 'username': user_doc['username'],
    #                 'hashed_password': user_doc['hashed_password'],
    #                 'is_active': user_doc['is_active'],
    #                 'is_admin': user_doc['is_admin']
    #             }
    #             return UserInDB(**user_doc)
    #         return None
    #     except Exception as e:
    #         print(f"Error fetching user by ID {user_id}: {e}")
    #         return None

    # Можно добавить методы update_user, delete_user и т.д. по необходимости
