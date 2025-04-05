from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
from typing import Optional

# UserCreate здесь не нужен, но для ясности
from app.models.user import UserInDB


class UserRepository:
    """Репозиторий для асинхронной работы с коллекцией users в MongoDB."""

    def __init__(self, database: AsyncIOMotorDatabase):
        self.collection: AsyncIOMotorCollection = database.users

    async def create_user(self, user_in_db: UserInDB) -> UserInDB:
        """Создает нового пользователя в БД."""

        user_doc = await self.collection.find_one({"_id": user_in_db.id})
        if user_doc:
            raise ValueError(
                f"Username '{user_in_db.username}' already exists.")
        else:
            try:
                new_user = {
                    'username': user_in_db.username,
                    'hashed_password': user_in_db.hashed_password,
                    'is_active': user_in_db.is_active,
                    'is_admin': user_in_db.is_admin
                }
                insert_result = await self.collection.insert_one(new_user)
                # Получаем созданный документ, чтобы вернуть его с _id
                created_doc = await self.collection.find_one({"_id": insert_result.inserted_id})
                if created_doc:
                    # Создаем модель Pydantic из документа
                    # print(created_doc['_id'])
                    created_doc = {
                        'id': created_doc['_id'],
                        'username': created_doc['username'],
                        'hashed_password': created_doc['hashed_password'],
                        'is_active': created_doc['is_active'],
                        'is_admin': created_doc['is_admin']
                    }
                    # print(created_doc)
                    return UserInDB(**created_doc)
                    # return created_doc
                else:
                    # Эта ситуация маловероятна, но стоит обработать
                    raise RuntimeError(
                        "Failed to retrieve created user document")
            except DuplicateKeyError:
                # Обрабатываем нарушение уникального индекса (username)
                raise ValueError(
                    f"Username '{user_in_db.username}' already exists.")
            except Exception as e:
                # Логирование или дальнейшая обработка ошибок БД
                print(f"Database error during user creation: {e}")
                raise  # Перевыброс исключения

        # # Преобразуем Pydantic модель в dict, используя alias для _id
        # # Исключаем 'id', используем '_id'
        # user_doc = user_in_db.model_dump(by_alias=True, exclude={"id"})
        # if user_doc.get("_id") is None:  # Если _id не был предоставлен
        #     del user_doc["_id"]  # Удаляем, чтобы MongoDB сгенерировала его

        # try:
        #     insert_result = await self.collection.insert_one(user_doc)
        #     # Получаем созданный документ, чтобы вернуть его с _id
        #     created_doc = await self.collection.find_one({"_id": insert_result.inserted_id})
        #     if created_doc:
        #         # Создаем модель Pydantic из документа
        #         return UserInDB(**created_doc)
        #     else:
        #         # Эта ситуация маловероятна, но стоит обработать
        #         raise RuntimeError("Failed to retrieve created user document")
        # except DuplicateKeyError:
        #     # Обрабатываем нарушение уникального индекса (username)
        #     raise ValueError(
        #         f"Username '{user_in_db.username}' already exists.")
        # except Exception as e:
        #     # Логирование или дальнейшая обработка ошибок БД
        #     print(f"Database error during user creation: {e}")
        #     raise  # Перевыброс исключения

    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Ищет пользователя по username."""
        user_doc = await self.collection.find_one({"username": username})
        if user_doc:
            user_doc = {
                'id': user_doc['_id'],
                'username': user_doc['username'],
                'hashed_password': user_doc['hashed_password'],
                'is_active': user_doc['is_active'],
                'is_admin': user_doc['is_admin']
            }
            return UserInDB(**user_doc)
        return None

    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Ищет пользователя по его MongoDB _id (строковое представление)."""
        if not ObjectId.is_valid(user_id):
            print(f"Invalid ObjectId format: {user_id}")
            return None  # Невалидный формат ID
        try:
            oid = ObjectId(user_id)
            user_doc = await self.collection.find_one({"_id": oid})
            if user_doc:
                user_doc = {
                    'id': user_doc['_id'],
                    'username': user_doc['username'],
                    'hashed_password': user_doc['hashed_password'],
                    'is_active': user_doc['is_active'],
                    'is_admin': user_doc['is_admin']
                }
                return UserInDB(**user_doc)
            return None
        except Exception as e:
            print(f"Error fetching user by ID {user_id}: {e}")
            return None

    # Можно добавить методы update_user, delete_user и т.д. по необходимости
