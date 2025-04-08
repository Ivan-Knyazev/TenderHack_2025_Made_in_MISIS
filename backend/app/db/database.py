import motor.motor_asyncio
from pymongo import ASCENDING
from typing import Optional
from app.core.config import settings

client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None


async def connect_db():
    """Устанавливает асинхронное соединение с MongoDB."""
    global client, db
    print("Connecting to MongoDB...")
    client = motor.motor_asyncio.AsyncIOMotorClient(
        settings['MONGO_URL'],
        uuidRepresentation="standard"  # Рекомендуется для работы с UUID
    )
    db = client[settings['DATABASE_NAME']]
    await create_db_indexes(db)  # Создаем индексы при старте
    print(f"Connected to MongoDB database: {settings['DATABASE_NAME']}")


async def close_db():
    """Закрывает соединение с MongoDB."""
    global client
    if client:
        print("Closing MongoDB connection...")
        client.close()
        print("MongoDB connection closed.")


def get_db_client() -> motor.motor_asyncio.AsyncIOMotorClient:
    if client is None:
        raise Exception("MongoDB client not initialized.")
    return client


def get_database() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    """Возвращает объект базы данных для использования в репозиториях."""
    if db is None:
        raise Exception("Database not initialized.")
    return db


async def create_db_indexes(database: motor.motor_asyncio.AsyncIOMotorDatabase):
    """Создает уникальный индекс для поля username в коллекции users."""
    try:
        await database.users.create_index(
            [("username", ASCENDING)],
            name="username_unique_idx",
            unique=True
        )
        print("User collection indexes ensured.")
    except Exception as e:
        print(f"Error creating indexes for 'users' collection: {e}")
