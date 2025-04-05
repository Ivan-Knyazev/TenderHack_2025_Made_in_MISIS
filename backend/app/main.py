from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.db.database import connect_db, close_db
from app.controllers import auth_controller, users_controller  # Импорт роутеров


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управляет ресурсами приложения (подключение к БД)."""
    print(f"Starting up {app.title}...")
    await connect_db()  # Подключаемся к БД при старте
    yield
    # Код после yield выполнится при остановке
    print(f"Shutting down {app.title}...")
    await close_db()  # Закрываем соединение с БД

app = FastAPI(
    title="AI Assistant for TenderHack",
    description="FastAPI backend application with MongoDB for TenderHack",
    version="1.0.0",
    lifespan=lifespan  # Используем новый механизм lifespan
)

# Подключение роутеров
api_prefix = "/api/v1"
app.include_router(auth_controller.router, prefix=api_prefix)
app.include_router(users_controller.router, prefix=api_prefix)


@app.get("/", tags=["Root (test)"])
async def read_root():
    """Корневой эндпоинт для простой проверки работы сервиса."""
    return {"message": f"Welcome to {app.title}!"}

# Для запуска: uvicorn app.main:app --reload --port 8001
# (порт можно выбрать любой)
