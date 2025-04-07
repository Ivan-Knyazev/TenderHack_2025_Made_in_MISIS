from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from app.db.database import connect_db, close_db
# Импорт роутеров
from app.controllers import auth_controller, users_controller, query_controller
import os
import aiofiles


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Подключение роутеров
api_prefix = "/api/v1"
app.include_router(auth_controller.router, prefix=api_prefix)
app.include_router(users_controller.router, prefix=api_prefix)
app.include_router(query_controller.router, prefix=api_prefix)


@app.get("/", tags=["Root (test)"])
async def read_root():
    """Корневой эндпоинт для простой проверки работы сервиса."""
    return {"message": f"Welcome to {app.title}!"}


# Files
async def generate_file_content(file_path: str):
    """
    Асинхронный генератор, который читает файл блоками.
    """
    try:
        # Используем aiofiles для асинхронного открытия файла
        async with aiofiles.open(file_path, mode='rb') as f:
            while True:
                # Читаем по 8KB за раз (можно настроить)
                chunk = await f.read(8192)
                if not chunk:
                    break
                yield chunk
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")


@app.get("/files/{file_path:path}")
async def get_file(file_path: str):
    """
    Асинхронная эндпойнт для скачивания файла.
    """
    full_file_path = os.path.join(
        "./files", file_path)  # Путь к файлу (настройте!)

    # Проверяем, существует ли файл (синхронно, чтобы избежать проблем с гонкой)
    if not os.path.exists(full_file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return StreamingResponse(
        generate_file_content(full_file_path),
        media_type="application/octet-stream",  # Тип файла (можно настроить)
        headers={
            # Имя файла для скачивания
            "Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"
        }
    )

# # Создадим директорию и файл для тестирования
# if not os.path.exists("./files"):
#     os.makedirs("./files")

# if not os.path.exists("./files/example.txt"):
#     with open("./files/example.txt", "w") as f:
#         # Создадим большой файл
#         f.write(
#             "This is an example file for testing async file serving in FastAPI.\n" * 1000)


# Для запуска: uvicorn app.main:app --reload --port 8001
# (порт можно выбрать любой)
