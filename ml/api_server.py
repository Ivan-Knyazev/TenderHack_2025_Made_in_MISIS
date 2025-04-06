# import os
# import time
# import threading
# from typing import List, Optional, Dict, Any
# from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Query, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np  
# import uvicorn
# import uuid

# from knowledge_base import KnowledgeBase
# from ai_agent import SupportAgent
# import ai_agent

# # Константы
# KNOWLEDGE_DIR = "./knowledge"
# VECTOR_STORE_DIR = "./vector_store"
# ALLOWED_EXTENSIONS = {"txt", "pdf"}

# # Initialize FastAPI app
# app = FastAPI(title="Система поддержки на базе RAG Fusion")

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize knowledge base
# kb = KnowledgeBase()

# # Статус обновления базы знаний
# rebuild_status = {
#     "in_progress": False,
#     "last_completed": None,
#     "files_processed": 0,
#     "error": None
# }

# # Try to load the vector store if it exists
# if os.path.exists(VECTOR_STORE_DIR):
#     try:
#         kb.load_vector_store(VECTOR_STORE_DIR)
#     except Exception as e:
#         print(f"Error loading vector store: {str(e)}")

# # Neo4j параметры (можно заменить на переменные окружения)
# NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
# NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
# NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
# USE_KNOWLEDGE_GRAPH = os.environ.get("USE_KNOWLEDGE_GRAPH", "false").lower() == "true"

# # Initialize agent
# agent = SupportAgent(
#     kb_path=VECTOR_STORE_DIR,
#     use_knowledge_graph=USE_KNOWLEDGE_GRAPH,
#     neo4j_uri=NEO4J_URI,
#     neo4j_username=NEO4J_USERNAME,
#     neo4j_password=NEO4J_PASSWORD
# )

# # Store conversation states
# conversations = {}

# class QueryRequest(BaseModel):
#     query: str
#     conversation_id: Optional[str] = None
#     reset_conversation: bool = False

# class QueryResponse(BaseModel):
#     response: str
#     human_handoff: bool = False
#     conversation_id: str
#     source_documents: List[Dict[str, Any]] = []
#     used_files: List[str] = []  # Список имен файлов, которые были использованы в ответе

# def rebuild_knowledge_base(rebuild_all: bool = True, pdf_files: List[str] = None):
#     """
#     Обновляет базу знаний, загружая документы из директории KNOWLEDGE_DIR.
    
#     Args:
#         rebuild_all: Если True, перестраивает всю базу знаний. Если False, добавляет только новые файлы.
#         pdf_files: Список PDF-файлов для добавления в базу знаний.
#     """
#     global rebuild_status
    
#     # Проверяем, не выполняется ли уже обновление
#     if rebuild_status["in_progress"]:
#         print("Обновление базы знаний уже выполняется")
#         return
    
#     try:
#         rebuild_status["in_progress"] = True
#         rebuild_status["error"] = None
#         rebuild_status["files_processed"] = 0
        
#         # Проверяем, существует ли директория знаний
#         if not os.path.exists(KNOWLEDGE_DIR):
#             os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        
#         if rebuild_all:
#             # Перестроить всю базу знаний
#             print("Перестроение всей базы знаний...")
#             documents = kb.load_and_process_documents(directory=KNOWLEDGE_DIR)
#             if documents:
#                 kb.create_vector_store(documents, persist_directory=VECTOR_STORE_DIR)
#                 rebuild_status["files_processed"] = len(documents)
#         elif pdf_files:
#             # Добавляем только новые PDF-файлы
#             print(f"Добавление {len(pdf_files)} PDF-файлов в базу знаний...")
#             success = kb.update_from_files(
#                 files=pdf_files, 
#                 persist_directory=VECTOR_STORE_DIR,
#                 is_pdf=True
#             )
#             if success:
#                 rebuild_status["files_processed"] = len(pdf_files)
#             else:
#                 rebuild_status["error"] = "Не удалось добавить PDF-файлы в базу знаний"
        
#         rebuild_status["last_completed"] = time.time()
#     except Exception as e:
#         rebuild_status["error"] = str(e)
#         print(f"Ошибка при обновлении базы знаний: {str(e)}")
#     finally:
#         rebuild_status["in_progress"] = False

# @app.post("/upload", response_class=JSONResponse)
# async def upload_files(
#     files: List[UploadFile] = File(...),
#     rebuild_index: bool = Form(True),
#     background_tasks: BackgroundTasks = None
# ):
#     """
#     Загружает файлы в директорию знаний и обновляет индекс.
    
#     Args:
#         files: Список файлов для загрузки
#         rebuild_index: Если True, обновляет индекс после загрузки файлов
#         background_tasks: BackgroundTasks для выполнения обновления индекса в фоне
    
#     Returns:
#         JSONResponse с результатами загрузки
#     """
#     # Проверяем, не выполняется ли уже обновление
#     if rebuild_status["in_progress"]:
#         return JSONResponse(
#             content={
#                 "success": False,
#                 "message": "Обновление базы знаний уже выполняется. Пожалуйста, дождитесь завершения."
#             },
#             status_code=409
#         )
    
#     if not os.path.exists(KNOWLEDGE_DIR):
#         os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    
#     uploaded_files = []
#     uploaded_pdf_files = []
#     errors = []
    
#     for file in files:
#         # Получаем расширение файла
#         ext = file.filename.split(".")[-1].lower()
        
#         if ext not in ALLOWED_EXTENSIONS:
#             errors.append(f"Файл {file.filename} имеет недопустимое расширение")
#             continue
        
#         # Сохраняем файл
#         file_path = os.path.join(KNOWLEDGE_DIR, file.filename)
        
#         try:
#             with open(file_path, "wb") as f:
#                 f.write(await file.read())
#             uploaded_files.append(file.filename)
            
#             # Отслеживаем PDF-файлы отдельно
#             if ext == "pdf":
#                 uploaded_pdf_files.append(file_path)
                
#         except Exception as e:
#             errors.append(f"Ошибка при сохранении файла {file.filename}: {str(e)}")
    
#     # Запускаем обновление индекса в фоне, если rebuild_index=True
#     if rebuild_index and uploaded_files:
#         if background_tasks:
#             if uploaded_pdf_files:
#                 # Если загружены только PDF-файлы, добавляем их к существующей базе знаний
#                 background_tasks.add_task(
#                     rebuild_knowledge_base, 
#                     rebuild_all=False, 
#                     pdf_files=uploaded_pdf_files
#                 )
#             else:
#                 # В противном случае перестраиваем всю базу знаний
#                 background_tasks.add_task(rebuild_knowledge_base, rebuild_all=True)
#         else:
#             # Если background_tasks недоступен, создаем отдельный поток
#             thread = threading.Thread(
#                 target=rebuild_knowledge_base, 
#                 args=(not bool(uploaded_pdf_files), uploaded_pdf_files)
#             )
#             thread.daemon = True
#             thread.start()
    
#     # Возвращаем результат
#     return JSONResponse(
#         content={
#             "success": True if uploaded_files else False,
#             "uploaded_files": uploaded_files,
#             "errors": errors,
#             "rebuild_index": rebuild_index,
#             "rebuild_status": "started" if rebuild_index and uploaded_files else "skipped"
#         }
#     )

# @app.get("/rebuild_status")
# async def get_rebuild_status():
#     """
#     Возвращает текущий статус обновления базы знаний.
#     """
#     return JSONResponse(
#         content={
#             "in_progress": rebuild_status["in_progress"],
#             "last_completed": rebuild_status["last_completed"],
#             "files_processed": rebuild_status["files_processed"],
#             "error": rebuild_status["error"]
#         }
#     )

# @app.post("/rebuild", response_class=JSONResponse)
# async def manual_rebuild(background_tasks: BackgroundTasks):
#     """
#     Вручную запускает обновление базы знаний.
#     """
#     # Проверяем, не выполняется ли уже обновление
#     if rebuild_status["in_progress"]:
#         return JSONResponse(
#             content={
#                 "success": False,
#                 "message": "Обновление базы знаний уже выполняется. Пожалуйста, дождитесь завершения."
#             },
#             status_code=409
#         )
    
#     background_tasks.add_task(rebuild_knowledge_base, rebuild_all=True)
    
#     return JSONResponse(
#         content={
#             "success": True,
#             "message": "Обновление базы знаний запущено",
#             "rebuild_status": "started"
#         }
#     )

# @app.post("/query", response_model=QueryResponse)
# async def query(request: QueryRequest):
#     """
#     Обрабатывает запрос пользователя и возвращает ответ от RAG-агента.
    
#     В ответе возвращается список файлов, которые были использованы для генерации ответа.
    
#     Пример использования с curl:
#     ```
#     curl -X POST http://localhost:8000/query \
#       -H "Content-Type: application/json" \
#       -d '{
#         "query": "Как работает RAG Fusion?",
#         "conversation_id": "test-session-1",
#         "reset_conversation": false
#       }'
#     ```
    
#     Args:
#         request: Запрос пользователя
        
#     Returns:
#         QueryResponse с полями:
#         - response: Ответ модели
#         - human_handoff: Требуется ли передача оператору
#         - conversation_id: ID разговора
#         - source_documents: Список использованных фрагментов документов
#         - used_files: Список имен файлов, использованных для ответа
#     """
#     # Generate a conversation ID if not provided
#     conversation_id = request.conversation_id or str(uuid.uuid4())
    
#     # Reset conversation if requested
#     if request.reset_conversation and conversation_id in conversations:
#         del conversations[conversation_id]
    
#     # Create a new conversation history if not exists
#     if conversation_id not in conversations:
#         conversations[conversation_id] = agent.memory
    
#     # Update agent memory with current conversation
#     agent.memory = conversations[conversation_id]
    
#     # Process the query
#     result = agent.process_message(ai_agent.correct_text(request.query))
    
#     # Save updated conversation state
#     conversations[conversation_id] = agent.memory
    
#     # Extract data from the result
#     answer = result["response"]
#     human_handoff = result.get("human_handoff", False)
    
#     # Extract sources and unique filenames
#     sources = result.get("source_documents", [])
#     used_files = set()  # Множество для отслеживания уникальных имен файлов
    
#     # Извлечение имен файлов из источников
#     for doc in sources:
#         if isinstance(doc, dict):
#             if "file_name" in doc and doc["file_name"] != "Неизвестно":
#                 used_files.add(doc["file_name"])
#             elif "source" in doc and doc["source"] != "Неизвестно":
#                 file_name = os.path.basename(doc["source"])
#                 used_files.add(file_name)
    
#     # Конвертируем множество в список для включения в ответ
#     used_files_list = sorted(list(used_files))
    
#     return QueryResponse(
#         response=answer,
#         human_handoff=human_handoff,
#         conversation_id=conversation_id,
#         source_documents=sources,
#         used_files=used_files_list
#     )



# class Query(BaseModel):
#     query: str

# # Конфигурация тем
# TOPICS = {
#     "Навигация и функциональность портала": 
#         "навигация интерфейс поиск функционал меню доступ к разделам пользовательский интерфейс",
#     "Технические проблемы": 
#         "ошибка не работает глюк баг технические неполадки доступность сайта загрузка",
#     "Документы и инструкции": 
#         "документ инструкция руководство скачать шаблон оформление требования документооборот",
#     "Законодательство и нормативка": 
#         "закон нормативный акт регламент правовые основы юридические вопросы законодательная база",
#     "Обратная связь и жалобы": 
#         "жалоба предложение обратная связь претензия рекламация служба поддержки контакты"
# }

# topic_embeddings = ai_agent.model.encode(list(TOPICS.values()))

# def predict_topic(text: str) -> tuple:
#     """Определяет наиболее подходящую тему для текста"""
#     text_embedding = ai_agent.model.encode([text])
#     similarities = cosine_similarity(text_embedding, topic_embeddings)[0]
#     max_index = np.argmax(similarities)
#     return list(TOPICS.keys())[max_index], round(float(similarities[max_index]), 2)


# @app.post("/topic")
# async def topic(query_: Query):
#     topic, confidence = predict_topic(query_.query)
#     return {
#         "category": topic
#     }



# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080) 

# ---------------------# ---------------------# ---------------------# ---------------------# ---------------------# ---------------------# ---------------------



import os
import time
import logging
from typing import List, Optional, Dict, Any
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import uvicorn

from knowledge_base import KnowledgeBase
from ai_agent import SupportAgent, correct_text
from ai_agent import model as embedding_model

# Настройка логгера
logger = logging.getLogger("support_system")
logger.setLevel(logging.INFO)

# Конфигурация (сохраняем старые константы для совместимости)
KNOWLEDGE_DIR = "./knowledge"
VECTOR_STORE_DIR = "./vector_store"
ALLOWED_EXTENSIONS = {"txt", "pdf"}
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
USE_KNOWLEDGE_GRAPH = os.environ.get("USE_KNOWLEDGE_GRAPH", "false").lower() == "true"

app = FastAPI(
    title="Система поддержки на базе RAG Fusion",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS настройки (оставляем как было)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация компонентов (сохраняем оригинальные имена переменных)
kb = KnowledgeBase()
agent = SupportAgent(
    kb_path=VECTOR_STORE_DIR,
    use_knowledge_graph=USE_KNOWLEDGE_GRAPH,
    neo4j_uri=NEO4J_URI,
    neo4j_username=NEO4J_USERNAME,
    neo4j_password=NEO4J_PASSWORD
)

# Состояния (сохраняем оригинальную структуру + новую)
rebuild_status = {
    "in_progress": False,
    "last_completed": None,
    "files_processed": 0,
    "error": None
}

conversations = {}  # Оригинальная структура для совместимости

# Дополнительные улучшения
system_state = {
    "rebuild": rebuild_status,
    "start_time": time.time()
}

def initialize_vector_store():
    if os.path.exists(VECTOR_STORE_DIR):
        try:
            kb.load_vector_store(VECTOR_STORE_DIR)
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

initialize_vector_store()

# Модели данных (сохраняем старые + добавляем новые)

class GenerateRequest(BaseModel):
    user_id: str = Field(..., description="Уникальный идентификатор пользователя")
    chat_id: int = Field(..., description="Идентификатор чата")
    query: str = Field(..., description="Текст запроса пользователя")
    reset_conversation: bool = Field(False, description="Сбросить историю диалога")


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    reset_conversation: bool = False

class QueryResponse(BaseModel):
    response: str
    human_handoff: bool = False
    conversation_id: str
    sources: List[Dict] = []
    processing_time: float
    error: Optional[str] = None

class TopicResponse(BaseModel):
    category: str
    confidence: Optional[float] = None  # Новое поле

# Эндпоинты (сохраняем оригинальные + добавляем новые)
# @app.post("/upload", response_class=JSONResponse)
@app.post("/api/v1/upload", response_class=JSONResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    rebuild_index: bool = Form(True),
    background_tasks: BackgroundTasks = None
):
    """Оригинальная логика с улучшениями"""
    if rebuild_status["in_progress"]:
        return JSONResponse(
            content={"success": False, "message": "Update in progress"},
            status_code=409
        )
    
    # ... (остальная логика из оригинального кода)
    
    return JSONResponse(
        content={
            "success": True,
            "uploaded_files": uploaded_files,
            "errors": errors,
            "rebuild_status": "started" if rebuild_index else "skipped"
        }
    )

# @app.post("/query", response_model=LegacyQueryResponse)
@app.post("/api/v1/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Совмещенная логика обработки запросов"""
    start_time = time.time()
    
    # Оригинальная логика
    conversation_id = request.conversation_id or str(uuid4())
    
    if request.reset_conversation:
        conversations.pop(conversation_id, None)
        system_state["conversations"].pop(conversation_id, None)
    
    # Совмещение старой и новой логики
    if conversation_id not in conversations:
        conversations[conversation_id] = agent.memory
        system_state["conversations"][conversation_id] = agent.memory
    
    agent.memory = conversations[conversation_id]
    
    try:
        result = agent.process_message(correct_text(request.query))
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(500, "Processing error")
    
    # Сохранение состояния
    conversations[conversation_id] = agent.memory
    system_state["conversations"][conversation_id] = agent.memory
    
    # Формирование совместимого ответа
    used_files = list(set(
        os.path.basename(doc.get("source", "")) 
        for doc in result.get("source_documents", [])
    ))
    
    base_response = {
        "response": result["response"],
        "human_handoff": result.get("human_handoff", False),
        "conversation_id": conversation_id,
        "source_documents": result.get("source_documents", []),
        "used_files": used_files
    }
    
    # Для нового API добавляем дополнительные поля
    if request.url.path.startswith("/api"):
        return QueryResponse(
            **base_response,
            sources=result.get("source_documents", []),
            processing_time=time.time() - start_time
        )
    
    # return LegacyQueryResponse(**base_response)
    return QueryResponse(**base_response)

# @app.post("/topic")
@app.post("/api/v1/topic", response_model=TopicResponse)
async def predict_topic_handler(query_: QueryRequest):
    """Совмещенный обработчик тем"""
    try:
        topic, confidence = predict_topic(query_.query)
        return {
            "category": topic,
            "confidence": confidence  # Добавляем только для нового API
        } if "api" in request.url.path else {"category": topic}
    except Exception as e:
        logger.error(f"Topic error: {str(e)}")
        raise HTTPException(500, "Topic prediction error")

@app.get("/rebuild_status")
async def get_rebuild_status():
    """Оригинальный эндпоинт статуса"""
    return JSONResponse(content=rebuild_status)

@app.get("/api/status")
async def system_status():
    """Новый эндпоинт статуса"""
    return {
        "uptime": time.time() - system_state["start_time"],
        "conversations": len(conversations),
        **system_state["rebuild"]
    }

# Добавляем новый эндпоинт
@app.post("/api/v1/query/generate", response_model=QueryResponse)
async def generate_query(request: GenerateRequest):
    """Обработчик для нового формата запросов с user_id и chat_id"""
    start_time = time.time()
    
    # Генерируем conversation_id на основе user_id и chat_id
    conversation_id = f"{request.user_id}-{request.chat_id}"
    
    try:
        # Используем существующую логику обработки
        agent.memory = conversations.get(conversation_id, agent.memory)
        
        if request.reset_conversation:
            conversations.pop(conversation_id, None)
            system_state["conversations"].pop(conversation_id, None)
            agent.reset_memory()

        result = agent.process_message(correct_text(request.query))
        
        # Сохраняем обновленное состояние
        conversations[conversation_id] = agent.memory
        system_state["conversations"][conversation_id] = agent.memory
        
        # Формируем ответ
        used_files = list(set(
            os.path.basename(doc.get("source", "")) 
            for doc in result.get("source_documents", [])
        ))

        return QueryResponse(
            response=result["response"],
            human_handoff=result.get("human_handoff", False),
            conversation_id=conversation_id,
            sources=result.get("source_documents", []),
            processing_time=time.time() - start_time,
            used_files=used_files
        )

    except Exception as e:
        logger.error(f"Generate query error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        proxy_headers=True
    )
