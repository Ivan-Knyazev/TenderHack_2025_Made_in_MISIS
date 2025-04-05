import os
import time
import threading
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid

from knowledge_base import KnowledgeBase
from ai_agent import SupportAgent

# Константы
KNOWLEDGE_DIR = "./knowledge"
VECTOR_STORE_DIR = "./vector_store"
ALLOWED_EXTENSIONS = {"txt", "pdf"}

# Initialize FastAPI app
app = FastAPI(title="Система поддержки на базе RAG Fusion")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize knowledge base
kb = KnowledgeBase()

# Статус обновления базы знаний
rebuild_status = {
    "in_progress": False,
    "last_completed": None,
    "files_processed": 0,
    "error": None
}

# Try to load the vector store if it exists
if os.path.exists(VECTOR_STORE_DIR):
    try:
        kb.load_vector_store(VECTOR_STORE_DIR)
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")

# Initialize agent
agent = SupportAgent(kb_path=VECTOR_STORE_DIR)

# Store conversation states
conversations = {}

class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    reset_conversation: bool = False

class QueryResponse(BaseModel):
    response: str
    human_handoff: bool = False
    conversation_id: str
    source_documents: List[Dict[str, Any]] = []

def rebuild_knowledge_base(rebuild_all: bool = True, pdf_files: List[str] = None):
    """
    Обновляет базу знаний, загружая документы из директории KNOWLEDGE_DIR.
    
    Args:
        rebuild_all: Если True, перестраивает всю базу знаний. Если False, добавляет только новые файлы.
        pdf_files: Список PDF-файлов для добавления в базу знаний.
    """
    global rebuild_status
    
    # Проверяем, не выполняется ли уже обновление
    if rebuild_status["in_progress"]:
        print("Обновление базы знаний уже выполняется")
        return
    
    try:
        rebuild_status["in_progress"] = True
        rebuild_status["error"] = None
        rebuild_status["files_processed"] = 0
        
        # Проверяем, существует ли директория знаний
        if not os.path.exists(KNOWLEDGE_DIR):
            os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        
        if rebuild_all:
            # Перестроить всю базу знаний
            print("Перестроение всей базы знаний...")
            documents = kb.load_and_process_documents(directory=KNOWLEDGE_DIR)
            if documents:
                kb.create_vector_store(documents, persist_directory=VECTOR_STORE_DIR)
                rebuild_status["files_processed"] = len(documents)
        elif pdf_files:
            # Добавляем только новые PDF-файлы
            print(f"Добавление {len(pdf_files)} PDF-файлов в базу знаний...")
            success = kb.update_from_files(
                files=pdf_files, 
                persist_directory=VECTOR_STORE_DIR,
                is_pdf=True
            )
            if success:
                rebuild_status["files_processed"] = len(pdf_files)
            else:
                rebuild_status["error"] = "Не удалось добавить PDF-файлы в базу знаний"
        
        rebuild_status["last_completed"] = time.time()
    except Exception as e:
        rebuild_status["error"] = str(e)
        print(f"Ошибка при обновлении базы знаний: {str(e)}")
    finally:
        rebuild_status["in_progress"] = False

@app.post("/upload", response_class=JSONResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    rebuild_index: bool = Form(True),
    background_tasks: BackgroundTasks = None
):
    """
    Загружает файлы в директорию знаний и обновляет индекс.
    
    Args:
        files: Список файлов для загрузки
        rebuild_index: Если True, обновляет индекс после загрузки файлов
        background_tasks: BackgroundTasks для выполнения обновления индекса в фоне
    
    Returns:
        JSONResponse с результатами загрузки
    """
    # Проверяем, не выполняется ли уже обновление
    if rebuild_status["in_progress"]:
        return JSONResponse(
            content={
                "success": False,
                "message": "Обновление базы знаний уже выполняется. Пожалуйста, дождитесь завершения."
            },
            status_code=409
        )
    
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    
    uploaded_files = []
    uploaded_pdf_files = []
    errors = []
    
    for file in files:
        # Получаем расширение файла
        ext = file.filename.split(".")[-1].lower()
        
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"Файл {file.filename} имеет недопустимое расширение")
            continue
        
        # Сохраняем файл
        file_path = os.path.join(KNOWLEDGE_DIR, file.filename)
        
        try:
            with open(file_path, "wb") as f:
                f.write(await file.read())
            uploaded_files.append(file.filename)
            
            # Отслеживаем PDF-файлы отдельно
            if ext == "pdf":
                uploaded_pdf_files.append(file_path)
                
        except Exception as e:
            errors.append(f"Ошибка при сохранении файла {file.filename}: {str(e)}")
    
    # Запускаем обновление индекса в фоне, если rebuild_index=True
    if rebuild_index and uploaded_files:
        if background_tasks:
            if uploaded_pdf_files:
                # Если загружены только PDF-файлы, добавляем их к существующей базе знаний
                background_tasks.add_task(
                    rebuild_knowledge_base, 
                    rebuild_all=False, 
                    pdf_files=uploaded_pdf_files
                )
            else:
                # В противном случае перестраиваем всю базу знаний
                background_tasks.add_task(rebuild_knowledge_base, rebuild_all=True)
        else:
            # Если background_tasks недоступен, создаем отдельный поток
            thread = threading.Thread(
                target=rebuild_knowledge_base, 
                args=(not bool(uploaded_pdf_files), uploaded_pdf_files)
            )
            thread.daemon = True
            thread.start()
    
    # Возвращаем результат
    return JSONResponse(
        content={
            "success": True if uploaded_files else False,
            "uploaded_files": uploaded_files,
            "errors": errors,
            "rebuild_index": rebuild_index,
            "rebuild_status": "started" if rebuild_index and uploaded_files else "skipped"
        }
    )

@app.get("/rebuild_status")
async def get_rebuild_status():
    """
    Возвращает текущий статус обновления базы знаний.
    """
    return JSONResponse(
        content={
            "in_progress": rebuild_status["in_progress"],
            "last_completed": rebuild_status["last_completed"],
            "files_processed": rebuild_status["files_processed"],
            "error": rebuild_status["error"]
        }
    )

@app.post("/rebuild", response_class=JSONResponse)
async def manual_rebuild(background_tasks: BackgroundTasks):
    """
    Вручную запускает обновление базы знаний.
    """
    # Проверяем, не выполняется ли уже обновление
    if rebuild_status["in_progress"]:
        return JSONResponse(
            content={
                "success": False,
                "message": "Обновление базы знаний уже выполняется. Пожалуйста, дождитесь завершения."
            },
            status_code=409
        )
    
    background_tasks.add_task(rebuild_knowledge_base, rebuild_all=True)
    
    return JSONResponse(
        content={
            "success": True,
            "message": "Обновление базы знаний запущено",
            "rebuild_status": "started"
        }
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query and return a response from the RAG agent.
    """
    # Generate a conversation ID if not provided
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Reset conversation if requested
    if request.reset_conversation and conversation_id in conversations:
        del conversations[conversation_id]
    
    # Create a new conversation history if not exists
    if conversation_id not in conversations:
        conversations[conversation_id] = agent.memory
    
    # Update agent memory with current conversation
    agent.memory = conversations[conversation_id]
    
    # Process the query
    result = agent.process_message(request.query)
    
    # Save updated conversation state
    conversations[conversation_id] = agent.memory
    
    # Extract data from the result
    answer = result["response"]
    human_handoff = result.get("human_handoff", False)
    
    # Convert source documents if available
    sources = []
    if "source_documents" in result and result["source_documents"]:
        for doc in result["source_documents"]:
            if isinstance(doc, dict):
                sources.append(doc)
            elif hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                sources.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Неизвестно")
                })
    
    return QueryResponse(
        response=answer,
        human_handoff=human_handoff,
        conversation_id=conversation_id,
        source_documents=sources
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 