import os
from knowledge_base import KnowledgeBase

def main():
    print("Создание базы знаний для демонстрации...")
    
    # Проверка существования директории знаний
    if not os.path.exists("./knowledge"):
        os.makedirs("./knowledge")
        print("Создана директория ./knowledge/")
    
    # Проверка существования демонстрационных файлов
    knowledge_files = ["faq.txt", "about_service.txt", "pricing.txt", "installation_guide.txt", "features.txt"]
    missing_files = []
    
    for file in knowledge_files:
        if not os.path.exists(f"./knowledge/{file}"):
            missing_files.append(file)
    
    if missing_files:
        print("Ошибка: Следующие демонстрационные файлы не найдены в директории knowledge/:")
        for file in missing_files:
            print(f"- {file}")
        print("Убедитесь, что все необходимые файлы находятся в директории knowledge/")
        return False
    
    # Инициализация базы знаний
    print("Инициализация базы знаний...")
    kb = KnowledgeBase()
    
    # Загрузка и обработка документов
    print("Загрузка и обработка документов...")
    documents = kb.load_and_process_documents(
        directory="./knowledge"
    )
    
    print(f"Загружено документов: {len(documents)}")
    
    # Подсчет типов документов
    file_types = {}
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        ext = os.path.splitext(source)[1].lower()
        if ext:
            file_types[ext] = file_types.get(ext, 0) + 1
    
    print("\nТипы загруженных документов:")
    for ext, count in file_types.items():
        print(f"- {ext}: {count} документов")
    
    # Примеры содержимого нескольких документов
    print("\nПримеры загруженных документов:")
    for i, doc in enumerate(documents[:3], 1):
        source = doc.metadata.get('source', 'Неизвестно')
        page_info = f" (стр. {doc.metadata.get('page', 1)})" if 'page' in doc.metadata else ""
        print(f"\nДокумент {i}:")
        print(f"Содержание: {doc.page_content[:150]}...")
        print(f"Источник: {source}{page_info}")
    
    # Создание векторного хранилища
    print("\nСоздание векторного хранилища...")
    kb.create_vector_store(documents, persist_directory="./vector_store")
    
    print("\nБаза знаний успешно создана и сохранена в директории ./vector_store/")
    print("Теперь вы можете запустить ИИ-агента командой 'python ai_agent.py'")
    print("или API сервер командой 'python api_server.py'")
    
    # Демонстрация поиска
    print("\nДемонстрация поиска в базе знаний:")
    
    test_queries = [
        "Как работает технология RAG Fusion?",
        "Сколько стоит тариф Премиум?",
        "Как установить систему ПоддержкаПро на сервер?",
        "Что делать, если система работает медленно?",
        "Как связаться с технической поддержкой?"
    ]
    
    for query in test_queries:
        print(f"\nЗапрос: '{query}'")
        results = kb.similarity_search(query, k=1)
        for doc in results:
            source = doc.metadata.get('source', 'Неизвестно')
            page_info = f" (стр. {doc.metadata.get('page', 1)})" if 'page' in doc.metadata else ""
            print(f"Найдено: {doc.page_content[:150]}...")
            print(f"Источник: {source}{page_info}")
    
    return True

if __name__ == "__main__":
    main() 