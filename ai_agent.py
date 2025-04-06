import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from langchain.schema import Document, AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ParentDocumentRetriever, BM25Retriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
from knowledge_base import KnowledgeBase
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from langchain.graphs import Neo4jGraph
from transformers import T5ForConditionalGeneration, T5Tokenizer


model_t5 = T5ForConditionalGeneration.from_pretrained("UrukHan/t5-russian-spell")
tokenizer_t5 = T5Tokenizer.from_pretrained("UrukHan/t5-russian-spell")


def correct_text(text):
    input_ids = tokenizer_t5(
        f"Spell correct: {text}", 
        return_tensors="pt"
    ).input_ids
    outputs = model_t5.generate(input_ids, max_length=512)
    return tokenizer_t5.decode(outputs[0], skip_special_tokens=True)


# Функции для извлечения сущностей и связей
def extract_entities(text):
    """
    Извлекает сущности из текста используя простые правила.
    В реальном проекте здесь можно использовать NER модели.
    
    Args:
        text: Текст для извлечения сущностей
        
    Returns:
        Список словарей с информацией о сущностях
    """
    entities = []
    
    # Простой пример извлечения сущностей
    # В реальном проекте лучше использовать NER (Named Entity Recognition)
    
    # Шаблоны для поиска сущностей
    product_pattern = r'(тариф|план|пакет)\s+["\']?([А-Яа-я\w\s]+)["\']?'
    feature_pattern = r'функция\s+["\']?([А-Яа-я\w\s]+)["\']?'
    
    # Извлечение продуктов
    for match in re.finditer(product_pattern, text, re.IGNORECASE):
        entity_type = match.group(1).lower()
        entity_name = match.group(2).strip()
        cleaned_name = re.sub(r'\W+', '_', entity_name.lower())
        entity_id = f"{entity_type}_{cleaned_name}"
        
        entities.append({
            'id': entity_id,
            'type': entity_type.capitalize(),
            'props': {
                'name': entity_name,
                'mention': match.group(0)
            }
        })
    
    # Извлечение функций
    for match in re.finditer(feature_pattern, text, re.IGNORECASE):
        feature_name = match.group(1).strip()
        cleaned_name = re.sub(r'\W+', '_', feature_name.lower())
        feature_id = f"feature_{cleaned_name}"
        
        entities.append({
            'id': feature_id,
            'type': 'Feature',
            'props': {
                'name': feature_name,
                'mention': match.group(0)
            }
        })
    
    return entities

def extract_relations(text):
    """
    Извлекает отношения между сущностями из текста.
    
    Args:
        text: Текст для извлечения отношений
        
    Returns:
        Список словарей с информацией о связях
    """
    relations = []
    
    # Простой шаблон для поиска отношений между тарифами и функциями
    pattern = r'(тариф|план|пакет)\s+["\']?([А-Яа-я\w\s]+)["\']?\s+(?:включает|содержит|имеет)\s+функцию\s+["\']?([А-Яа-я\w\s]+)["\']?'
    
    for match in re.finditer(pattern, text, re.IGNORECASE):
        entity_type = match.group(1).lower()
        entity_name = match.group(2).strip()
        feature_name = match.group(3).strip()
        
        cleaned_entity_name = re.sub(r'\W+', '_', entity_name.lower())
        cleaned_feature_name = re.sub(r'\W+', '_', feature_name.lower())
        
        entity_id = f"{entity_type}_{cleaned_entity_name}"
        feature_id = f"feature_{cleaned_feature_name}"
        
        relations.append({
            'source': entity_id,
            'target': feature_id,
            'type': 'INCLUDES'
        })
    
    return relations

def build_knowledge_graph(docs):
    """
    Создает граф знаний на основе документов.
    
    Args:
        docs: Список документов для обработки
        
    Returns:
        Объект Neo4jGraph с построенным графом знаний
    """
    graph = Neo4jGraph()
    
    for doc in docs:
        # Извлечение сущностей из текста документа
        entities = extract_entities(doc.page_content)
        
        for entity in entities:
            graph.add_node(entity['id'], entity['type'], entity['props'])
            
        # Установка связей
        for rel in extract_relations(doc.page_content):
            graph.add_relationship(rel['source'], rel['target'], rel['type'])
    
    return graph


model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


class SemanticChunker:
    """Класс для семантического разбиения текста на связанные части"""
    
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Инициализация семантического чанкера.
        
        Args:
            model_name: Название модели для эмбеддингов предложений
        """
        self.model = model
        self.threshold = 0.85  # Порог семантической схожести

    def split_text(self, text):
        """
        Разбивает текст на семантически связанные части.
        
        Args:
            text: Текст для разбиения
            
        Returns:
            Список семантически связанных кусков текста
        """
        sentences = text.split('. ')
        if not sentences:
            return []
            
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = []
        
        for sent, emb in zip(sentences, embeddings):
            if current_chunk:
                last_sent, last_emb = current_chunk[-1]
                similarity = np.dot(emb, last_emb) / (np.linalg.norm(emb) * np.linalg.norm(last_emb))
                if similarity < self.threshold:
                    chunks.append(". ".join([s for s, _ in current_chunk]) + ".")
                    current_chunk = []
            current_chunk.append((sent, emb))
            
        if current_chunk:
            chunks.append(". ".join([s for s, _ in current_chunk]) + ".")
            
        return chunks
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Обрабатывает документы, разбивая их на семантически связанные части.
        
        Args:
            documents: Список документов для обработки
            
        Returns:
            Список документов с текстом, разбитым на семантические части
        """
        processed_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            if not chunks:  # Если разбиение не удалось, сохраняем исходный документ
                processed_docs.append(doc)
                continue
                
            # Создаем новые документы для каждого семантического чанка
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "semantic_chunk": True
                    }
                )
                processed_docs.append(new_doc)
                
        return processed_docs

class HybridRetriever:
    """Класс для гибридного поиска с динамическим взвешиванием"""
    
    def __init__(self, vector_store, documents, vector_weight=0.6, bm25_weight=0.4):
        """
        Инициализация гибридного поисковика.
        
        Args:
            vector_store: Векторное хранилище
            documents: Список документов для создания BM25 индекса
            vector_weight: Вес для результатов векторного поиска
            bm25_weight: Вес для результатов BM25 поиска
        """
        self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 5
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2))
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
    def _calculate_similarity(self, query, texts):
        """
        Рассчитывает similarity между запросом и текстами с помощью TF-IDF.
        
        Args:
            query: Запрос пользователя
            texts: Список текстов для сравнения
            
        Returns:
            Массив значений схожести
        """
        if not texts:
            return []
            
        try:
            # Добавляем запрос в корпус для обучения TF-IDF
            all_texts = [query] + texts
            tfidf_matrix = self.tfidf.fit_transform(all_texts)
            
            # Извлекаем вектор запроса и векторы текстов
            query_vec = tfidf_matrix[0:1]
            docs_vecs = tfidf_matrix[1:]
            
            # Вычисляем косинусное сходство
            return (query_vec * docs_vecs.T).toarray()[0]
        except Exception as e:
            print(f"Ошибка при расчете TF-IDF similarity: {str(e)}")
            return [1.0] * len(texts)  # Возвращаем единичные веса в случае ошибки

    def get_relevant_documents(self, query):
        """
        Получает релевантные документы с помощью гибридного поиска.
        
        Args:
            query: Запрос пользователя
            
        Returns:
            Список релевантных документов с весами
        """
        # Получаем результаты из разных поисковиков
        vector_results = self.vector_retriever.get_relevant_documents(query)
        bm25_results = self.bm25_retriever.get_relevant_documents(query)
        
        # Если нет результатов из какого-то поисковика, возвращаем результаты другого
        if not vector_results and not bm25_results:
            return []
        if not vector_results:
            return [(doc, 1.0) for doc in bm25_results]
        if not bm25_results:
            return [(doc, 1.0) for doc in vector_results]
        
        # Динамическое взвешивание
        vector_scores = self._calculate_similarity(query, [d.page_content for d in vector_results])
        bm25_scores = self._calculate_similarity(query, [d.page_content for d in bm25_results])
        
        combined = {}
        doc_map = {}
        
        # Сохраняем документы в словарь для быстрого доступа
        for doc in vector_results + bm25_results:
            doc_map[doc.page_content] = doc
        
        # Комбинируем результаты с учетом весов
        for doc, score in zip(vector_results, vector_scores):
            combined[doc.page_content] = score * self.vector_weight
            
        for doc, score in zip(bm25_results, bm25_scores):
            combined[doc.page_content] = combined.get(doc.page_content, 0) + score * self.bm25_weight
        
        # Сортируем результаты по релевантности
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        # Возвращаем документы с их весами
        return [(doc_map[text], score) for text, score in sorted_results[:5]]

class GraphEnhancedRetriever:
    """Класс для поиска с использованием графа знаний"""
    
    def __init__(self, graph, vector_retriever):
        """
        Инициализация ретривера с графом знаний.
        
        Args:
            graph: Объект графа знаний Neo4jGraph
            vector_retriever: Векторный ретривер для комбинирования результатов
        """
        self.graph = graph
        self.vector_retriever = vector_retriever
        
    def get_relevant_documents(self, query):
        """
        Получает релевантные документы с помощью графа знаний и векторного поиска.
        
        Args:
            query: Запрос пользователя
            
        Returns:
            Список релевантных документов
        """
        try:
            # Поиск в графе (безопасный запрос с параметром)
            sanitized_query = query.replace("'", "")
            graph_results = self.graph.query(
                f"MATCH (n) WHERE toLower(n.name) CONTAINS toLower($query) "
                f"RETURN n, id(n) as id LIMIT 3",
                {"query": sanitized_query}
            )
            
            # Векторный поиск
            vector_results = self.vector_retriever.get_relevant_documents(query)
            
            return self._merge_results(graph_results, vector_results)
        except Exception as e:
            print(f"Ошибка при поиске в графе знаний: {str(e)}")
            # В случае ошибки возвращаем только результаты векторного поиска
            return vector_results
            
    def _merge_results(self, graph_results, vector_results):
        """
        Объединяет результаты поиска из графа и векторного поиска.
        
        Args:
            graph_results: Результаты из графа знаний
            vector_results: Результаты векторного поиска
            
        Returns:
            Объединенный список документов
        """
        if not graph_results:
            return vector_results
            
        # Преобразуем результаты графа в документы LangChain
        graph_docs = []
        for result in graph_results:
            if 'n' in result and isinstance(result['n'], dict) and 'properties' in result['n']:
                props = result['n']['properties']
                content = f"Тип: {result['n'].get('labels', ['Неизвестно'])[0]}\n"
                
                # Добавляем свойства узла
                for prop_name, prop_value in props.items():
                    content += f"{prop_name}: {prop_value}\n"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "knowledge_graph",
                        "node_id": result.get('id', 'unknown'),
                        "node_type": result['n'].get('labels', ['Неизвестно'])[0]
                    }
                )
                graph_docs.append(doc)
        
        # Объединяем результаты, сначала идут результаты из графа
        return graph_docs + vector_results

class SupportAgent:
    def __init__(self, 
                 kb_path: str = "./vector_store", 
                 #ollama_base_url: str = "http://46.227.68.167:22077/",
                 ollama_base_url: str = "http://localhost:11434/",
                 model_name: str = "deepseek-r1:7b",
                 use_semantic_chunking: bool = True,
                 use_hybrid_search: bool = True,
                 use_knowledge_graph: bool = False,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_username: str = "neo4j",
                 neo4j_password: str = "password"):
        """
        Initialize the AI support agent with RAG Fusion.
        
        Args:
            kb_path: Path to the knowledge base vector store
            ollama_base_url: Base URL for Ollama API
            model_name: Name of the model to use
            use_semantic_chunking: Whether to use semantic chunking
            use_hybrid_search: Whether to use hybrid search
            use_knowledge_graph: Whether to use knowledge graph
            neo4j_uri: Neo4j server URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
        """
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.use_semantic_chunking = use_semantic_chunking
        self.use_hybrid_search = use_hybrid_search
        self.use_knowledge_graph = use_knowledge_graph
        self.kb_path = kb_path
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.3
        )
        
        # Initialize knowledge base
        self.kb = KnowledgeBase(ollama_base_url=ollama_base_url)
        
        # Initialize source documents storage
        self.source_documents = []
        
        # Initialize semantic chunker
        if self.use_semantic_chunking:
            self.semantic_chunker = SemanticChunker()
        
        # Load vector store if it exists
        if os.path.exists(kb_path):
            self.kb.load_vector_store(kb_path)
        else:
            raise ValueError(f"Knowledge base not found at {kb_path}. Please create it first.")
        
        # Load all documents
        self.all_docs = self._load_all_documents()
            
        # Initialize hybrid retriever if enabled
        if self.use_hybrid_search:
            try:
                self.hybrid_retriever = HybridRetriever(self.kb.vector_store, self.all_docs)
                print(f"Гибридный поисковик инициализирован с {len(self.all_docs)} документами")
            except Exception as e:
                print(f"Ошибка при инициализации гибридного поисковика: {str(e)}")
                self.use_hybrid_search = False
        
        # Initialize knowledge graph if enabled
        if self.use_knowledge_graph:
            try:
                # Подключаемся к Neo4j
                self.graph = Neo4jGraph(
                    url=neo4j_uri,
                    username=neo4j_username,
                    password=neo4j_password
                )
                
                # Проверяем, существуют ли уже узлы в графе
                result = self.graph.query("MATCH (n) RETURN count(n) as count")
                node_count = result[0]['count'] if result else 0
                
                if node_count == 0:
                    # Строим граф знаний на основе документов
                    print("Построение графа знаний...")
                    self.graph = build_knowledge_graph(self.all_docs)
                    print(f"Граф знаний построен")
                else:
                    print(f"Граф знаний уже существует и содержит {node_count} узлов")
                
                # Инициализируем ретривер на основе графа
                self.graph_retriever = GraphEnhancedRetriever(
                    self.graph, 
                    self.kb.vector_store.as_retriever(search_kwargs={"k": 3})
                )
                
            except Exception as e:
                print(f"Ошибка при инициализации графа знаний: {str(e)}")
                self.use_knowledge_graph = False
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup system prompt
        self.system_prompt = """Вы - полезный помощник службы поддержки. 
Отвечайте на вопросы пользователя на основе информации, предоставленной в извлеченных документах. Есть три типа ответа:
ответ на термин 
ответ по проблеме   
ответ по работе пользователя

Если вы не знаете ответ или не можете найти его в извлеченных документах, честно скажите об этом и предложите перевести диалог на оператора-человека.
Всегда будьте вежливы, профессиональны и лаконичны.
Отвечайте пользователю ТОЛЬКО на русском языке, независимо от языка запроса.
В начале задай тему диалога в таком формате <topic>Тема диалога</topic>, а ответ модели в <answer>Ответ модели</answer>, обязательно соблюдай формат.
"""
        
        # Set up the chain
        self._setup_rag_fusion_chain()
        
        # Tracking if human support is requested
        self.human_support_requested = False
        
    def _load_all_documents(self):
        """
        Загружает все документы из базы знаний для создания BM25 индекса.
        
        Returns:
            Список всех документов
        """
        try:
            # Пытаемся получить документы из текстовых файлов
            docs_from_dir = self.kb.load_and_process_documents(directory=os.path.dirname(self.kb_path))
            
            # Если не получилось, пробуем получить все документы из векторного хранилища
            if not docs_from_dir and self.kb.vector_store:
                # Получаем эмбеддинги всех документов из векторного хранилища
                # Это хак, но должен работать для большинства реализаций
                if hasattr(self.kb.vector_store, 'docstore') and hasattr(self.kb.vector_store.docstore, '_dict'):
                    return list(self.kb.vector_store.docstore._dict.values())
                return []
                
            return docs_from_dir
        except Exception as e:
            print(f"Ошибка при загрузке всех документов: {str(e)}")
            return []
    
    def _setup_rag_fusion_chain(self):
        """Setup the RAG Fusion retrieval chain"""
        # Setup RAG Fusion with multiple queries
        self.query_generation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Вы - ассистент по генерации поисковых запросов. Ваша цель - генерировать несколько поисковых запросов, связанных с вопросом пользователя, для поиска наиболее релевантной информации."),
                ("human", "Сгенерируйте 3 разных поисковых запроса, связанных со следующим вопросом пользователя. Верните ТОЛЬКО поисковые запросы, разделенные новыми строками, без объяснений или вступлений:\n\n{question}")
            ]
        )
        
        self.query_generation_chain = self.query_generation_prompt | self.llm | StrOutputParser()
        
        # Define the main prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
                ("system", "Вот релевантная информация из базы знаний:\n{context}")
            ]
        )
        
        # Build the chain
        self.chain = (
            {
                "question": RunnablePassthrough(),
                "context": self._rag_fusion_retriever,
                "chat_history": lambda _: self.memory.load_memory_variables({})["chat_history"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _rag_fusion_retriever(self, question: str) -> str:
        """
        Implement RAG Fusion retrieval strategy.
        
        Generate multiple queries from the user question, retrieve documents for each query,
        and merge the results with reciprocal rank fusion.
        """
        # Generate multiple search queries
        queries_text = self.query_generation_chain.invoke({"question": question})
        queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
        queries.append(question)  # Include the original question
        
        all_docs = []
        all_retrieved_ids = set()
        
        # Если включен граф знаний, получаем документы из него
        if self.use_knowledge_graph and hasattr(self, 'graph_retriever'):
            try:
                print("Поиск в графе знаний...")
                graph_results = self.graph_retriever.get_relevant_documents(question)
                
                # Добавляем результаты из графа знаний
                for doc in graph_results:
                    if doc.page_content not in all_retrieved_ids:
                        all_retrieved_ids.add(doc.page_content)
                        # Результаты из графа получают высокий приоритет (низкий score)
                        all_docs.append((doc, 0.1))
                        
                print(f"Найдено {len(graph_results)} результатов в графе знаний")
            except Exception as e:
                print(f"Ошибка при поиске в графе знаний: {str(e)}")
        
        # Если включен гибридный поиск, используем его для основного запроса
        if self.use_hybrid_search and hasattr(self, 'hybrid_retriever'):
            try:
                hybrid_results = self.hybrid_retriever.get_relevant_documents(question)
                
                # Добавляем результаты гибридного поиска
                for doc, score in hybrid_results:
                    if doc.page_content not in all_retrieved_ids:
                        all_retrieved_ids.add(doc.page_content)
                        all_docs.append((doc, score))
            except Exception as e:
                print(f"Ошибка при гибридном поиске: {str(e)}")
                # В случае ошибки продолжаем обычный поиск
        
        # Get documents for each query
        for query in queries:
            docs = self.kb.similarity_search_with_score(query, k=4)
            
            # Extract documents and scores
            for doc, score in docs:
                if doc.page_content not in all_retrieved_ids:
                    all_retrieved_ids.add(doc.page_content)
                    all_docs.append((doc, score))
        
        # Sort by relevance score and take top 5
        all_docs.sort(key=lambda x: x[1])
        top_docs = all_docs[:5]
        
        # Apply semantic chunking if enabled
        if self.use_semantic_chunking and top_docs:
            raw_docs = [doc for doc, _ in top_docs]
            try:
                # Разбиваем найденные документы на семантические части
                chunked_docs = self.semantic_chunker.process_documents(raw_docs)
                
                if chunked_docs:
                    # Отбираем только наиболее релевантные семантические части
                    doc_embeddings = self.semantic_chunker.model.encode([question])
                    chunk_embeddings = self.semantic_chunker.model.encode([doc.page_content for doc in chunked_docs])
                    
                    # Рассчитываем косинусную близость
                    similarities = [np.dot(doc_embeddings[0], chunk_emb) / 
                                   (np.linalg.norm(doc_embeddings[0]) * np.linalg.norm(chunk_emb)) 
                                   for chunk_emb in chunk_embeddings]
                    
                    # Сортируем документы по релевантности
                    chunk_scores = list(zip(chunked_docs, similarities))
                    chunk_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Берем топ-5 наиболее релевантных частей   
                    top_docs = [(doc, score) for doc, score in chunk_scores[:5]]
            except Exception as e:
                print(f"Ошибка при семантическом разбиении: {str(e)}")
                # В случае ошибки используем исходные документы
        
        # Store the source documents for later reference
        self.source_documents = [doc for doc, _ in top_docs]
        
        # Format the context from documents
        context_texts = []
        for i, (doc, _) in enumerate(top_docs, 1):
            source = doc.metadata.get("source", "Неизвестный источник")
            if source == "knowledge_graph":
                # Особый формат для узлов графа знаний
                context_texts.append(f"Информация из графа знаний ({doc.metadata.get('node_type', 'Узел')}):\n{doc.page_content}\n")
            else:
                # Стандартный формат для обычных документов
                chunk_info = f" (часть {doc.metadata.get('chunk_id', 0) + 1})" if doc.metadata.get("semantic_chunk") else ""
                context_texts.append(f"Документ {i} (Источник: {source}{chunk_info}):\n{doc.page_content}\n")
        
        return "\n".join(context_texts)
    
    def check_for_human_handoff(self, response: str) -> bool:
        """
        Check if the response indicates a need for human handoff.
        
        Returns:
            bool: True if human handoff is needed, False otherwise
        """
        handoff_phrases = [
            "передать оператору", 
            "человеком-оператором",
            "оператором поддержки",
            "недостаточно информации",
            "не могу ответить на основе",
            "человеком из поддержки",
            "связаться с оператором",
            "сотрудник поддержки",
            "перевести на человека",
            # English phrases (for fallback)
            "transfer to a human", 
            "human agent", 
            "speak with a representative",
            "don't have enough information",
            "can't answer that based on",
            "human support",
            "transfer to a representative"
        ]
        
        return any(phrase in response.lower() for phrase in handoff_phrases)
    
    def transfer_to_human(self) -> None:
        """Handle the process of transferring to a human support agent"""
        self.human_support_requested = True
        
        # In a real system, this would trigger an alert to a human support team
        # For this example, we'll just return a message
        return "Я перевожу вас на оператора поддержки. Пожалуйста, подождите."
    
    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and return the appropriate response.
        
        Args:
            user_message: The user's message
            
        Returns:
            Dict containing the response and metadata
        """
        # Reset source documents for this new query
        self.source_documents = []
        
        if self.human_support_requested:
            return {
                "response": "Ваш запрос поставлен в очередь для обработки оператором. Специалист поддержки свяжется с вами в ближайшее время.",
                "human_handoff": True,
                "source_documents": []
            }
        
        # Add user message to memory
        self.memory.chat_memory.add_user_message(user_message)
        
        # Get response from the chain
        response = self.chain.invoke(user_message)
        
        # Add the agent's response to memory
        self.memory.chat_memory.add_ai_message(response)
        
        # Check if we need to hand off to human support
        needs_human = self.check_for_human_handoff(response)
        if needs_human:
            handoff_message = self.transfer_to_human()
            return {
                "response": f"{response}\n\n{handoff_message}",
                "human_handoff": True,
                "source_documents": []
            }
        
        # Prepare source info for return
        sources = []
        for doc in self.source_documents:
            # Получаем метаданные документа
            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
            source_path = metadata.get("source", "Неизвестно")
            
            # Формируем полную информацию об источнике
            source_info = {
                "content": doc.page_content,
                "source": source_path,
                "file_name": os.path.basename(source_path) if source_path != "Неизвестно" else "Неизвестно",
                "chunk_id": metadata.get("chunk_id", 0) if metadata.get("semantic_chunk", False) else None,
                "page": metadata.get("page", None),
                "is_semantic_chunk": metadata.get("semantic_chunk", False)
            }
            
            # Удаляем None значения для более чистого JSON
            source_info = {k: v for k, v in source_info.items() if v is not None}
            
            sources.append(source_info)
        
        return {
            "response": response,
            "human_handoff": False,
            "source_documents": sources
        }
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.memory.clear()
        self.human_support_requested = False

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = SupportAgent(kb_path="./vector_store")
    
    print("ИИ-агент поддержки (введите 'exit' для выхода, 'reset' для начала нового разговора)")
    print("-" * 50)
    
    while True:
        user_input = input("Вы: ")
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'reset':
            agent.reset_conversation()
            print("Разговор сброшен.")
            continue
        
        result = agent.process_message(user_input)
        print("\nАгент:", result["response"])
        
        if result["human_handoff"]:
            print("\nВаш разговор переведен на оператора поддержки.")
            agent.reset_conversation()
        
        print("-" * 50) 
