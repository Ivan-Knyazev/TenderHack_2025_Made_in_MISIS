import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader, JSONLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

class KnowledgeBase:
    def __init__(self, ollama_base_url: str = "http://46.227.68.167:22065/"):
        """Initialize the knowledge base with Ollama embeddings."""
        self.embeddings = OllamaEmbeddings(
            model="qwen2.5:14b",
            base_url=ollama_base_url
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=2000
        )

    def load_and_process_documents(self, 
                                   directory: str = None, 
                                   files: List[str] = None, 
                                   json_files: Dict[str, str] = None,
                                   pdf_files: List[str] = None) -> List[Document]:
        """
        Load documents from various sources and process them.
        
        Args:
            directory: Directory containing text files to load
            files: List of specific file paths to load
            json_files: Dict mapping file paths to jq-like content extractors
            pdf_files: List of PDF file paths to load
            
        Returns:
            List of processed Document objects
        """
        documents = []
        
        # Load from directory (both TXT and PDF files)
        if directory and os.path.exists(directory):
            # Load txt files
            txt_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
            documents.extend(txt_loader.load())
            
            # Load pdf files
            pdf_loader = DirectoryLoader(
                directory, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents.extend(pdf_loader.load())
            
        # Load specific text files
        if files:
            for file_path in files:
                if os.path.exists(file_path) and file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
        
        # Load specific PDF files
        if pdf_files:
            for file_path in pdf_files:
                if os.path.exists(file_path) and file_path.endswith('.pdf'):
                    print(f"Загрузка PDF: {os.path.basename(file_path)}")
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    
        # Load JSON files with specific extractors
        if json_files:
            for file_path, jq_schema in json_files.items():
                if os.path.exists(file_path) and file_path.endswith('.json'):
                    loader = JSONLoader(file_path, jq_schema=jq_schema)
                    documents.extend(loader.load())
        
        # Split documents into chunks
        if documents:
            chunked_docs = self.text_splitter.split_documents(documents)

            # Добавляем уникальные doc_id каждому чанку
            for i, doc in enumerate(chunked_docs):
                doc.metadata["doc_id"] = f"doc_{i}"
                doc.metadata["original_index"] = i  # сохраняем исходный порядок


            print(f"Загружено {len(documents)} документов, разбито на {len(chunked_docs)} чанков")
            return chunked_docs

        
        return []

    def create_vector_store(self, documents: List[Document], persist_directory: Optional[str] = None):
        """
        Create a FAISS vector store from the documents.
        
        Args:
            documents: List of Document objects
            persist_directory: Optional directory to persist the vector store
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
            
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.vector_store.save_local(persist_directory)
            print(f"Vector store saved to {persist_directory}")

    def load_vector_store(self, persist_directory: str):
        """
        Load a FAISS vector store from a directory.
        
        Args:
            persist_directory: Directory containing the persisted vector store
        """
        if not os.path.exists(persist_directory):
            raise ValueError(f"Directory {persist_directory} does not exist")
            
        self.vector_store = FAISS.load_local(persist_directory, self.embeddings, allow_dangerous_deserialization=True)
        print(f"Vector store loaded from {persist_directory}")

    def add_documents(self, 
                     documents: List[Document], 
                     persist_directory: Optional[str] = None) -> bool:
        """
        Add new documents to an existing vector store.
        
        Args:
            documents: List of Document objects to add
            persist_directory: Optional directory to persist the updated vector store
            
        Returns:
            bool: True if documents were added successfully
        """
        if not documents:
            print("Нет документов для добавления в базу знаний")
            return False
            
        if not self.vector_store:
            print("Векторное хранилище не инициализировано. Создаем новое...")
            self.create_vector_store(documents, persist_directory)
            return True
            
        try:
            print(f"Добавление {len(documents)} документов в существующее векторное хранилище...")
            self.vector_store.add_documents(documents)
            
            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)
                self.vector_store.save_local(persist_directory)
                print(f"Обновленное векторное хранилище сохранено в {persist_directory}")
            
            return True
        except Exception as e:
            print(f"Ошибка при добавлении документов: {str(e)}")
            return False

    def update_from_files(self, 
                         files: Union[List[str], str],
                         persist_directory: Optional[str] = None,
                         is_pdf: bool = False) -> bool:
        """
        Update the vector store from files directly.
        
        Args:
            files: List of file paths or a single file path to process
            persist_directory: Optional directory to persist the updated vector store
            is_pdf: Whether the files are PDFs
            
        Returns:
            bool: True if vector store was updated successfully
        """
        if isinstance(files, str):
            files = [files]
        
        if is_pdf:
            documents = self.load_and_process_documents(pdf_files=files)
        else:
            documents = self.load_and_process_documents(files=files)
        
        if not documents:
            return False
            
        if not self.vector_store:
            if persist_directory and os.path.exists(persist_directory):
                self.load_vector_store(persist_directory)
                return self.add_documents(documents, persist_directory)
            else:
                return self.create_vector_store(documents, persist_directory)
        else:
            return self.add_documents(documents, persist_directory)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform a similarity search on the vector store.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Create or load a vector store first.")
            
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform a similarity search with scores.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Create or load a vector store first.")
            
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def similarity_search_with_context(self, query: str, k: int = 4, context_window: int = 3) -> List[Document]:
            """
            Perform similarity search and return matching chunks with surrounding context.
            
            Args:
                query: The search query
                k: Number of top results to return
                context_window: Number of surrounding chunks to include before and after each match
            
            Returns:
                List of Document objects with surrounding context
            """
            if not self.vector_store:
                raise ValueError("Vector store not initialized.")
                
            results = self.vector_store.similarity_search_with_score(query, k=k)
            all_documents = self.vector_store.docstore._dict  # access to all chunks

            doc_keys = list(all_documents.keys())

            matched_indices = []
            for doc, _ in results:
                doc_id = doc.metadata.get("doc_id")
                if doc_id and doc_id in doc_keys:
                    matched_indices.append(doc_keys.index(doc_id))

            final_indices = set()
            for idx in matched_indices:
                for offset in range(-context_window, context_window + 1):
                    neighbor_idx = idx + offset
                    if 0 <= neighbor_idx < len(doc_keys):
                        final_indices.add(neighbor_idx)

            return [all_documents[doc_keys[i]] for i in sorted(final_indices)]
    
    def similarity_search_with_context(self, query: str, k: int = 4, context_window: int = 10) -> List[Document]:
        """
        Perform similarity search and return matching chunks with surrounding context.

        Args:
            query: The search query
            k: Number of top results to return
            context_window: Number of surrounding chunks to include before and after each match

        Returns:
            List of Document objects with surrounding context, in proper order
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")

        # Основной поиск
        results = self.vector_store.similarity_search_with_score(query, k=k)

        # Все чанки, отсортированные по original_index
        all_documents = list(self.vector_store.docstore._dict.values())
        all_documents.sort(key=lambda d: d.metadata.get("original_index", 0))

        # Получаем индексы совпавших чанков
        matched_indices = []
        for doc, _ in results:
            idx = doc.metadata.get("original_index")
            if idx is not None:
                matched_indices.append(idx)

        # Собираем индексы с учетом контекста
        final_indices = set()
        for idx in matched_indices:
            for offset in range(-context_window, context_window + 1):
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(all_documents):
                    final_indices.add(neighbor_idx)

        # Возвращаем чанки в правильном порядке
        return [all_documents[i] for i in sorted(final_indices)]



# Example usage
if __name__ == "__main__":
    kb = KnowledgeBase()
    
    # Example: Convert and load documents
    documents = kb.load_and_process_documents(
        directory="./knowledge",
        files=["./faq.txt"],
        json_files={"./data.json": ".content"}
    )
    
    # Create and save vector store
    kb.create_vector_store(documents, persist_directory="./vector_store")
    
    # Alternative: Load existing vector store
    # kb.load_vector_store("./vector_store")
    
    # Search the knowledge base
    results = kb.similarity_search("How do I reset my password?")
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("-" * 50) 