import os
import time
from typing import List, Dict, Any, Optional, Tuple
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from langchain.schema import Document, AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
from knowledge_base import KnowledgeBase

class SupportAgent:
    def __init__(self, 
                 kb_path: str = "./vector_store", 
                 ollama_base_url: str = "http://localhost:11434",
                 model_name: str = "deepseek-r1:7b"):
        """
        Initialize the AI support agent with RAG Fusion.
        
        Args:
            kb_path: Path to the knowledge base vector store
            ollama_base_url: Base URL for Ollama API
            model_name: Name of the model to use
        """
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.1
        )
        
        # Initialize knowledge base
        self.kb = KnowledgeBase(ollama_base_url=ollama_base_url)
        
        # Load vector store if it exists
        if os.path.exists(kb_path):
            self.kb.load_vector_store(kb_path)
        else:
            raise ValueError(f"Knowledge base not found at {kb_path}. Please create it first.")
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup system prompt
        self.system_prompt = """Вы - полезный помощник службы поддержки. 
Отвечайте на вопросы пользователя на основе информации, предоставленной в извлеченных документах.
Если вы не знаете ответ или не можете найти его в извлеченных документах, честно скажите об этом и предложите перевести диалог на оператора-человека.
Всегда будьте вежливы, профессиональны и лаконичны.
Отвечайте пользователю ТОЛЬКО на русском языке, независимо от языка запроса.
"""
        
        # Set up the chain
        self._setup_rag_fusion_chain()
        
        # Tracking if human support is requested
        self.human_support_requested = False
    
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
        
        # Format the context from documents
        context_texts = []
        for i, (doc, _) in enumerate(top_docs, 1):
            source = doc.metadata.get("source", "Неизвестный источник")
            context_texts.append(f"Документ {i} (Источник: {source}):\n{doc.page_content}\n")
        
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
        
        return {
            "response": response,
            "human_handoff": False,
            "source_documents": []  # In a real implementation, you'd return the actual sources
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