from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

class BaseRetrieval:
    def __init__(self, config: dict):
        self.vectorstore_search_k = config.get("search_k", 5)
        self.context_max_length = config.get("context_max_length", 8000)
        self.vector_db_path = config.get("vector_db_path")
        self.collection_name = config.get("collection_name")
        self.embedding_model_path = config.get("embedding_model_path")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={'device': 'cuda'}
        )
        self.chroma_vectorstore = self.load_chroma_vectorstore()

    def load_chroma_vectorstore(self):
        absolute_path = os.path.abspath(self.vector_db_path)
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"벡터 스토어 경로가 존재하지 않습니다: {absolute_path}")
        return Chroma(
            persist_directory=absolute_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

    def retrieve_context(self, query: str, filter: Optional[dict] = None) -> list:
        search_kwargs = {"query": query, "k": self.vectorstore_search_k}
        if filter:
            search_kwargs["filter"] = filter
        results = self.chroma_vectorstore.similarity_search(**search_kwargs)
        return results