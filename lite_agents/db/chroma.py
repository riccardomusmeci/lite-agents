from typing import Any
import uuid
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from lite_agents.db.db import VectorDB

DEFAULT_METADATA = {"hnsw:space": "cosine"}

class ChromaDB(VectorDB):
    """ChromaDB implementation of VectorDB.
    
    Args:
        collection_name (str): name of the collection.
        path (str): path to store the database (if persistent).
        persistent (bool): whether to use persistent storage or in-memory.
        metadata (dict | None): metadata for the collection. If None, defaults are used (hnsw:space = cosine). Defaults to None.
    """

    def __init__(
        self, 
        collection_name: str = "knowledge_base", 
        path: str = "./chroma_db", 
        persistent: bool = True, 
        metadata: dict | None = None
    ) -> None:
        """Initialize ChromaDB."""
        if chromadb is None:
            raise ImportError("chromadb is not installed. Please install it with `pip install chromadb`.")

        if persistent:
            self.client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        else:
            self.client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
            
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata or DEFAULT_METADATA
        )

    def add_documents(
        self, 
        documents: list[str],
        embeddings: list[list[float]], 
        metadatas: list[dict] | None = None, 
        ids: list[str] | None = None
    ) -> None:
        """Add documents to the ChromaDB collection.

        Args:
            documents (list[str]): list of text documents to add.
            embeddings (list[list[float]]): list of embeddings corresponding to documents.
            metadatas (list[dict] | None, optional): list of metadata dictionaries corresponding to documents. Defaults to None.
            ids (list[str] | None, optional): list of IDs for the documents. Defaults to None.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    
    def query(self, query_embeddings: list[float], n_results: int = 5, threshold: float | None = None) -> list[dict[str, Any]]:
        """Query the ChromaDB collection for relevant documents.

        Args:
            query_embeddings (list[float]): the embedding vector of the query.
            n_results (int, optional): number of results to return. Defaults to 5.
            threshold (float, optional): similarity threshold for filtering results. Defaults to None.

        Returns:
            list[dict[str, Any]]: a list of dictionaries containing 'content', 'metadata', and 'distance' for each result.
        """
        
        threshold = threshold if threshold is not None else 0.0
        
        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        if not results['documents'] or not results['documents'][0]:
            return []

        formatted_results = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            # Convert distance to similarity (assuming cosine distance)
            similarity = max(0, 1 - dist)
            
            if similarity >= threshold:
                formatted_results.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity": similarity # changed from distance to similarity
                })
                
        return formatted_results


