from typing import List, Optional, Dict, Any
import uuid
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from lite_agents.db.db import VectorDB

class ChromaDB(VectorDB):
    """ChromaDB implementation of VectorDB.
    
    Args:
        collection_name (str): Name of the collection.
        path (str): Path to store the database (if persistent).
        persistent (bool): Whether to use persistent storage or in-memory.
    """

    def __init__(self, collection_name: str = "knowledge_base", path: str = "./chroma_db", persistent: bool = True):
        """Initialize ChromaDB."""
        if chromadb is None:
            raise ImportError("chromadb is not installed. Please install it with `pip install chromadb`.")

        if persistent:
            self.client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        else:
            self.client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
            
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(
        self, 
        documents: List[str],
        embeddings: List[List[float]], 
        metadatas: Optional[List[dict]] = None, 
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to the ChromaDB collection.

        Args:
            documents (List[str]): List of text documents to add.
            embeddings (List[List[float]]): List of embeddings corresponding to the documents.
            metadatas (Optional[List[dict]], optional): List of metadata dictionaries. Defaults to None.
            ids (Optional[List[str]], optional): List of document IDs. If None, UUIDs are generated. Defaults to None.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query(
        self, 
        query_embeddings: List[float], 
        n_results: int = 5,
        max_distance: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Query the ChromaDB collection for relevant documents.

        Args:
            query_embeddings (List[float]): The embedding vector of the query.
            n_results (int, optional): Number of results to return. Defaults to 5.
            max_distance (Optional[float], optional): Maximum distance for results. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing 'content', 'metadata', and 'distance' for each result.
        """
        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        
        # results['documents'] is a list of lists (one list per query)
        if not results['documents']:
            return []
            
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        
        formatted_results = []
        for doc, meta, dist in zip(docs, metas, dists):
            if max_distance is not None and dist > max_distance:
                continue
            formatted_results.append({
                "content": doc,
                "metadata": meta,
                "distance": dist
            })
            
        return formatted_results

