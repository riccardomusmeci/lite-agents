from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict

class VectorDB(ABC):
    """Abstract base class for Vector Databases."""

    @abstractmethod
    def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None, 
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to the vector database.

        Args:
            documents (List[str]): List of text documents to add.
            embeddings (List[List[float]]): List of embeddings corresponding to documents.
            metadatas (Optional[List[dict]]): List of metadata dictionaries corresponding to documents.
            ids (Optional[List[str]]): List of IDs for the documents.
        """
        pass

    @abstractmethod
    def query(
        self, 
        query_embeddings: List[float], 
        n_results: int = 5,
        max_distance: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Query the vector database for relevant documents.

        Args:
            query_embeddings (List[float]): the query embedding.
            n_results (int): number of results to return.
            max_distance (Optional[float]): maximum distance for results.

        Returns:
            List[Dict[str, Any]]: list of results, each containing 'content', 'metadata', and 'distance'.
        """
        pass
