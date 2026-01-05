from abc import ABC, abstractmethod
from typing import Any

class VectorDB(ABC):
    """Abstract base class for Vector Databases."""

    @abstractmethod
    def add_documents(
        self, 
        documents: list[str], 
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None, 
        ids: list[str] | None = None
    ) -> None:
        """Add documents to the vector database.

        Args:
            documents (list[str]): list of text documents to add.
            embeddings (list[list[float]]): list of embeddings corresponding to documents.
            metadatas (list[dict] | None): list of metadata dictionaries corresponding to documents.
            ids (list[str] | None): list of IDs for the documents.
        """
        pass

    @abstractmethod
    def query(
        self, 
        query_embeddings: list[float], 
        n_results: int = 5,
        threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """Query the vector database for relevant documents.

        Args:
            query_embeddings (list[float]): the query embedding.
            n_results (int): number of results to return.
            threshold (float | None): similarity threshold for filtering results.

        Returns:
            list[dict[str, Any]]: list of results, each containing 'content', 'metadata', and 'distance'.
        """
        pass
