from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Callable, Any

from lite_agents.llm.lite import LiteLLM
from lite_agents.core.message import ChatMessage, ChatRole
from lite_agents.core.chunk import DocumentChunk
from lite_agents.db.db import VectorDB
from lite_agents.prompts.ingestion import get_document_summary_prompt, get_chunk_context_prompt
from lite_agents.logger import setup_logger
from lite_agents.readers import get_reader_for_file
import time

logger = setup_logger()


class LiteIngestion:
    """Handles document ingestion with optional contextual retrieval enhancement.
    
    This class can optionally implement the Contextual Retrieval technique from Anthropic,
    which generates context for each chunk to improve retrieval accuracy.
    
    Args:
        llm (LiteLLM): the LiteLLM instance for context generation
        vector_db (VectorDB): the vector database instance
        embedding_function (Callable[[list[str]], list[list[float]]]): function to create embeddings
        chunk_size (int, optional): target size of each chunk in characters. Defaults to 800.
        chunk_overlap (int, optional): overlap between consecutive chunks. Defaults to 200.
        add_context (bool, optional): whether to add contextual retrieval. Defaults to True.
    """
    def __init__(
        self,
        llm: LiteLLM,
        vector_db: VectorDB,
        embedding_function: Callable[[list[str]], list[list[float]]],
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        add_context: bool = True,
    ) -> None:
        """Initialize the LiteIngestion class."""
        self.llm = llm
        self.vector_db = vector_db
        self.embedding_function = embedding_function
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_context = add_context
        self._document_summaries: dict[str, str] = {}
    
    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks with overlap, respecting paragraphs.
        
        Args:
            text (str): the text to chunk
            
        Returns:
            list[str]: list of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If single paragraph exceeds chunk_size, split by sentences
            if len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= self.chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            # Start new chunk with overlap
                            overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                            current_chunk = overlap_text + " " + sentence + " "
                        else:
                            current_chunk = sentence + " "
            else:
                # If adding paragraph exceeds chunk_size, save current chunk
                if len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text
                
                current_chunk += paragraph + "\n\n"
        
        # Add last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_document_summary(self, document_title: str, content: str) -> str:
        """Generate a summary of the document to help with contextualization.
        
        Args:
            document_title (str): the document title
            content (str): the document content (or preview)
            
        Returns:
            str: the generated summary
        """
        t_start = time.time() 
        logger.warning("⚠️ Currently all the content is used for summary generation. Consider using a preview for large documents.")
        prompt = get_document_summary_prompt(
            document_title=document_title, 
            content=content
        )
        messages = [ChatMessage(role=ChatRole.USER, content=prompt)]
        response = self.llm.generate(messages=messages, tools=None)
        t_end = time.time()
        logger.info(f"⏱️ Document summary generated in {t_end - t_start:.2f} seconds.")
        return response.content.strip()
    
    def generate_chunk_context(
        self,
        chunk_content: str,
        document_title: str,
        section_header: str,
        document_summary: str = ""
    ) -> str:
        """Generate context for a chunk using the LLM.
        
        Args:
            chunk_content (str): the chunk content
            document_title (str): the document title
            section_header (str): the section header
            document_summary (str, optional): the document summary. Defaults to "".
            
        Returns:
            str: the generated context (2-3 sentences)
        """
        prompt = get_chunk_context_prompt(
            document_title=document_title,
            section_header=section_header,
            chunk_content=chunk_content,
            document_summary=document_summary
        )
        messages = [ChatMessage(role=ChatRole.USER, content=prompt)]
        response = self.llm.generate(messages=messages, tools=None)
        return response.content.strip()

    def process_document(self, file_path: Path) -> list[DocumentChunk]:
        """Process a single document and create chunks with optional context.
        
        Args:
            file_path (Path): path to the document
            
        Returns:
            list[DocumentChunk]: list of processed chunks
        """
        logger.info(f"📄 Processing document: {file_path.name}")
        
        # Get appropriate reader
        reader = get_reader_for_file(file_path)
        
        # Read document
        doc_name, doc_title, content = reader.read(file_path)
        
        # Generate document summary
        document_summary = None
        if self.add_context:
            logger.info(f"📝 Generating summary for: {doc_title}")
            document_summary = self.generate_document_summary(doc_title, content)
            self._document_summaries[doc_name] = document_summary
        
        # Split by sections
        sections = reader.split(content) # sections is list[(section_header, section_content)]
        
        # Create chunks
        all_chunks = []
        for section_header, section_content in sections:
            chunks = self.chunk_text(section_content)
            for chunk in chunks:
                all_chunks.append({
                    'content': chunk,
                    'document_name': doc_name,
                    'document_title': doc_title,
                    'section_header': section_header,
                    'document_summary': document_summary
                })
        
        logger.info(f"✂️ Created {len(all_chunks)} chunks from {file_path.name}")
        
        # Generate context for each chunk
        document_chunks = []
        t_start = time.time()
        for idx, chunk_info in enumerate(all_chunks):
            if self.add_context:
                context_t_start = time.time()
                context = self.generate_chunk_context(
                    chunk_content=chunk_info['content'],
                    document_title=doc_title,
                    section_header=chunk_info['section_header'],
                    document_summary=document_summary
                )
                context_t_end = time.time()
                logger.info(f"⏱️ Generated context for chunk {idx + 1}/{len(all_chunks)} in {context_t_end - context_t_start:.2f} seconds.")
            else:
                context = None
            
            doc_chunk = DocumentChunk(
                content=chunk_info['content'],
                context=context,
                document_name=doc_name,
                document_title=doc_title,
                chunk_index=idx,
                total_chunks=len(all_chunks),
                section_header=chunk_info['section_header'],
                document_summary=document_summary
            )
            document_chunks.append(doc_chunk)
        t_end = time.time()
        
        # Log time taken
        if self.add_context:
            logger.info(f"⏱️ Generated context for {len(document_chunks)} chunks in {t_end - t_start:.2f} seconds.")
        else:
            logger.info(f"⏱️ Skipped context generation for {len(document_chunks)} chunks. Took {t_end - t_start:.2f} seconds.")
        
        logger.info(f"✅ Completed processing {file_path.name}")
        return document_chunks

    def process_directory(
        self,
        directory: Path,
        file_pattern: str = "*.md"
    ) -> list[DocumentChunk]:
        """Process all documents in a directory.
        
        Args:
            directory (Path): directory containing documents
            file_pattern (str, optional): glob pattern for files. Defaults to "*.md".
            
        Returns:
            list[DocumentChunk]: list of all processed chunks
        """
        files = sorted(directory.glob(file_pattern))
        logger.info(f"📂 Found {len(files)} files in {directory}")
        
        all_chunks = []
        all_t_start = time.time()
        for file_path in files:
            t_start = time.time()
            chunks = self.process_document(file_path)
            t_end = time.time()
            logger.info(f"⏱️ Processed document '{file_path.name}' in {t_end - t_start:.2f} seconds.")
            all_chunks.extend(chunks)
        all_t_end = time.time()
        logger.info(f"⏱️ Processed {len(files)} documents with total {len(all_chunks)} chunks in {all_t_end - all_t_start:.2f} seconds.")
        
            
        return all_chunks

    def ingest_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Ingest chunks into the vector database.
        
        Args:
            chunks (list[DocumentChunk]): list of chunks to ingest
        """
        logger.info(f"🚀 Starting ingestion of {len(chunks)} chunks")
        
        if not chunks:
            return

        # Prepare data
        documents = [chunk.get_contextualized_content() for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]
        ids = [f"{chunk.document_name}_{chunk.chunk_index}" for chunk in chunks]
        
        # Create embeddings
        logger.info("🧠 Creating embeddings...")
        embeddings = self.embedding_function(documents)
        
        # Add to vector database
        logger.info("💾 Adding to vector database...")
        self.vector_db.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def ingest_directory(
        self,
        directory: Path,
        file_pattern: str = "*.md"
    ) -> int:
        """Process and ingest all documents from a directory.
        
        Args:
            directory (Path): directory containing documents
            file_pattern (str, optional): glob pattern for files. Defaults to "*.md".
            
        Returns:
            int: number of chunks ingested
        """
        chunks = self.process_directory(directory, file_pattern)
        self.ingest_chunks(chunks)
        return len(chunks)
    
    def get_statistics(self, chunks: list[DocumentChunk]) -> dict[str, Any]:
        """Get statistics about the processed chunks.
        
        Args:
            chunks (list[DocumentChunk]): list of chunks
            
        Returns:
            dict[str, Any]: statistics dictionary
        """
        docs = {}
        for chunk in chunks:
            if chunk.document_name not in docs:
                docs[chunk.document_name] = {
                    'title': chunk.document_title,
                    'chunks': 0,
                    'total_chars': 0,
                    'context_chars': 0
                }
            docs[chunk.document_name]['chunks'] += 1
            docs[chunk.document_name]['total_chars'] += len(chunk.content)
            docs[chunk.document_name]['context_chars'] += len(chunk.context)
        
        return {
            'total_documents': len(docs),
            'total_chunks': len(chunks),
            'avg_chunks_per_document': len(chunks) / len(docs) if docs else 0,
            'documents': docs
        }
    
    def save_chunks_to_json(
        self,
        chunks: list[DocumentChunk],
        output_path: Path | str,
        pretty: bool = True
    ) -> None:
        """Save chunks to a JSON file for easy visualization.
        
        Args:
            chunks (list[DocumentChunk]): list of chunks to save
            output_path (Path | str): path to the output JSON file
            pretty (bool, optional): whether to use pretty formatting. Defaults to True.
        """
        output_path = Path(output_path)
        
        chunks_data = [chunk.to_dict() for chunk in chunks]
        
        logger.info(f"💾 Saving {len(chunks)} chunks to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(chunks_data, f, ensure_ascii=False)
        
        logger.info(f"✅ Chunks saved successfully to {output_path}")
    
    @staticmethod
    def load_chunks_from_json(input_path: Path | str) -> list[DocumentChunk]:
        """Load chunks from a JSON file.
        
        Args:
            input_path (Path | str): path to the input JSON file
            
        Returns:
            list[DocumentChunk]: list of loaded chunks
        """
        input_path = Path(input_path)
        
        logger.info(f"📂 Loading chunks from {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks = [DocumentChunk.from_dict(chunk_dict) for chunk_dict in chunks_data]
        
        logger.info(f"✅ Loaded {len(chunks)} chunks from {input_path}")
        return chunks

