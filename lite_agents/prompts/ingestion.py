"""Prompt templates for document ingestion with contextual retrieval."""

DOCUMENT_SUMMARY_PROMPT = """Document: "{document_title}"

{content}

---

Generate a concise summary (2-3 sentences, max 80 words) that describes:
- The main objective of this document
- The key topics covered
- Who it applies to

Write in the same language as the document. Respond ONLY with the summary."""


CHUNK_CONTEXT_PROMPT = """You are an assistant helping to improve document retrieval.

Document: "{document_title}"
Section: "{section_header}"
{document_summary_line}

Here is a chunk from the document:

{chunk_content}

---

Generate a brief context (2-3 sentences, max 100 words) that:
1. Specifies which document and section this chunk belongs to
2. Describes the main topic covered in the chunk
3. Explains how it relates to the document's general theme

The context will be used to improve semantic search.
Write clearly and concisely in the same language as the document.
Respond ONLY with the context, no introductions or explanations."""


def get_document_summary_prompt(document_title: str, content: str) -> str:
    """Get the prompt for document summary generation.
    
    Args:
        document_title (str): the document title
        content (str): the content of the document
        
    Returns:
        str: the formatted prompt
    """
    return DOCUMENT_SUMMARY_PROMPT.format(
        document_title=document_title,
        content=content
    )


def get_chunk_context_prompt(
    document_title: str,
    section_header: str,
    chunk_content: str,
    document_summary: str = ""
) -> str:
    """Get the prompt for chunk context generation.
    
    Args:
        document_title (str): the document title
        section_header (str): the section header
        chunk_content (str): the chunk content
        document_summary (str, optional): the document summary. Defaults to "".
        
    Returns:
        str: the formatted prompt
    """
    summary_line = f'Document summary: {document_summary}' if document_summary else ''
    
    return CHUNK_CONTEXT_PROMPT.format(
        document_title=document_title,
        section_header=section_header,
        document_summary_line=summary_line,
        chunk_content=chunk_content
    )
