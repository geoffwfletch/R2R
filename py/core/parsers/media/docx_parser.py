# type: ignore
import asyncio
from io import BytesIO
from typing import AsyncGenerator

from docx import Document

from core.base.parsers.base_parser import AsyncParser
from core.base.providers import (
    CompletionProvider,
    DatabaseProvider,
    IngestionConfig,
)
from core.parsers.media.pdf_parser import fix_heading_hierarchy_from_result


class DOCXParser(AsyncParser[str | bytes]):
    """A parser for DOCX data."""

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
    ):
        self.database_provider = database_provider
        self.llm_provider = llm_provider
        self.config = config
        self.Document = Document

    async def ingest(
        self, data: str | bytes, *args, **kwargs
    ) -> AsyncGenerator[str, None]:  # type: ignore
        """Ingest DOCX data and yield text from each paragraph."""
        if isinstance(data, str):
            raise ValueError("DOCX data must be in bytes format.")

        doc = self.Document(BytesIO(data))
        for paragraph in doc.paragraphs:
            yield paragraph.text


class DoclingHybridDOCXParser(AsyncParser[str | bytes]):
    """DOCX parser using Docling extraction + HybridChunker (pre-chunked)."""

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
    ):
        self.config = config
        self.database_provider = database_provider
        self.llm_provider = llm_provider

    async def ingest(
        self, data: str | bytes, **kwargs
    ) -> AsyncGenerator[dict, None]:
        import os
        import tempfile

        from docling.document_converter import DocumentConverter
        from docling_core.transforms.chunker import HybridChunker

        if isinstance(data, str):
            temp_path = data
        else:
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".docx"
            )
            tmp.write(data)
            tmp.flush()
            tmp.close()
            temp_path = tmp.name

        try:
            converter = DocumentConverter()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, converter.convert, temp_path)
            doc = result.document

            yield {
                "full_markdown": doc.export_to_markdown(),
                "type": "markdown_preview",
            }

            chunker = HybridChunker()
            for i, chunk in enumerate(chunker.chunk(doc)):
                yield {
                    "content": chunker.serialize(chunk),
                    "metadata": {"chunk_order": i},
                    "pre_chunked": True,
                }
        finally:
            if not isinstance(data, str):
                os.unlink(temp_path)


class DoclingMarkdownDOCXParser(AsyncParser[str | bytes]):
    """DOCX parser using Docling extraction → markdown (chunked by MarkdownChunker)."""

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
    ):
        self.config = config
        self.database_provider = database_provider
        self.llm_provider = llm_provider

    async def ingest(
        self, data: str | bytes, **kwargs
    ) -> AsyncGenerator[str | dict, None]:
        import os
        import tempfile

        from docling.document_converter import DocumentConverter

        if isinstance(data, str):
            temp_path = data
        else:
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".docx"
            )
            tmp.write(data)
            tmp.flush()
            tmp.close()
            temp_path = tmp.name

        try:
            converter = DocumentConverter()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, converter.convert, temp_path)
            md = result.document.export_to_markdown()
            md = fix_heading_hierarchy_from_result(md, result)

            yield {
                "full_markdown": md,
                "type": "markdown_preview",
            }

            yield md
        finally:
            if not isinstance(data, str):
                os.unlink(temp_path)
