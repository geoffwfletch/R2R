# Docling Extraction + Azure Blob Storage Changes

## Overview

Three parser options for PDF/DOCX, Azure Blob file storage, markdown preview flow.

---

## Part A: Parser Options

### Selection via `parser_overrides` in ingestion config

| Override value | PDF parser | DOCX parser | Chunking |
|---|---|---|---|
| *(default/none)* | `DoclingHybridPDFParser` | `DoclingHybridDOCXParser` | HybridChunker (pre-chunked, skip chunking step) |
| `"docling_markdown"` | `DoclingMarkdownPDFParser` | `DoclingMarkdownDOCXParser` | MarkdownChunker (header split + recursive) |
| `"text"` | `BasicPDFParser` | `DOCXParser` | RecursiveCharacterTextSplitter (original behavior) |
| `"ocr"` | `OCRPDFParser` | n/a | one-page-per-chunk or text splitter |
| `"zerox"` | `VLMPDFParser` | n/a | one-page-per-chunk or text splitter |

### Files changed

**`py/core/parsers/media/pdf_parser.py`** — Added `DoclingHybridPDFParser`, `DoclingMarkdownPDFParser`

**`py/core/parsers/media/docx_parser.py`** — Added `DoclingHybridDOCXParser`, `DoclingMarkdownDOCXParser`

All 4 parsers:
- Write bytes to `NamedTemporaryFile(delete=False)`, close it, then pass path to Docling
- Yield `{"type": "markdown_preview", "full_markdown": ...}` for preview storage
- Hybrid variants yield `{"pre_chunked": True, "content": ..., "metadata": {"chunk_order": i}}`
- Markdown variants yield raw markdown string (chunked by MarkdownChunker)
- Cleanup temp file in `finally` block

**`py/core/parsers/__init__.py`** + **`py/core/parsers/media/__init__.py`** — Exports for all 4 new parsers

**`py/shared/utils/splitter/text.py`** — Added `MarkdownChunker` class at end of file
- Two-stage: `MarkdownHeaderTextSplitter` (H1-H3) then `RecursiveCharacterTextSplitter`
- `create_documents(texts)` returns `list[SplitterDocument]` with header metadata

**`py/core/base/providers/ingestion.py`** — Added `MARKDOWN = "markdown"` to `ChunkingStrategy` enum

**`py/core/providers/ingestion/r2r/base.py`** — Main changes:
- `DEFAULT_PARSERS`: PDF → `DoclingHybridPDFParser`, DOCX → `DoclingHybridDOCXParser`
- `EXTRA_PARSERS`: Added `"docling_markdown"` and `"text"` entries for PDF and DOCX
- `_initialize_parsers()`: try/except TypeError for parsers that don't accept `ocr_provider`
- `_build_text_splitter()`: Added `ChunkingStrategy.MARKDOWN` → `MarkdownChunker`
- `chunk()`: Added `preserve_metadata` param to yield raw `SplitterDocument` objects
- `parse()`: Generalized override lookup (`f"{override_name}_{doc_type_val}"`), handles `pre_chunked` items (skip chunking), `markdown_preview` items (stash for caller), auto-selects MARKDOWN strategy for `docling_markdown` override

**`py/pyproject.toml`** — Added `"docling >=2.70.0"`, `"azure-storage-blob >=12.0.0"` to core extras

**`py/r2r/r2r.toml`** — Updated `extra_parsers` with `docling_markdown` and `text` for pdf/docx

---

## Part B: Azure Blob Storage

**`py/core/base/providers/file.py`**
- `FileConfig`: Added `azure_account_name`, `azure_account_key`, `azure_container_name` fields; added `"azure_blob"` to `supported_providers`; added validation
- `FileProvider`: Added non-abstract `store_markdown_preview()` and `retrieve_markdown_preview()` with default no-op/None

**`py/core/providers/file/azure_blob.py`** — NEW file, `AzureBlobFileProvider`
- Blob paths: `ragengine/documents/{document_id}/{file_name}`
- Preview path: `ragengine/documents/{document_id}/preview.md`
- Implements all `FileProvider` abstract methods + markdown preview methods
- `delete_file` cleans up all blobs under prefix (including preview)
- Config via env vars: `AZURE_BLOB_ACCOUNT_NAME`, `AZURE_BLOB_ACCOUNT_KEY`, `AZURE_BLOB_CONTAINER_NAME`

**`py/core/providers/file/__init__.py`** — Added `AzureBlobFileProvider` export

**`py/core/providers/__init__.py`** — Added `AzureBlobFileProvider` to imports and `__all__`

**`py/core/main/abstractions.py`** — Added `AzureBlobFileProvider` to `file:` type union in `R2RProviders`

**`py/core/main/assembly/factory.py`** — Added `elif config.provider == "azure_blob"` case in `create_file_provider()`

---

## Part C: Markdown Preview Flow

**`py/core/main/services/ingestion_service.py`**
- After parse loop in `parse_file()`, pops `_markdown_preview` from `ingestion_config_override` and calls `file_provider.store_markdown_preview()`

**`py/core/providers/file/postgres.py`**
- `initialize()`: Creates `markdown_previews` table + update trigger
- `store_markdown_preview()`: Creates large object, upserts row. **Unlinks old large object on re-ingestion** to prevent storage leak
- `retrieve_markdown_preview()`: Reads large object, decodes UTF-8
- `delete_file()`: Also deletes associated preview large object + row

**`py/core/providers/file/s3.py`**
- `store_markdown_preview()`: `put_object` at `documents/{id}/preview.md`
- `retrieve_markdown_preview()`: `get_object`, returns decoded body
- `delete_file()`: Also deletes preview key

**`py/core/main/api/v3/documents_router.py`** — Added `GET /documents/{id}/preview` endpoint
- Auth checks (owner or collection access)
- Returns `{"document_id": id, "markdown": markdown}`

---

## Data Flow

```
PDF/DOCX (default — docling_hybrid):
  Upload → store original in blob
  → Docling DocumentConverter → DoclingDocument
  → export_to_markdown() → store as preview
  → HybridChunker → pre-chunked DocumentChunks (skip chunking step)
  → embed + store

PDF/DOCX (docling_markdown):
  Upload → store original
  → Docling → markdown → store preview
  → MarkdownChunker (header split → recursive size split)
  → DocumentChunks with section_headers metadata

PDF/DOCX (text — backward compat):
  Upload → store original
  → pypdf/python-docx → plain text (no preview)
  → RecursiveCharacterTextSplitter → DocumentChunks

PDF (ocr/zerox — unchanged):
  → Mistral OCR / VLM → markdown per page
  → one-page-per-chunk or text splitter
```

---

## Config Example (r2r.toml)

```toml
[file]
provider = "azure_blob"
# azure_account_name = "myaccount"        # or AZURE_BLOB_ACCOUNT_NAME env
# azure_account_key = "mykey"             # or AZURE_BLOB_ACCOUNT_KEY env
# azure_container_name = "mycontainer"    # or AZURE_BLOB_CONTAINER_NAME env
```

```toml
[ingestion]
extra_parsers = { pdf = ["zerox", "ocr", "docling_markdown", "text"], docx = ["docling_markdown", "text"] }
```

---

## Known Considerations

- `docling` and `azure-storage-blob` added as hard deps — consider moving to optional extras if bundle size matters
- Azure/S3 providers use sync SDK calls in async methods (matches existing S3 pattern)
- Default parser changed from BasicPDFParser/DOCXParser to Docling — use `parser_overrides: {"pdf": "text", "docx": "text"}` for old behavior
