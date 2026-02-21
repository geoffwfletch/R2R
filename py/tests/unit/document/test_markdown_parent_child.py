"""Unit tests for MarkdownParentChildChunker and expand_chunks_to_parents."""
import uuid

import pytest

from shared.utils.splitter.text import MarkdownParentChildChunker
from shared.utils.base_utils import expand_chunks_to_parents
from shared.abstractions.search import ChunkSearchResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_MD = """\
# AWS S3

## Authentication

Some intro text about authentication.

### API Keys

Content about API keys here. This is relatively short.

## Storage

Content about storage configuration.
"""

# Large section: >1500 words under ## Big Section, with ### subsections
# 600 words × 3 blocks = 1800 words total in ## Big Section → exceeds 1500 limit
_WORD_FILLER = ("word " * 600).strip()  # 600 words

LARGE_MD = f"""\
# Doc Title

## Big Section

{_WORD_FILLER}

### Sub A

{_WORD_FILLER}

### Sub B

{_WORD_FILLER}

## Small Section

Just a few words here.
"""

TABLE_MD = """\
# Report

## Data Table

Here is the data:

| Name | Value |
|------|-------|
| Foo  | 1     |
| Bar  | 2     |

End of section.
"""

CODE_MD = """\
# Guide

## Setup

Install the package:

```bash
pip install mypackage
echo "done"
```

Then configure it.
"""

H1_ONLY_MD = """\
# First Section

Some content here.

# Second Section

More content here.
"""

MULTI_PARENT_MD = """\
# Doc

## Section Alpha

Content alpha.

## Section Beta

Content beta.
"""


def _make_chunk(
    text: str,
    score: float = 0.5,
    parent_key: str | None = None,
    parent_content: str | None = None,
) -> ChunkSearchResult:
    meta: dict = {}
    if parent_key is not None:
        meta["parent_key"] = parent_key
        meta["parent_content"] = parent_content or text
    return ChunkSearchResult(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        owner_id=None,
        collection_ids=[],
        score=score,
        text=text,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# MarkdownParentChildChunker tests
# ---------------------------------------------------------------------------


def test_basic_section_becomes_parent():
    chunker = MarkdownParentChildChunker()
    docs = chunker.create_documents(SIMPLE_MD)

    auth_docs = [d for d in docs if d.metadata.get("parent_key") == "AWS S3__Authentication"]
    assert auth_docs, "Expected chunks with parent_key 'AWS S3__Authentication'"
    assert auth_docs[0].metadata["breadcrumb"] == "AWS S3 > Authentication"


def test_large_section_uses_h3_parents():
    chunker = MarkdownParentChildChunker()
    docs = chunker.create_documents(LARGE_MD)

    parent_keys = {d.metadata["parent_key"] for d in docs}
    # Big Section has >1500 words → Sub A and Sub B become parents
    assert any("Sub A" in k for k in parent_keys), f"Expected Sub A parent, got {parent_keys}"
    assert any("Sub B" in k for k in parent_keys), f"Expected Sub B parent, got {parent_keys}"
    # Big Section itself should NOT be a parent key
    assert "Doc Title__Big Section" not in parent_keys


def test_small_section_single_child():
    chunker = MarkdownParentChildChunker()
    docs = chunker.create_documents(SIMPLE_MD)

    storage_docs = [d for d in docs if d.metadata.get("parent_key") == "AWS S3__Storage"]
    assert len(storage_docs) == 1, f"Expected 1 child for small section, got {len(storage_docs)}"


def test_table_not_split():
    chunker = MarkdownParentChildChunker()
    docs = chunker.create_documents(TABLE_MD)

    # Find chunks that contain table content
    table_chunks = [d for d in docs if "|" in d.page_content]
    assert table_chunks, "No chunk contains table content"
    # The whole table should be in a single chunk (not straddling two)
    for chunk in table_chunks:
        rows = [l for l in chunk.page_content.splitlines() if l.strip().startswith("|")]
        # If a table starts in a chunk it should all be there
        if "Foo" in chunk.page_content:
            assert "Bar" in chunk.page_content, "Table rows split across chunks"


def test_code_block_not_split():
    chunker = MarkdownParentChildChunker()
    docs = chunker.create_documents(CODE_MD)

    code_chunks = [d for d in docs if "pip install" in d.page_content]
    assert code_chunks, "No chunk contains code block"
    # Both lines of the code block should be in the same chunk
    assert all("echo" in d.page_content for d in code_chunks), \
        "Code block split across chunks"


def test_child_text_has_no_breadcrumb():
    chunker = MarkdownParentChildChunker()
    docs = chunker.create_documents(SIMPLE_MD)

    for doc in docs:
        breadcrumb = doc.metadata.get("breadcrumb", "")
        # The breadcrumb should be in metadata, NOT prepended to text
        if breadcrumb:
            assert not doc.page_content.startswith(breadcrumb), \
                f"Breadcrumb found prepended to child text: {doc.page_content[:80]}"


def test_parent_content_is_full_section():
    chunker = MarkdownParentChildChunker()
    docs = chunker.create_documents(TABLE_MD)

    data_docs = [d for d in docs if d.metadata.get("parent_key") == "Report__Data Table"]
    assert data_docs
    parent_content = data_docs[0].metadata["parent_content"]
    assert "| Foo" in parent_content
    assert "| Bar" in parent_content
    assert "## Data Table" in parent_content


def test_chunk_order_sequential():
    chunker = MarkdownParentChildChunker()
    docs = chunker.create_documents(LARGE_MD)

    orders = [d.metadata["chunk_order"] for d in docs]
    assert orders == list(range(len(orders))), f"chunk_order not sequential: {orders}"


def test_multiple_parents_independent():
    chunker = MarkdownParentChildChunker()
    docs = chunker.create_documents(MULTI_PARENT_MD)

    alpha_keys = {d.metadata["parent_key"] for d in docs if "Alpha" in d.metadata.get("parent_key", "")}
    beta_keys = {d.metadata["parent_key"] for d in docs if "Beta" in d.metadata.get("parent_key", "")}
    assert alpha_keys and beta_keys
    assert not alpha_keys.intersection(beta_keys), "Parent keys should be independent"


def test_h1_only_doc():
    chunker = MarkdownParentChildChunker()
    # Should not crash and return at least one chunk
    docs = chunker.create_documents(H1_ONLY_MD)
    assert len(docs) >= 1


# ---------------------------------------------------------------------------
# expand_chunks_to_parents tests
# ---------------------------------------------------------------------------


def test_deduplicates_children_same_parent():
    chunks = [
        _make_chunk("child1", 0.9, "P__A", "full parent content"),
        _make_chunk("child2", 0.7, "P__A", "full parent content"),
        _make_chunk("child3", 0.5, "P__A", "full parent content"),
    ]
    result = expand_chunks_to_parents(chunks)
    assert len(result) == 1


def test_highest_score_preserved():
    chunks = [
        _make_chunk("child1", 0.9, "P__A", "full parent"),
        _make_chunk("child2", 0.7, "P__A", "full parent"),
        _make_chunk("child3", 0.5, "P__A", "full parent"),
    ]
    result = expand_chunks_to_parents(chunks)
    assert result[0].score == 0.9


def test_passthrough_no_parent_key():
    chunks = [
        _make_chunk("plain chunk 1", 0.8),
        _make_chunk("plain chunk 2", 0.6),
    ]
    result = expand_chunks_to_parents(chunks)
    assert len(result) == 2
    assert result[0].text == "plain chunk 1"
    assert result[1].text == "plain chunk 2"


def test_mixed_chunks():
    chunks = [
        _make_chunk("plain high", 0.95),
        _make_chunk("child a1", 0.8, "P__A", "parent A content"),
        _make_chunk("child a2", 0.6, "P__A", "parent A content"),
        _make_chunk("plain low", 0.3),
    ]
    result = expand_chunks_to_parents(chunks)
    # 1 passthrough high + 1 expanded parent + 1 passthrough low = 3
    assert len(result) == 3
    # Sorted by score desc
    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=True)


def test_parent_content_is_text():
    chunks = [
        _make_chunk("child snippet", 0.7, "P__A", "THE FULL PARENT CONTENT"),
    ]
    result = expand_chunks_to_parents(chunks)
    assert result[0].text == "THE FULL PARENT CONTENT"


def test_empty_input():
    result = expand_chunks_to_parents([])
    assert result == []
