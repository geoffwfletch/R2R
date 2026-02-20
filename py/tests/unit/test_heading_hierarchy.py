"""Tests for heading hierarchy fixing in docling markdown output."""

import importlib.util
import os
import sys

import pytest

# Load the module directly to avoid pulling in the entire core package tree.
_HERE = os.path.dirname(__file__)
_PY_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

_spec = importlib.util.spec_from_file_location(
    "pdf_parser_heading",
    os.path.join(_PY_ROOT, "core", "parsers", "media", "pdf_parser.py"),
    submodule_search_locations=[],
)
_mod = importlib.util.module_from_spec(_spec)
# Stub out heavy imports that the module file has at the top level.
# We only need the pure-python heading functions, not the parser classes.
class _SubscriptableMeta(type):
    def __getitem__(cls, item):
        return cls


class _Dummy(metaclass=_SubscriptableMeta):
    pass


for _stub in (
    "pdf2image",
    "mistralai", "mistralai.models",
    "pypdf",
    "core", "core.base", "core.base.abstractions",
    "core.base.parsers", "core.base.parsers.base_parser",
    "core.base.providers",
):
    if _stub not in sys.modules:
        sys.modules[_stub] = type(sys)(_stub)
    m = sys.modules[_stub]
    for attr in (
        "GenerationConfig", "AsyncParser", "CompletionProvider",
        "DatabaseProvider", "IngestionConfig", "OCRProvider",
        "PdfReader", "OCRResponse",
    ):
        setattr(m, attr, _Dummy)
_spec.loader.exec_module(_mod)

_heading_numbering_depth = _mod._heading_numbering_depth
_fix_heading_levels_by_numbering = _mod._fix_heading_levels_by_numbering
_promote_empty_body_headings = _mod._promote_empty_body_headings
fix_heading_hierarchy = _mod.fix_heading_hierarchy
_normalize_heading_text = _mod._normalize_heading_text
_infer_levels_from_font_sizes = _mod._infer_levels_from_font_sizes
_apply_font_size_levels = _mod._apply_font_size_levels
_extract_heading_font_sizes = _mod._extract_heading_font_sizes
fix_heading_hierarchy_from_result = _mod.fix_heading_hierarchy_from_result


# ---------------------------------------------------------------------------
# _heading_numbering_depth
# ---------------------------------------------------------------------------
class TestHeadingNumberingDepth:
    def test_dotted_two_levels(self):
        assert _heading_numbering_depth("1.2 Background") == 2

    def test_dotted_three_levels(self):
        assert _heading_numbering_depth("1.2.3 Details") == 3

    def test_dotted_four_levels(self):
        assert _heading_numbering_depth("1.2.3.4 Sub-details") == 4

    def test_simple_numerical_period(self):
        assert _heading_numbering_depth("1. Introduction") == 1

    def test_simple_numerical_paren(self):
        assert _heading_numbering_depth("3) Scope") == 1

    def test_roman_period(self):
        assert _heading_numbering_depth("IV. Requirements") == 1

    def test_roman_paren(self):
        assert _heading_numbering_depth("III) Definitions") == 1

    def test_roman_lowercase(self):
        assert _heading_numbering_depth("iv. Requirements") == 1

    def test_roman_single_i(self):
        assert _heading_numbering_depth("I. Introduction") == 1

    def test_alphabetic_period(self):
        assert _heading_numbering_depth("A. First section") == 1

    def test_alphabetic_paren(self):
        assert _heading_numbering_depth("b) Second section") == 1

    def test_no_numbering(self):
        assert _heading_numbering_depth("Introduction") is None

    def test_no_numbering_plain_text(self):
        assert _heading_numbering_depth("Schedule of Prices") is None

    def test_no_numbering_no_delimiter(self):
        # "1Introduction" - no space/delimiter after number
        assert _heading_numbering_depth("1Introduction") is None

    def test_non_roman_letters(self):
        # "ZZ." is not a valid roman numeral
        assert _heading_numbering_depth("ZZ. Heading") is None


# ---------------------------------------------------------------------------
# _fix_heading_levels_by_numbering
# ---------------------------------------------------------------------------
class TestFixHeadingLevelsByNumbering:
    def test_promotes_depth1_to_h1(self):
        md = "## 1. Introduction\n\nBody text\n\n## 1.1 Background\n\nMore text\n"
        result = _fix_heading_levels_by_numbering(md)
        assert result.startswith("# 1. Introduction")
        assert "## 1.1 Background" in result

    def test_demotes_depth3_to_h3(self):
        md = (
            "## 1. Intro\n\nText\n\n"
            "## 1.1 Sub\n\nText\n\n"
            "## 1.1.1 Detail\n\nText\n"
        )
        result = _fix_heading_levels_by_numbering(md)
        assert "# 1. Intro" in result
        assert "## 1.1 Sub" in result
        assert "### 1.1.1 Detail" in result

    def test_below_threshold_no_change(self):
        # Only 1 out of 4 headings numbered = 25% < 30%
        md = (
            "## 1. Numbered\n\nText\n\n"
            "## Plain A\n\nText\n\n"
            "## Plain B\n\nText\n\n"
            "## Plain C\n\nText\n"
        )
        result = _fix_heading_levels_by_numbering(md)
        assert result == md

    def test_above_threshold_applies(self):
        # 3 out of 4 numbered = 75% > 30%
        md = (
            "## 1. First\n\nText\n\n"
            "## 2. Second\n\nText\n\n"
            "## 3. Third\n\nText\n\n"
            "## Appendix\n\nText\n"
        )
        result = _fix_heading_levels_by_numbering(md)
        assert "# 1. First" in result
        assert "# 2. Second" in result

    def test_no_headings_returns_unchanged(self):
        md = "Just some body text\n\nMore text\n"
        assert _fix_heading_levels_by_numbering(md) == md

    def test_unnumbered_headings_left_as_h2(self):
        md = (
            "## 1. Intro\n\nText\n\n"
            "## 1.1 Sub\n\nText\n\n"
            "## Appendix\n\nText\n"
        )
        result = _fix_heading_levels_by_numbering(md)
        assert "## Appendix" in result


# ---------------------------------------------------------------------------
# _promote_empty_body_headings
# ---------------------------------------------------------------------------
class TestPromoteEmptyBodyHeadings:
    def test_promotes_parent_heading(self):
        md = "## Parent Section\n\n## Child Section\n\nBody text\n"
        result = _promote_empty_body_headings(md)
        assert result.startswith("# Parent Section")
        assert "## Child Section" in result

    def test_no_promotion_when_body_exists(self):
        md = "## Section A\n\nSome body text here.\n\n## Section B\n\nMore text\n"
        result = _promote_empty_body_headings(md)
        assert "## Section A" in result
        assert "## Section B" in result

    def test_chain_of_empty_headings(self):
        md = "## Top\n\n## Middle\n\n## Bottom\n\nBody\n"
        result = _promote_empty_body_headings(md)
        assert "# Top" in result
        assert "# Middle" in result
        assert "## Bottom" in result

    def test_no_headings(self):
        md = "Just text\n\nMore text\n"
        assert _promote_empty_body_headings(md) == md


# ---------------------------------------------------------------------------
# fix_heading_hierarchy (text-only, no ConversionResult)
# ---------------------------------------------------------------------------
class TestFixHeadingHierarchy:
    def test_numbering_takes_priority_over_structural(self):
        md = (
            "## 1. Overview\n\n"
            "## 1.1 Details\n\nBody\n\n"
            "## 2. Next\n\nBody\n"
        )
        result = fix_heading_hierarchy(md)
        assert "# 1. Overview" in result
        assert "## 1.1 Details" in result
        assert "# 2. Next" in result

    def test_structural_fallback(self):
        md = "## Description of works\n\n## Jason Armstrong\n\nDirector\n"
        result = fix_heading_hierarchy(md)
        assert "# Description of works" in result
        assert "## Jason Armstrong" in result

    def test_real_world_docling_pattern(self):
        md = (
            "## Schedule 1A -Schedule of Prices\n\n"
            "## Tenderer's Name\n\nDonald Cant Watts Corke\n"
        )
        result = fix_heading_hierarchy(md)
        assert "# Schedule 1A -Schedule of Prices" in result
        assert "## Tenderer's Name" in result


# ---------------------------------------------------------------------------
# _normalize_heading_text
# ---------------------------------------------------------------------------
class TestNormalizeHeadingText:
    def test_collapses_whitespace(self):
        assert _normalize_heading_text("  hello   world  ") == "hello world"

    def test_lowercase(self):
        assert _normalize_heading_text("HELLO") == "hello"

    def test_preserves_content(self):
        assert _normalize_heading_text("Section 1.2") == "section 1.2"


# ---------------------------------------------------------------------------
# _infer_levels_from_font_sizes
# ---------------------------------------------------------------------------
class TestInferLevelsFromFontSizes:
    def test_two_distinct_sizes(self):
        headings = [
            ("Big Heading", 18.0),
            ("Small Heading", 12.0),
            ("Another Big", 18.0),
        ]
        levels = _infer_levels_from_font_sizes(headings)
        assert levels["big heading"] == 1
        assert levels["small heading"] == 2
        assert levels["another big"] == 1

    def test_three_sizes(self):
        headings = [
            ("Title", 20.0),
            ("Section", 16.0),
            ("Subsection", 12.0),
        ]
        levels = _infer_levels_from_font_sizes(headings)
        assert levels["title"] == 1
        assert levels["section"] == 2
        assert levels["subsection"] == 3

    def test_caps_at_level_3(self):
        headings = [
            ("H1", 20.0),
            ("H2", 16.0),
            ("H3", 12.0),
            ("H4", 8.0),
        ]
        levels = _infer_levels_from_font_sizes(headings)
        assert levels["h1"] == 1
        assert levels["h2"] == 2
        assert levels["h3"] == 3
        assert levels["h4"] == 3  # capped

    def test_single_size_returns_empty(self):
        headings = [("A", 14.0), ("B", 14.0)]
        assert _infer_levels_from_font_sizes(headings) == {}

    def test_empty_input(self):
        assert _infer_levels_from_font_sizes([]) == {}

    def test_sizes_within_tolerance_same_level(self):
        headings = [
            ("Heading A", 18.2),
            ("Heading B", 18.0),
            ("Small", 12.0),
        ]
        levels = _infer_levels_from_font_sizes(headings, tolerance=1.0)
        assert levels["heading a"] == 1
        assert levels["heading b"] == 1
        assert levels["small"] == 2

    def test_custom_tolerance(self):
        headings = [
            ("A", 18.0),
            ("B", 17.0),
            ("C", 12.0),
        ]
        # Tight tolerance: 18 and 17 are different levels
        levels = _infer_levels_from_font_sizes(headings, tolerance=0.5)
        assert levels["a"] == 1
        assert levels["b"] == 2
        assert levels["c"] == 3


# ---------------------------------------------------------------------------
# _apply_font_size_levels
# ---------------------------------------------------------------------------
class TestApplyFontSizeLevels:
    def test_promotes_to_h1(self):
        md = "## Big Title\n\nBody\n\n## Small Section\n\nText\n"
        level_map = {"big title": 1, "small section": 2}
        result = _apply_font_size_levels(md, level_map)
        assert "# Big Title" in result
        assert "## Small Section" in result

    def test_demotes_to_h3(self):
        md = "## Sub Detail\n\nText\n"
        level_map = {"sub detail": 3}
        result = _apply_font_size_levels(md, level_map)
        assert "### Sub Detail" in result

    def test_unmatched_headings_unchanged(self):
        md = "## Unknown Heading\n\nText\n"
        level_map = {"something else": 1}
        result = _apply_font_size_levels(md, level_map)
        assert "## Unknown Heading" in result

    def test_empty_map_no_change(self):
        md = "## Heading\n\nText\n"
        assert _apply_font_size_levels(md, {}) == md

    def test_whitespace_normalization_matching(self):
        md = "## Some   Spaced   Heading\n\nText\n"
        level_map = {"some spaced heading": 1}
        result = _apply_font_size_levels(md, level_map)
        assert "# Some   Spaced   Heading" in result


# ---------------------------------------------------------------------------
# _extract_heading_font_sizes (mock ConversionResult)
# ---------------------------------------------------------------------------
class _MockRect:
    def __init__(self, height):
        self.height = height


class _MockCell:
    def __init__(self, text, height):
        self.text = text
        self.rect = _MockRect(height)


class _MockCluster:
    def __init__(self, label, cells):
        self.label = label
        self.cells = cells


class _MockLayout:
    def __init__(self, clusters):
        self.clusters = clusters


class _MockPredictions:
    def __init__(self, layout):
        self.layout = layout


class _MockPage:
    def __init__(self, clusters):
        self.predictions = _MockPredictions(_MockLayout(clusters))


class _MockResult:
    def __init__(self, pages):
        self.pages = pages


class TestExtractHeadingFontSizes:
    def test_extracts_section_headers(self):
        result = _MockResult([
            _MockPage([
                _MockCluster("section_header", [_MockCell("Introduction", 18.0)]),
                _MockCluster("text", [_MockCell("body text", 12.0)]),
                _MockCluster("section_header", [_MockCell("Details", 14.0)]),
            ])
        ])
        headings = _extract_heading_font_sizes(result)
        assert len(headings) == 2
        assert headings[0] == ("Introduction", 18.0)
        assert headings[1] == ("Details", 14.0)

    def test_multi_cell_heading(self):
        result = _MockResult([
            _MockPage([
                _MockCluster("section_header", [
                    _MockCell("Part", 16.0),
                    _MockCell("One", 16.0),
                ]),
            ])
        ])
        headings = _extract_heading_font_sizes(result)
        assert headings[0] == ("Part One", 16.0)

    def test_skips_non_header_clusters(self):
        result = _MockResult([
            _MockPage([
                _MockCluster("table", [_MockCell("data", 10.0)]),
            ])
        ])
        assert _extract_heading_font_sizes(result) == []

    def test_skips_empty_clusters(self):
        result = _MockResult([
            _MockPage([
                _MockCluster("section_header", []),
            ])
        ])
        assert _extract_heading_font_sizes(result) == []

    def test_multiple_pages(self):
        result = _MockResult([
            _MockPage([
                _MockCluster("section_header", [_MockCell("Page 1 H", 18.0)]),
            ]),
            _MockPage([
                _MockCluster("section_header", [_MockCell("Page 2 H", 14.0)]),
            ]),
        ])
        headings = _extract_heading_font_sizes(result)
        assert len(headings) == 2

    def test_missing_predictions_graceful(self):
        """Pages without predictions attribute are skipped."""

        class _BrokenPage:
            pass

        result = _MockResult([_BrokenPage()])
        assert _extract_heading_font_sizes(result) == []


# ---------------------------------------------------------------------------
# fix_heading_hierarchy_from_result (full pipeline with mock)
# ---------------------------------------------------------------------------
class TestFixHeadingHierarchyFromResult:
    def test_font_sizes_used_when_available(self):
        md = "## Introduction\n\nBody\n\n## Details\n\nMore\n"
        result = _MockResult([
            _MockPage([
                _MockCluster("section_header", [_MockCell("Introduction", 18.0)]),
                _MockCluster("section_header", [_MockCell("Details", 12.0)]),
            ])
        ])
        out = fix_heading_hierarchy_from_result(md, result)
        assert "# Introduction" in out
        assert "## Details" in out

    def test_falls_back_to_numbering(self):
        """When font sizes are uniform, numbering heuristic kicks in."""
        md = "## 1. Intro\n\nText\n\n## 1.1 Sub\n\nText\n"
        result = _MockResult([
            _MockPage([
                _MockCluster("section_header", [_MockCell("1. Intro", 14.0)]),
                _MockCluster("section_header", [_MockCell("1.1 Sub", 14.0)]),
            ])
        ])
        out = fix_heading_hierarchy_from_result(md, result)
        assert "# 1. Intro" in out
        assert "## 1.1 Sub" in out

    def test_falls_back_to_structural(self):
        """When no font-size signal and no numbering, structural pass runs."""
        md = "## Parent\n\n## Child\n\nBody\n"
        result = _MockResult([
            _MockPage([
                _MockCluster("section_header", [_MockCell("Parent", 14.0)]),
                _MockCluster("section_header", [_MockCell("Child", 14.0)]),
            ])
        ])
        out = fix_heading_hierarchy_from_result(md, result)
        assert "# Parent" in out
        assert "## Child" in out

    def test_structural_pass_runs_after_font_sizes(self):
        """Structural pass catches remaining flat headings that font-size missed."""
        md = (
            "## Known Big\n\nBody\n\n"
            "## Unknown Parent\n\n"
            "## Unknown Child\n\nBody\n"
        )
        result = _MockResult([
            _MockPage([
                _MockCluster("section_header", [_MockCell("Known Big", 18.0)]),
                # Unknown headings not in clusters (e.g. missed by layout model)
                _MockCluster("section_header", [_MockCell("Unknown Parent", 12.0)]),
                _MockCluster("section_header", [_MockCell("Unknown Child", 12.0)]),
            ])
        ])
        out = fix_heading_hierarchy_from_result(md, result)
        assert "# Known Big" in out
        # Unknown Parent has no body before Unknown Child → structural promotes it
        # But both are level 2 from font-size, so structural check applies
        # to remaining ## headings
        assert "Unknown Parent" in out
        assert "Unknown Child" in out


# ---------------------------------------------------------------------------
# Integration: MarkdownChunker propagates Header 1 metadata
# ---------------------------------------------------------------------------

# Load MarkdownChunker via importlib to avoid heavy dependency chain.
_splitter_spec = importlib.util.spec_from_file_location(
    "splitter_text",
    os.path.join(_PY_ROOT, "shared", "utils", "splitter", "text.py"),
)
try:
    _splitter_mod = importlib.util.module_from_spec(_splitter_spec)
    _splitter_spec.loader.exec_module(_splitter_mod)
    MarkdownChunker = _splitter_mod.MarkdownChunker
    _HAS_CHUNKER = True
except Exception:
    _HAS_CHUNKER = False
    MarkdownChunker = None


@pytest.mark.skipif(not _HAS_CHUNKER, reason="MarkdownChunker not importable")
class TestMarkdownChunkerIntegration:
    def test_header1_propagates_to_child_chunks(self):
        md = (
            "# Parent Section\n\n"
            "## Child Section\n\n"
            "This is the body text under the child section. "
            "It has enough content to form a chunk.\n"
        )
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=0)
        docs = chunker.create_documents([md])
        body_chunks = [
            d for d in docs if "body text" in d.page_content
        ]
        assert body_chunks, "Expected chunk with body text"
        meta = body_chunks[0].metadata
        assert meta.get("Header 1") == "Parent Section"
        assert meta.get("Header 2") == "Child Section"

    def test_fix_then_chunk_produces_header1(self):
        """End-to-end: fix_heading_hierarchy + MarkdownChunker."""
        raw_md = (
            "## Description of works\n\n"
            "## Jason Armstrong\n\n"
            "Director in Charge\n"
            "Associate Diploma\n"
        )
        fixed = fix_heading_hierarchy(raw_md)
        assert "# Description of works" in fixed

        chunker = MarkdownChunker(chunk_size=1024, chunk_overlap=0)
        docs = chunker.create_documents([fixed])
        body_chunks = [
            d for d in docs if "Director in Charge" in d.page_content
        ]
        assert body_chunks
        meta = body_chunks[0].metadata
        assert meta.get("Header 1") == "Description of works"

    def test_font_size_fix_then_chunk(self):
        """End-to-end: font-size fix + MarkdownChunker."""
        md = (
            "## Big Section\n\n"
            "## Small Sub\n\n"
            "Detail text here for chunking.\n"
        )
        result = _MockResult([
            _MockPage([
                _MockCluster("section_header", [_MockCell("Big Section", 18.0)]),
                _MockCluster("section_header", [_MockCell("Small Sub", 12.0)]),
            ])
        ])
        fixed = fix_heading_hierarchy_from_result(md, result)
        chunker = MarkdownChunker(chunk_size=1024, chunk_overlap=0)
        docs = chunker.create_documents([fixed])
        body_chunks = [d for d in docs if "Detail text" in d.page_content]
        assert body_chunks
        meta = body_chunks[0].metadata
        assert meta.get("Header 1") == "Big Section"
        assert meta.get("Header 2") == "Small Sub"
