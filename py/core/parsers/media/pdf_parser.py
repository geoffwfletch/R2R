# type: ignore
import asyncio
import base64
import json
import logging
import re
import string
import time
import unicodedata
from io import BytesIO
from typing import AsyncGenerator

import pdf2image
from mistralai.models import OCRResponse
from pypdf import PdfReader

from core.base.abstractions import GenerationConfig
from core.base.parsers.base_parser import AsyncParser
from core.base.providers import (
    CompletionProvider,
    DatabaseProvider,
    IngestionConfig,
    OCRProvider,
)

logger = logging.getLogger()


def _get_usage_field(usage, field: str, default: int = 0) -> int:
    """Extract a field from a usage object that may be a dict or pydantic model."""
    if isinstance(usage, dict):
        return usage.get(field, default) or default
    return getattr(usage, field, default) or default


_ROMAN_NUMERALS = frozenset({
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
})


def _heading_numbering_depth(text: str) -> int | None:
    """Return numbering depth from heading text, or None if unnumbered.

    "1.2.3 Heading" -> 3, "1. Heading" -> 1, "IV. Heading" -> 1
    """
    text = text.strip()

    # Dotted numerical: 1.2.3 -> depth 3
    m = re.match(r"^(\d+(?:\.\d+)+)\s", text)
    if m:
        return len(m.group(1).split("."))

    # Simple numerical: 1. or 3) -> depth 1
    if re.match(r"^\d+[.)]\s", text):
        return 1

    # Roman: IV. or III) -> depth 1
    m = re.match(r"^([IVXLCDM]+)[.)]\s", text, re.IGNORECASE)
    if m and m.group(1).upper() in _ROMAN_NUMERALS:
        return 1

    # Alphabetic: A. or b) -> depth 1
    if re.match(r"^[A-Za-z][.)]\s", text):
        return 1

    return None


def _fix_heading_levels_by_numbering(md: str) -> str:
    """Adjust ## heading levels using numbering depth when >30% are numbered."""
    h2_re = re.compile(r"^## (.+)$", re.MULTILINE)
    matches = list(h2_re.finditer(md))
    if not matches:
        return md

    depths = [_heading_numbering_depth(m.group(1)) for m in matches]
    numbered = sum(1 for d in depths if d is not None)
    if numbered / len(matches) <= 0.3:
        return md

    for match, depth in reversed(list(zip(matches, depths))):
        if depth is None or depth == 2:
            continue
        prefix = "#" * min(depth, 3) + " "
        if prefix == "## ":
            continue
        md = md[:match.start()] + prefix + match.group(1) + md[match.end():]

    return md


def _promote_empty_body_headings(md: str) -> str:
    """Promote ## with no body before next ## to #."""
    return re.sub(
        r"^(## .+)\n\s*\n(?=## )",
        lambda m: m.group(1).replace("## ", "# ", 1) + "\n\n",
        md,
        flags=re.MULTILINE,
    )


def _normalize_heading_text(text: str) -> str:
    """Normalize heading text for fuzzy matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_heading_font_sizes(result) -> list[tuple[str, float]]:
    """Extract ``(text, font_size)`` pairs for section headers from a
    docling ``ConversionResult``.  Font size is approximated by the
    first cell's bounding-box height in the layout cluster.
    """
    headings: list[tuple[str, float]] = []
    for page in result.pages:
        preds = getattr(page, "predictions", None)
        layout = getattr(preds, "layout", None) if preds else None
        if layout is None:
            continue
        for cluster in layout.clusters:
            if cluster.label != "section_header" or not cluster.cells:
                continue
            text = " ".join(c.text for c in cluster.cells).strip()
            size = cluster.cells[0].rect.height
            if text:
                headings.append((text, size))
    return headings


def _infer_levels_from_font_sizes(
    heading_sizes: list[tuple[str, float]],
    tolerance: float = 1.0,
) -> dict[str, int]:
    """Map *normalised* heading text -> level (1-3) based on font size.

    Largest font cluster -> level 1, next -> level 2, etc.
    Sizes within *tolerance* are treated as the same level.
    Returns an empty dict when all headings share one size level
    (i.e. font size provides no hierarchy signal).
    """
    if not heading_sizes:
        return {}

    unique_sizes = sorted(set(s for _, s in heading_sizes), reverse=True)

    # Build clusters: each cluster is a representative (largest) size
    cluster_reps: list[float] = []
    for size in unique_sizes:
        if not cluster_reps or cluster_reps[-1] - size > tolerance:
            cluster_reps.append(size)

    # Only useful when there are multiple distinct size levels
    if len(cluster_reps) <= 1:
        return {}

    # Map every unique size to the cluster level it belongs to
    size_to_level: dict[float, int] = {}
    for size in unique_sizes:
        for idx, rep in enumerate(cluster_reps):
            if rep - size <= tolerance:
                size_to_level[size] = min(idx + 1, 3)
                break

    return {
        _normalize_heading_text(text): size_to_level[size]
        for text, size in heading_sizes
        if size in size_to_level
    }


def _apply_font_size_levels(md: str, level_map: dict[str, int]) -> str:
    """Replace ``##`` headings with the level dictated by *level_map*."""
    if not level_map:
        return md

    h2_re = re.compile(r"^## (.+)$", re.MULTILINE)

    def _replace(match):
        level = level_map.get(_normalize_heading_text(match.group(1)))
        if level is None or level == 2:
            return match.group(0)
        return "#" * level + " " + match.group(1)

    return h2_re.sub(_replace, md)


def fix_heading_hierarchy(md: str) -> str:
    """Fix flattened heading hierarchy using text heuristics only.

    Two strategies applied in order:
    1. Numbering-based: if >30% of H2s have numbering (1.2.3, IV., A.),
       adjust levels by numbering depth.
    2. Structural: promote empty-body H2s to H1 (parent headings).
    """
    md = _fix_heading_levels_by_numbering(md)
    md = _promote_empty_body_headings(md)
    return md


def fix_heading_hierarchy_from_result(md: str, result) -> str:
    """Fix heading hierarchy using font-size data from ConversionResult,
    falling back to text heuristics when font sizes are unavailable.
    """
    heading_sizes = _extract_heading_font_sizes(result)
    level_map = _infer_levels_from_font_sizes(heading_sizes)
    if level_map:
        md = _apply_font_size_levels(md, level_map)
    else:
        md = _fix_heading_levels_by_numbering(md)
    # Always run structural pass for remaining flat headings
    md = _promote_empty_body_headings(md)
    return md


class OCRPDFParser(AsyncParser[str | bytes]):
    """
    A parser for PDF documents using Mistral's OCR for page processing.

    Mistral supports directly processing PDF files, so this parser is a simple wrapper around the Mistral OCR API.
    """

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
        ocr_provider: OCRProvider,
    ):
        self.config = config
        self.database_provider = database_provider
        self.ocr_provider = ocr_provider

    async def ingest(
        self, data: str | bytes, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Ingest PDF data and yield text from each page."""
        try:
            logger.info("Starting PDF ingestion using MistralOCRParser")

            if isinstance(data, str):
                response: OCRResponse = await self.ocr_provider.process_pdf(
                    file_path=data
                )
            else:
                response: OCRResponse = await self.ocr_provider.process_pdf(
                    file_content=data
                )

            for page in response.pages:
                yield {
                    "content": page.markdown,
                    "page_number": page.index + 1,  # Mistral is 0-indexed
                }

        except Exception as e:
            logger.error(f"Error processing PDF with Mistral OCR: {str(e)}")
            raise


class VLMPDFParser(AsyncParser[str | bytes]):
    """A parser for PDF documents using vision models for page processing."""

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
        ocr_provider: OCRProvider,
    ):
        self.database_provider = database_provider
        self.llm_provider = llm_provider
        self.config = config
        self.vision_prompt_text = None
        self.vlm_batch_size = self.config.vlm_batch_size or 5
        self.vlm_max_tokens_to_sample = (
            self.config.vlm_max_tokens_to_sample or 1024
        )
        self.max_concurrent_vlm_tasks = (
            self.config.max_concurrent_vlm_tasks or 5
        )
        self.vlm_pages_per_call = getattr(
            self.config, "vlm_pages_per_call", None
        ) or 1
        self.semaphore = None
        self._vlm_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._chunk_selection_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    @staticmethod
    def _clean_vlm_output(text: str) -> str:
        """Strip code fences and metadata lines from VLM output."""
        # Remove ```markdown ... ``` wrapping
        text = re.sub(r"^```(?:markdown)?\s*\n?", "", text.strip())
        text = re.sub(r"\n?```\s*$", "", text.strip())
        # Remove metadata lines the model sometimes adds
        text = re.sub(
            r"^(Page number|Header/Footer|Sidebars?/Callouts?):.*$\n?",
            "",
            text,
            flags=re.MULTILINE,
        )
        return text.strip()

    async def process_page(self, image, page_num: int) -> dict[str, str]:
        """Process a single PDF page using the vision model."""
        page_start = time.perf_counter()
        try:
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="JPEG")
            image_data = img_byte_arr.getvalue()
            # Convert image bytes to base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            model = getattr(self, '_vlm_override', None) or self.config.vlm or self.config.app.vlm

            # Configure generation parameters
            generation_config = GenerationConfig(
                model=model,
                stream=False,
                max_tokens_to_sample=self.vlm_max_tokens_to_sample,
            )

            is_anthropic = model and "anthropic/" in model

            # Prepare message with image content
            if is_anthropic:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.vision_prompt_text},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                        ],
                    }
                ]
            else:
                # Use OpenAI format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.vision_prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ]

            logger.debug(f"Sending page {page_num} to vision model.")

            if is_anthropic:
                response = await self.llm_provider.aget_completion(
                    messages=messages,
                    generation_config=generation_config,
                    apply_timeout=True,
                    tools=[
                        {
                            "name": "parse_pdf_page",
                            "description": "Parse text content from a PDF page",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "page_content": {
                                        "type": "string",
                                        "description": "Extracted text from the PDF page, transcribed into markdown",
                                    },
                                    "thoughts": {
                                        "type": "string",
                                        "description": "Any thoughts or comments on the text",
                                    },
                                },
                                "required": ["page_content"],
                            },
                        }
                    ],
                    tool_choice={"type": "tool", "name": "parse_pdf_page"},
                )

                if hasattr(response, 'usage') and response.usage:
                    self._vlm_usage["prompt_tokens"] += _get_usage_field(response.usage, 'prompt_tokens')
                    self._vlm_usage["completion_tokens"] += _get_usage_field(response.usage, 'completion_tokens')

                if (
                    response.choices
                    and response.choices[0].message
                    and response.choices[0].message.tool_calls
                ):
                    tool_call = response.choices[0].message.tool_calls[0]
                    args = json.loads(tool_call.function.arguments)
                    content = self._clean_vlm_output(args.get("page_content", ""))
                    page_elapsed = time.perf_counter() - page_start
                    logger.debug(
                        f"Processed page {page_num} in {page_elapsed:.2f} seconds."
                    )
                    return {"page": str(page_num), "content": content}
                else:
                    logger.warning(
                        f"No valid tool call in response for page {page_num}, document might be missing text."
                    )
                    return {"page": str(page_num), "content": ""}
            else:
                response = await self.llm_provider.aget_completion(
                    messages=messages,
                    generation_config=generation_config,
                    apply_timeout=True,
                )

                if hasattr(response, 'usage') and response.usage:
                    self._vlm_usage["prompt_tokens"] += _get_usage_field(response.usage, 'prompt_tokens')
                    self._vlm_usage["completion_tokens"] += _get_usage_field(response.usage, 'completion_tokens')

                if response.choices and response.choices[0].message:
                    content = self._clean_vlm_output(
                        response.choices[0].message.content or ""
                    )
                    page_elapsed = time.perf_counter() - page_start
                    logger.debug(
                        f"Processed page {page_num} in {page_elapsed:.2f} seconds."
                    )
                    return {"page": str(page_num), "content": content}
                else:
                    msg = f"No response content for page {page_num}"
                    logger.error(msg)
                    return {"page": str(page_num), "content": ""}
        except Exception as e:
            logger.error(
                f"Error processing page {page_num} with vision model: {str(e)}"
            )
            # Return empty content rather than raising to avoid failing the entire batch
            return {
                "page": str(page_num),
                "content": f"Error processing page: {str(e)}",
            }

    async def process_pages(
        self, images: list, page_nums: list[int]
    ) -> list[dict[str, str]]:
        """Process multiple PDF pages in a single VLM call."""
        call_start = time.perf_counter()
        n = len(images)
        try:
            image_blocks = []
            model = getattr(self, '_vlm_override', None) or self.config.vlm or self.config.app.vlm
            is_anthropic = model and "anthropic/" in model

            for image in images:
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format="JPEG")
                image_base64 = base64.b64encode(
                    img_byte_arr.getvalue()
                ).decode("utf-8")

                if is_anthropic:
                    image_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        }
                    )
                else:
                    image_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        }
                    )

            prompt = (
                f"{self.vision_prompt_text}\n\n"
                f"You are given {n} consecutive pages "
                f"(pages {page_nums[0]}-{page_nums[-1]}). "
                f"Convert ALL pages. Separate each page with "
                f"exactly this marker on its own line: "
                f"<!--PAGE_BREAK-->"
            )

            content_blocks = [{"type": "text", "text": prompt}]
            content_blocks.extend(image_blocks)
            messages = [{"role": "user", "content": content_blocks}]

            generation_config = GenerationConfig(
                model=model,
                stream=False,
                max_tokens_to_sample=self.vlm_max_tokens_to_sample * n,
            )

            logger.debug(
                f"Sending pages {page_nums[0]}-{page_nums[-1]} "
                f"to vision model in single call."
            )

            response = await self.llm_provider.aget_completion(
                messages=messages,
                generation_config=generation_config,
                apply_timeout=True,
            )

            if hasattr(response, 'usage') and response.usage:
                self._vlm_usage["prompt_tokens"] += getattr(response.usage, 'prompt_tokens', 0) or 0
                self._vlm_usage["completion_tokens"] += getattr(response.usage, 'completion_tokens', 0) or 0

            if response.choices and response.choices[0].message:
                full_text = response.choices[0].message.content or ""
                parts = full_text.split("<!--PAGE_BREAK-->")
                elapsed = time.perf_counter() - call_start
                logger.debug(
                    f"Processed pages {page_nums[0]}-{page_nums[-1]} "
                    f"in {elapsed:.2f}s, got {len(parts)} parts."
                )

                results = []
                for i, page_num in enumerate(page_nums):
                    text = (
                        self._clean_vlm_output(parts[i])
                        if i < len(parts)
                        else ""
                    )
                    results.append(
                        {"page": str(page_num), "content": text}
                    )
                return results
            else:
                return [
                    {"page": str(p), "content": ""}
                    for p in page_nums
                ]
        except Exception as e:
            logger.error(
                f"Error processing pages {page_nums[0]}-{page_nums[-1]}: "
                f"{e}"
            )
            return [
                {
                    "page": str(p),
                    "content": f"Error processing page: {e}",
                }
                for p in page_nums
            ]

    async def process_and_yield(self, image, page_num: int):
        """Process a page and yield the result."""
        async with self.semaphore:
            result = await self.process_page(image, page_num)
            return {
                "content": result.get("content", "") or "",
                "page_number": page_num,
            }

    async def process_and_yield_multi(
        self, images: list, page_nums: list[int]
    ):
        """Process multiple pages in one VLM call and yield results."""
        async with self.semaphore:
            results = await self.process_pages(images, page_nums)
            return [
                {
                    "content": r.get("content", "") or "",
                    "page_number": int(r["page"]),
                }
                for r in results
            ]

    async def _llm_chunk_selection(
        self, full_md: str, chunk_model: str
    ) -> tuple[list[str], str]:
        """Use an LLM to insert semantic chunk boundaries into the document.

        Returns (chunks, raw_llm_output_with_markers).
        """
        logger.info(
            f"Running LLM chunk selection with model={chunk_model}, "
            f"doc_len={len(full_md)} chars"
        )
        try:
            prompt_template = (
                await self.database_provider.prompts_handler.get_cached_prompt(
                    prompt_name="llm_chunk_selection"
                )
            )

            # Sliding window for long docs (>80k chars)
            window_size = 80_000
            overlap = 2_000
            if len(full_md) <= window_size:
                segments = [full_md]
            else:
                segments = []
                start = 0
                while start < len(full_md):
                    end = min(start + window_size, len(full_md))
                    segments.append(full_md[start:end])
                    start = end - overlap
                    if start + overlap >= len(full_md):
                        break

            all_chunks: list[str] = []
            raw_segments: list[str] = []
            for seg in segments:
                prompt_text = prompt_template.replace(
                    "{document_text}", seg
                )
                messages = [{"role": "user", "content": prompt_text}]
                generation_config = GenerationConfig(
                    model=chunk_model,
                    stream=False,
                    temperature=0,
                    max_tokens_to_sample=16_384,
                )
                response = await self.llm_provider.aget_completion(
                    messages=messages,
                    generation_config=generation_config,
                    apply_timeout=True,
                )
                if hasattr(response, 'usage') and response.usage:
                    self._chunk_selection_usage["prompt_tokens"] += _get_usage_field(response.usage, 'prompt_tokens')
                    self._chunk_selection_usage["completion_tokens"] += _get_usage_field(response.usage, 'completion_tokens')
                if response.choices and response.choices[0].message:
                    result_text = response.choices[0].message.content or ""
                    raw_segments.append(result_text)
                    parts = [
                        p.strip()
                        for p in result_text.split("<CHUNK_BREAK>")
                        if p.strip()
                    ]
                    all_chunks.extend(parts)
                else:
                    raw_segments.append(seg)
                    all_chunks.append(seg)

            raw_output = "\n\n".join(raw_segments)
            logger.info(
                f"LLM chunk selection produced {len(all_chunks)} chunks"
            )
            return (
                all_chunks if all_chunks else [full_md],
                raw_output,
            )
        except Exception as e:
            logger.warning(
                f"LLM chunk selection failed, falling back to single chunk: {e}"
            )
            return [full_md], full_md

    async def ingest(
        self, data: str | bytes, **kwargs
    ) -> AsyncGenerator[dict[str, str | int], None]:
        """Process PDF as images using pdf2image."""
        ingest_start = time.perf_counter()
        logger.info("Starting PDF ingestion using VLMPDFParser.")

        self._vlm_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._chunk_selection_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        # Per-request VLM model override (from frontend ingestion settings)
        app_override = kwargs.get("app", {})
        if isinstance(app_override, dict) and app_override.get("vlm"):
            self._vlm_override = app_override["vlm"]
            logger.info(f"Using per-request VLM override: {self._vlm_override}")
        else:
            self._vlm_override = None

        if not self.vision_prompt_text:
            self.vision_prompt_text = (
                await self.database_provider.prompts_handler.get_cached_prompt(
                    prompt_name="vision_pdf"
                )
            )
            logger.info("Retrieved vision prompt text from database.")

        self.semaphore = asyncio.Semaphore(self.max_concurrent_vlm_tasks)

        try:
            if isinstance(data, str):
                pdf_info = pdf2image.pdfinfo_from_path(data)
            else:
                pdf_bytes = BytesIO(data)
                pdf_info = pdf2image.pdfinfo_from_bytes(pdf_bytes.getvalue())

            max_pages = pdf_info["Pages"]
            logger.info(f"PDF has {max_pages} pages to process")

            # Create a task queue to process pages in order
            pending_tasks = []
            completed_tasks = []
            all_page_contents = []
            next_page_to_yield = 1

            # Process pages with a sliding window, in batches
            for batch_start in range(1, max_pages + 1, self.vlm_batch_size):
                batch_end = min(
                    batch_start + self.vlm_batch_size - 1, max_pages
                )
                logger.debug(
                    f"Preparing batch of pages {batch_start}-{batch_end}/{max_pages}"
                )

                # Convert the batch of pages to images
                if isinstance(data, str):
                    images = pdf2image.convert_from_path(
                        data,
                        dpi=150,
                        first_page=batch_start,
                        last_page=batch_end,
                    )
                else:
                    pdf_bytes = BytesIO(data)
                    images = pdf2image.convert_from_bytes(
                        pdf_bytes.getvalue(),
                        dpi=150,
                        first_page=batch_start,
                        last_page=batch_end,
                    )

                # Create tasks — group pages per VLM call
                ppc = kwargs.get(
                    "vlm_pages_per_call", self.vlm_pages_per_call
                )
                if ppc > 1:
                    for i in range(0, len(images), ppc):
                        group_imgs = images[i : i + ppc]
                        group_nums = [
                            batch_start + i + j
                            for j in range(len(group_imgs))
                        ]
                        task = asyncio.create_task(
                            self.process_and_yield_multi(
                                group_imgs, group_nums
                            )
                        )
                        task.page_num = group_nums[0]
                        task._is_multi = True
                        pending_tasks.append(task)
                else:
                    for i, image in enumerate(images):
                        page_num = batch_start + i
                        task = asyncio.create_task(
                            self.process_and_yield(image, page_num)
                        )
                        task.page_num = page_num
                        task._is_multi = False
                        pending_tasks.append(task)

                # Check if any tasks have completed and yield them in order
                while pending_tasks:
                    # Get the first done task without waiting
                    done_tasks, pending_tasks_set = await asyncio.wait(
                        pending_tasks,
                        timeout=0.01,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if not done_tasks:
                        break

                    # Add completed tasks to our completed list
                    pending_tasks = list(pending_tasks_set)
                    completed_tasks.extend(iter(done_tasks))

                    # Sort completed tasks by page number
                    completed_tasks.sort(key=lambda t: t.page_num)

                    # Yield results in order
                    while (
                        completed_tasks
                        and completed_tasks[0].page_num == next_page_to_yield
                    ):
                        task = completed_tasks.pop(0)
                        result = await task
                        if isinstance(result, list):
                            for r in result:
                                all_page_contents.append(r)
                                yield r
                            next_page_to_yield += len(result)
                        else:
                            all_page_contents.append(result)
                            yield result
                            next_page_to_yield += 1

            # Wait for and yield any remaining tasks in order
            while pending_tasks:
                done_tasks, _ = await asyncio.wait(pending_tasks)
                completed_tasks.extend(done_tasks)
                pending_tasks = []

                # Sort and yield remaining completed tasks
                completed_tasks.sort(key=lambda t: t.page_num)

                # Yield results in order
                while (
                    completed_tasks
                    and completed_tasks[0].page_num == next_page_to_yield
                ):
                    task = completed_tasks.pop(0)
                    result = await task
                    if isinstance(result, list):
                        for r in result:
                            all_page_contents.append(r)
                            yield r
                        next_page_to_yield += len(result)
                    else:
                        all_page_contents.append(result)
                        yield result
                        next_page_to_yield += 1

            # Assemble full markdown preview from all pages
            if all_page_contents:
                all_page_contents.sort(
                    key=lambda x: x.get("page_number", 0)
                )
                full_md = "\n\n".join(
                    p.get("content", "")
                    for p in all_page_contents
                    if p.get("content")
                )
                if full_md:
                    yield {
                        "full_markdown": full_md,
                        "type": "markdown_preview",
                    }

                    llm_chunk_model = kwargs.get("llm_chunk_model")
                    if llm_chunk_model:
                        chunks, raw_output = (
                            await self._llm_chunk_selection(
                                full_md, llm_chunk_model
                            )
                        )
                        yield {
                            "full_markdown": raw_output,
                            "type": "chunked_markdown_preview",
                        }
                        for i, chunk_text in enumerate(chunks):
                            yield {
                                "content": chunk_text,
                                "metadata": {"chunk_order": i},
                                "pre_chunked": True,
                            }

            # Yield LLM usage signals for cost tracking
            if self._vlm_usage["prompt_tokens"] > 0:
                yield {
                    "type": "llm_usage_vlm",
                    "usage": dict(self._vlm_usage),
                    "model": getattr(self, '_vlm_override', None) or self.config.vlm or self.config.app.vlm,
                    "pages": max_pages,
                }
            if self._chunk_selection_usage["prompt_tokens"] > 0:
                yield {
                    "type": "llm_usage_chunk_selection",
                    "usage": dict(self._chunk_selection_usage),
                    "model": kwargs.get("llm_chunk_model", ""),
                }

            total_elapsed = time.perf_counter() - ingest_start
            logger.info(
                f"Completed PDF conversion in {total_elapsed:.2f} seconds"
            )

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise


class BasicPDFParser(AsyncParser[str | bytes]):
    """A parser for PDF data."""

    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
    ):
        self.database_provider = database_provider
        self.llm_provider = llm_provider
        self.config = config
        self.PdfReader = PdfReader

    async def ingest(
        self, data: str | bytes, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Ingest PDF data and yield text from each page."""
        if isinstance(data, str):
            raise ValueError("PDF data must be in bytes format.")
        pdf = self.PdfReader(BytesIO(data))
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text is not None:
                page_text = "".join(
                    filter(
                        lambda x: (
                            unicodedata.category(x)
                            in [
                                "Ll",
                                "Lu",
                                "Lt",
                                "Lm",
                                "Lo",
                                "Nl",
                                "No",
                            ]  # Keep letters and numbers
                            or "\u4e00" <= x <= "\u9fff"  # Chinese characters
                            or "\u0600" <= x <= "\u06ff"  # Arabic characters
                            or "\u0400" <= x <= "\u04ff"  # Cyrillic letters
                            or "\u0370" <= x <= "\u03ff"  # Greek letters
                            or "\u0e00" <= x <= "\u0e7f"  # Thai
                            or "\u3040" <= x <= "\u309f"  # Japanese Hiragana
                            or "\u30a0" <= x <= "\u30ff"  # Katakana
                            or "\uff00"
                            <= x
                            <= "\uffef"  # Halfwidth and Fullwidth Forms
                            or x in string.printable
                        ),
                        page_text,
                    )
                )  # Keep characters in common languages ; # Filter out non-printable characters
                yield page_text


class DoclingHybridPDFParser(AsyncParser[str | bytes]):
    """PDF parser using Docling extraction + HybridChunker (pre-chunked)."""

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
                delete=False, suffix=".pdf"
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


class DoclingMarkdownPDFParser(AsyncParser[str | bytes]):
    """PDF parser using Docling extraction → markdown (chunked by MarkdownChunker)."""

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
                delete=False, suffix=".pdf"
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


class PDFParserUnstructured(AsyncParser[str | bytes]):
    def __init__(
        self,
        config: IngestionConfig,
        database_provider: DatabaseProvider,
        llm_provider: CompletionProvider,
        ocr_provider: OCRProvider,
    ):
        self.database_provider = database_provider
        self.llm_provider = llm_provider
        self.config = config
        try:
            from unstructured.partition.pdf import partition_pdf

            self.partition_pdf = partition_pdf

        except ImportError as e:
            logger.error("PDFParserUnstructured ImportError :  ", e)

    async def ingest(
        self,
        data: str | bytes,
        partition_strategy: str = "hi_res",
        chunking_strategy="by_title",
    ) -> AsyncGenerator[str, None]:
        # partition the pdf
        elements = self.partition_pdf(
            file=BytesIO(data),
            partition_strategy=partition_strategy,
            chunking_strategy=chunking_strategy,
        )
        for element in elements:
            yield element.text
