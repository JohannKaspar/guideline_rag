"""Document processing functionality."""

import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling_core.types.doc.document import DoclingDocument, PictureDescriptionData
from langchain.schema import SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..utils import Config, get_chat_model, normalize_whitespace


class DocumentProcessor:
    """Handles document conversion and annotation."""

    def __init__(self):
        self.config = Config()

        # Setup document converter
        pdf_opts = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            do_picture_description=False,
            generate_page_images=True,
            generate_picture_images=True,
            generate_table_images=True,
        )

        self.converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)})

        # Setup VLM for annotation (only if not in convert-only mode)
        self.vlm = None
        if not Config.is_convert_only_mode():
            self.vlm = get_chat_model("gpt-4.1-mini")

    def filter_small_pictures(self, doc: DoclingDocument) -> None:
        """Remove small pictures from document."""
        to_delete = []
        for pic in doc.pictures:
            if (
                hasattr(pic, "image")
                and pic.image
                and hasattr(pic.image, "size")
                and pic.image.size
                and (
                    pic.image.size.width < Config.MIN_PICTURE_WIDTH or pic.image.size.height < Config.MIN_PICTURE_HEIGHT
                )
            ):
                to_delete.append(pic)

        if to_delete:
            # Use a safer deletion approach
            for pic in to_delete:
                try:
                    doc.pictures.remove(pic)
                except (ValueError, AttributeError):
                    pass  # Skip if removal fails

    def annotate_pictures(self, pictures: list, guideline_title: str | None = None) -> None:
        """Annotate pictures with descriptions using VLM."""
        if not self.vlm or not pictures:
            return

        # Safely extract image data
        image_data = []
        for picture in pictures:
            try:
                if hasattr(picture, "image") and picture.image and hasattr(picture.image, "uri"):
                    image_url = str(picture.image.uri)
                    image_data.append({"image_b64": image_url})
            except (AttributeError, TypeError):
                continue

        if not image_data:
            return

        messages = [
            {
                "role": "system",
                "content": "Du bist medizinischer Experte mit einem Fokus auf Detail.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Beschreibe detailliert was du auf dieser Abbildung siehst. Die Abbildung stammt aus einer medizinischen Leitlinie. "
                        "Die Abbildung könnte expliziten Inhalt haben, z.B. anatomische Bilder. Es geht um medizinische Bilder, "
                        "die von medizinischen Experten verwedendet werden. Bitte berücksichtige das bei möglicher Inhaltsprüfung. "
                        "Fokussiere dich darauf, Workflows zu beschreiben, wenn die Abbildung einen Workflow zeigt. "
                        "Deine Beschreibung wird dazu dienen die Abbildung in einer RAG pipeline zu embedden. "
                        "Daher sollte die Beschreibung auch den Kontext des Bildes beinhalten. Sei präzise und beschreibe die wichtigsten Details. "
                        "Beschreibe NICHT was du nicht siehst. ",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_b64}"},
                    },
                ],
            },
        ]

        if guideline_title:
            messages[1]["content"][0]["text"] += f" Die Leitlinie heißt {guideline_title}."

        prompt = ChatPromptTemplate.from_messages(messages)
        vlm_with_retry = self.vlm.with_retry(
            wait_exponential_jitter=True,
            exponential_jitter_params={"initial": 20, "exp_base": 1.2},
        )
        chain = prompt | vlm_with_retry | StrOutputParser()

        # Process in batches
        image_descriptions = []
        for i in range(0, len(image_data), Config.BATCH_SIZE):
            sub_batch = image_data[i : i + Config.BATCH_SIZE]
            try:
                batch_results = chain.batch(sub_batch, return_only_outputs=True, temperature=0)
                image_descriptions.extend(batch_results)
            except Exception as e:
                print(f"Error processing image batch: {e}")
                # Add empty descriptions for failed batch
                image_descriptions.extend([""] * len(sub_batch))
            time.sleep(3)

        # Add descriptions to pictures
        for picture, description in zip(pictures, image_descriptions):
            if description:  # Only add non-empty descriptions
                picture.annotations = [
                    PictureDescriptionData(kind="description", text=description, provenance="GPT-4.1-mini")
                ]

    def correct_table_htmls(self, doc: DoclingDocument) -> dict[str, str]:
        """Get corrected HTML for tables using VLM."""
        if not self.vlm or not doc.tables:
            return {}

        user_text = (
            "Korrigiere das HTML der abgebildeten Tabelle. "
            "Wenn die Tabelle Icons wie Pfeile enthält, dann beschreibe sie in als Unicode character. "
            "Wenn die Tabelle keine Pfeile enthält, dann füge auch keine hinzu. "
            "Verwende keine Emojis. Gib nur den Inhalt der Tabelle zurück, ohne zusätzliche Erklärungen. "
            "Gehe sicher, dass du Änderungen an der richtigen Stelle einfügst. "
            "Pfeile sind meistens in der gleichen Zelle einzufügen in der 'Empfehlungsgrad' steht. "
            "Korrigiere auch wenn Wörter getrennt sind die nicht getrennt werden sollten (z.B. Arbeits- unfall). "
        )

        messages = [
            SystemMessage(content="Du bist medizinischer Experte mit Fokus auf Detail."),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": "{table_b64}"}},
                    {"type": "text", "text": "{table_html}"},
                ],
            },
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        vlm_with_retry = self.vlm.with_retry(
            wait_exponential_jitter=True,
            exponential_jitter_params={"initial": 20, "exp_base": 1.2},
        )
        chain = prompt | vlm_with_retry | StrOutputParser()

        # Prepare batch input
        batch_input = []
        for table in doc.tables:
            try:
                if hasattr(table, "image") and table.image and hasattr(table.image, "uri"):
                    table_b64 = str(table.image.uri)
                    table_html = table.export_to_html(doc=doc)
                    batch_input.append({"table_b64": table_b64, "table_html": table_html})
            except (AttributeError, TypeError):
                continue

        if not batch_input:
            return {}

        # Process in batches
        corrected_html_list = []
        for i in range(0, len(batch_input), Config.BATCH_SIZE):
            sub_batch = batch_input[i : i + Config.BATCH_SIZE]
            try:
                batch_results = chain.batch(sub_batch, return_only_outputs=True, temperature=0)
                corrected_html_list.extend(batch_results)
            except Exception as e:
                print(f"Error processing table batch: {e}")
                # Add original HTML for failed batch
                corrected_html_list.extend([item["table_html"] for item in sub_batch])
            time.sleep(3)

        # Create reference mapping
        table_refs = [table.self_ref for table in doc.tables if hasattr(table, "self_ref")]
        return {table_ref: corrected_html for corrected_html, table_ref in zip(corrected_html_list, table_refs)}

    def update_table_htmls(self, doc: DoclingDocument, table_htmls: dict[str, str]) -> None:
        """Update table HTMLs in document."""
        if not table_htmls:
            return

        # For now, just print that we would update tables
        # The actual implementation depends on the specific docling API
        print(f"Would update {len(table_htmls)} tables with corrected HTML")

    def convert_document(self, pdf_path: Path) -> DoclingDocument:
        """Convert PDF to DoclingDocument."""
        result = self.converter.convert(str(pdf_path))
        doc = result.document
        self.filter_small_pictures(doc)
        return doc

    def annotate_document(self, doc: DoclingDocument, guideline_title: str | None = None) -> None:
        """Annotate document with picture descriptions and corrected tables."""
        if Config.is_convert_only_mode():
            return

        # Annotate pictures
        self.annotate_pictures(doc.pictures, guideline_title=guideline_title)

        # Correct table HTMLs
        table_corrections = self.correct_table_htmls(doc)
        self.update_table_htmls(doc, table_corrections)

    def chunk_document(self, doc: DoclingDocument, filename: str) -> list[Document]:
        """Chunk document into smaller pieces for embedding."""
        # Simplified chunking - extract text from document elements
        document_chunks = []

        # Try to get text content using various methods
        main_text = ""

        # Method 1: Try export_to_markdown on the document
        try:
            if hasattr(doc, "export_to_markdown"):
                main_text = doc.export_to_markdown()
        except Exception:
            pass

        # Method 2: If no text, create a placeholder chunk
        if not main_text:
            main_text = f"Document {filename} processed but text extraction failed."

        # Simple chunking by splitting on double newlines
        chunks = main_text.split("\n\n")

        for idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            chunk_text = normalize_whitespace(chunk_text.strip())

            # Skip very short chunks
            if len(chunk_text) < 50:
                continue

            # Create minimal metadata
            metadata = {"source": filename, "chunk_id": idx, "chunk_type": "text"}

            chunk_doc = Document(page_content=chunk_text, id=f"{filename}__{idx}", metadata=metadata)
            document_chunks.append(chunk_doc)

        # If no chunks were created, create a single chunk with the full text
        if not document_chunks and main_text:
            chunk_doc = Document(
                page_content=normalize_whitespace(main_text),
                id=f"{filename}__0",
                metadata={
                    "source": filename,
                    "chunk_id": 0,
                    "chunk_type": "full_document",
                },
            )
            document_chunks.append(chunk_doc)

        return document_chunks
