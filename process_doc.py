from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)
import base64
import io
from PIL import Image as PILImage

from typing import Iterable, Optional
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.types.doc.labels import DocItemLabel
from rich.console import Console
from rich.panel import Panel
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
)
from docling.datamodel.base_models import InputFormat
# Initialize the proxy client and models
proxy_client = get_proxy_client("gen-ai-hub")

model = ChatOpenAI(proxy_model_name="gpt-4o", proxy_client=proxy_client, temperature=0)
embedding_model = OpenAIEmbeddings(
    proxy_model_name="text-embedding-3-small", proxy_client=proxy_client
)

console = Console(
    width=200,  # for getting Markdown tables rendered nicely
)

EMBED_MODEL_ID = "jinaai/jina-embeddings-v3"

tokenizer: BaseTokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID, trust_remote_code=True),
)
chunker = HybridChunker(tokenizer=tokenizer)

def filter_small_pictures(doc):
    MIN_W, MIN_H = 50, 50

    valid_pics = []
    for pic in doc.pictures:
        w, h = pic.image.size.width, pic.image.size.height
        if w >= MIN_W and h >= MIN_H:
            valid_pics.append(pic)
    doc.pictures = valid_pics

    valid_refs = {doc.body.self_ref}
    for seq in (doc.texts, doc.tables, doc.pictures, doc.groups):
        valid_refs.update(item.self_ref for item in seq)

    all_groups = list(doc.groups) + [doc.body]
    for grp in all_groups:
        grp.children = [r for r in grp.children if r.cref in valid_refs]
        

pdf_opts = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    do_picture_description=False,  # set to FALSE, will do this later
    generate_page_images=True,  # Seiten‐Renders erzeugen, damit Crops möglich sind
    generate_picture_images=True,
    generate_table_images=True,
    
    # picture_description_options=PictureDescriptionVlmOptions(
    #     repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    #     prompt="Erklär, was auf dem Bild zu sehen ist. Sei präzise und beschreibe die wichtigsten Details.",
    #     generation_config={
    #         "max_new_tokens": 500,
    #     },
    #     picture_area_threshold=0.1,
    # ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pdf_opts,
        )
    }
)

INPUT_PDFS = [
    "pdfs/001_044l_S1Praevention-Therapie-systemischen-Lokalanaesthetika-Intoxikation-LAST_2025-01-abgelaufen.pdf",
    "pdfs/test/184-001l_S2e_Soziale-Teilhabe-Lebensqualitaet-stationaere-Altenhilfe-Pandemie_2025-03.pdf",
]

res = converter.convert(INPUT_PDFS[1])
doc = res.document