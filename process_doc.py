from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    HTMLFormatOption,
    DocumentStream,
)
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.document import PictureDescriptionData
from langchain.schema import SystemMessage
from docling_core.types.doc.document import DoclingDocument
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
import os


user_name = os.path.expanduser("~")
if "joli13" in user_name:
    CONVERT_ONLY = True
    CONVERTED_DIR = "work/guideline_rag/converted/"
    ANNOTATED_DIR = "work/guideline_rag/annotated/"
    PDF_DIR = "work/guideline_rag/pdfs/"
    accel_opts = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA
    )
    gpt4_1_mini = None
else:
    CONVERT_ONLY = False
    CONVERTED_DIR = "converted/"
    ANNOTATED_DIR = "annotated/"
    PDF_DIR = "pdfs/"
    accel_opts = AcceleratorOptions()

    proxy_client = get_proxy_client("gen-ai-hub")
    gpt4_1_mini = ChatOpenAI(
        proxy_model_name="gpt-4.1-mini", proxy_client=proxy_client, temperature=0
    )



pdf_opts = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    do_picture_description=False,  # set to FALSE, will do this later
    generate_page_images=True,  # Seiten‐Renders erzeugen, damit Crops möglich sind
    generate_picture_images=True,
    generate_table_images=True,
    accelerator_options = accel_opts
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pdf_opts,
        )
    }
)


def filter_small_pictures(doc, min_w=75, min_h=75):
    to_delete = [
        pic
        for pic in doc.pictures
        if pic.image.size.width < min_w or pic.image.size.height < min_h
    ]
    if to_delete:
        doc.delete_items(node_items=to_delete)


def annotate_pictures_remote(pictures, vlm, guideline_title=None):
    image_data = [{"image_b64": str(picture.image.uri._url)} for picture in pictures]

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

    vlm_with_retry = vlm.with_retry(
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 20, "exp_base": 1.2},
    )

    chain = prompt | vlm_with_retry | StrOutputParser()

    # print(f"Annotating {len(image_data)} images...")
    image_descriptions = chain.batch(
        image_data,
        return_only_outputs=True,
        temperature=0,
    )

    if any(len(r) == 0 for r in image_descriptions):
        print("WARNING: Some responses are empty.")
    if any(len(r) > 3000 for r in image_descriptions):
        print("WARNING: Some responses are very long.")

    for picture, image_description in zip(pictures, image_descriptions):
        picture.annotations = [
            PictureDescriptionData(
                kind="description", text=image_description, provenance="GPT-4.1-mini"
            )
        ]


def get_correct_table_htmls(doc_object: DoclingDocument, vlm) -> dict[str, str]:
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
                {
                    "type": "image_url",
                    "image_url": {"url": "{table_b64}"},
                },
                {"type": "text", "text": "{table_html}"},
            ],
        },
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    vlm_with_retry = vlm.with_retry(
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 20, "exp_base": 1.2},
    )

    chain = prompt | vlm_with_retry | StrOutputParser()

    batch_input = []
    for table in doc_object.tables:
        table_b64 = str(table.image.uri._url)
        table_html = table.export_to_html(doc=doc_object)
        batch_input.append({"table_b64": table_b64, "table_html": table_html})

    if not batch_input:
        return {}

    # print(f"Checking {len(batch_input)} tables...")
    corrected_html_list = chain.batch(
        batch_input,
        return_only_outputs=True,
        temperature=0,
    )

    table_refs = [table.self_ref for table in doc_object.tables]

    ref_html = {
        table_ref: corrected_html
        for corrected_html, table_ref in zip(corrected_html_list, table_refs)
    }

    return ref_html


def update_table_htmls(doc_object: DoclingDocument, table_htmls: dict[str, str]):
    # Converter für HTML konfigurieren
    converter_html = DocumentConverter(
        allowed_formats=[InputFormat.HTML],
        format_options={InputFormat.HTML: HTMLFormatOption()},
    )

    for table_ref, corrected_html in list(table_htmls.items()):
        html_bytes = corrected_html.encode("utf-16")
        doc_stream = DocumentStream(
            stream=BytesIO(html_bytes),
            name=f"table_{table_ref}.html",
            content_type="text/html; charset=utf-16",
        )

        result = converter_html.convert(
            source=doc_stream,
        )
        new_doc: DoclingDocument = result.document

        new_table = new_doc.tables[0]

        idx = next(
            i for i, t in enumerate(doc_object.tables) if t.self_ref == table_ref
        )
        old_table = doc_object.tables[idx]
        new_table.self_ref = old_table.self_ref
        new_table.image = old_table.image
        doc_object.tables[idx] = new_table


def process_doc(pdf_path: str, file_hash, gpt4_1_mini=None):
    if file_hash:
        doc = DoclingDocument.load_from_json(f"{file_hash}.json")
    else:
        res = converter.convert(pdf_path)
        doc = res.document

        filter_small_pictures(doc)

        file_path_converted = Path(
            Path("converted/") / f"{doc.origin.binary_hash}.json"
        )
        doc.save_as_json(file_path_converted)

    if not CONVERT_ONLY:
        annotate_pictures_remote(
            doc.pictures, vlm=gpt4_1_mini
        )  # TODO add guideline title

        table_html_corrections = get_correct_table_htmls(doc, vlm=gpt4_1_mini)
        update_table_htmls(doc, table_html_corrections)

        file_path_annotated = Path(
            Path("annotated/") / f"{doc.origin.binary_hash}.json"
        )
        doc.save_as_json(file_path_annotated)

        os.remove(file_path_converted)

    return doc.origin.binary_hash


if __name__ == "__main__":
    import json
    import os
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler("convert.log"),  # Output to log file
        ],
    )

    with open("doc_hashes.csv", "r") as f:
        lines = [tuple(l.split(",")) for l in f.readlines()[1:]]
        file_hashes = {file_name.strip(): hash.strip() for (hash, file_name) in lines}
        hash_files = {hash.strip(): file_name.strip() for (hash, file_name) in lines}

    todo_pdfs = []

    annotated_pdfs = [
        hash_files.get(os.path.splitext(file)[0])
        for file in os.listdir(ANNOTATED_DIR)
        if file.endswith(".json")
    ]
    converted_pdfs = [
        hash_files.get(os.path.splitext(file)[0])
        for file in os.listdir(CONVERTED_DIR)
        if file.endswith(".json")
    ]
    processed_pdfs = converted_pdfs + annotated_pdfs

    awmf_guidelines = json.load(open("awmf.json", "r"))
    for guideline in awmf_guidelines["records"]:
        for link in guideline["links"]:
            if link["type"] == "longVersion":
                file_name = link["media"]
                if CONVERT_ONLY:
                    if file_name not in processed_pdfs:
                        todo_pdfs.append(file_name)
                else:
                    if file_name not in annotated_pdfs:
                        todo_pdfs.append(file_name)

    for file_name in tqdm(todo_pdfs):
        try:
            pdf_path = os.path.join(PDF_DIR, file_name)
            logging.info(f"Processing: {pdf_path}")
            file_hash = file_hashes.get(pdf_path)
            if file_hash and CONVERT_ONLY:
                raise ValueError(
                    f"File {pdf_path} already converted. Please remove the file or set CONVERT_ONLY to False."
                )
            doc_hash = process_doc(
                pdf_path=pdf_path, file_hash=file_hash, gpt4_1_mini=gpt4_1_mini
            )
            with open("doc_hashes.csv", "a") as f:
                f.write(f"\n{doc_hash},{file_name}")
            processed_pdfs.append(file_name)
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")
            with open("doc_hashes.csv", "a") as f:
                f.write(f"\nERROR,{file_name}")
            continue
