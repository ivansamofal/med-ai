"""Loaders for the three knowledge-base sources: guideline PDFs, the
drug-interaction dict (JSON), and lab reference ranges (CSV). Each loader
returns plain LlamaIndex `Document`s tagged with `source_type`/`source_title`
metadata so retrieved passages can always be cited back to their source.
"""

import csv
import json
from pathlib import Path

from llama_index.core import Document


def load_guideline_documents(directory: Path) -> list[Document]:
    from llama_index.readers.file import PDFReader

    reader = PDFReader()
    documents: list[Document] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        fallback_title = pdf_path.stem.replace("_", " ").title()
        for doc in reader.load_data(file=pdf_path):
            first_line = doc.text.strip().splitlines()[0].strip() if doc.text.strip() else ""
            doc.metadata["source_type"] = "guideline"
            doc.metadata["source_title"] = first_line or fallback_title
            documents.append(doc)
    return documents


def load_drug_interaction_documents(path: Path) -> list[Document]:
    entries = json.loads(path.read_text())
    documents = []
    for entry in entries:
        text = (
            f"{entry['drug_a']} + {entry['drug_b']} ({entry['severity']} interaction): "
            f"{entry['description']}"
        )
        documents.append(
            Document(
                text=text,
                metadata={
                    "source_type": "drug_interaction",
                    "source_title": f"{entry['drug_a']} / {entry['drug_b']} interaction",
                    "drug_a": entry["drug_a"].lower(),
                    "drug_b": entry["drug_b"].lower(),
                    "severity": entry["severity"],
                },
            )
        )
    return documents


def load_icd10_documents(path: Path) -> list[Document]:
    documents = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        code, _, description = line.partition(" - ")
        documents.append(
            Document(
                text=f"ICD-10 {code}: {description}",
                metadata={
                    "source_type": "icd10_code",
                    "source_title": f"ICD-10 {code}",
                    "icd10_code": code,
                },
            )
        )
    return documents


def load_medical_reference_documents(path: Path) -> list[Document]:
    sections = path.read_text().strip().split("\n\n")
    documents = []
    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue
        title, body = lines[0].strip(), "\n".join(lines[1:]).strip()
        documents.append(
            Document(
                text=f"{title}\n{body}",
                metadata={"source_type": "medical_reference", "source_title": title},
            )
        )
    return documents


def load_lab_reference_range_documents(path: Path) -> list[Document]:
    documents = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            text = (
                f"{row['test_name']} ({row['test_code']}) reference range: "
                f"{row['low']}-{row['high']} {row['unit']}. Critical if below "
                f"{row['critical_low']} or above {row['critical_high']} {row['unit']}."
            )
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source_type": "reference_range",
                        "source_title": f"{row['test_name']} reference range",
                        "test_code": row["test_code"],
                    },
                )
            )
    return documents
