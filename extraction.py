import pdfminer.high_level
import docx
import re

from nlp_processing import nettoyer_texte
def extract_text(file_path):
    """Extrait le texte selon le type de fichier"""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    raise ValueError("Format non supporté")

def extract_text_from_pdf(pdf_path):
    """Extrait le texte d'un PDF"""
    return pdfminer.high_level.extract_text(pdf_path)

def extract_text_from_docx(docx_path):
    """Extrait le texte d'un DOCX"""
    doc = docx.Document(docx_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_rules_with_regex(text):
    """Extraction des règles avec expressions régulières"""
    patterns = [
        r"(Si|Lorsqu’|Quand|Dès que|En cas de).*?(alors|doit|devra|est tenu de).*?\.",
        r"(Le système|L'utilisateur|Le client).*?(doit|devra|peut|est tenu de).*?\."
    ]
    return [m[0] for p in patterns for m in re.findall(p, text, re.IGNORECASE)]