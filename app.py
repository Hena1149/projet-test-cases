import streamlit as st
import pdfminer.high_level
import docx
from io import BytesIO

st.set_page_config(page_title="Extracteur de Texte", layout="wide")
st.title("📝 Extracteur de Texte (PDF/DOCX)")

# Fonctions d'extraction (votre code original)
def extract_text_from_pdf(pdf_path):
    """Extrait le texte d'un fichier PDF"""
    try:
        text = pdfminer.high_level.extract_text(pdf_path)
        if text is None:
            raise ValueError("Le texte n'a pas pu être extrait du PDF")
        return text
    except Exception as e:
        raise ValueError(f"Erreur PDF: {str(e)}")

def extract_text_from_docx(docx_path):
    """Extrait le texte d'un fichier Word"""
    try:
        doc = docx.Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        raise ValueError(f"Erreur DOCX: {str(e)}")

# Interface Streamlit
uploaded_file = st.file_uploader("Déposez un fichier PDF ou DOCX", type=["pdf", "docx"])

if uploaded_file:
    try:
        # Création d'un fichier temporaire en mémoire
        with BytesIO() as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file.seek(0)
            
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(temp_file)
            else:
                text = extract_text_from_docx(temp_file)
        
        # Affichage du résultat
        st.success("Texte extrait avec succès !")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label="📥 Télécharger le texte",
                data=text.encode('utf-8'),
                file_name="texte_extrait.txt",
                mime="text/plain"
            )
        with col2:
            st.expander("Afficher le texte complet").write(text[:5000] + ("..." if len(text) > 5000 else ""))
            
    except Exception as e:
        st.error(f"Erreur : {str(e)}")
else:
    st.info("Veuillez uploader un fichier pour commencer")

st.markdown("---")
st.caption("Appuyez sur 'R' pour rafraîchir l'application")
