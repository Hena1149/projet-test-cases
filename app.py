# import streamlit as st
# import pdfminer.high_level
# import docx
# from io import BytesIO
# import pandas as pd
# st.set_page_config(page_title="Extracteur de Texte", layout="wide")
# st.title("📝 Extracteur de Texte (PDF/DOCX)")

# # Fonctions d'extraction (votre code original)
# def extract_text_from_pdf(pdf_path):
#     """Extrait le texte d'un fichier PDF"""
#     try:
#         text = pdfminer.high_level.extract_text(pdf_path)
#         if text is None:
#             raise ValueError("Le texte n'a pas pu être extrait du PDF")
#         return text
#     except Exception as e:
#         raise ValueError(f"Erreur PDF: {str(e)}")

# def extract_text_from_docx(docx_path):
#     """Extrait le texte d'un fichier Word"""
#     try:
#         doc = docx.Document(docx_path)
#         return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
#     except Exception as e:
#         raise ValueError(f"Erreur DOCX: {str(e)}")

# # Interface Streamlit
# uploaded_file = st.file_uploader("Déposez un fichier PDF ou DOCX", type=["pdf", "docx"])

# if uploaded_file:
#     try:
#         # Création d'un fichier temporaire en mémoire
#         with BytesIO() as temp_file:
#             temp_file.write(uploaded_file.getvalue())
#             temp_file.seek(0)
            
#             if uploaded_file.type == "application/pdf":
#                 text = extract_text_from_pdf(temp_file)
#             else:
#                 text = extract_text_from_docx(temp_file)
        
#         # Affichage du résultat
#         st.success("Texte extrait avec succès !")
        
#         col1, col2 = st.columns([1, 3])
#         with col1:
#             st.download_button(
#                 label="📥 Télécharger le texte",
#                 data=text.encode('utf-8'),
#                 file_name="texte_extrait.txt",
#                 mime="text/plain"
#             )
#         with col2:
#             st.expander("Afficher le texte complet").write(text[:5000] + ("..." if len(text) > 5000 else ""))
            
#     except Exception as e:
#         st.error(f"Erreur : {str(e)}")
# else:
#     st.info("Veuillez uploader un fichier pour commencer")

# st.markdown("---")
# st.caption("Appuyez sur 'R' pour rafraîchir l'application")







# # # Fonction de nettoyage (optimisée pour Streamlit)
# # def nettoyer_texte(text, nlp, stopwords_fr, mots_inutiles):
# #     """Version simplifiée pour Streamlit"""
# #     text = text.lower()
# #     text = re.sub(r"[\n\t\xa0«»\"']", " ", text)
# #     text = re.sub(r"\b[lLdDjJcCmM]'(\w+)", r"\1", text)
# #     text = text.translate(str.maketrans("", "", string.punctuation))
    
# #     doc = nlp(text)
# #     return " ".join([
# #         token.lemma_ for token in doc
# #         if (token.text not in stopwords_fr and 
# #             token.lemma_ not in mots_inutiles and 
# #             len(token.lemma_) > 2 and 
# #             not token.is_digit)
# #     ])

# # # Interface Streamlit
# # st.title("🔠 Analyse Textuelle")
# # st.subheader("Nettoyage et Fréquences des Mots")

# # if 'text' not in st.session_state:
# #     st.warning("Veuillez d'abord extraire un texte dans l'onglet précédent")
# # else:
# #     # Chargement des ressources NLP une seule fois
# #     if 'nlp' not in st.session_state:
# #         import spacy
# #         st.session_state.nlp = spacy.load("fr_core_news_md")
# #         st.session_state.stopwords_fr = set(stopwords.words('french'))
# #         st.session_state.mots_inutiles = {
# #             "les", "des", "aux", "une", "dans", "sur", "par", "avec", "pour",
# #             "ce", "ces", "ses", "leur", "leurs", "sous", "comme", "plus"
# #         }
    
# #     # Options utilisateur
# #     min_word_length = st.slider("Longueur minimale des mots", 2, 6, 3)
# #     top_n = st.slider("Nombre de mots à afficher", 5, 30, 10)

# #     # Traitement
# #     with st.spinner("Nettoyage du texte..."):
# #         text_clean = nettoyer_texte(
# #             st.session_state.text,
# #             st.session_state.nlp,
# #             st.session_state.stopwords_fr,
# #             st.session_state.mots_inutiles
# #         )
        
# #         # Calcul des fréquences
# #         vectorizer = CountVectorizer()
# #         X = vectorizer.fit_transform([text_clean])
# #         df_bow = pd.DataFrame(
# #             X.toarray(), 
# #             columns=vectorizer.get_feature_names_out(), 
# #             index=["Texte"]
# #         )
# #         freq = df_bow.loc["Texte"].sort_values(ascending=False).head(top_n)

# #     # Affichage des résultats
# #     col1, col2 = st.columns(2)
    
# #     with col1:
# #         st.markdown("### Mots les plus fréquents")
# #         st.dataframe(freq)
        
# #     with col2:
# #         st.markdown("### Nuage de mots")
# #         fig, ax = plt.subplots()
# #         wc = WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(df_bow.iloc[0].to_dict())
# #         ax.imshow(wc, interpolation='bilinear')
# #         ax.axis("off")
# #         st.pyplot(fig)

# #     # Sauvegarde pour les étapes suivantes
# #     st.session_state.text_clean = text_clean
# #     st.success("Analyse terminée !")


import streamlit as st
import re
import string
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt

# Configuration de base
st.set_page_config(page_title="Cas de Test", layout="wide")
st.title("📑 Génération automatique des cas de test à partir du CDC")

# ----------------------------
# PARTIE 1: Extraction du texte
# ----------------------------
def extract_text(uploaded_file):
    """Fonction unifiée d'extraction pour PDF/DOCX"""
    try:
        file_bytes = uploaded_file.getvalue()
        
        if uploaded_file.type == "application/pdf":
            import pdfminer.high_level
            with BytesIO(file_bytes) as f:
                text = pdfminer.high_level.extract_text(f)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            import docx
            with BytesIO(file_bytes) as f:
                doc = docx.Document(f)
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            raise ValueError("Format non supporté")
            
        return text if text.strip() else None
        
    except Exception as e:
        st.error(f"Erreur d'extraction : {str(e)}")
        return None

# ----------------------------
# PARTIE 2: Nettoyage du texte
# ----------------------------
def simple_text_clean(text):
    """Version allégée du nettoyage sans spaCy"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # Supprime la ponctuation
    text = re.sub(r"\s+", " ", text)      # Supprime les espaces multiples
    return text.strip()

def show_word_frequencies(text, top_n=10):
    """Analyse des fréquences sans sklearn"""
    words = [word for word in text.split() if len(word) > 2]
    freq = pd.Series(words).value_counts().head(top_n)
    
    fig, ax = plt.subplots()
    freq.plot.bar(ax=ax, color='skyblue')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    return freq

# ----------------------------
# INTERFACE UTILISATEUR
# ----------------------------
tab1, tab2 = st.tabs(["📤 Extraction", "🔍 Analyse"])

with tab1:
    uploaded_file = st.file_uploader("Téléversez un document", type=["pdf", "docx"])
    
    if uploaded_file:
        if st.button("Extraire le texte"):
            with st.spinner("Extraction en cours..."):
                extracted_text = extract_text(uploaded_file)
                
                if extracted_text:
                    st.session_state.text = extracted_text
                    st.success("Texte extrait avec succès !")
                    st.download_button(
                        "💾 Télécharger le texte brut",
                        data=extracted_text,
                        file_name="texte_extrait.txt"
                    )
                    
                    with st.expander("Aperçu du texte"):
                        st.text(extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""))

with tab2:
    if 'text' not in st.session_state:
        st.warning("Veuillez d'abord extraire un texte")
    else:
        st.header("Analyse Simplifiée")
        clean_text = simple_text_clean(st.session_state.text)
        
        st.subheader("Fréquence des mots")
        freq = show_word_frequencies(clean_text)
        
        st.subheader("Données brutes")
        st.dataframe(freq.rename("Occurences"))
