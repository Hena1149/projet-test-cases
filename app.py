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




# # 2ème code source

# import streamlit as st
# import re
# import string
# from io import BytesIO
# import pandas as pd
# import matplotlib.pyplot as plt

# # Configuration de base
# st.set_page_config(page_title="Cas de Test", layout="wide")
# st.title("📑 Génération automatique des cas de test à partir du CDC")

# # ----------------------------
# # PARTIE 1: Extraction du texte
# # ----------------------------
# def extract_text(uploaded_file):
#     """Fonction unifiée d'extraction pour PDF/DOCX"""
#     try:
#         file_bytes = uploaded_file.getvalue()
        
#         if uploaded_file.type == "application/pdf":
#             import pdfminer.high_level
#             with BytesIO(file_bytes) as f:
#                 text = pdfminer.high_level.extract_text(f)
#         elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#             import docx
#             with BytesIO(file_bytes) as f:
#                 doc = docx.Document(f)
#                 text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
#         else:
#             raise ValueError("Format non supporté")
            
#         return text if text.strip() else None
        
#     except Exception as e:
#         st.error(f"Erreur d'extraction : {str(e)}")
#         return None

# # ----------------------------
# # PARTIE 2: Nettoyage du texte
# # ----------------------------
# def simple_text_clean(text):
#     """Version allégée du nettoyage sans spaCy"""
#     text = text.lower()
#     text = re.sub(r"[^\w\s]", " ", text)  # Supprime la ponctuation
#     text = re.sub(r"\s+", " ", text)      # Supprime les espaces multiples
#     return text.strip()

# def show_word_frequencies(text, top_n=10):
#     """Analyse des fréquences sans sklearn"""
#     words = [word for word in text.split() if len(word) > 2]
#     freq = pd.Series(words).value_counts().head(top_n)
    
#     fig, ax = plt.subplots()
#     freq.plot.bar(ax=ax, color='skyblue')
#     plt.xticks(rotation=45)
#     st.pyplot(fig)
    
#     return freq

# # ----------------------------
# # INTERFACE UTILISATEUR
# # ----------------------------
# tab1, tab2 = st.tabs(["📤 Extraction", "🔍 Analyse"])

# with tab1:
#     uploaded_file = st.file_uploader("Téléversez un document", type=["pdf", "docx"])
    
#     if uploaded_file:
#         if st.button("Extraire le texte"):
#             with st.spinner("Extraction en cours..."):
#                 extracted_text = extract_text(uploaded_file)
                
#                 if extracted_text:
#                     st.session_state.text = extracted_text
#                     st.success("Texte extrait avec succès !")
#                     st.download_button(
#                         "💾 Télécharger le texte brut",
#                         data=extracted_text,
#                         file_name="texte_extrait.txt"
#                     )
                    
#                     with st.expander("Aperçu du texte"):
#                         st.text(extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""))

# with tab2:
#     if 'text' not in st.session_state:
#         st.warning("Veuillez d'abord extraire un texte")
#     else:
#         st.header("Analyse Simplifiée")
#         clean_text = simple_text_clean(st.session_state.text)
        
#         st.subheader("Fréquence des mots")
#         freq = show_word_frequencies(clean_text)
        
#         st.subheader("Données brutes")
#         st.dataframe(freq.rename("Occurences"))





#3ème Bloc

import streamlit as st
import re
import string
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pdfminer.high_level
import docx
import spacy

# Configuration de l'application
st.set_page_config(page_title="Cas de Test", layout="wide", page_icon="📑")

# ----------------------------
# FONCTIONS UTILITAIRES
# ----------------------------
def extract_text(uploaded_file):
    """Extrait le texte depuis PDF ou DOCX"""
    try:
        file_bytes = uploaded_file.getvalue()
        
        if uploaded_file.type == "application/pdf":
            with BytesIO(file_bytes) as f:
                text = pdfminer.high_level.extract_text(f)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            with BytesIO(file_bytes) as f:
                doc = docx.Document(f)
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            raise ValueError("Format non supporté")
            
        return text if text and text.strip() else None
        
    except Exception as e:
        st.error(f"Erreur d'extraction : {str(e)}")
        return None

def clean_text(text):
    """Nettoyage basique du texte"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # Supprime la ponctuation
    text = re.sub(r"\s+", " ", text)      # Espaces multiples -> simple
    return text.strip()

def calculate_frequencies(text):
    """Calcule les fréquences des mots"""
    words = [word for word in text.split() if len(word) > 2]  # Filtre mots courts
    return pd.Series(words).value_counts()

def generate_wordcloud(freq_dict, width=800, height=400, background_color="white", colormap="viridis"):
    """Génère un nuage de mots"""
    fig, ax = plt.subplots(figsize=(10, 5))
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=100
    ).generate_from_frequencies(freq_dict)
    
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# ----------------------------
# NOUVELLES FONCTIONS POUR L'EXTRACTION DE RÈGLES
# ----------------------------
@st.cache_resource
def load_spacy_model():
    """Charge le modèle spaCy une seule fois"""
    try:
        nlp = spacy.load("fr_core_news_md")
        return nlp
    except Exception as e:
        st.error(f"Erreur de chargement du modèle NLP: {str(e)}")
        return None

def extract_business_rules(text, nlp_model):
    """Nouvelle fonction optimisée pour l'extraction de règles"""
    # Extraction par motifs regex
    patterns = [
        r"(Si|Lorsqu['’]|Quand|Dès que|En cas de).*?(alors|doit|devra|est tenu de|nécessite|implique|entraîne|peut).*?\.",
        r"(L['’]utilisateur|Le client|Le système).*?(doit|devra|peut|est tenu de|ne peut pas).*?\."
    ]
    
    rules = set()
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        rules.update(match.group().strip() for match in matches)
    
    # Extraction NLP si modèle disponible
    if nlp_model:
        doc = nlp_model(text)
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in ["doit", "obligatoire", "interdit", "si ", "alors"]):
                rules.add(sent.text.strip())
    
    return sorted([clean_rule(rule) for rule in rules if len(rule.split()) > 3], key=len, reverse=True)

def clean_rule(rule_text):
    """Nettoie une règle de gestion"""
    rule_text = re.sub(r"\s+", " ", rule_text).strip()
    return rule_text if rule_text.endswith(".") else f"{rule_text}."

def create_rule_docx(rules):
    """Crée un document Word des règles"""
    doc = docx.Document()
    doc.add_heading('Règles Métier Extraites', level=1)
    
    for i, rule in enumerate(rules, 1):
        doc.add_paragraph(f"{i}. {rule}", style='ListBullet')
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# ----------------------------
# INTERFACE UTILISATEUR
# ----------------------------
st.title("📑 Génération automatique des cas de test à partir du CDC")
# Chargement du modèle NLP
nlp = load_spacy_model()
tab1, tab2, tab3, tab4 = st.tabs(["📤 Extraction", "🔍 Analyse", "☁️ WordCloud", "📑 Règles Métier"])

with tab1:
    st.header("Extraction de Texte")
    uploaded_file = st.file_uploader("Téléversez un document (PDF ou DOCX)", type=["pdf", "docx"])
    
    if uploaded_file and st.button("Extraire le texte"):
        with st.spinner("Extraction en cours..."):
            extracted_text = extract_text(uploaded_file)
            
            if extracted_text:
                st.session_state.text = extracted_text
                st.success("Texte extrait avec succès !")
                
                with st.expander("Aperçu du texte"):
                    st.text(extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""))

with tab2:
    st.header("Analyse Textuelle")
    
    if 'text' not in st.session_state:
        st.warning("Veuillez d'abord extraire un texte dans l'onglet 'Extraction'")
    else:
        with st.spinner("Nettoyage du texte..."):
            st.session_state.text_clean = clean_text(st.session_state.text)
            st.session_state.freq = calculate_frequencies(st.session_state.text_clean)
        
        st.subheader("Fréquence des mots")
        top_n = st.slider("Nombre de mots à afficher", 5, 50, 20)
        st.dataframe(st.session_state.freq.head(top_n))

with tab3:
    st.header("Visualisation WordCloud")
    
    if 'text_clean' not in st.session_state:
        st.warning("Veuillez d'abord analyser un texte dans l'onglet 'Analyse'")
    else:
        with st.expander("Paramètres avancés"):
            col1, col2 = st.columns(2)
            with col1:
                width = st.slider("Largeur", 400, 1200, 800, key="wc_width")
                height = st.slider("Hauteur", 200, 800, 400, key="wc_height")
            with col2:
                bg_color = st.color_picker("Couleur de fond", "#FFFFFF", key="wc_bg")
                colormap = st.selectbox("Palette", ["viridis", "plasma", "inferno", "magma", "cividis"], key="wc_cmap")
        
        if st.button("Générer le WordCloud"):
            freq_dict = st.session_state.freq.to_dict()
            fig = generate_wordcloud(
                freq_dict,
                width=width,
                height=height,
                background_color=bg_color,
                colormap=colormap
            )
            
            st.pyplot(fig)
            
            # Téléchargement
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            st.download_button(
                label="💾 Télécharger l'image",
                data=img_buffer.getvalue(),
                file_name="wordcloud.png",
                mime="image/png"
            )

    with tab4:
        st.header("Extraction des Règles Métier")
        st.warning("ℹ️ Vous devez d'abord extraire du texte dans l'onglet précédent")
        
        if 'extracted_text' in st.session_state and nlp:
            if st.button("Analyser les règles", type="primary"):
                with st.spinner("Recherche des règles métier..."):
                    rules = extract_business_rules(st.session_state.extracted_text, nlp)
                    
                    if rules:
                        st.session_state.rules = rules
                        st.success(f"{len(rules)} règles identifiées !")
                        
                        # Affichage avec pagination
                        page_size = 10
                        total_pages = (len(rules) + page_size - 1) // page_size
                        page = st.number_input("Page", 1, total_pages, 1)
                        
                        start_idx = (page - 1) * page_size
                        end_idx = min(start_idx + page_size, len(rules))
                        
                        for i in range(start_idx, end_idx):
                            st.markdown(f"**Règle {i+1}**")
                            st.info(rules[i])
                        
                        # Téléchargement
                        docx_file = create_rule_docx(rules)
                        st.download_button(
                            "💾 Exporter les règles (DOCX)",
                            data=docx_file,
                            file_name="regles_metier.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    else:
                        st.warning("Aucune règle métier détectée")
        elif not nlp:
            st.error("Le modèle NLP n'est pas disponible")

# ----------------------------
# PIED DE PAGE
# ----------------------------
st.markdown("---")
st.caption("Application développée avec Streamlit - Mise à jour : %s" % pd.Timestamp.now().strftime("%d/%m/%Y"))
