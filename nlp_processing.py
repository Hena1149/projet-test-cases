import re
import spacy
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

@st.cache_resource
def load_spacy_model():
    return spacy.load("fr_core_news_md")

nlp = load_spacy_model()

# Téléchargements NLTK (un seul appel)
nltk.download('punkt')
nltk.download('stopwords')

# Chargement du modèle spaCy
nlp = spacy.load("fr_core_news_md")

# Configuration des stopwords
stopwords_fr = set(stopwords.words('french'))
mots_inutiles = {
    "les", "des", "aux", "une", "dans", "sur", "par", "avec", "pour", 
    "ce", "ces", "ses", "leur", "leurs", "sous", "comme", "plus", 
    "tous", "tout", "sans", "non", "peu", "donc", "ainsi", "même", 
    "alors", "or"
}

def nettoyer_texte(text):
    """
    Nettoie et prétraite le texte avec spaCy
    Args:
        text (str): Texte brut à nettoyer
    Returns:
        str: Texte nettoyé et lemmatisé
    """
    # Nettoyage de base
    text = text.lower()
    text = re.sub(r"[\n\t\xa0«»\"']", " ", text)
    text = re.sub(r"\b[lLdDjJcCmM]'(\w+)", r"\1", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Traitement NLP avec spaCy
    doc = nlp(text)
    
    mots_nets = [
        token.lemma_ for token in doc
        if token.text not in stopwords_fr
        and token.lemma_ not in mots_inutiles
        and len(token.lemma_) > 2
        and not token.is_digit
    ]

    return " ".join(mots_nets)

def generer_wordcloud(text, largeur=800, hauteur=400):
    """
    Génère un nuage de mots à partir du texte nettoyé
    Args:
        text (str): Texte à visualiser
        largeur (int): Largeur de l'image
        hauteur (int): Hauteur de l'image
    Returns:
        matplotlib.figure.Figure: Figure contenant le wordcloud
    """
    # Nettoyage du texte
    text_clean = nettoyer_texte(text)
    
    # Calcul des fréquences
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text_clean])
    freq = pd.DataFrame(X.toarray(), 
                       columns=vectorizer.get_feature_names_out(), 
                       index=["Texte"]).loc["Texte"].sort_values(ascending=False)
    
    # Génération du wordcloud
    wordcloud = WordCloud(
        width=largeur,
        height=hauteur,
        background_color="white",
        colormap="viridis"
    ).generate_from_frequencies(freq.to_dict())
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    
    return fig

def extraire_phrases_cles(text, n_phrases=5):
    """
    Extrait les phrases les plus importantes du texte
    Args:
        text (str): Texte brut
        n_phrases (int): Nombre de phrases à extraire
    Returns:
        list: Liste des phrases clés
    """
    doc = nlp(text)
    phrases = [sent.text.strip() for sent in doc.sents]
    
    # Score basé sur la présence de mots importants
    scores = []
    for phrase in phrases:
        score = 0
        mots = [token.lemma_ for token in nlp(phrase)]
        for mot in mots:
            if (mot not in stopwords_fr and 
                mot not in mots_inutiles and 
                len(mot) > 3):
                score += 1
        scores.append(score / len(mots) if mots else 0)
    
    # Tri des phrases par score
    phrases_triees = [p for _, p in sorted(zip(scores, phrases), reverse=True)]
    
    return phrases_triees[:n_phrases]

def comparer_ressemblances(texte1, texte2):
    """
    Compare deux textes et retourne un score de similarité
    Args:
        texte1 (str): Premier texte
        texte2 (str): Deuxième texte
    Returns:
        float: Score de similarité entre 0 et 1
    """
    doc1 = nlp(texte1)
    doc2 = nlp(texte2)
    return doc1.similarity(doc2)

def tokenize_and_lemmatize(text):
    """
    Tokenisation et lemmatisation avancée avec spaCy
    Args:
        text (str): Texte à traiter
    Returns:
        list: Liste de tokens lemmatisés
    """
    doc = nlp(text)
    return [
        token.lemma_ for token in doc
        if not token.is_stop 
        and not token.is_punct 
        and not token.is_space
    ]