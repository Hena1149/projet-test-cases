import streamlit as st
import os
from extraction import (
    extract_text,
    extract_rules_with_regex,
    extract_rules_with_nlp
)
from nlp_processing import (
    nettoyer_texte,
    generer_wordcloud,
    extraire_phrases_cles,
    comparer_ressemblances
)
from nlp_processing import clean_text, clean_short_rules
from test_cases import generate_test_cases, save_test_cases_to_docx
import tempfile

# Configuration de la page
st.set_page_config(
    page_title="Générateur de Cas de Test",
    page_icon="🧪",
    layout="wide"
)

def main():
    st.title("🛠️ Générateur Automatique de Cas de Test")
    st.markdown("""
    **Chargez votre document (PDF/DOCX) pour extraire les règles de gestion et générer automatiquement des cas de test.**
    """)

    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("Paramètres")
        min_rule_length = st.slider("Longueur minimale des règles", 5, 50, 10)
        similarity_threshold = st.slider("Seuil de similarité", 0.1, 1.0, 0.6)
        generate_pdc = st.checkbox("Générer des PDC automatiques", True)

    # Onglets principaux
    tab1, tab2, tab3 = st.tabs(["📤 Chargement", "🔍 Règles extraites", "🧪 Cas de test"])

    with tab1:
        st.subheader("Charger vos documents")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Document principal (CDC)")
            cdc_file = st.file_uploader("Charger CDC", type=["pdf", "docx"])
        
        with col2:
            st.markdown("### Matrice PDC (Optionnel)")
            pdc_file = st.file_uploader("Charger PDC", type=["txt", "docx"])

    # Traitement lorsque fichiers chargés
    if cdc_file:
        with tab2:
            with st.spinner("Extraction des règles en cours..."):
                try:
                    # Sauvegarde temporaire du fichier
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(cdc_file.name)[1]) as tmp:
                        tmp.write(cdc_file.getbuffer())
                        tmp_path = tmp.name

                    # Extraction du texte
                    text = extract_text(tmp_path)
                    
                    # Extraction des règles
                    rules_regex = extract_rules_with_regex(text)
                    rules_nlp = extract_rules_with_nlp(text)
                    all_rules = list(set(rules_regex + rules_nlp))
                    all_rules = clean_short_rules(all_rules, min_length=min_rule_length)
                    
                    st.success(f"{len(all_rules)} règles extraites !")
                    
                    # Affichage des règles
                    for i, rule in enumerate(all_rules, 1):
                        st.markdown(f"**Règle {i}**")
                        st.info(rule)
                    
                    # Sauvegarde en session
                    st.session_state['rules'] = all_rules
                    
                except Exception as e:
                    st.error(f"Erreur lors de l'extraction : {str(e)}")
                finally:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        # Traitement des PDC si fourni
        pdc_text = None
        if pdc_file:
            with st.spinner("Traitement du PDC..."):
                try:
                    if pdc_file.name.endswith('.txt'):
                        pdc_text = pdc_file.read().decode("utf-8")
                    elif pdc_file.name.endswith('.docx'):
                        doc = docx.Document(pdc_file)
                        pdc_text = "\n".join([p.text for p in doc.paragraphs])
                    
                    st.session_state['pdc_text'] = pdc_text
                    st.success("PDC chargé avec succès !")
                except Exception as e:
                    st.error(f"Erreur PDC : {str(e)}")

        # Génération des cas de test
        with tab3:
            if 'rules' in st.session_state:
                st.subheader("Génération des cas de test")
                
                if st.button("Générer les cas de test", type="primary"):
                    with st.spinner("Génération en cours..."):
                        try:
                            pdc = st.session_state.get('pdc_text', None)
                            test_cases = generate_test_cases(
                                st.session_state['rules'],
                                exigences_pdc=pdc.split("\n") if pdc else None,
                                generate_pdc=generate_pdc
                            )
                            
                            st.success(f"{len(test_cases)} cas de test générés !")
                            
                            # Affichage d'un aperçu
                            st.dataframe(
                                pd.DataFrame(test_cases, columns=["ID", "Description", "Préconditions", "Étapes", "Résultat", "Statut"]),
                                height=300
                            )
                            
                            # Bouton de téléchargement
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                                save_test_cases_to_docx(test_cases, tmp.name)
                                with open(tmp.name, "rb") as f:
                                    st.download_button(
                                        "📥 Télécharger les cas de test",
                                        f,
                                        file_name="cas_de_test_generes.docx"
                                    )
                        except Exception as e:
                            st.error(f"Erreur génération : {str(e)}")

if __name__ == "__main__":
    main()