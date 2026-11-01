import streamlit as st

from frontend.api_client import upload_csv

st.title("Importer des transactions")
st.write("Importez un fichier CSV pour vérifier sa structure et l’ajouter à l’espace de travail.")
st.caption("Colonne requise : amt ou amount. Import en flux par blocs; limite par défaut : 1 Gio.")
uploaded_file = st.file_uploader("Fichier de transactions", type=["csv"])
if uploaded_file and st.button("Importer le fichier", type="primary"):
    with st.spinner("Validation et import en cours…"):
        try:
            result = upload_csv(uploaded_file.name, uploaded_file.getvalue())
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Impossible de joindre l’API. Vérifiez qu’elle est démarrée.")
        else:
            st.success("Fichier importé avec succès.")
            st.metric("Lignes", result["row_count"])
            st.metric("Cellules manquantes", result["missing_cells"])
            st.write("Colonnes détectées : " + ", ".join(result["columns"]))
st.divider()
st.caption("Les étiquettes éventuelles du fichier sont conservées dans le CSV mais ne sont pas utilisées pour produire des prédictions.")
