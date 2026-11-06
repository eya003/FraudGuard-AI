"""Streamlit renderers for exact full-dataset aggregates and real sample rows."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import get_public_demo_sample

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = go = None


def _layout(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=15, r=15, t=45, b=25))
    return fig


def _amount_chart(summary):
    data = summary["amount_histogram"]
    edges = data["edges"]
    figure = go.Figure()
    for key, label, color in (("normal", "Normales", "#25D0C2"), ("fraud", "Fraudes", "#F07178")):
        counts = data[key]
        widths = [edges[i + 1] - edges[i] for i in range(len(counts))]
        denominator = sum(counts)
        figure.add_trace(go.Bar(
            x=[(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))],
            y=[count / denominator / widths[i] if denominator and widths[i] else 0
               for i, count in enumerate(counts)],
            width=widths,
            name=label, marker_color=color, opacity=0.65,
        ))
    figure.update_layout(title="Montants · dataset complet", barmode="overlay",
                         xaxis_title="Montant", yaxis_title="Densité")
    return _layout(figure)


def _distance_chart(summary):
    data = summary["distance_histogram"]
    edges = data["edges"]
    figure = go.Figure()
    for key, label, color in (("normal", "Normales", "#25D0C2"), ("fraud", "Fraudes", "#F07178")):
        counts = data[key]
        widths = [edges[i + 1] - edges[i] for i in range(len(counts))]
        denominator = sum(counts)
        figure.add_trace(go.Bar(
            x=[(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))],
            y=[count / denominator / widths[i] if denominator and widths[i] else 0
               for i, count in enumerate(counts)],
            width=widths,
            name=label, marker_color=color, opacity=0.65,
        ))
    figure.update_layout(title="Distances Haversine · dataset complet", barmode="overlay",
                         xaxis_title="Distance (km)", yaxis_title="Densité")
    return _layout(figure)


def render_public_dashboard(summary: dict, *, alerts_today: int = 0) -> None:
    st.caption("Sparkov · agrégats exacts du dataset complet. Les graphiques descriptifs ne servent pas à régler un modèle.")
    amounts = summary["amount"]
    kpis = st.columns(5)
    kpis[0].metric("Transactions", f"{summary['rows']:,}".replace(",", " "))
    kpis[1].metric("Fraudes étiquetées", f"{summary['fraud_count']:,}".replace(",", " "))
    kpis[2].metric("Taux observé", f"{summary['fraud_rate']:.3%}")
    kpis[3].metric("Montant moyen", f"{amounts['mean']:,.2f}".replace(",", " "))
    kpis[4].metric("Alertes aujourd'hui", alerts_today)
    st.caption("Les taux utilisent les labels réels du CSV. Aucune partition Train/Validation/Test n'est déduite de ce fichier fusionné.")
    left, right = st.columns(2)
    if go:
        left.plotly_chart(_amount_chart(summary), width="stretch")
        right.plotly_chart(_distance_chart(summary), width="stretch")
    if px:
        months = pd.DataFrame(summary["months"])
        figure = px.line(months, x="month", y="fraud_rate", markers=True,
                         title="Taux de fraude mensuel · dataset complet")
        st.plotly_chart(_layout(figure), width="stretch")
        col1, col2 = st.columns(2)
        hours = pd.DataFrame(summary["hours"])
        col1.plotly_chart(_layout(px.bar(hours, x="hour", y="fraud_rate",
                                          title="Taux par heure · dataset complet")), width="stretch")
        weekdays = pd.DataFrame(summary["weekdays"])
        weekdays["weekday"] = weekdays["weekday"].map({0:"Lun",1:"Mar",2:"Mer",3:"Jeu",4:"Ven",5:"Sam",6:"Dim"})
        col2.plotly_chart(_layout(px.bar(weekdays, x="weekday", y="transactions",
                                         color="fraud_rate", title="Transactions par jour · dataset complet",
                                         color_continuous_scale=["#173c4b", "#25D0C2"])), width="stretch")
        categories = pd.DataFrame(summary["categories"]).sort_values("fraud_rate", ascending=False)
        st.plotly_chart(_layout(px.bar(categories, x="fraud_rate", y="category", orientation="h",
                                         hover_data=["transactions", "frauds"],
                                         title="Taux de fraude par catégorie · dataset complet")), width="stretch")
        channels = pd.DataFrame(summary["channels"])
        col1, col2 = st.columns(2)
        col1.plotly_chart(_layout(px.bar(channels, x="channel", y="fraud_rate",
                                          title="Taux par canal · dataset complet")), width="stretch")
        col2.plotly_chart(_layout(px.pie(names=["Normales", "Fraudes"],
                                          values=[summary["normal_count"], summary["fraud_count"]],
                                          hole=0.65, color_discrete_sequence=["#25D0C2", "#F07178"],
                                          title="Répartition réelle des classes")), width="stretch")
    class_summary = amounts["by_class"]
    st.subheader("Montants par classe")
    st.dataframe(pd.DataFrame(class_summary).T, width="stretch")


def render_public_explorer(summary: dict) -> None:
    render_public_dashboard(summary)
    st.caption("Le tableau utilise un échantillon réel déterministe filtrable; les agrégats exacts ci-dessus couvrent le dataset complet.")
    cats = [item["category"] for item in summary["categories"]]
    c1, c2, c3 = st.columns(3)
    category = c1.selectbox("Catégorie", ["Toutes", *cats])
    class_label = c2.selectbox("Classe", ["Toutes", "Normales", "Fraudes"])
    class_filter = {"Toutes": "", "Normales": "normal", "Fraudes": "fraud"}[class_label]
    month_options = [item["month"] for item in summary["months"]]
    month = c3.selectbox("Période de l'échantillon", ["Toutes", *month_options])
    start, end = (f"{month}-01", f"{month}-01") if month != "Toutes" else ("", "")
    page = st.number_input("Page de l'échantillon", min_value=1, max_value=16, value=1, step=1)
    try:
        sample = get_public_demo_sample(offset=(int(page) - 1) * 500, limit=500,
                                       category="" if category == "Toutes" else category,
                                       class_filter=class_filter, start=start, end=end)
    except Exception as exc:
        st.error(f"Échantillon Sparkov indisponible : {exc}")
        return
    st.caption(f"{sample['total']:,} lignes dans l'échantillon filtré · seed {sample['sample_seed']}. Les KPI/graphiques au-dessus portent sur le dataset complet.".replace(",", " "))
    frame = pd.DataFrame(sample["items"])
    if not frame.empty:
        st.dataframe(frame, width="stretch", hide_index=True)
    else:
        st.info("Aucune ligne de l'échantillon ne correspond à ces filtres.")
    st.subheader("Features V2")
    st.info("Le CSV Sparkov brut ne contient pas les variables historiques V2. Importez l'export de features du notebook pour explorer tx_count_1h/24h/30d, avg_amount_30d et amount_ratio_30d.")
