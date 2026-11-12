"""Plotly figures built from verified aggregate values, without raw CSV loading."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

ACCENT = "#44dac7"
PINK = "#ff6b92"
BLUE = "#64a5ed"
GRID = "rgba(160,195,218,.16)"
DAYS = ("Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim")

def finish(fig: go.Figure, title: str, ytitle: str = "") -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 17}},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 14, "r": 14, "t": 76, "b": 25},
        hovermode="closest", legend={"orientation": "h", "y": 1.02, "x": 0.01, "yanchor": "bottom"},
        font={"family": "Arial", "size": 12},
        yaxis={"title": ytitle, "gridcolor": GRID},
        xaxis={"gridcolor": GRID},
    )
    return fig

def classes_chart(data: dict) -> go.Figure:
    cls = data["eda_train"]["class_distribution"]
    fig = go.Figure()
    fig.add_bar(x=["Normales", "Fraudes"], y=[cls["normal"], cls["fraud"]],
                marker_color=[BLUE, PINK],
                customdata=[[cls["normal_percent"]], [cls["fraud_percent"]]],
                hovertemplate="%{x} : %{y:,} transactions<br>Part : %{customdata[0]:.4f} %<extra></extra>")
    fig.update_yaxes(type="log", tickmode="array", tickvals=[1e3, 1e4, 1e5, 1e6, 1e7], ticktext=["1 000", "10 000", "100 000", "1 M", "10 M"], title="Transactions (échelle log)")
    return finish(fig, "01 · Répartition des classes — Train", "Transactions · logarithmique")

def amount_chart(data: dict) -> go.Figure:
    hist = data["eda_train"]["amount_histogram"]
    edges = hist["bin_edges"]
    widths = [edges[i+1]-edges[i] for i in range(len(edges)-1)]
    centers = [(edges[i]+edges[i+1])/2 for i in range(len(widths))]
    fig = go.Figure()
    for name, label, color in (("normal", "Normales", BLUE), ("fraud", "Fraudes", PINK)):
        fig.add_bar(x=centers, y=hist["classes"][name]["density"],
            width=widths, name=label, marker_color=color, opacity=0.55,
            hovertemplate="log10(montant) : %{x:.3f}<br>Densité : %{y:.5f}<extra>"+label+"</extra>")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="log10(montant) · bins communs")
    fig = finish(fig, "02 · Distribution des montants par classe — Train", "Densité normalisée par classe")
    # Keep the legend below the chart: its former top position overlapped the title in Streamlit.
    fig.update_layout(
        margin={"l":14, "r":14, "t":74, "b":108},
        legend={"orientation":"h", "x":0.01, "y":-0.25, "yanchor":"top"},
    )
    return fig

def records_chart(data: dict, group: str, *, metric: str = "fraud_rate_percent") -> go.Figure:
    rows = data["eda_train"][group]
    frame = pd.DataFrame(rows)
    labels = {
        "monthly": ("03 · Évolution mensuelle de la fraude — Train", "Mois"),
        "hourly": ("04 · Fraude par heure — Train", "Heure de la journée"),
        "weekday": ("05 · Fraude par jour de semaine — Train", "Jour"),
        "category": ("06 · Fraude par catégorie commerciale — Train", "Catégorie"),
        "channel": ("07 · Fraude par canal dérivé — Train", "Canal"),
    }
    if group not in labels or metric not in {"fraud_rate_percent", "frauds", "transactions"}:
        raise ValueError("Graphique ou métrique non pris en charge.")
    title, axis = labels[group]
    if metric != "fraud_rate_percent":
        title = title.replace("Fraude ", "Effectifs · ").replace("Évolution mensuelle de la fraude", "Évolution mensuelle des transactions") + (" · Fraudes" if metric == "frauds" else " · Transactions")
    frame["display"] = frame["key"]
    if group == "weekday": frame["display"] = frame["key"].astype(int).map(lambda i: DAYS[i])
    if group == "hourly": frame["display"] = frame["key"].astype(int).map(lambda i: f"{i:02d}h")
    frame["display"] = pd.Categorical(frame["display"], categories=frame["display"].tolist(), ordered=True)
    values = frame[metric].tolist()
    custom = frame[["transactions", "frauds", "fraud_rate_percent"]].to_numpy()
    fig = go.Figure()
    style = {"marker_color": ACCENT if metric == "fraud_rate_percent" else BLUE}
    hover = ("Transactions : %{customdata[0]:,}<br>Fraudes : %{customdata[1]:,}"
             "<br>Taux : %{customdata[2]:.4f} %<extra></extra>")
    if group == "monthly":
        fig.add_scatter(x=frame["display"].astype(str), y=values, mode="lines+markers",
                        line={"color": ACCENT, "width": 3}, marker={"size": 7},
                        customdata=custom, hovertemplate=hover)
    elif group == "category":
        fig.add_bar(y=frame["display"].astype(str), x=values, orientation="h",
                    customdata=custom, hovertemplate=hover, **style)
        fig.update_yaxes(autorange="reversed", title=axis)
        fig.update_xaxes(title={"fraud_rate_percent": "Taux de fraude (%)", "frauds": "Fraudes (nombre)", "transactions": "Transactions (nombre)"}[metric])
    else:
        fig.add_bar(x=frame["display"].astype(str), y=values, customdata=custom,
                    hovertemplate=hover, **style)
        fig.update_xaxes(title=axis)
    label = {"fraud_rate_percent": "Taux de fraude (%)", "frauds": "Fraudes (nombre)", "transactions": "Transactions (nombre)"}[metric]
    fig = finish(fig, title, label if group != "category" else "")
    if group != "category": fig.update_yaxes(rangemode="tozero")
    if group == "monthly": fig.add_annotation(text="Dernier mois Train partiel",xref="paper",yref="paper",
                    x=1,y=1.06,showarrow=False,font={"size":10,"color":"#b9c7d3"},xanchor="right")
    return fig
