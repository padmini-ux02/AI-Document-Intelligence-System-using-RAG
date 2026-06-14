"""
analytics.py — Plotly-based analytics and visualizations for Nexus AI.
All functions return Plotly figures that can be rendered with st.plotly_chart().
"""
import re
from typing import List, Dict, Any

import numpy as np

# Shared dark theme colors
_BG        = "#0a0a1a"
_PAPER_BG  = "rgba(5,5,30,0.0)"
_GRID      = "rgba(0,242,255,0.07)"
_CYAN      = "#00f2ff"
_PURPLE    = "#7b2fff"
_PINK      = "#ff2ff7"
_TEXT      = "#8baabb"
_FONT      = dict(family="Inter, sans-serif", color=_TEXT)

_DARK_LAYOUT = dict(
    paper_bgcolor=_PAPER_BG,
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=_FONT,
    margin=dict(l=20, r=20, t=40, b=20),
)


# ── 1. Keyword Frequency Bar Chart ───────────────────────────────────────────
def keyword_frequency_chart(keywords: List[str], scores: List[float]):
    """Horizontal bar chart of keyword TF-IDF scores."""
    import plotly.graph_objects as go

    if not keywords:
        return _empty_fig("No keywords extracted yet")

    # Sort ascending so highest score is at top
    pairs = sorted(zip(scores, keywords), key=lambda x: x[0])
    sorted_scores = [p[0] for p in pairs]
    sorted_kw     = [p[1] for p in pairs]

    colors = [
        f"rgba({int(0 + 123 * i/max(1,len(sorted_kw)-1))},{int(242 - 195 * i/max(1,len(sorted_kw)-1))},{int(255 - 0 * i/max(1,len(sorted_kw)-1))},0.85)"
        for i in range(len(sorted_kw))
    ]

    fig = go.Figure(go.Bar(
        x=sorted_scores,
        y=sorted_kw,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=_CYAN, width=0.5),
        ),
        hovertemplate="<b>%{y}</b><br>TF-IDF Score: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="📊 Keyword Importance (TF-IDF)", font=dict(color=_CYAN, size=14), x=0),
        xaxis=dict(showgrid=True, gridcolor=_GRID, title="TF-IDF Score", color=_TEXT),
        yaxis=dict(showgrid=False, color=_TEXT),
        height=380,
    )
    return fig


# ── 2. Chunk Distribution Per Document ───────────────────────────────────────
def chunk_distribution_chart(chunk_counts: Dict[str, int]):
    """Bar chart showing how many chunks came from each document."""
    import plotly.graph_objects as go

    if not chunk_counts:
        return _empty_fig("No documents indexed yet")

    docs   = list(chunk_counts.keys())
    counts = list(chunk_counts.values())
    # Truncate long filenames
    short_docs = [d[:25] + "…" if len(d) > 25 else d for d in docs]

    gradient_colors = [
        f"rgba(0,{int(180 + 62 * i/max(1,len(docs)-1))},{int(216 + 39 * i/max(1,len(docs)-1))},0.8)"
        for i in range(len(docs))
    ]

    fig = go.Figure(go.Bar(
        x=short_docs,
        y=counts,
        marker=dict(color=gradient_colors, line=dict(color=_CYAN, width=0.5)),
        text=counts,
        textposition="outside",
        textfont=dict(color=_CYAN, size=11),
        hovertemplate="<b>%{x}</b><br>Chunks: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="📁 Chunk Distribution per Document", font=dict(color=_CYAN, size=14), x=0),
        xaxis=dict(showgrid=False, color=_TEXT),
        yaxis=dict(showgrid=True, gridcolor=_GRID, title="Chunk Count", color=_TEXT),
        height=320,
    )
    return fig


# ── 3. Document Similarity Heatmap ───────────────────────────────────────────
def document_similarity_heatmap(chunks_by_doc: Dict[str, List[str]]):
    """Cosine-similarity heatmap between documents based on their TF-IDF representations."""
    import plotly.graph_objects as go
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    if len(chunks_by_doc) < 2:
        return _empty_fig("Need at least 2 documents for similarity heatmap")

    docs   = list(chunks_by_doc.keys())
    bodies = [" ".join(chunks_by_doc[d])[:10000] for d in docs]
    short  = [d[:20] + "…" if len(d) > 20 else d for d in docs]

    try:
        vec = TfidfVectorizer(stop_words="english", max_features=3000)
        mat = vec.fit_transform(bodies)
        sim = cos_sim(mat).tolist()
    except Exception:
        return _empty_fig("Could not compute document similarity")

    fig = go.Figure(go.Heatmap(
        z=sim,
        x=short,
        y=short,
        colorscale=[[0, "#050518"], [0.5, "#7b2fff"], [1, "#00f2ff"]],
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in sim],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>Similarity: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="🔗 Document Similarity Heatmap", font=dict(color=_CYAN, size=14), x=0),
        height=360,
        xaxis=dict(color=_TEXT),
        yaxis=dict(color=_TEXT),
    )
    return fig


# ── 4. Topic Cluster Scatter ──────────────────────────────────────────────────
def topic_cluster_scatter(chunks: List[Any], n_clusters: int = 4):
    """
    2-D scatter of document chunks colored by cluster (k-means on TF-IDF + PCA).
    `chunks` is a list of DocChunk objects with .text and .source attributes.
    """
    import plotly.graph_objects as go
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    if not chunks or len(chunks) < n_clusters:
        return _empty_fig("Not enough chunks for cluster analysis")

    texts   = [c.text for c in chunks]
    sources = [c.source for c in chunks]

    try:
        vec    = TfidfVectorizer(stop_words="english", max_features=2000)
        mat    = vec.fit_transform(texts)
        norm   = normalize(mat)
        svd    = TruncatedSVD(n_components=min(50, norm.shape[1] - 1, len(texts) - 1))
        reduced = svd.fit_transform(norm)
        # Final 2-D via second SVD pass for plotting
        svd2   = TruncatedSVD(n_components=2, random_state=42)
        coords = svd2.fit_transform(reduced)
        km     = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42, n_init="auto")
        labels = km.fit_predict(reduced)
    except Exception as e:
        return _empty_fig(f"Cluster error: {e}")

    cluster_colors = [_CYAN, _PURPLE, _PINK, "#ffb300", "#00ff88", "#ff6600"]
    unique_sources = list(dict.fromkeys(sources))
    source_symbols = ["circle", "square", "diamond", "cross", "star"]

    traces = []
    for cl in range(min(n_clusters, len(texts))):
        idx = [i for i, l in enumerate(labels) if l == cl]
        traces.append(go.Scatter(
            x=[coords[i, 0] for i in idx],
            y=[coords[i, 1] for i in idx],
            mode="markers",
            name=f"Cluster {cl+1}",
            marker=dict(
                color=cluster_colors[cl % len(cluster_colors)],
                size=7,
                opacity=0.75,
                line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
            ),
            text=[f"[{sources[i]}] {texts[i][:80]}…" for i in idx],
            hovertemplate="<b>%{text}</b><extra>Cluster %{fullData.name}</extra>",
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="🌐 Topic Cluster Map (SVD + K-Means)", font=dict(color=_CYAN, size=14), x=0),
        xaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT, title="Component 1"),
        yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT, title="Component 2"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=_CYAN, borderwidth=0.5, font=dict(color=_TEXT)),
        height=420,
    )
    return fig


# ── 5. Confidence Timeline ─────────────────────────────────────────────────────
def confidence_timeline(history: List[Dict]):
    """Line chart of response confidence over Q&A turns."""
    import plotly.graph_objects as go

    bot_turns = [m for m in history if m.get("role") == "assistant"]
    if not bot_turns:
        return _empty_fig("Ask some questions first to see the confidence trend")

    confidences = [m.get("confidence", 0) * 100 for m in bot_turns]
    turns = list(range(1, len(confidences) + 1))
    questions = [history[history.index(m) - 1]["content"][:50] + "…"
                 if history.index(m) > 0 else "?"
                 for m in bot_turns]

    colors = [_CYAN if c >= 70 else "#ffb300" if c >= 40 else "#ff4444" for c in confidences]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turns, y=confidences,
        mode="lines+markers",
        line=dict(color=_CYAN, width=2, dash="dot"),
        marker=dict(color=colors, size=10, line=dict(color="white", width=1)),
        text=questions,
        hovertemplate="<b>Q%{x}</b>: %{text}<br>Confidence: %{y:.1f}%<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(0,242,255,0.05)",
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="📈 Response Confidence Over Time", font=dict(color=_CYAN, size=14), x=0),
        xaxis=dict(title="Turn #", showgrid=True, gridcolor=_GRID, color=_TEXT),
        yaxis=dict(title="Confidence %", range=[0, 105], showgrid=True, gridcolor=_GRID, color=_TEXT),
        height=280,
    )
    return fig


# ── Helper ───────────────────────────────────────────────────────────────────
def _empty_fig(message: str):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=_TEXT, size=13),
    )
    fig.update_layout(
        **_DARK_LAYOUT,
        height=280,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ── Utility: keyword scores from TF-IDF ──────────────────────────────────────
def get_keyword_scores(chunks: List[Any], top_n: int = 15):
    """Return (keywords, scores) tuples from a list of DocChunk objects."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not chunks:
        return [], []
    text = " ".join(c.text for c in chunks[:30])
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=500)
        m = vec.fit_transform([text[:20000]])
        scores_arr = m.toarray().flatten()
        names = vec.get_feature_names_out()
        top_idx = np.argsort(scores_arr)[::-1][:top_n]
        return [names[i] for i in top_idx], [float(scores_arr[i]) for i in top_idx]
    except Exception:
        return [], []


def get_chunk_counts_by_doc(chunks: List[Any]) -> Dict[str, int]:
    """Count chunks per source document."""
    counts: Dict[str, int] = {}
    for c in chunks:
        counts[c.source] = counts.get(c.source, 0) + 1
    return counts


def get_chunks_by_doc(chunks: List[Any]) -> Dict[str, List[str]]:
    """Group chunk texts by source document."""
    groups: Dict[str, List[str]] = {}
    for c in chunks:
        groups.setdefault(c.source, []).append(c.text)
    return groups
