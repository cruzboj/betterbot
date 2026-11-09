"""The result figures: one message each, shared colours (palette checked for colour-blind safety)."""
import matplotlib.pyplot as plt
import numpy as np

from analysis.cluster import in_right_cluster
from qwen_config import FIGURES_DIR

KIND_COLORS = {"factual": "#2a78d6", "creative": "#eb6834"}
FAMILY_COLORS = {"guess meaning": "#6250d6", "model confidence": "#1baf7a"}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"


def clean_axes(ax, grid_axis="x"):
    """Light grid on one axis, no top/right frame, soft tick labels."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, file_name, tries=3):
    """Save to results/figures. OneDrive sometimes refuses to overwrite a synced file (OSError 22);
    then the old file is removed and the save retried."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / file_name
    for attempt in range(tries):
        try:
            fig.savefig(path, dpi=110, bbox_inches="tight")
            return path
        except OSError:
            if attempt == tries - 1:
                raise
            path.unlink(missing_ok=True)


def short_name(text_name):
    """'factual_wiki_volcano' -> 'wiki volcano'."""
    for prefix in ("factual_", "creative_"):
        if text_name.startswith(prefix):
            return text_name[len(prefix):].replace("_", " ")
    return text_name


# ------------------------------------------------------------------------------------------
# figure 1: what a curve is
# ------------------------------------------------------------------------------------------
def plot_example_curves(signal_name, examples):
    """examples: [(text name, kind, curve), ...]. Each curve drawn as it is, one panel per text,
    same y-axis, so the two kinds can be compared by eye."""
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 2.2 * len(examples)), sharey=True)
    for ax, (name, text_kind, values) in zip(axes, examples):
        positions = np.arange(1, len(values) + 1)
        ax.plot(positions, values, color=KIND_COLORS[text_kind], lw=1.2)
        ax.set_title(f"{text_kind.upper()}: {short_name(name)}  ({len(values)} measured words)", loc="left",
                     fontsize=10, color=INK)
        ax.set_xlim(1, len(values))
        clean_axes(ax, grid_axis="y")
    axes[-1].set_xlabel("measured word (noun/verb) in reading order", color=INK_SOFT)
    fig.supylabel(signal_name, color=INK_SOFT, fontsize=10)
    fig.suptitle("A curve = one value per word, in reading order. Each text is one curve.", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------
# figure 2: every signal in one chart
# ------------------------------------------------------------------------------------------
def plot_summary(results):
    """One bar per signal: texts in their own kind's cluster (of 25). Colour = signal family."""
    ordered = results.sort_values("correct")
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(ordered) + 1.6))
    for row, (_index, result) in enumerate(ordered.iterrows()):
        ax.barh(row, result["correct"], height=0.65, color=FAMILY_COLORS[result["family"]])
        ax.text(result["correct"] + 0.2, row, f"{result['correct']}/{result['texts']}", va="center", fontsize=8.5,
                color=INK)
    texts = int(ordered["texts"].iloc[0])
    ax.axvline(texts / 2, color=INK_SOFT, lw=1, ls="--")
    ax.text(texts / 2, len(ordered) - 0.2, " half (coin flip)", fontsize=8, color=INK_SOFT, va="bottom")
    ax.set_yticks(range(len(ordered)), ordered["name"], fontsize=9)
    ax.set_xlim(0, texts + 2)
    ax.set_xlabel(f"texts in the cluster of their own kind (out of {texts})", color=INK_SOFT)
    handles = []
    for family, color in FAMILY_COLORS.items():
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=family))
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    best = ordered.iloc[-1]
    ax.set_title(f"How well each signal groups the texts by kind. Best: {best['name']} "
                 f"({best['correct']}/{best['texts']})", loc="left", fontsize=11, color=INK)
    clean_axes(ax)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------
# figure 3: the text map of one signal
# ------------------------------------------------------------------------------------------
def plot_text_map(signal_name, coordinates, names, kinds, labels, centres):
    """Each text is a dot with its name; close dots = similar curves (DTW). Colour = real kind. Filled / hollow =
    the two clusters k-medoids found; a black ring marks each cluster's centre text. Texts in the other kind's
    cluster have their name in bold."""
    kinds = np.asarray(kinds)
    correct = in_right_cluster(kinds, labels)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for index in range(len(names)):
        color = KIND_COLORS[kinds[index]]
        x, y = coordinates[index]
        if labels[index] == 0:
            ax.scatter(x, y, s=110, color=color, edgecolor=INK, linewidth=0.6, zorder=3)
        else:
            ax.scatter(x, y, s=110, facecolor="white", edgecolor=color, linewidth=2.2, zorder=3)
        if correct[index]:
            weight = "normal"
        else:
            weight = "bold"
        label = short_name(names[index])
        if index in centres:
            ax.scatter(x, y, s=420, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=2)
            label += " (centre)"
        ax.annotate(label, (x, y), xytext=(8, 6), textcoords="offset points", fontsize=8,
                    color=INK, fontweight=weight)
    legend_items = [
        plt.scatter([], [], s=90, color=KIND_COLORS["factual"], label="factual text"),
        plt.scatter([], [], s=90, color=KIND_COLORS["creative"], label="creative text"),
        plt.scatter([], [], s=90, color="#999999", edgecolor=INK, label="cluster A (filled)"),
        plt.scatter([], [], s=90, facecolor="white", edgecolor="#999999", linewidth=2, label="cluster B (hollow)"),
        plt.scatter([], [], s=200, facecolor="none", edgecolor=INK, linewidth=1.6, label="cluster centre"),
    ]
    ax.legend(handles=legend_items, frameon=False, fontsize=9, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_title(f"{signal_name}: {int(correct.sum())} of {len(names)} texts sit with their own kind\n"
                 "close dots = similar curves (bold: texts in the other kind's cluster)", loc="left", fontsize=11,
                 color=INK)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------
# figure 5: dendrogram (hierarchical clustering, complete linkage)
# ------------------------------------------------------------------------------------------
def plot_dendrogram(signal_name, joins, names, kinds, correct_when_cut):
    """The tree of joins: texts on the left (name coloured by real kind), joined further right the less
    similar they are. The two branches left before the last join are drawn black and grey."""
    from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

    set_link_color_palette([INK, "#8a8880"])     # the two branches: neutral, so only the names carry kind colours
    kinds = np.asarray(kinds)
    labels = []
    for index in range(len(names)):
        labels.append(short_name(names[index]))
    last_join_height = joins[-1, 2]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(names) + 1.6))
    result = dendrogram(joins, labels=labels, orientation="left", ax=ax, color_threshold=last_join_height * 0.999,
                        above_threshold_color="#cfcdc6")
    for tick, leaf in zip(ax.get_yticklabels(), result["leaves"]):
        tick.set_color(KIND_COLORS[kinds[leaf]])
        tick.set_fontsize(9)
    ax.set_xlabel("DTW distance between the farthest texts of the two joined clusters", color=INK_SOFT)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=8)
    ax.set_title(f"{signal_name}: tree of the most similar texts (complete linkage)\n"
                 f"names: blue = factual, orange = creative.  Cut into 2 branches: {correct_when_cut} of {len(names)} "
                 f"texts sit with their own kind", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------
# figure 4: the same map in 3D, interactive (rotate / zoom in a browser)
# ------------------------------------------------------------------------------------------
def plot_text_map_3d(signal_name, coordinates, names, kinds, labels, centres, fit):
    """Interactive 3D text map (plotly). Colour = real kind; filled / open circle = the two clusters; a black
    diamond marks each cluster's centre text; names of texts in the other kind's cluster are marked with *."""
    import plotly.graph_objects as go

    kinds = np.asarray(kinds)
    correct = in_right_cluster(kinds, labels)
    fig = go.Figure()
    for text_kind in ["factual", "creative"]:
        for cluster_label, cluster_name, symbol in [(0, "cluster A", "circle"), (1, "cluster B", "circle-open")]:
            members = []
            for index in range(len(names)):
                if kinds[index] == text_kind and labels[index] == cluster_label:
                    members.append(index)
            if not members:
                continue
            texts = []
            for index in members:
                name = short_name(names[index])
                if not correct[index]:
                    name = "*" + name
                texts.append(name)
            fig.add_trace(go.Scatter3d(
                x=coordinates[members, 0], y=coordinates[members, 1], z=coordinates[members, 2],
                mode="markers+text", text=texts, textposition="top center", textfont={"size": 10, "color": INK},
                marker={"size": 7, "color": KIND_COLORS[text_kind], "symbol": symbol,
                        "line": {"width": 2, "color": KIND_COLORS[text_kind]}},
                name=f"{text_kind}, {cluster_name}", hovertemplate="%{text}<extra></extra>"))
    centre_names = []
    for index in centres:
        centre_names.append(f"centre: {short_name(names[index])}")
    fig.add_trace(go.Scatter3d(
        x=coordinates[centres, 0], y=coordinates[centres, 1], z=coordinates[centres, 2], mode="markers",
        marker={"size": 16, "symbol": "diamond-open", "color": INK, "line": {"width": 2, "color": INK}},
        text=centre_names, name="cluster centre", hovertemplate="%{text}<extra></extra>"))
    axis = {"showticklabels": False, "title": "", "showbackground": False, "gridcolor": GRID}
    fig.update_layout(
        title=f"{signal_name}: {int(correct.sum())} of {len(names)} texts sit with their own kind<br>"
              f"<sup>close dots = similar curves; * = in the other kind's cluster; map fit {fit:.2f} "
              f"(1 = perfect). Drag to rotate.</sup>",
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis}, width=900, height=750,
        legend={"itemsizing": "constant"}, paper_bgcolor="white")
    return fig


def save_html(fig, file_name):
    """Save an interactive plotly figure to results/figures/3d (plotly is loaded from its CDN when opened)."""
    folder = FIGURES_DIR / "3d"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name
    fig.write_html(path, include_plotlyjs="cdn")
    return path
