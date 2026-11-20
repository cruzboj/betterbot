"""The result figures: one message each, shared colours (palette checked for colour-blind safety)."""
import matplotlib.pyplot as plt
import numpy as np

from analysis.cluster import in_right_cluster
from qwen_config import EVOLVING_WINDOW, FIGURES_DIR

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
            fig.savefig(path, dpi=220, bbox_inches="tight")     # sharp enough for full-slide use
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


# ------------------------------------------------------------------------------------------
# figure 6: one hallucinated text, word by word
# ------------------------------------------------------------------------------------------
HALLUCINATION_COLOR = "#c0392b"


def plot_word_curves(title, panels, marks, switch_position=None):
    """Stacked panels, one pair per signal: the text's raw curve (the clean original's faint underneath, when there
    is one) and, below it, the gap to the clean original at every word. A panel can also be a matrix profile
    (how unusual each stretch of words is), with its most unusual stretches shaded yellow. Red lines mark hallucinations
    (planted or found by reading), labelled with the word; a dashed line marks the temperature switch point.

    panels: [{"signal": name, "curve": values, "original": values or None, "gaps": values or None,
              "stretches": [start, ...] (optional), "window": stretch length}, ...]
    marks:  [(position on the curve, label), ...] (positions of the first panel's curve are used for all panels
            when a word has no value on another curve it is simply not drawn there)"""
    rows = []
    for panel in panels:
        rows.append(("curve", panel))
        if panel.get("gaps") is not None:
            rows.append(("gaps", panel))
    heights = [2.4 if kind == "curve" else 1.3 for kind, _panel in rows]
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, sum(heights) + 0.8), sharex=True,
                             gridspec_kw={"height_ratios": heights})
    if len(rows) == 1:
        axes = [axes]
    for ax, (kind, panel) in zip(axes, rows):
        if kind == "curve":
            if panel.get("original") is not None:
                original = panel["original"]
                ax.plot(np.arange(1, len(original) + 1), original, color="#b8b6ae", lw=1.0, label="clean original")
            values = panel["curve"]
            ax.plot(np.arange(1, len(values) + 1), values, color=INK, lw=1.1, label="this text")
            ax.set_ylabel(panel["signal"], color=INK_SOFT, fontsize=9)
            for start in panel.get("stretches", []):      # the most unusual stretches (matrix profile)
                ax.axvspan(start + 0.5, start + panel["window"] + 0.5, color="#f0b429", alpha=0.35, lw=0)
        else:
            gaps = panel["gaps"]
            ax.bar(np.arange(1, len(gaps) + 1), gaps, color=INK_SOFT, width=1.0)
            ax.axhline(0, color=GRID, lw=0.8)
            ax.set_ylabel("gap to\noriginal", color=INK_SOFT, fontsize=9)
        for position, label in marks.get(panel["signal"], []):
            ax.axvline(position + 1, color=HALLUCINATION_COLOR, lw=1.2, alpha=0.8)
            if kind == "curve":
                ax.annotate(label, (position + 1, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -12),
                            textcoords="offset points", fontsize=8, color=HALLUCINATION_COLOR)
        if switch_position is not None:
            ax.axvline(switch_position + 1, color=HALLUCINATION_COLOR, lw=1.4, ls="--")
            if kind == "curve":
                ax.annotate("temperature switch", (switch_position + 1, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(3, -12), textcoords="offset points", fontsize=8, color=HALLUCINATION_COLOR)
        clean_axes(ax, grid_axis="y")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("measured word (noun/verb) in reading order", color=INK_SOFT)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def save_to(fig, folder, file_name, tries=3):
    """Save a figure into another folder than results/figures (e.g. the hallucination results), with the same
    OneDrive retry as save()."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name
    for attempt in range(tries):
        try:
            fig.savefig(path, dpi=220, bbox_inches="tight")
            return path
        except OSError:
            if attempt == tries - 1:
                raise
            path.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------
# figure 7: the evolving graph (notebook 06)
# ------------------------------------------------------------------------------------------
def plot_evolving(title, profiles, ceilings, real_nearest, switch_word, marks):
    """Two stacked panels per signal. Top: the text's raw curve. Bottom: its distance to real factual (blue) and
    creative (orange) writing at every word, every real text's own distance to the nearest writing (thin grey, each
    compared without itself), the normal ceiling (dashed) and the first alarm (red dot). Dashed red line: the
    temperature switch; red lines: marked hallucinations. All arguments are dicts keyed by signal name;
    profiles hold evolving.evolving_profile tables, marks are [(word, label), ...]."""
    from analysis.evolving import first_alarm
    signal_names = list(profiles)
    fig, axes = plt.subplots(2 * len(signal_names), 1, figsize=(13, 4.6 * len(signal_names) + 0.8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2.4] * len(signal_names)})
    for number, signal_name in enumerate(signal_names):
        profile = profiles[signal_name]
        words = profile["word"].to_numpy() + 1
        curve_ax, distance_ax = axes[2 * number], axes[2 * number + 1]
        curve_ax.plot(words, profile["value"], color=INK, lw=1.0)
        curve_ax.set_ylabel(signal_name, color=INK_SOFT, fontsize=9)
        for real in real_nearest.get(signal_name, []):
            distance_ax.plot(np.arange(1, len(real) + 1), real, color="#cfcdc6", lw=0.6)
        distance_ax.plot(words, profile["to factual"], color=KIND_COLORS["factual"], lw=1.4,
                         label="distance to factual writing")
        distance_ax.plot(words, profile["to creative"], color=KIND_COLORS["creative"], lw=1.4,
                         label="distance to creative writing")
        ceiling = ceilings[signal_name]
        distance_ax.axhline(ceiling, color=INK_SOFT, lw=1.0, ls="--", label="highest any real text reaches")
        alarm = first_alarm(profile["to nearest"].to_numpy(), ceiling)
        if alarm is not None:
            distance_ax.scatter([alarm + 1], [profile["to nearest"].iloc[alarm]], s=60, color=HALLUCINATION_COLOR,
                                zorder=5, label=f"first alarm (word {alarm + 1})")
        distance_ax.set_ylabel(f"distance to real writing\n(last {EVOLVING_WINDOW} words)", color=INK_SOFT, fontsize=9)
        for ax in [curve_ax, distance_ax]:
            for word, label in marks.get(signal_name, []):
                ax.axvline(word + 1, color=HALLUCINATION_COLOR, lw=1.0, alpha=0.7)
                if ax is curve_ax:
                    ax.annotate(label, (word + 1, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -12),
                                textcoords="offset points", fontsize=8, color=HALLUCINATION_COLOR)
            if switch_word.get(signal_name) is not None:
                ax.axvline(switch_word[signal_name] + 1, color=HALLUCINATION_COLOR, lw=1.4, ls="--")
                if ax is curve_ax:
                    ax.annotate("temperature switch", (switch_word[signal_name] + 1, 0.02),
                                xycoords=("data", "axes fraction"), xytext=(3, 3), textcoords="offset points",
                                fontsize=8, color=HALLUCINATION_COLOR)
            clean_axes(ax, grid_axis="y")
        distance_ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    axes[-1].set_xlabel("measured word (noun/verb) in reading order", color=INK_SOFT)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def animate_evolving(title, profile, ceiling, switch_word, step=5):
    """The evolving graph of one text and signal as it is read: a plotly figure with a frame at words step,
    2*step, ... (the last frame is always the whole text), a play button and a slider."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    n = len(profile)
    ends = list(range(step, n, step)) + [n]
    words = profile["word"].to_numpy() + 1

    def traces(end):
        return [go.Scatter(x=words[:end], y=profile["value"].iloc[:end], line=dict(color=INK, width=1),
                           name="curve"),
                go.Scatter(x=words[:end], y=profile["to factual"].iloc[:end],
                           line=dict(color=KIND_COLORS["factual"], width=2), name="distance to factual writing"),
                go.Scatter(x=words[:end], y=profile["to creative"].iloc[:end],
                           line=dict(color=KIND_COLORS["creative"], width=2), name="distance to creative writing")]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for row, trace in zip([1, 2, 2], traces(ends[0])):
        fig.add_trace(trace, row=row, col=1)
    fig.frames = [go.Frame(data=traces(end), traces=[0, 1, 2], name=str(end)) for end in ends]
    fig.add_hline(y=ceiling, line=dict(color=INK_SOFT, dash="dash"), row=2, col=1,
                  annotation_text="highest any real text reaches")
    if switch_word is not None:
        fig.add_vline(x=switch_word + 1, line=dict(color=HALLUCINATION_COLOR, dash="dash"),
                      annotation_text="temperature switch")
    top = np.nanmax(profile[["to factual", "to creative"]].to_numpy()) if n else 1.0
    fig.update_xaxes(range=[0, n + 1])
    fig.update_yaxes(range=[np.nanmin(profile["value"]) - 0.05, np.nanmax(profile["value"]) + 0.05], row=1, col=1)
    fig.update_yaxes(range=[0, max(top, ceiling) * 1.1], row=2, col=1)
    fig.update_layout(
        title=title, template="plotly_white", height=600,
        updatemenus=[dict(type="buttons", buttons=[dict(label="play", method="animate",
                                                        args=[None, dict(frame=dict(duration=120), fromcurrent=True)])])],
        sliders=[dict(steps=[dict(label=str(end), method="animate",
                                  args=[[str(end)], dict(mode="immediate", frame=dict(duration=0))]) for end in ends],
                      currentvalue=dict(prefix="words read: "))])
    return fig


def plot_burst(title, burst_profiles, cool_profiles, ceilings, burst_words):
    """A text with one hot burst against its clean original, two stacked panels per signal. Top: both raw curves
    (clean grey, burst black; identical up to the burst). Bottom: both texts' distance to the nearest real writing,
    with the normal ceiling dashed. The burst's measured words are shaded red. All arguments are dicts keyed by
    signal name; profiles are evolving.evolving_profile tables, burst_words is (first, last) curve position or None."""
    signal_names = list(burst_profiles)
    fig, axes = plt.subplots(2 * len(signal_names), 1, figsize=(13, 4.4 * len(signal_names) + 0.8), sharex=True)
    for number, signal_name in enumerate(signal_names):
        burst, cool = burst_profiles[signal_name], cool_profiles[signal_name]
        curve_ax, distance_ax = axes[2 * number], axes[2 * number + 1]
        curve_ax.plot(cool["word"] + 1, cool["value"], color="#a9a7a0", lw=1.0, label="clean original")
        curve_ax.plot(burst["word"] + 1, burst["value"], color=INK, lw=1.0, label="with a hot burst")
        curve_ax.set_ylabel(signal_name, color=INK_SOFT, fontsize=9)
        distance_ax.plot(cool["word"] + 1, cool["to nearest"], color="#a9a7a0", lw=1.4, label="clean original")
        distance_ax.plot(burst["word"] + 1, burst["to nearest"], color=INK, lw=1.4, label="with a hot burst")
        distance_ax.axhline(ceilings[signal_name], color=INK_SOFT, lw=1.0, ls="--",
                            label="highest any real text reaches")
        distance_ax.set_ylabel(f"distance to real writing\n(last {EVOLVING_WINDOW} words)", color=INK_SOFT,
                               fontsize=9)
        for ax in [curve_ax, distance_ax]:
            if burst_words.get(signal_name) is not None:
                first, last = burst_words[signal_name]
                ax.axvspan(first + 0.5, last + 1.5, color=HALLUCINATION_COLOR, alpha=0.15, lw=0,
                           label="hot burst" if ax is curve_ax else None)
            clean_axes(ax, grid_axis="y")
        curve_ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
        distance_ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    axes[-1].set_xlabel("measured word (noun/verb) in reading order", color=INK_SOFT)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig
