"""
formation_display.py — Display formation detection results.

Provides Streamlit-compatible components for showing:
    - Formation label with confidence bar
    - Formation change timeline across the match
    - Side-by-side team formation comparison
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def create_formation_summary(
    team_a_formation: str,
    team_a_confidence: float,
    team_b_formation: str,
    team_b_confidence: float,
    figsize: Tuple[int, int] = (8, 2),
) -> np.ndarray:
    """
    Create a visual summary of both teams' formations.

    Returns:
        np.ndarray: RGB image of the formation summary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")

    for ax, name, formation, conf, colour in [
        (ax1, "Team A", team_a_formation, team_a_confidence, "#e74c3c"),
        (ax2, "Team B", team_b_formation, team_b_confidence, "#3498db"),
    ]:
        ax.set_facecolor("#1a1a2e")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Team name
        ax.text(5, 2.3, name, fontsize=12, color=colour,
               ha="center", fontweight="bold")

        # Formation label
        ax.text(5, 1.3, formation, fontsize=20, color="white",
               ha="center", fontweight="bold")

        # Confidence bar
        bar_y = 0.4
        bar_h = 0.3
        ax.add_patch(plt.Rectangle((1, bar_y), 8, bar_h,
                     facecolor="#333333", edgecolor="none"))
        ax.add_patch(plt.Rectangle((1, bar_y), 8 * conf, bar_h,
                     facecolor=colour, edgecolor="none"))
        ax.text(5, bar_y - 0.2, f"{conf:.0%} confidence",
               fontsize=8, color="#aaaaaa", ha="center")

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)

    return img


def create_formation_timeline(
    timestamps: List[float],
    formations_a: List[str],
    formations_b: List[str],
    figsize: Tuple[int, int] = (10, 3),
) -> np.ndarray:
    """
    Create a timeline showing how formations changed during the match.

    Args:
        timestamps: list of time points (seconds)
        formations_a: team A formation at each time
        formations_b: team B formation at each time

    Returns:
        np.ndarray: RGB image of the timeline
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.patch.set_facecolor("#1a1a2e")

    for ax, formations, name, colour in [
        (ax1, formations_a, "Team A", "#e74c3c"),
        (ax2, formations_b, "Team B", "#3498db"),
    ]:
        ax.set_facecolor("#1a1a2e")
        ax.set_ylabel(name, color=colour, fontsize=10)
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_color("#333333")

        if not timestamps or not formations:
            continue

        # Find formation change points
        changes = [0]
        for i in range(1, len(formations)):
            if formations[i] != formations[i-1]:
                changes.append(i)

        # Draw coloured segments
        unique = list(set(formations))
        colour_map = plt.cm.Set3(np.linspace(0, 1, max(len(unique), 1)))

        for j in range(len(changes)):
            start = timestamps[changes[j]]
            end = timestamps[changes[j+1]] if j+1 < len(changes) else timestamps[-1]
            form = formations[changes[j]]

            c_idx = unique.index(form) % len(colour_map)
            ax.axvspan(start, end, alpha=0.4, color=colour_map[c_idx])
            ax.text((start + end) / 2, 0.5, form,
                   ha="center", va="center", fontsize=9,
                   color="white", fontweight="bold",
                   transform=ax.get_xaxis_transform())

        ax.set_yticks([])

    ax2.set_xlabel("Time (seconds)", color="#aaaaaa", fontsize=10)
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)

    return img


if __name__ == "__main__":
    # Test formation summary
    img = create_formation_summary("4-3-3", 0.85, "4-4-2", 0.72)
    from PIL import Image
    Image.fromarray(img).save("formation_summary_test.png")
    print(f"Saved formation_summary_test.png ({img.shape})")

    # Test timeline
    timestamps = list(range(0, 300, 10))
    forms_a = ["4-3-3"] * 15 + ["4-2-3-1"] * 15
    forms_b = ["4-4-2"] * 20 + ["4-5-1"] * 10
    img2 = create_formation_timeline(timestamps, forms_a, forms_b)
    Image.fromarray(img2).save("formation_timeline_test.png")
    print(f"Saved formation_timeline_test.png ({img2.shape})")