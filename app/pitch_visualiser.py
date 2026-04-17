"""
pitch_visualiser.py — Draw 2D pitch with player positions.

Creates a matplotlib figure of the pitch with:
    - Colour-coded player dots by team
    - Player ID labels
    - Ball position
    - Formation lines connecting players in the same line
    - Formation label overlay

Returns the figure as a numpy image for Streamlit display.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Dict


def draw_pitch(
    ax: plt.Axes,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
):
    """Draw pitch markings on a matplotlib axes."""
    ax.set_xlim(-3, pitch_length + 3)
    ax.set_ylim(-3, pitch_width + 3)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d8c3c")

    lw = 1.5
    white = "white"

    # Boundary
    ax.plot([0, pitch_length, pitch_length, 0, 0],
            [0, 0, pitch_width, pitch_width, 0], color=white, lw=lw)

    # Halfway line
    ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], color=white, lw=lw)

    # Centre circle
    circle = plt.Circle((pitch_length/2, pitch_width/2), 9.15,
                        fill=False, color=white, lw=lw)
    ax.add_patch(circle)
    ax.plot(pitch_length/2, pitch_width/2, "o", color=white, markersize=3)

    # Penalty areas
    pa_w, pa_h = 16.5, 40.3
    pa_y = (pitch_width - pa_h) / 2
    ax.add_patch(patches.Rectangle((0, pa_y), pa_w, pa_h,
                fill=False, edgecolor=white, lw=lw))
    ax.add_patch(patches.Rectangle((pitch_length - pa_w, pa_y), pa_w, pa_h,
                fill=False, edgecolor=white, lw=lw))

    # Goal areas
    ga_w, ga_h = 5.5, 18.3
    ga_y = (pitch_width - ga_h) / 2
    ax.add_patch(patches.Rectangle((0, ga_y), ga_w, ga_h,
                fill=False, edgecolor=white, lw=lw))
    ax.add_patch(patches.Rectangle((pitch_length - ga_w, ga_y), ga_w, ga_h,
                fill=False, edgecolor=white, lw=lw))

    # Penalty spots
    ax.plot(11, pitch_width/2, "o", color=white, markersize=3)
    ax.plot(pitch_length - 11, pitch_width/2, "o", color=white, markersize=3)

    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])


def create_pitch_map(
    player_positions: List[Tuple[float, float]],
    player_teams: List[str],
    player_ids: Optional[List[int]] = None,
    ball_position: Optional[Tuple[float, float]] = None,
    formation_a: Optional[str] = None,
    formation_b: Optional[str] = None,
    title: str = "",
    team_colours: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 7),
) -> np.ndarray:
    """
    Create a 2D pitch map with player positions.

    Args:
        player_positions: list of (pitch_x, pitch_y) in metres
        player_teams: list of team labels per player
        player_ids: optional player ID labels
        ball_position: (pitch_x, pitch_y) or None
        formation_a: formation string for team A (e.g. "4-3-3")
        formation_b: formation string for team B
        title: figure title
        team_colours: colour per team name
        figsize: figure size

    Returns:
        np.ndarray: RGB image of the pitch map
    """
    if team_colours is None:
        team_colours = {
            "team_a": "#e74c3c",
            "team_b": "#3498db",
            "referee": "#f1c40f",
            "unknown": "#95a5a6",
        }

    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax)

    # Draw players
    for i, (px, py) in enumerate(player_positions):
        if px is None or py is None:
            continue

        team = player_teams[i] if i < len(player_teams) else "unknown"
        colour = team_colours.get(team, "#95a5a6")

        ax.plot(px, py, "o", color=colour, markersize=10,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)

        if player_ids and i < len(player_ids):
            ax.annotate(str(player_ids[i]), (px, py),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=7, color="white", fontweight="bold")

    # Draw ball
    if ball_position and ball_position[0] is not None:
        bx, by = ball_position
        ax.plot(bx, by, "o", color="white", markersize=7,
                markeredgecolor="black", markeredgewidth=1, zorder=6)

    # Formation labels
    if formation_a:
        ax.text(25, -1.5, f"Team A: {formation_a}",
               fontsize=11, color=team_colours["team_a"],
               fontweight="bold", ha="center")
    if formation_b:
        ax.text(80, -1.5, f"Team B: {formation_b}",
               fontsize=11, color=team_colours["team_b"],
               fontweight="bold", ha="center")

    if title:
        ax.set_title(title, fontsize=12, color="white", pad=10)

    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()

    # Convert to numpy image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)

    return img


if __name__ == "__main__":
    # Quick test
    positions = [(25, 15), (25, 35), (25, 55), (50, 20), (50, 34),
                 (50, 48), (75, 20), (75, 34), (75, 48), (3, 34),
                 (80, 15), (80, 35), (80, 55), (55, 20), (55, 34),
                 (55, 48), (30, 20), (30, 34), (30, 48), (102, 34)]
    teams = ["team_a"]*10 + ["team_b"]*10

    img = create_pitch_map(positions, teams, formation_a="4-3-3", formation_b="4-3-3")
    from PIL import Image
    Image.fromarray(img).save("pitch_vis_test.png")
    print(f"Saved pitch_vis_test.png ({img.shape})")