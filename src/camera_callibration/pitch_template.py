"""
pitch_template.py — FIFA pitch model with all field markings.

Defines the football pitch in world coordinates (metres).
Origin (0, 0) is at the bottom-left corner of the pitch.
X-axis runs along the length (0 to 105m).
Y-axis runs along the width (0 to 68m).

All markings are stored as lists of (x, y) point sequences.
Lines are pairs of endpoints. Circles and arcs are sampled
into dense point sequences for projection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PitchTemplate:
    """FIFA standard pitch template with all field markings."""

    # Pitch dimensions (metres)
    length: float = 105.0
    width: float = 68.0

    # Penalty area dimensions
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.3

    # Goal area dimensions
    goal_area_length: float = 5.5
    goal_area_width: float = 18.3

    # Goal dimensions
    goal_width: float = 7.32

    # Circle/arc parameters
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0
    penalty_arc_radius: float = 9.15

    # How many points to sample along curves
    circle_sample_points: int = 100

    def get_all_markings(self) -> dict:
        """
        Returns all pitch markings as a dictionary.
        Each entry is a list of (x, y) numpy arrays representing
        connected line segments or curves.

        Returns:
            dict: keys are marking names, values are lists of
                  numpy arrays of shape (N, 2)
        """
        markings = {}

        # --- Boundary lines ---
        markings["touchline_bottom"] = self._line(0, 0, self.length, 0)
        markings["touchline_top"] = self._line(0, self.width, self.length, self.width)
        markings["goal_line_left"] = self._line(0, 0, 0, self.width)
        markings["goal_line_right"] = self._line(self.length, 0, self.length, self.width)

        # --- Halfway line ---
        half_x = self.length / 2
        markings["halfway_line"] = self._line(half_x, 0, half_x, self.width)

        # --- Centre circle ---
        markings["centre_circle"] = self._circle(
            half_x, self.width / 2, self.centre_circle_radius
        )

        # --- Centre spot ---
        markings["centre_spot"] = np.array([[half_x, self.width / 2]])

        # --- Left penalty area ---
        pa_y_start = (self.width - self.penalty_area_width) / 2
        pa_y_end = pa_y_start + self.penalty_area_width
        markings["penalty_area_left"] = self._rectangle_open_left(
            0, pa_y_start, self.penalty_area_length, pa_y_end
        )

        # Right penalty area 
        markings["penalty_area_right"] = self._rectangle_open_right(
            self.length - self.penalty_area_length, pa_y_start,
            self.length, pa_y_end
        )

        # Left goal area 
        ga_y_start = (self.width - self.goal_area_width) / 2
        ga_y_end = ga_y_start + self.goal_area_width
        markings["goal_area_left"] = self._rectangle_open_left(
            0, ga_y_start, self.goal_area_length, ga_y_end
        )

        # Right goal area
        markings["goal_area_right"] = self._rectangle_open_right(
            self.length - self.goal_area_length, ga_y_start,
            self.length, ga_y_end
        )

        #Penalty spots 
        markings["penalty_spot_left"] = np.array([
            [self.penalty_spot_distance, self.width / 2]
        ])
        markings["penalty_spot_right"] = np.array([
            [self.length - self.penalty_spot_distance, self.width / 2]
        ])

        #  Penalty arcs (the curved part outside the penalty area) 
        markings["penalty_arc_left"] = self._penalty_arc(
            self.penalty_spot_distance, self.width / 2,
            self.penalty_arc_radius, side="left"
        )
        markings["penalty_arc_right"] = self._penalty_arc(
            self.length - self.penalty_spot_distance, self.width / 2,
            self.penalty_arc_radius, side="right"
        )

        return markings

    def get_all_points(self) -> np.ndarray:
        """
        Returns all marking points concatenated into a single array.
        Useful for bulk projection through a camera model.

        Returns:
            np.ndarray: shape (N, 2) — all pitch marking points
        """
        markings = self.get_all_markings()
        all_points = []
        for name, points in markings.items():
            all_points.append(points)
        return np.vstack(all_points)

    def get_lines_for_drawing(self) -> List[np.ndarray]:
        """
        Returns markings as separate line segments for drawing.
        Each element is a connected sequence of points that
        should be drawn as a continuous line.

        Returns:
            list: each element is np.ndarray of shape (N, 2)
        """
        markings = self.get_all_markings()
        lines = []
        for name, points in markings.items():
            # Skip single-point markings (spots)
            if len(points) > 1:
                lines.append(points)
        return lines

    # --- Helper methods ---

    def _line(self, x1: float, y1: float, x2: float, y2: float,
              num_points: int = 50) -> np.ndarray:
        """Generate evenly spaced points along a straight line."""
        x = np.linspace(x1, x2, num_points)
        y = np.linspace(y1, y2, num_points)
        return np.column_stack([x, y])

    def _circle(self, cx: float, cy: float, radius: float) -> np.ndarray:
        """Generate points along a full circle."""
        angles = np.linspace(0, 2 * np.pi, self.circle_sample_points)
        x = cx + radius * np.cos(angles)
        y = cy + radius * np.sin(angles)
        return np.column_stack([x, y])

    def _rectangle_open_left(self, x1: float, y1: float,
                              x2: float, y2: float) -> np.ndarray:
        """
        Three sides of a rectangle (open on the left/goal line side).
        Goes: bottom-left → bottom-right → top-right → top-left
        """
        bottom = self._line(x1, y1, x2, y1, 30)
        right = self._line(x2, y1, x2, y2, 30)
        top = self._line(x2, y2, x1, y2, 30)
        return np.vstack([bottom, right, top])

    def _rectangle_open_right(self, x1: float, y1: float,
                               x2: float, y2: float) -> np.ndarray:
        """
        Three sides of a rectangle (open on the right/goal line side).
        Goes: top-right → top-left → bottom-left → bottom-right
        """
        top = self._line(x2, y2, x1, y2, 30)
        left = self._line(x1, y2, x1, y1, 30)
        bottom = self._line(x1, y1, x2, y1, 30)
        return np.vstack([top, left, bottom])

    def _penalty_arc(self, cx: float, cy: float, radius: float,
                     side: str) -> np.ndarray:
        """
        Generate the penalty arc — the part of the circle around
        the penalty spot that falls OUTSIDE the penalty area.

        Args:
            cx, cy: penalty spot coordinates
            radius: arc radius (9.15m)
            side: 'left' or 'right'
        """
        angles = np.linspace(-np.pi / 2, np.pi / 2, self.circle_sample_points)

        if side == "left":
            # Arc faces right (towards centre of pitch)
            x = cx + radius * np.cos(angles)
            y = cy + radius * np.sin(angles)
            # Keep only points outside the penalty area
            mask = x > self.penalty_area_length
        else:
            # Arc faces left (towards centre of pitch)
            x = cx - radius * np.cos(angles)
            y = cy + radius * np.sin(angles)
            # Keep only points outside the penalty area
            mask = x < (self.length - self.penalty_area_length)

        return np.column_stack([x[mask], y[mask]])


def draw_pitch_template(save_path: str = None):
    """
    Visualise the pitch template using matplotlib.
    Useful for verifying the template is correct.

    Args:
        save_path: if provided, save the figure to this path
    """
    import matplotlib.pyplot as plt

    pitch = PitchTemplate()
    markings = pitch.get_all_markings()

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_facecolor("#2d8c3c")  # Grass green
    ax.set_xlim(-5, pitch.length + 5)
    ax.set_ylim(-5, pitch.width + 5)
    ax.set_aspect("equal")
    ax.set_title("FIFA Pitch Template", fontsize=14)
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres)")

    for name, points in markings.items():
        if "spot" in name:
            ax.plot(points[:, 0], points[:, 1], "wo", markersize=4)
        else:
            ax.plot(points[:, 0], points[:, 1], "w-", linewidth=1.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Pitch template saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Quick test: print summary and draw
    pitch = PitchTemplate()
    markings = pitch.get_all_markings()
    total_points = sum(len(pts) for pts in markings.values())

    print(f"Pitch: {pitch.length}m x {pitch.width}m")
    print(f"Markings: {len(markings)} elements, {total_points} total points")
    print(f"Marking names: {list(markings.keys())}")

    # Draw and save
    draw_pitch_template("pitch_template.png")