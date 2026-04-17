"""
pitch_projection.py — Project player positions to 2D pitch coordinates.

Takes detected player bounding boxes and the homography matrix,
projects each player's foot position (bottom-centre of bbox) to
real-world pitch coordinates in metres.

Also provides visualisation: draws a top-down pitch diagram with
coloured dots for each player, showing their actual positions.

This is what Mavrogiannis shows in Figure 4 — the broadcast view
alongside a 2D pitch minimap with player positions.

Reference: Mavrogiannis & Maglogiannis (2022) Section 3.7
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .homography import HomographyManager


@dataclass
class ProjectedPlayer:
    """A player's position in pitch coordinates."""
    player_id: int
    pitch_x: float          # Metres from left goal line (0-105)
    pitch_y: float          # Metres from bottom touchline (0-68)
    pixel_x: int            # Original pixel foot position x
    pixel_y: int            # Original pixel foot position y
    team: str               # "team_a", "team_b", "referee"
    is_valid: bool          # Whether projection is within pitch bounds


class PitchProjector:
    """
    Project player positions from image pixels to pitch coordinates.

    Usage:
        projector = PitchProjector()
        projector.set_homography(H)

        # Project all detections
        players = projector.project_players(
            foot_positions=[(640, 500), (300, 450), ...],
            player_ids=[0, 1, ...],
            teams=["team_a", "team_b", ...]
        )

        # Visualise on 2D pitch
        pitch_image = projector.draw_pitch_map(players)
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        position_margin: float = 5.0,
    ):
        """
        Args:
            pitch_length: FIFA pitch length in metres
            pitch_width: FIFA pitch width in metres
            position_margin: accept positions this far outside pitch bounds
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.position_margin = position_margin
        self.homography_manager = HomographyManager()

    def set_homography(self, H: np.ndarray) -> None:
        """Set the homography matrix from camera calibration."""
        self.homography_manager.set_homography(H)

    def project_players(
        self,
        foot_positions: List[Tuple[int, int]],
        player_ids: Optional[List[int]] = None,
        teams: Optional[List[str]] = None,
    ) -> List[ProjectedPlayer]:
        """
        Project player foot positions to pitch coordinates.

        Args:
            foot_positions: list of (pixel_x, pixel_y) foot positions
                          (bottom-centre of each bounding box)
            player_ids: optional list of player IDs
            teams: optional list of team labels

        Returns:
            list: ProjectedPlayer for each input position
        """
        if not self.homography_manager.is_valid:
            return []

        results = []

        for i, (px, py) in enumerate(foot_positions):
            player_id = player_ids[i] if player_ids else i
            team = teams[i] if teams else "unknown"

            pitch_pos = self.homography_manager.pixel_to_pitch(px, py)

            if pitch_pos is not None:
                pitch_x, pitch_y = pitch_pos
                is_valid = self.homography_manager.is_valid_pitch_position(
                    pitch_x, pitch_y,
                    self.pitch_length, self.pitch_width,
                    self.position_margin
                )
            else:
                pitch_x, pitch_y = 0.0, 0.0
                is_valid = False

            results.append(ProjectedPlayer(
                player_id=player_id,
                pitch_x=pitch_x,
                pitch_y=pitch_y,
                pixel_x=px,
                pixel_y=py,
                team=team,
                is_valid=is_valid,
            ))

        return results

    def project_ball(
        self,
        pixel_x: int,
        pixel_y: int,
    ) -> Optional[Tuple[float, float]]:
        """
        Project ball position to pitch coordinates.

        Args:
            pixel_x, pixel_y: ball centre in pixels

        Returns:
            tuple: (pitch_x, pitch_y) or None if invalid
        """
        pos = self.homography_manager.pixel_to_pitch(pixel_x, pixel_y)
        if pos is None:
            return None

        if self.homography_manager.is_valid_pitch_position(
            pos[0], pos[1], self.pitch_length, self.pitch_width,
            self.position_margin
        ):
            return pos

        return None

    def get_valid_players(
        self,
        players: List[ProjectedPlayer],
    ) -> List[ProjectedPlayer]:
        """Return only players with valid pitch positions."""
        return [p for p in players if p.is_valid]

    def get_team_positions(
        self,
        players: List[ProjectedPlayer],
        team: str,
    ) -> np.ndarray:
        """
        Get pitch positions for a specific team as an array.

        Args:
            players: list of projected players
            team: "team_a" or "team_b"

        Returns:
            np.ndarray: (N, 2) array of (pitch_x, pitch_y) positions
        """
        team_players = [p for p in players if p.team == team and p.is_valid]
        if not team_players:
            return np.array([]).reshape(0, 2)

        return np.array([[p.pitch_x, p.pitch_y] for p in team_players])

    def draw_pitch_map(
        self,
        players: List[ProjectedPlayer],
        ball_position: Optional[Tuple[float, float]] = None,
        map_width: int = 700,
        map_height: int = 453,
        team_colours: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Draw a 2D top-down pitch map with player positions.

        Args:
            players: list of projected players
            ball_position: (pitch_x, pitch_y) of the ball, or None
            map_width: output image width in pixels
            map_height: output image height in pixels
            team_colours: BGR colours per team. Defaults provided.

        Returns:
            np.ndarray: BGR image of the pitch map
        """
        if team_colours is None:
            team_colours = {
                "team_a": (0, 0, 255),      # Red
                "team_b": (255, 0, 0),      # Blue
                "referee": (0, 255, 255),   # Yellow
                "unknown": (200, 200, 200), # Grey
            }

        # Create green pitch background
        pitch = np.zeros((map_height, map_width, 3), dtype=np.uint8)
        pitch[:, :] = [45, 140, 50]  # Grass green

        # Padding
        pad = 30
        draw_w = map_width - 2 * pad
        draw_h = map_height - 2 * pad

        # Scale factors: pitch metres to map pixels
        sx = draw_w / self.pitch_length
        sy = draw_h / self.pitch_width

        def to_map(px, py):
            """Convert pitch coordinates to map pixel coordinates."""
            mx = int(pad + px * sx)
            my = int(pad + py * sy)
            return (mx, my)

        # Draw pitch markings
        white = (255, 255, 255)
        line_t = 1

        # Boundary
        cv2.rectangle(pitch, to_map(0, 0), to_map(self.pitch_length, self.pitch_width),
                      white, line_t)

        # Halfway line
        cv2.line(pitch, to_map(self.pitch_length/2, 0),
                 to_map(self.pitch_length/2, self.pitch_width), white, line_t)

        # Centre circle
        centre = to_map(self.pitch_length/2, self.pitch_width/2)
        radius = int(9.15 * sx)
        cv2.circle(pitch, centre, radius, white, line_t)
        cv2.circle(pitch, centre, 3, white, -1)

        # Penalty areas
        pa_w = 16.5
        pa_h = 40.3
        pa_y = (self.pitch_width - pa_h) / 2

        # Left penalty area
        cv2.rectangle(pitch, to_map(0, pa_y), to_map(pa_w, pa_y + pa_h),
                      white, line_t)
        # Right penalty area
        cv2.rectangle(pitch, to_map(self.pitch_length - pa_w, pa_y),
                      to_map(self.pitch_length, pa_y + pa_h), white, line_t)

        # Goal areas
        ga_w = 5.5
        ga_h = 18.3
        ga_y = (self.pitch_width - ga_h) / 2

        cv2.rectangle(pitch, to_map(0, ga_y), to_map(ga_w, ga_y + ga_h),
                      white, line_t)
        cv2.rectangle(pitch, to_map(self.pitch_length - ga_w, ga_y),
                      to_map(self.pitch_length, ga_y + ga_h), white, line_t)

        # Draw players
        for player in players:
            if not player.is_valid:
                continue

            colour = team_colours.get(player.team, (200, 200, 200))
            pos = to_map(player.pitch_x, player.pitch_y)

            # Player dot
            cv2.circle(pitch, pos, 6, colour, -1)
            cv2.circle(pitch, pos, 6, (0, 0, 0), 1)  # Black outline

            # Player ID label
            label = str(player.player_id)
            cv2.putText(pitch, label, (pos[0] + 8, pos[1] + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Draw ball
        if ball_position is not None:
            bx, by = ball_position
            if self.homography_manager.is_valid_pitch_position(
                bx, by, self.pitch_length, self.pitch_width, self.position_margin
            ):
                ball_pos = to_map(bx, by)
                cv2.circle(pitch, ball_pos, 5, (255, 255, 255), -1)
                cv2.circle(pitch, ball_pos, 5, (0, 0, 0), 1)

        return pitch


if __name__ == "__main__":
    print("=== Pitch Projector Test ===\n")

    projector = PitchProjector()

    # Set up a simple homography
    # Maps pixel (0,0)→pitch (0,0), pixel (1050,680)→pitch (105,68)
    H = np.array([
        [10.0,  0.0,  0.0],
        [ 0.0, 10.0,  0.0],
        [ 0.0,  0.0,  1.0],
    ])
    projector.set_homography(H)

    # Test projection
    print("1. Testing player projection...")
    foot_positions = [
        (250, 340),   # Should be near (25, 34) on pitch
        (525, 340),   # Should be near (52.5, 34) — centre
        (800, 200),   # Should be near (80, 20)
    ]
    player_ids = [0, 1, 2]
    teams = ["team_a", "team_b", "referee"]

    players = projector.project_players(foot_positions, player_ids, teams)
    for p in players:
        print(f"   Player {p.player_id} ({p.team}): "
              f"pixel ({p.pixel_x},{p.pixel_y}) → "
              f"pitch ({p.pitch_x:.1f}, {p.pitch_y:.1f}) "
              f"valid={p.is_valid}")

    # Test ball
    print("\n2. Testing ball projection...")
    ball = projector.project_ball(500, 300)
    print(f"   Ball: pixel (500,300) → pitch ({ball[0]:.1f}, {ball[1]:.1f})")

    # Test team positions
    print("\n3. Testing team positions...")
    team_a_pos = projector.get_team_positions(players, "team_a")
    print(f"   Team A positions: {team_a_pos}")

    # Test pitch map drawing
    print("\n4. Drawing pitch map...")
    pitch_map = projector.draw_pitch_map(players, ball_position=ball)
    cv2.imwrite("pitch_map_test.png", pitch_map)
    print(f"   Saved pitch_map_test.png ({pitch_map.shape})")

    print("\n=== Tests complete ===")