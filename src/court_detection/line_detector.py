"""
line_detector.py — Detect pitch lines using Hough Line Transform.

Detects straight lines in the edge image that correspond to pitch
markings (touchlines, penalty area borders, halfway line, etc.).

Lines are filtered by:
    - Minimum length (short edges from players/noise are rejected)
    - Angle (pitch lines are roughly horizontal or vertical in
      broadcast footage)
    - Position within the grass mask

The detected lines can be used:
    1. As features for camera calibration (alternative to U-Net)
    2. For visualising the pitch overlay on the broadcast frame
    3. As a sanity check against the U-Net field marking output

"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectedLine:
    """A single detected line segment."""
    x1: int
    y1: int
    x2: int
    y2: int
    angle: float        # Angle in degrees (0=horizontal, 90=vertical)
    length: float       # Line length in pixels

    @property
    def midpoint(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def endpoints(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return ((self.x1, self.y1), (self.x2, self.y2))


class LineDetector:
    """
    Detect pitch lines using Hough Line Transform.

    Usage:
        detector = LineDetector()
        lines = detector.detect(edges, grass_mask=mask)

        # Draw on frame
        detector.draw_lines(frame, lines)
    """

    def __init__(
        self,
        rho: float = 1.0,
        theta_resolution: float = np.pi / 180,
        threshold: int = 80,
        min_line_length: int = 50,
        max_line_gap: int = 20,
        angle_tolerance: float = 15.0,
        min_length_filter: float = 30.0,
    ):
        """
        Args:
            rho: distance resolution in pixels
            theta_resolution: angle resolution in radians
            threshold: minimum number of votes for a line
            min_line_length: minimum line segment length (pixels)
            max_line_gap: maximum gap between segments to merge
            angle_tolerance: degrees from horizontal/vertical to accept
            min_length_filter: reject lines shorter than this
        """
        self.rho = rho
        self.theta_resolution = theta_resolution
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.angle_tolerance = angle_tolerance
        self.min_length_filter = min_length_filter

    def detect(
        self,
        edge_image: np.ndarray,
        grass_mask: Optional[np.ndarray] = None,
    ) -> List[DetectedLine]:
        """
        Detect lines in an edge image.

        Args:
            edge_image: binary edge image (H, W), uint8
            grass_mask: optional mask to restrict detection to pitch area

        Returns:
            list: DetectedLine objects
        """
        # Apply grass mask if provided
        if grass_mask is not None:
            edges = cv2.bitwise_and(edge_image, grass_mask)
        else:
            edges = edge_image

        # Run Hough Line Transform (probabilistic version)
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta_resolution,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if hough_lines is None:
            return []

        # Convert to DetectedLine objects and filter
        lines = []
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]

            # Compute angle (0=horizontal, 90=vertical)
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(abs(dy), abs(dx)))

            # Compute length
            length = np.sqrt(dx**2 + dy**2)

            # Filter by length
            if length < self.min_length_filter:
                continue

            lines.append(DetectedLine(
                x1=x1, y1=y1, x2=x2, y2=y2,
                angle=angle, length=length,
            ))

        return lines

    def detect_filtered(
        self,
        edge_image: np.ndarray,
        grass_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[DetectedLine], List[DetectedLine]]:
        """
        Detect and separate lines into horizontal and vertical groups.

        Pitch lines in broadcast footage are either roughly horizontal
        (touchlines, penalty area top/bottom) or roughly vertical
        (goal lines, penalty area sides, halfway line).

        Returns:
            tuple: (horizontal_lines, vertical_lines)
        """
        all_lines = self.detect(edge_image, grass_mask)

        horizontal = []
        vertical = []

        for line in all_lines:
            if line.angle <= self.angle_tolerance:
                horizontal.append(line)
            elif line.angle >= (90 - self.angle_tolerance):
                vertical.append(line)
            # Lines at intermediate angles are rejected
            # (likely not pitch markings)

        return horizontal, vertical

    def merge_similar_lines(
        self,
        lines: List[DetectedLine],
        distance_threshold: float = 20.0,
        angle_threshold: float = 10.0,
    ) -> List[DetectedLine]:
        """
        Merge lines that are close together and at similar angles.

        Multiple Hough detections often correspond to the same
        physical pitch line. This merges them into a single line.
        """
        if len(lines) <= 1:
            return lines

        merged = []
        used = set()

        for i, line_i in enumerate(lines):
            if i in used:
                continue

            group = [line_i]
            used.add(i)

            for j, line_j in enumerate(lines):
                if j in used:
                    continue

                # Check angle similarity
                angle_diff = abs(line_i.angle - line_j.angle)
                if angle_diff > angle_threshold:
                    continue

                # Check distance between midpoints
                mid_i = line_i.midpoint
                mid_j = line_j.midpoint
                dist = np.sqrt(
                    (mid_i[0] - mid_j[0])**2 + (mid_i[1] - mid_j[1])**2
                )

                if dist < distance_threshold:
                    group.append(line_j)
                    used.add(j)

            # Merge group: use the longest line
            best = max(group, key=lambda l: l.length)
            merged.append(best)

        return merged

    def draw_lines(
        self,
        frame: np.ndarray,
        lines: List[DetectedLine],
        colour: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        show_info: bool = False,
    ) -> np.ndarray:
        """
        Draw detected lines on a frame.

        Args:
            frame: BGR image (modified in place)
            lines: detected lines to draw
            colour: BGR line colour
            thickness: line thickness
            show_info: annotate each line with angle and length

        Returns:
            np.ndarray: frame with lines drawn
        """
        for line in lines:
            cv2.line(
                frame,
                (line.x1, line.y1), (line.x2, line.y2),
                colour, thickness
            )

            if show_info:
                mid = line.midpoint
                label = f"{line.angle:.0f}deg {line.length:.0f}px"
                cv2.putText(
                    frame, label, (mid[0] + 5, mid[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1
                )

        return frame


if __name__ == "__main__":
    print("=== Line Detector Test ===\n")

    detector = LineDetector()

    # Create test frame with pitch-like lines
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]

    # Horizontal lines (touchlines)
    cv2.line(frame, (50, 100), (1230, 100), (255, 255, 255), 3)
    cv2.line(frame, (50, 620), (1230, 620), (255, 255, 255), 3)

    # Vertical lines
    cv2.line(frame, (50, 100), (50, 620), (255, 255, 255), 3)
    cv2.line(frame, (1230, 100), (1230, 620), (255, 255, 255), 3)
    cv2.line(frame, (640, 100), (640, 620), (255, 255, 255), 3)

    # Centre circle
    cv2.circle(frame, (640, 360), 80, (255, 255, 255), 3)

    # Edge detection first
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines
    lines = detector.detect(edges)
    print(f"1. All lines detected: {len(lines)}")
    for line in lines[:5]:
        print(f"   angle={line.angle:.1f}deg length={line.length:.0f}px")

    # Filtered into horizontal/vertical
    h_lines, v_lines = detector.detect_filtered(edges)
    print(f"\n2. Filtered lines:")
    print(f"   Horizontal: {len(h_lines)}")
    print(f"   Vertical: {len(v_lines)}")

    # Merge similar
    merged_h = detector.merge_similar_lines(h_lines)
    merged_v = detector.merge_similar_lines(v_lines)
    print(f"\n3. After merging:")
    print(f"   Horizontal: {len(merged_h)}")
    print(f"   Vertical: {len(merged_v)}")

    # Draw
    vis = frame.copy()
    detector.draw_lines(vis, merged_h, colour=(0, 0, 255))
    detector.draw_lines(vis, merged_v, colour=(255, 0, 0))
    cv2.imwrite("line_detection_test.png", vis)
    print(f"\n4. Saved line_detection_test.png")

    print("\n=== Tests complete ===")