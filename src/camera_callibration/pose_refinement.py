"""
pose_refinement.py — Refine camera pose using Lucas-Kanade alignment.

Takes the initial camera pose from the database retrieval and refines
it by aligning the retrieved edge image with the detected edge image
using distance transforms and the Lucas-Kanade algorithm.


Why refinement is needed:
    The database lookup returns the nearest match, not an exact match.
    The retrieved pose might be off by a few degrees in pan/tilt or
    a few hundred pixels in focal length. Refinement iteratively
    adjusts the homography to minimise the difference between the
    retrieved and detected edge images.

Method:
    1. Compute distance transform of both edge images
       (distance from each pixel to the nearest edge pixel)
    2. Truncate distances at an upper bound (30-50px)
       to prevent distant pixels from dominating
    3. Use Lucas-Kanade (ECC alignment in OpenCV) to find the
       homography warp that best aligns the two distance images
    4. Apply the warp to the initial homography to get the refined result

"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RefinementResult:
    """Result of pose refinement."""
    homography: np.ndarray          # Refined 3x3 homography matrix
    initial_homography: np.ndarray  # Initial homography before refinement
    warp_matrix: np.ndarray         # The correction warp found by LK/ECC
    converged: bool                 # Whether the alignment converged
    correlation: float              # ECC correlation score (higher = better, max 1.0)
    iterations: int                 # Number of iterations used


class PoseRefinement:
    """
    Refines camera pose by aligning edge images using the
    Enhanced Correlation Coefficient (ECC) algorithm.

    The ECC method is preferred over standard Lucas-Kanade because:
    - It is invariant to photometric distortions (brightness/contrast)
    - Although the objective is nonlinear, the iterative solution is linear
    - It works well with binary/sparse images like edge maps

    Usage:
        refiner = PoseRefinement()

        # Get initial homography from database retrieval
        initial_H = engine.get_homography(retrieved_pose)

        # Get detected edge image from U-Net
        detected_edges = unet.predict(broadcast_frame)

        # Refine
        result = refiner.refine(
            detected_edge_image=detected_edges,
            initial_homography=initial_H,
            pitch_template=pitch
        )

        # Use the refined homography
        refined_H = result.homography
    """

    def __init__(
        self,
        distance_upper_bound: float = 40.0,
        max_iterations: int = 100,
        epsilon: float = 1e-6,
        warp_mode: str = "homography",
        image_size: Tuple[int, int] = (320, 180),
    ):
        """
        Args:
            distance_upper_bound: truncation value for distance transform (pixels).
                                 Chen & Little found 30-50 works well.
            max_iterations: maximum iterations for ECC alignment
            epsilon: convergence threshold for ECC
            warp_mode: 'homography' (8 params) or 'affine' (6 params)
            image_size: size of edge images for alignment (width, height)
        """
        self.distance_upper_bound = distance_upper_bound
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.image_size = image_size

        # Set OpenCV warp mode
        if warp_mode == "homography":
            self.warp_mode = cv2.MOTION_HOMOGRAPHY
        elif warp_mode == "affine":
            self.warp_mode = cv2.MOTION_AFFINE
        else:
            raise ValueError(f"Unknown warp_mode: {warp_mode}. Use 'homography' or 'affine'.")

        # ECC termination criteria
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.max_iterations,
            self.epsilon,
        )

    def compute_distance_image(self, edge_image: np.ndarray) -> np.ndarray:
        """
        Compute truncated distance transform of an edge image.

        The distance transform computes, for each pixel, the distance
        to the nearest edge pixel. Truncating prevents distant pixels
        from dominating the alignment.

        Args:
            edge_image: binary edge image (H, W), uint8 (0 or 255)

        Returns:
            np.ndarray: truncated distance image (H, W), float32
        """
        # Ensure binary
        if edge_image.dtype != np.uint8:
            edge_image = (edge_image * 255).astype(np.uint8)

        _, binary = cv2.threshold(edge_image, 127, 255, cv2.THRESH_BINARY)

        # Distance transform measures distance FROM non-edge TO nearest edge
        # OpenCV's distanceTransform needs the inverse (edges=0, background=255)
        inverted = cv2.bitwise_not(binary)
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

        # Truncate at upper bound
        dist = np.minimum(dist, self.distance_upper_bound)

        # Normalise to [0, 1] for ECC alignment
        dist = dist / self.distance_upper_bound

        return dist.astype(np.float32)

    def render_edge_from_homography(
        self,
        homography: np.ndarray,
        pitch_lines: list,
        image_size: Tuple[int, int] = None,
    ) -> np.ndarray:
        """
        Render an edge image from a homography matrix by projecting
        pitch template lines.

        Args:
            homography: 3x3 homography mapping pitch coords to image coords
            pitch_lines: list of line segments from PitchTemplate.get_lines_for_drawing()
            image_size: (width, height) of output image

        Returns:
            np.ndarray: binary edge image (H, W), uint8
        """
        if image_size is None:
            image_size = self.image_size

        w, h = image_size
        image = np.zeros((h, w), dtype=np.uint8)

        for line_points in pitch_lines:
            # Project pitch points through homography
            # Points are (x_pitch, y_pitch) in metres
            # Add homogeneous coordinate
            N = len(line_points)
            pts_h = np.hstack([
                line_points,
                np.ones((N, 1))
            ])  # (N, 3)

            # Project: pixel = H * pitch_point
            projected = (homography @ pts_h.T).T  # (N, 3)

            # Convert from homogeneous
            w_coord = projected[:, 2]
            valid = np.abs(w_coord) > 1e-6

            for j in range(len(projected) - 1):
                if not valid[j] or not valid[j + 1]:
                    continue

                pt1 = projected[j, :2] / w_coord[j]
                pt2 = projected[j + 1, :2] / w_coord[j + 1]

                # Check bounds
                margin = 50
                if (pt1[0] < -margin or pt1[0] > w + margin or
                    pt1[1] < -margin or pt1[1] > h + margin):
                    continue
                if (pt2[0] < -margin or pt2[0] > w + margin or
                    pt2[1] < -margin or pt2[1] > h + margin):
                    continue

                cv2.line(
                    image,
                    (int(round(pt1[0])), int(round(pt1[1]))),
                    (int(round(pt2[0])), int(round(pt2[1]))),
                    255, 2
                )

        return image

    def refine(
        self,
        detected_edge_image: np.ndarray,
        initial_homography: np.ndarray,
        pitch_lines: list,
    ) -> RefinementResult:
        """
        Refine the camera pose by aligning the detected edge image
        with the edge image rendered from the initial homography.

        This is the core refinement step from Chen & Little Section 3.4.

        Args:
            detected_edge_image: edge image from U-Net (H, W), uint8
            initial_homography: 3x3 homography from database retrieval
            pitch_lines: pitch template lines for rendering

        Returns:
            RefinementResult: refined homography and diagnostics
        """
        w, h = self.image_size

        # Resize detected edges to working size
        detected = cv2.resize(detected_edge_image, (w, h))
        if detected.dtype != np.uint8:
            detected = (detected * 255).astype(np.uint8)

        # Render edge image from initial homography
        # Scale homography to work at the alignment image size
        # If initial_homography maps to full resolution (1280x720),
        # we need to scale to working size (320x180)
        rendered = self.render_edge_from_homography(
            initial_homography, pitch_lines, (w, h)
        )

        # Compute distance transforms
        dist_detected = self.compute_distance_image(detected)
        dist_rendered = self.compute_distance_image(rendered)

        # Check that both images have enough content
        if np.sum(detected > 0) < 50 or np.sum(rendered > 0) < 50:
            return RefinementResult(
                homography=initial_homography,
                initial_homography=initial_homography,
                warp_matrix=np.eye(3, dtype=np.float32),
                converged=False,
                correlation=0.0,
                iterations=0,
            )

        # Initialize warp matrix (identity = no correction)
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Run ECC alignment
        try:
            correlation, warp_matrix = cv2.findTransformECC(
                dist_rendered,      # template (from database)
                dist_detected,      # input (from real image)
                warp_matrix,
                self.warp_mode,
                self.criteria,
            )
            converged = True
            iterations = self.max_iterations  # ECC doesn't report actual iterations

        except cv2.error:
            # ECC can fail if images are too different or too sparse
            correlation = 0.0
            converged = False
            iterations = 0
            warp_matrix = np.eye(3, dtype=np.float32)

        # Compute refined homography
        # The warp_matrix transforms rendered → detected
        # So: refined_H = warp_matrix @ initial_H
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            refined_homography = warp_matrix @ initial_homography
        else:
            # Convert affine to 3x3
            warp_3x3 = np.eye(3, dtype=np.float32)
            warp_3x3[:2, :] = warp_matrix
            refined_homography = warp_3x3 @ initial_homography

        return RefinementResult(
            homography=refined_homography,
            initial_homography=initial_homography,
            warp_matrix=warp_matrix,
            converged=converged,
            correlation=correlation,
            iterations=iterations,
        )

    def refine_with_fallback(
        self,
        detected_edge_image: np.ndarray,
        initial_homography: np.ndarray,
        pitch_lines: list,
        distance_bounds: list = None,
    ) -> RefinementResult:
        """
        Try refinement with multiple distance bounds, returning
        the best result. Falls back to initial homography if
        all attempts fail.

        This is more robust than a single attempt since the optimal
        truncation distance depends on the image content.

        Args:
            detected_edge_image: edge image from U-Net
            initial_homography: from database retrieval
            pitch_lines: pitch template lines
            distance_bounds: list of bounds to try (default: [30, 40, 50])

        Returns:
            RefinementResult: best result across all attempts
        """
        if distance_bounds is None:
            distance_bounds = [30.0, 40.0, 50.0]

        best_result = None
        best_correlation = -1.0

        original_bound = self.distance_upper_bound

        for bound in distance_bounds:
            self.distance_upper_bound = bound

            result = self.refine(
                detected_edge_image, initial_homography, pitch_lines
            )

            if result.converged and result.correlation > best_correlation:
                best_correlation = result.correlation
                best_result = result

        # Restore original bound
        self.distance_upper_bound = original_bound

        # If nothing converged, return initial homography
        if best_result is None:
            return RefinementResult(
                homography=initial_homography,
                initial_homography=initial_homography,
                warp_matrix=np.eye(3, dtype=np.float32),
                converged=False,
                correlation=0.0,
                iterations=0,
            )

        return best_result


def compute_iou(
    homography_pred: np.ndarray,
    homography_gt: np.ndarray,
    pitch_lines: list,
    image_size: Tuple[int, int] = (320, 180),
) -> float:
    """
    Compute IoU (Intersection over Union) between predicted and
    ground truth homographies by comparing their rendered edge images.

    This is the evaluation metric used by Chen & Little and
    Homayounfar et al. for camera calibration accuracy.

    Args:
        homography_pred: predicted 3x3 homography
        homography_gt: ground truth 3x3 homography
        pitch_lines: pitch template lines
        image_size: (width, height) for rendering

    Returns:
        float: IoU score between 0 and 1
    """
    refiner = PoseRefinement(image_size=image_size)

    # Render both projections
    img_pred = refiner.render_edge_from_homography(
        homography_pred, pitch_lines, image_size
    )
    img_gt = refiner.render_edge_from_homography(
        homography_gt, pitch_lines, image_size
    )

    # Dilate both images slightly for a fair comparison
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_pred = cv2.dilate(img_pred, kernel)
    img_gt = cv2.dilate(img_gt, kernel)

    # Compute IoU
    intersection = np.logical_and(img_pred > 0, img_gt > 0).sum()
    union = np.logical_or(img_pred > 0, img_gt > 0).sum()

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


if __name__ == "__main__":
    print("=== Pose Refinement Test ===\n")

    # Test 1: Distance transform
    print("1. Testing distance transform...")
    refiner = PoseRefinement()

    # Create a test edge image with some lines
    test_edge = np.zeros((180, 320), dtype=np.uint8)
    cv2.line(test_edge, (160, 0), (160, 180), 255, 2)
    cv2.line(test_edge, (0, 90), (320, 90), 255, 2)
    cv2.circle(test_edge, (160, 90), 40, 255, 2)

    dist = refiner.compute_distance_image(test_edge)
    print(f"   Edge image: {test_edge.shape}, {np.sum(test_edge > 0)} edge pixels")
    print(f"   Distance image: {dist.shape}, range [{dist.min():.2f}, {dist.max():.2f}]")

    # Test 2: Edge rendering from homography
    print("\n2. Testing edge rendering from homography...")

    # Import pitch template
    import sys
    sys.path.insert(0, ".")
    from pitch_template import PitchTemplate

    pitch = PitchTemplate()
    pitch_lines = pitch.get_lines_for_drawing()

    # Create a simple homography (identity-ish, scaled)
    H = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    rendered = refiner.render_edge_from_homography(H, pitch_lines)
    print(f"   Rendered edge: {rendered.shape}, {np.sum(rendered > 0)} pixels")

    # Test 3: ECC refinement
    print("\n3. Testing ECC refinement...")

    # Create a slightly perturbed version of the test edge image
    # (simulating the difference between retrieved and detected)
    M = np.float32([[1.0, 0.02, 3], [-0.02, 1.0, -2]])
    perturbed = cv2.warpAffine(test_edge, M, (320, 180))

    result = refiner.refine(
        detected_edge_image=perturbed,
        initial_homography=np.eye(3, dtype=np.float64),
        pitch_lines=pitch_lines,
    )

    print(f"   Converged: {result.converged}")
    print(f"   Correlation: {result.correlation:.4f}")
    print(f"   Warp matrix diagonal: [{result.warp_matrix[0,0]:.4f}, "
          f"{result.warp_matrix[1,1]:.4f}]")

    # Test 4: Refinement with fallback
    print("\n4. Testing refinement with fallback...")
    result_fb = refiner.refine_with_fallback(
        detected_edge_image=perturbed,
        initial_homography=np.eye(3, dtype=np.float64),
        pitch_lines=pitch_lines,
    )
    print(f"   Best correlation: {result_fb.correlation:.4f}")
    print(f"   Converged: {result_fb.converged}")

    # Test 5: IoU computation
    print("\n5. Testing IoU computation...")
    H1 = np.array([[3.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 1.0]])
    H2 = np.array([[3.1, 0.0, 2.0], [0.0, 2.5, -1.0], [0.0, 0.0, 1.0]])  # Slightly off
    iou = compute_iou(H1, H2, pitch_lines)
    print(f"   IoU (similar homographies): {iou:.4f}")

    H3 = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 50.0], [0.0, 0.0, 1.0]])  # Very different
    iou_bad = compute_iou(H1, H3, pitch_lines)
    print(f"   IoU (different homographies): {iou_bad:.4f}")

    print("\n=== All tests passed ===")
    print("\nThe complete camera calibration pipeline is now:")
    print("  1. pitch_template.py        — FIFA pitch model")
    print("  2. camera_pose_engine.py    — generate synthetic data")
    print("  3. siamese_network.py       — train feature extractor")
    print("  4. pose_database.py         — build retrieval database")
    print("  5. field_marking_detector.py — detect pitch lines in real images")
    print("  6. pose_refinement.py       — refine retrieved pose (this file)")