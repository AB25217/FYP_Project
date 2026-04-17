"""
test_camera_calibration.py — End-to-end test of the camera calibration pipeline.

Runs the full pipeline on synthetic data to verify everything works:
    1. Generate synthetic edge images (camera pose engine)
    2. Train siamese network (small scale, CPU)
    3. Build pose database (KD-tree)
    4. Query with held-out test images
    5. Measure retrieval accuracy (pan/tilt/focal errors)
    6. Test pose tracker across simulated frames
    7. Visualise results

Usage:
    python test_camera_calibration.py
"""

import numpy as np
import cv2
import os
import time
import sys

# Ensure both the package dir and src/ are on sys.path so bare module
# imports resolve when running as a script from any working directory.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_pkg_dir)
for _p in (_pkg_dir, _src_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pitch_template import PitchTemplate, draw_pitch_template
from camera_pose_engine import CameraPoseEngine, CameraParams, visualise_sample
from siamese_network import SiameseNetwork, train_siamese, load_trained_model
from pose_database import PoseDatabase
from pose_refinement import PoseRefinement, compute_iou
from pose_tracker import CameraPoseTracker


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    output_dir = "test_calibration_pipeline"
    os.makedirs(output_dir, exist_ok=True)

    # Pitch Template
    separator("STEP 1: Pitch Template")

    pitch = PitchTemplate()
    markings = pitch.get_all_markings()
    pitch_lines = pitch.get_lines_for_drawing()

    total_points = sum(len(pts) for pts in markings.values())
    print(f"Pitch: {pitch.length}m x {pitch.width}m")
    print(f"Markings: {len(markings)} elements, {total_points} points")

    # Save pitch visualisation
    draw_pitch_template(os.path.join(output_dir, "pitch_template.png"))
    print("Saved pitch_template.png")


    # Generate Synthetic Data
    separator("STEP 2: Generate Synthetic Data")

    engine = CameraPoseEngine()

    # Generate training set (500 samples — small for testing)
    train_dir = os.path.join(output_dir, "train_data")
    print("Generating 500 training samples...")
    start = time.time()
    engine.generate_dataset(train_dir, num_samples=500, print_every=250)
    print(f"Time: {time.time() - start:.1f}s")

    # Generate separate test set (50 samples)
    test_dir = os.path.join(output_dir, "test_data")
    print("\nGenerating 50 test samples...")
    engine.generate_dataset(test_dir, num_samples=50, print_every=50)

    # Visualise some samples
    print("\nSaving sample edge images...")
    for i in range(3):
        edge_img, params = engine.generate_sample()
        visualise_sample(
            edge_img, params,
            os.path.join(output_dir, f"sample_edge_{i}.png")
        )

    # 
    # Train Siamese Network

    separator("STEP 3: Train Siamese Network")

    model_path = os.path.join(output_dir, "siamese.pth")

    print("Training siamese network (10 epochs, CPU)...")
    print("This takes 2-5 minutes on a laptop.\n")

    start = time.time()
    model = train_siamese(
        data_dir=train_dir,
        output_path=model_path,
        num_epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        num_pairs_per_epoch=2000,
        save_every=10,
    )
    train_time = time.time() - start
    print(f"\nTraining time: {train_time:.1f}s")

    
    # Build Pose Database
    
    separator("STEP 4: Build Pose Database")

    db = PoseDatabase()
    db.build_from_synthetic(
        model_path=model_path,
        data_dir=train_dir,
        batch_size=32,
        print_every=250,
    )

    db_path = os.path.join(output_dir, "pose_database.npz")
    db.save(db_path)
    print(f"Database: {db.num_entries} entries")

    #Test Retrieval Accuracy
    separator("STEP 5: Test Retrieval Accuracy")

    # Reload database (tests save/load cycle)
    db_loaded = PoseDatabase.load(
        database_path=db_path,
        model_path=model_path,
    )

    # Evaluate on held-out test set
    stats = db_loaded.evaluate_retrieval(
        test_data_dir=test_dir,
        num_tests=30,
    )

    print(f"\nRetrieval quality assessment:")
    if stats["pan_error_mean"] < 5.0:
        print(f"  Pan error {stats['pan_error_mean']:.1f}° — GOOD (< 5°)")
    else:
        print(f"  Pan error {stats['pan_error_mean']:.1f}° — needs more training data")

    if stats["tilt_error_mean"] < 2.0:
        print(f"  Tilt error {stats['tilt_error_mean']:.1f}° — GOOD (< 2°)")
    else:
        print(f"  Tilt error {stats['tilt_error_mean']:.1f}° — needs more training data")


    #  Test Single Image Pipeline
    separator("STEP 6: Test Single Image Pipeline")

    # Generate a test image with known parameters
    test_params = CameraParams(
        pan=10.0, tilt=-10.0, focal_length=3000.0,
        cx=52.0, cy=-45.0, cz=17.0, roll=0.0
    )

    P = engine.build_projection_matrix(test_params)
    test_edge = engine.render_edge_image(P)

    print(f"Ground truth: pan={test_params.pan}°, tilt={test_params.tilt}°, "
          f"focal={test_params.focal_length}px")

    # Query database
    result = db_loaded.query(test_edge, k=1)
    print(f"Retrieved:    pan={result.pan:.1f}°, tilt={result.tilt:.1f}°, "
          f"focal={result.focal_length:.0f}px")
    print(f"Distance:     {result.distance:.4f}")

    # Errors
    pan_err = abs(result.pan - test_params.pan)
    tilt_err = abs(result.tilt - test_params.tilt)
    focal_err = abs(result.focal_length - test_params.focal_length)
    print(f"Errors:       pan={pan_err:.1f}°, tilt={tilt_err:.1f}°, "
          f"focal={focal_err:.0f}px")

    # Test top-5 retrieval
    result_k5 = db_loaded.query(test_edge, k=5)
    print(f"\nTop-5 avg:    pan={result_k5.pan:.1f}°, tilt={result_k5.tilt:.1f}°, "
          f"focal={result_k5.focal_length:.0f}px")

    #  Test Pose Refinement
    
    separator("STEP 7: Test Pose Refinement")

    refiner = PoseRefinement()

    # Get initial homography from retrieved pose
    retrieved_params = CameraParams(
        pan=result.pan, tilt=result.tilt,
        focal_length=result.focal_length,
        cx=result.cx, cy=result.cy, cz=result.cz,
        roll=result.roll
    )
    initial_H = engine.get_homography(retrieved_params)
    gt_H = engine.get_homography(test_params)

    # Compute IoU before refinement
    iou_before = compute_iou(initial_H, gt_H, pitch_lines)
    print(f"IoU before refinement: {iou_before:.4f}")

    # Run refinement
    refinement_result = refiner.refine_with_fallback(
        detected_edge_image=test_edge,
        initial_homography=initial_H,
        pitch_lines=pitch_lines,
    )

    # Compute IoU after refinement
    iou_after = compute_iou(refinement_result.homography, gt_H, pitch_lines)
    print(f"IoU after refinement:  {iou_after:.4f}")
    print(f"Refinement converged:  {refinement_result.converged}")
    print(f"ECC correlation:       {refinement_result.correlation:.4f}")

    if iou_after >= iou_before:
        print("Refinement IMPROVED the result")
    else:
        print("Refinement did not improve (common with small test data)")

    
    # Test Pose Tracker
    separator("STEP 8: Test Pose Tracker")

    tracker = CameraPoseTracker(estimation_interval=10)

    # Simulate a video sequence: camera slowly panning
    print("Simulating 20-frame video with slow camera pan...\n")

    for frame_idx in range(20):
        # Simulate camera panning from -10° to +10° over 20 frames
        current_pan = -10.0 + frame_idx * 1.0

        sim_params = CameraParams(
            pan=current_pan, tilt=-10.0, focal_length=3000.0,
            cx=52.0, cy=-45.0, cz=17.0
        )
        P_sim = engine.build_projection_matrix(sim_params)
        edge_sim = engine.render_edge_image(P_sim)

        if tracker.needs_estimation():
            # Full estimation (in real pipeline: siamese → database → refine)
            H_est = engine.get_homography(sim_params)
            state = tracker.set_estimation(H_est, edge_sim, frame_idx)
            status = "ESTIMATED"
        else:
            state = tracker.track(edge_sim, frame_idx)
            status = "tracked  "

        print(f"  Frame {frame_idx:2d}: {status} | "
              f"pan={current_pan:+5.1f}° | "
              f"corr={state.correlation:.3f} | "
              f"valid={state.tracking_valid}")

    # Print tracker stats
    tracker_stats = tracker.get_stats()
    print(f"\nTracker statistics:")
    print(f"  Total frames: {tracker_stats['total_frames']}")
    print(f"  Full estimations: {tracker_stats['estimation_frames']}")
    print(f"  Tracking ratio: {tracker_stats['tracking_ratio']:.1%}")
    print(f"  Mean correlation: {tracker_stats['mean_correlation']:.3f}")

    
    # SUMMARY
    
    separator("PIPELINE TEST COMPLETE")

    print("Results saved to:", output_dir)
    print()

if __name__ == "__main__":
    main()