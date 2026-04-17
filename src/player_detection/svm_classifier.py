"""
svm_classifier.py — SVM classifier for player/non-player classification.

Trains a Support Vector Machine on HOG features extracted from
positive (player) and negative (background) patches.

Supports:
    - Linear and RBF kernels
    - Cross-validation for hyperparameter tuning
    - Model saving/loading via joblib
    - Probability estimates for confidence scoring

Training is fast (minutes on CPU) since HOG features are compact.
"""

import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    roc_auc_score
)
from typing import Tuple, Optional


class SVMClassifier:
    """
    SVM classifier for player detection.

    Usage:
        classifier = SVMClassifier()
        classifier.train(features, labels)
        classifier.save("weights/svm_player.pkl")

        # Later:
        classifier = SVMClassifier.load("weights/svm_player.pkl")
        prediction = classifier.predict(hog_feature)
        confidence = classifier.predict_confidence(hog_feature)
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 10.0,
        gamma: str = "scale",
        probability: bool = True,
    ):
        """
        Args:
            kernel: 'linear' or 'rbf'
            C: regularisation parameter (higher = less regularisation)
            gamma: kernel coefficient for RBF ('scale' or 'auto')
            probability: enable probability estimates (needed for confidence)
        """
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                probability=probability,
                class_weight="balanced",
            )),
        ])
        self.is_trained = False

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        cross_validate: bool = True,
        cv_folds: int = 5,
    ) -> dict:
        """
        Train the SVM classifier.

        Args:
            features: (N, feature_dim) HOG feature matrix
            labels: (N,) labels (1=player, 0=background)
            cross_validate: run cross-validation before final training
            cv_folds: number of CV folds

        Returns:
            dict: training statistics
        """
        print(f"Training SVM on {len(features)} samples...")
        print(f"  Positive: {np.sum(labels == 1)}")
        print(f"  Negative: {np.sum(labels == 0)}")
        print(f"  Feature dim: {features.shape[1]}")

        stats = {}

        # Cross-validation
        if cross_validate and len(features) >= cv_folds * 2:
            print(f"\n  Running {cv_folds}-fold cross-validation...")
            cv_scores = cross_val_score(
                self.pipeline, features, labels,
                cv=cv_folds, scoring="f1"
            )
            stats["cv_f1_mean"] = float(np.mean(cv_scores))
            stats["cv_f1_std"] = float(np.std(cv_scores))
            print(f"  CV F1-score: {stats['cv_f1_mean']:.3f} "
                  f"(+/- {stats['cv_f1_std']:.3f})")

        # Train on full dataset
        print("\n  Training final model...")
        self.pipeline.fit(features, labels)
        self.is_trained = True

        # Training accuracy
        train_pred = self.pipeline.predict(features)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, train_pred, average="binary", pos_label=1
        )

        stats["train_precision"] = float(precision)
        stats["train_recall"] = float(recall)
        stats["train_f1"] = float(f1)

        print(f"\n  Training metrics:")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall:    {recall:.3f}")
        print(f"    F1-score:  {f1:.3f}")

        return stats

    def train_with_grid_search(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        cv_folds: int = 5,
    ) -> dict:
        """
        Train with hyperparameter grid search to find the best
        C and gamma values.

        Takes longer but finds better parameters.

        Args:
            features: HOG feature matrix
            labels: class labels

        Returns:
            dict: best parameters and scores
        """
        print("Running grid search over SVM parameters...")

        param_grid = {
            "svm__C": [0.1, 1.0, 10.0, 100.0],
            "svm__gamma": ["scale", "auto", 0.01, 0.001],
        }

        grid_search = GridSearchCV(
            self.pipeline, param_grid,
            cv=cv_folds, scoring="f1",
            n_jobs=-1, verbose=1
        )

        grid_search.fit(features, labels)

        self.pipeline = grid_search.best_estimator_
        self.is_trained = True

        print(f"\n  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV F1: {grid_search.best_score_:.3f}")

        return {
            "best_params": grid_search.best_params_,
            "best_f1": float(grid_search.best_score_),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            features: (N, feature_dim) or (feature_dim,) HOG features

        Returns:
            np.ndarray: predicted labels (1=player, 0=background)
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        return self.pipeline.predict(features)

    def predict_confidence(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (confidence scores).

        Args:
            features: (N, feature_dim) or (feature_dim,) HOG features

        Returns:
            np.ndarray: probability of being a player (0-1)
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        probs = self.pipeline.predict_proba(features)
        # Column 1 is the probability of class 1 (player)
        return probs[:, 1]

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """
        Evaluate the classifier on a test set.

        Args:
            features: test HOG features
            labels: test labels

        Returns:
            dict: precision, recall, F1, AUC-ROC
        """
        predictions = self.predict(features)
        confidences = self.predict_confidence(features)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", pos_label=1
        )

        try:
            auc = roc_auc_score(labels, confidences)
        except ValueError:
            auc = 0.0

        stats = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc),
        }

        print(f"Evaluation results:")
        print(f"  Precision: {stats['precision']:.3f}")
        print(f"  Recall:    {stats['recall']:.3f}")
        print(f"  F1-score:  {stats['f1']:.3f}")
        print(f"  AUC-ROC:   {stats['auc_roc']:.3f}")

        return stats

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")
        print(f"  File size: {os.path.getsize(path) / 1024:.1f} KB")

    @classmethod
    def load(cls, path: str) -> "SVMClassifier":
        """Load a trained model from disk."""
        instance = cls()
        instance.pipeline = joblib.load(path)
        instance.is_trained = True
        print(f"Model loaded from {path}")
        return instance


if __name__ == "__main__":
    print("=== SVM Classifier Test ===\n")

    # Generate synthetic data
    print("1. Generating synthetic training data...")
    np.random.seed(42)
    n_pos, n_neg = 200, 200
    dim = 1980  # Matches 48x96 HOG feature dim

    # Positive features: slightly different distribution
    pos_features = np.random.randn(n_pos, dim).astype(np.float32) + 0.5
    neg_features = np.random.randn(n_neg, dim).astype(np.float32) - 0.5

    features = np.vstack([pos_features, neg_features])
    labels = np.array([1] * n_pos + [0] * n_neg)

    # Shuffle
    idx = np.random.permutation(len(features))
    features = features[idx]
    labels = labels[idx]

    print(f"   Samples: {len(features)}, dim: {dim}")

    # Train
    print("\n2. Training SVM...")
    classifier = SVMClassifier(kernel="rbf", C=10.0)
    stats = classifier.train(features, labels, cross_validate=True)

    # Predict
    print("\n3. Testing prediction...")
    test_feat = np.random.randn(5, dim).astype(np.float32)
    preds = classifier.predict(test_feat)
    confs = classifier.predict_confidence(test_feat)
    print(f"   Predictions: {preds}")
    print(f"   Confidences: {np.round(confs, 3)}")

    # Save and load
    print("\n4. Testing save/load...")
    classifier.save("test_svm.pkl")
    loaded = SVMClassifier.load("test_svm.pkl")
    preds2 = loaded.predict(test_feat)
    print(f"   Predictions match: {np.array_equal(preds, preds2)}")

    # Cleanup
    os.remove("test_svm.pkl")

    print("\n=== Tests complete ===")