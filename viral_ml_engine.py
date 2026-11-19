#!/usr/bin/env python3
"""
Self-Learning Viral Scoring Engine (Decision Trees + Neural Network Ensemble)
=============================================================================
100% Commercial-Safe Open-Source ML Stack:
  - scikit-learn (BSD 3-Clause) : RandomForestRegressor (Decision Trees) + MLPRegressor (Neural Network)
  - numpy (BSD 3-Clause)        : Fast vectorized feature normalization
  - joblib (BSD 3-Clause)       : Model persistence across GitHub Actions runs

How the Learning Environment Works:
  1. Extracts 10 multimodal features per candidate clip (frame motion, scene cuts,
     brightness/flash spikes, face presence, audio loudness, audio dynamic punch,
     speech rate, chat velocity, heatmap score, and comment timestamp mentions).
  2. Auto-detects the video genre (gaming, podcast, comedy, horror, entertainment).
  3. Trains a Hybrid Ensemble:
       - Decision Tree Ensemble (RandomForestRegressor) for non-linear rules
       - Multi-Layer Perceptron Neural Network (MLPRegressor) for smooth feature fusion
  4. Saves every generated clip's features into `training_feedback.csv`. Whenever you
     fill in actual YouTube Shorts / Facebook Reels performance (0-100) in that CSV,
     the engine automatically retrains on your real channel data!
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FEATURE_NAMES = [
    "motion_score",       # 0-100: Frame-by-frame visual motion intensity
    "scene_cut_score",    # 0-100: Shot/camera transition rate
    "flash_spike_score",  # 0-100: Sudden brightness/flash changes (explosions, jump scares)
    "face_ratio",         # 0-100: % of frames with clear faces (podcasts/reactions)
    "audio_energy",       # 0-100: Mean audio loudness energy
    "audio_punch",        # 0-100: Peak-to-mean dynamic spike (screams, laughter, bangs)
    "speech_rate",        # 0-100: Words per second excitement
    "chat_score",         # 0-100: Live chat explosion score (livestreams)
    "heatmap_score",      # 0-100: YouTube Most Replayed heatmap score
    "comment_ts_score",   # 0-100: Regular comment timestamp mentions (normal videos)
]

GENRE_KEYWORDS = {
    "gaming": {
        "gta", "gta 6", "gameplay", "walkthrough", "stream", "twitch", "boss",
        "clutch", "valorant", "fortnite", "minecraft", "elden ring", "fps",
        "speedrun", "rank", "kill", "respawn", "mission", "xbox", "ps5", "gaming",
    },
    "podcast": {
        "podcast", "interview", "episode", "ep.", "guest", "host", "debate",
        "talk show", "conversation", "joe rogan", "lex fridman", "diary of a ceo",
        "full episode", "clips", "sit down", "deep dive",
    },
    "horror": {
        "horror", "scary", "ghost", "haunted", "creepy", "paranormal", "jumpscare",
        "jump scare", "nightmare", "conjuring", "terrifying", "backrooms", "dark",
        "spooky", "demon", "mystery",
    },
    "comedy": {
        "funny", "comedy", "prank", "try not to laugh", "hilarious", "standup",
        "stand-up", "skit", "meme", "memes", "roast", "fails", "blooper", "parody",
        "mimic", "sound mimic", "party", "jokes",
    },
}


def detect_video_genre(
    title: str,
    description: str,
    tags: List[str],
    avg_face_ratio: float,
    avg_motion: float,
    avg_brightness: float,
    speech_density: float = 50.0,
) -> str:
    """
    Hybrid rule + visual-signal genre detector.
    Identifies: 'gaming', 'podcast', 'horror', 'comedy', or 'entertainment'.
    """
    blob = " ".join([title or "", description or "", " ".join(tags or [])]).lower()
    scores = {g: 0.0 for g in ("gaming", "podcast", "horror", "comedy", "entertainment")}

    for genre, kw_set in GENRE_KEYWORDS.items():
        for kw in kw_set:
            if kw in blob:
                scores[genre] += 2.5

    # Visual & audio structural cues
    if avg_face_ratio >= 55.0 and avg_motion <= 35.0:
        scores["podcast"] += 3.5
    if avg_motion >= 45.0 and avg_face_ratio <= 40.0:
        scores["gaming"] += 2.5
    if avg_brightness <= 55.0:
        scores["horror"] += 2.0
    if speech_density >= 65.0 and avg_face_ratio >= 40.0:
        scores["podcast"] += 1.5
        scores["comedy"] += 1.0

    best_genre, best_val = max(scores.items(), key=lambda kv: kv[1])
    if best_val < 2.0:
        return "entertainment"
    return best_genre


def _generate_calibrated_prior_dataset(n_samples: int = 240, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a calibrated multi-genre synthetic viral prior dataset so the
    Decision Tree + Neural Network ensemble works intelligently on Day 1
    before the user has logged real channel feedback.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 100.0, size=(n_samples, len(FEATURE_NAMES)))

    # Non-linear viral scoring ground-truth prior reflecting real short-form retention:
    # - Social proof (chat / heatmap / comment timestamps) strongly lifts virality
    # - Audio punch + visual motion/flash spikes capture action, horror, and laughs
    # - Face presence + speech rate captures podcast & comedy hooks
    social = np.maximum.reduce([X[:, 7], X[:, 8], X[:, 9]])
    action_punch = 0.45 * X[:, 0] + 0.20 * X[:, 1] + 0.35 * X[:, 2]
    audio_excitement = 0.45 * X[:, 4] + 0.55 * X[:, 5]
    talk_hook = 0.45 * X[:, 3] + 0.55 * X[:, 6]

    # Non-linear synergy: high audio punch + high visual spike or social spike = breakout clip
    synergy = np.where((audio_excitement > 65) & (action_punch > 60), 12.0, 0.0)
    synergy += np.where((social > 70) & (audio_excitement > 55), 10.0, 0.0)

    y = (
        0.34 * social
        + 0.24 * audio_excitement
        + 0.20 * action_punch
        + 0.14 * talk_hook
        + 0.08 * np.minimum(X[:, 0], X[:, 5])
        + synergy
        + rng.normal(0.0, 3.0, size=n_samples)
    )
    y = np.clip(y, 5.0, 99.0)
    return X, y


class ViralMLEnsemble:
    """
    Hybrid Decision Tree (RandomForestRegressor) + Neural Network (MLPRegressor)
    viral scorer with automatic self-learning from `training_feedback.csv`.
    """

    def __init__(self, model_path: Path, feedback_csv: Path):
        self.model_path = model_path
        self.feedback_csv = feedback_csv
        self.tree_model = None
        self.nn_model = None
        self.scaler = None
        self.is_sklearn_available = False
        self.trained_on_user_rows = 0
        self._load_or_train()

    def _read_user_feedback(self) -> Tuple[List[List[float]], List[float]]:
        """Read user-labeled clips from training_feedback.csv if available."""
        X_user: List[List[float]] = []
        y_user: List[float] = []
        if not self.feedback_csv.exists():
            return X_user, y_user

        try:
            with open(self.feedback_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_target = (row.get("actual_viral_score_0_100") or "").strip()
                    if not raw_target:
                        continue
                    try:
                        target = float(raw_target)
                        feats = [float(row.get(name, 0.0) or 0.0) for name in FEATURE_NAMES]
                        X_user.append(feats)
                        y_user.append(max(0.0, min(100.0, target)))
                    except ValueError:
                        continue
        except Exception as exc:
            print(f"[ML WARN] Could not read {self.feedback_csv}: {exc}")

        return X_user, y_user

    def _load_or_train(self) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            import joblib
            self.is_sklearn_available = True
        except ImportError:
            print("[ML WARN] scikit-learn not installed; using analytical fallback scorer.")
            self.is_sklearn_available = False
            return

        X_user, y_user = self._read_user_feedback()
        self.trained_on_user_rows = len(X_user)

        # Build training set: combine prior knowledge + upweighted real user feedback
        X_prior, y_prior = _generate_calibrated_prior_dataset(n_samples=240)
        if len(X_user) >= 5:
            # Repeat user feedback rows so real channel performance strongly steers the models
            repeat_factor = max(3, min(12, 120 // len(X_user)))
            X_u = np.repeat(np.array(X_user, dtype=float), repeat_factor, axis=0)
            y_u = np.repeat(np.array(y_user, dtype=float), repeat_factor, axis=0)
            X_train = np.vstack([X_prior, X_u])
            y_train = np.concatenate([y_prior, y_u])
            print(f"[ML] Training Decision Tree + Neural Network on {len(X_user)} real channel clips + prior.")
        else:
            X_train, y_train = X_prior, y_prior
            print("[ML] Training Decision Tree + Neural Network on multi-genre viral prior (Day 1 mode).")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Model 1: Decision Tree Ensemble (RandomForest)
        self.tree_model = RandomForestRegressor(
            n_estimators=120,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.tree_model.fit(X_train, y_train)

        # Model 2: Multi-Layer Perceptron Neural Network (64 -> 32 hidden layers)
        self.nn_model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=350,
            random_state=42,
            early_stopping=False,
        )
        self.nn_model.fit(X_scaled, y_train)

        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "tree_model": self.tree_model,
                    "nn_model": self.nn_model,
                    "scaler": self.scaler,
                    "user_rows": self.trained_on_user_rows,
                },
                self.model_path,
            )
        except Exception as exc:
            print(f"[ML WARN] Could not save model artifact: {exc}")

    def predict_viral_score(self, features: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Predict viral score using Decision Tree + Neural Network ensemble.
        Returns: (ensemble_score, tree_score, nn_score)
        """
        vec = np.array([[float(features.get(name, 0.0)) for name in FEATURE_NAMES]], dtype=float)

        if not self.is_sklearn_available or self.tree_model is None or self.nn_model is None:
            # Clean fallback formula if scikit-learn is unavailable
            social = max(vec[0, 7], vec[0, 8], vec[0, 9])
            score = float(np.clip(0.35 * social + 0.30 * vec[0, 5] + 0.20 * vec[0, 0] + 0.15 * vec[0, 3], 0.0, 100.0))
            return round(score, 1), round(score, 1), round(score, 1)

        tree_pred = float(np.clip(self.tree_model.predict(vec)[0], 0.0, 100.0))
        nn_pred = float(np.clip(self.nn_model.predict(self.scaler.transform(vec))[0], 0.0, 100.0))
        ensemble = float(np.clip(0.55 * tree_pred + 0.45 * nn_pred, 0.0, 100.0))
        return round(ensemble, 1), round(tree_pred, 1), round(nn_pred, 1)

    def log_clip_for_future_learning(
        self,
        video_id: str,
        genre: str,
        clip_filename: str,
        features: Dict[str, float],
        predicted_score: float,
    ) -> None:
        """
        Append clip features to training_feedback.csv so the user can later enter
        actual performance (0-100) to continuously improve the AI on their channel.
        """
        self.feedback_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.feedback_csv.exists()
        fieldnames = (
            ["video_id", "genre", "clip_filename", "predicted_score"]
            + FEATURE_NAMES
            + ["actual_viral_score_0_100"]
        )
        with open(self.feedback_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            row = {
                "video_id": video_id,
                "genre": genre,
                "clip_filename": clip_filename,
                "predicted_score": predicted_score,
                "actual_viral_score_0_100": "",
            }
            for name in FEATURE_NAMES:
                row[name] = round(float(features.get(name, 0.0)), 2)
            writer.writerow(row)