#!/usr/bin/env python3
"""
Universal AI Viral Clipper v2 (Multi-Genre Open-Source OpusClip Alternative)
=============================================================================
100% Commercial-Safe Open-Source Stack for YouTube Shorts & Facebook Reels:
  - yt-dlp (Unlicense)          : Live chat, comment timestamp mining, heatmap, & HD slicing
  - OpenCV (Apache 2.0)         : Frame-by-frame motion, flash, scene-cut, & face tracking
  - scikit-learn (BSD 3-Clause) : Decision Tree + Neural Network self-learning viral ensemble
  - faster-whisper (MIT)        : Word-level speech transcription & speaking-rate scoring
  - Kokoro-82M (Apache 2.0)     : Optional commercial-safe AI voiceover hook narrator
  - FFmpeg (LGPL/GPL)           : Genre-adaptive 9:16 layouts, zoom motion, loudnorm & captions
  - rclone (MIT)                : Optional Google Drive sync

Works on BOTH:
  1. Long Livestreams (4-10h VODs) -> Live Chat Explosion + Chat-Lag Compensation + Heatmap
  2. Normal Videos (Gaming, Podcasts, Horror, Comedy, Entertainment) ->
     Full Frame-by-Frame Proxy Scan + Comment Timestamp Mining + Heatmap + AI ML Scoring!
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from viral_ml_engine import FEATURE_NAMES, ViralMLEnsemble, detect_video_genre


# ---------------------------------------------------------------------------
# Multi-Genre Hype Lexicon & Visual Themes
# ---------------------------------------------------------------------------

HYPE_KEYWORDS = {
    "omg", "wtf", "no way", "clip", "clip it", "insane", "crazy", "bro", "bruh",
    "lmao", "lmfao", "lol", "haha", "w", "ww", "www", "goat", "sheesh", "fire",
    "holy", "yooo", "pog", "lets go", "let's go", "scary", "jumpscare", "scream",
    "plot twist", "facts", "real", "truth", "genius", "hilarious", "dead",
    "cooked", "clutch", "wild", "unbelievable", "epic", "legendary", "gta",
}

HYPE_EMOJIS = ("🔥", "😂", "💀", "😱", "🤯", "😭", "🚨", "👑", "⚡", "💥", "🐐", "👀")

# Genre-specific caption colors (ASS &HBBGGRR format) & default layouts
GENRE_PROFILES = {
    "gaming": {
        "default_layout": "blur_stack",
        "highlight_color": "&H0000FFFF&",   # Bright Yellow
        "hook_bg": "&HB0000000",
        "add_vignette": False,
        "voice": "am_michael",
    },
    "podcast": {
        "default_layout": "auto_face_track",
        "highlight_color": "&H00FFFF00&",   # Electric Cyan
        "hook_bg": "&HB0101010",
        "add_vignette": False,
        "voice": "af_heart",
    },
    "horror": {
        "default_layout": "center_crop",
        "highlight_color": "&H002222FF&",   # Dramatic Red
        "hook_bg": "&HD0000000",
        "add_vignette": True,
        "voice": "am_adam",
    },
    "comedy": {
        "default_layout": "auto_face_track",
        "highlight_color": "&H0000FF7F&",   # Vibrant Neon Green
        "hook_bg": "&HB0000000",
        "add_vignette": False,
        "voice": "af_bella",
    },
    "entertainment": {
        "default_layout": "blur_stack",
        "highlight_color": "&H0000D7FF&",   # Gold
        "hook_bg": "&HB0000000",
        "add_vignette": False,
        "voice": "af_heart",
    },
}


@dataclass
class ClipCandidate:
    rank: int
    start_sec: float
    end_sec: float
    peak_sec: float
    genre: str = "entertainment"
    features: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in FEATURE_NAMES})
    tree_score: float = 0.0
    nn_score: float = 0.0
    final_score: float = 0.0
    face_x_norm: float = 0.5   # Smoothed horizontal face center [0.0..1.0] for auto_face_track
    top_chat_samples: List[str] = field(default_factory=list)
    transcript: str = ""
    viral_title: str = ""
    hook_text: str = ""
    voiceover_script: str = ""
    hashtags: List[str] = field(default_factory=lambda: ["#Shorts", "#Reels", "#Viral"])
    output_video: str = ""


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: List[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = min(99, int(round((seconds - int(seconds)) * 100)))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape_ffmpeg_path(path: Path) -> str:
    s = str(path.resolve()).replace("\\", "/")
    for ch in (":", "'", "=", ",", "[", "]"):
        s = s.replace(ch, "\\" + ch)
    return s


# ---------------------------------------------------------------------------
# Step 1: Metadata, Live Chat, Heatmap & Comment Timestamp Mining
# ---------------------------------------------------------------------------

def fetch_metadata_chat_comments(
    youtube_url: str,
    work_dir: Path,
    cookies_file: Optional[str] = None,
    max_comments: int = 300,
) -> Tuple[Optional[Path], Optional[Path], Dict[str, Any]]:
    """
    Download lightweight info.json (with heatmap & top comments for normal videos)
    AND live_chat.json (for livestreams) without downloading the full HD video.
    """
    meta_dir = work_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(meta_dir / "source.%(ext)s")

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-info-json",
        "--write-comments",
        "--extractor-args", f"youtube:max_comments={max_comments},all,100",
        "--write-subs",
        "--sub-langs", "live_chat",
        "--no-warnings",
        "-o", out_template,
    ]
    if cookies_file and Path(cookies_file).exists():
        cmd.extend(["--cookies", cookies_file])
    cmd.append(youtube_url)

    run_cmd(cmd, check=False)

    info_files = list(meta_dir.glob("*.info.json"))
    chat_files = list(meta_dir.glob("*.live_chat.json"))
    info_path = info_files[0] if info_files else None
    chat_path = chat_files[0] if chat_files else None

    info_data: Dict[str, Any] = {}
    if info_path and info_path.exists():
        try:
            info_data = json.loads(info_path.read_text(encoding="utf-8"))
            print(f"[INFO] Title: {info_data.get('title', 'Unknown')}")
            print(f"[INFO] Duration: {format_timestamp(float(info_data.get('duration') or 0))}")
            print(f"[INFO] Is Live/VOD: {bool(info_data.get('was_live') or chat_path)}")
        except Exception as exc:
            print(f"[WARN] Failed to read info.json: {exc}")

    return info_path, chat_path, info_data


def parse_comment_timestamps(info_data: Dict[str, Any], bucket_sec: int = 5) -> Tuple[Dict[int, float], Dict[int, List[str]]]:
    """
    For NORMAL YouTube videos (podcasts, horror, comedy, gaming highlights),
    viewers drop timestamps like '4:12 I died laughing' or '1:02:15 best part'.
    Mining these gives laser-accurate viral moments even without live chat!
    """
    ts_buckets: Dict[int, float] = defaultdict(float)
    ts_samples: Dict[int, List[str]] = defaultdict(list)
    comments = info_data.get("comments") or []
    ts_pattern = re.compile(r"\b(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)\b")

    hits = 0
    for c in comments:
        text = str(c.get("text") or "").strip()
        likes = int(c.get("like_count") or 0)
        if not text:
            continue
        matches = ts_pattern.findall(text)
        if not matches:
            continue
        weight = 1.0 + math.log1p(max(0, likes)) * 0.6
        for h_str, m_str, s_str in matches[:3]:
            sec = int(h_str or 0) * 3600 + int(m_str) * 60 + int(s_str)
            b_idx = int(sec // bucket_sec)
            ts_buckets[b_idx] += weight
            hits += 1
            if len(ts_samples[b_idx]) < 4 and len(text) <= 140:
                ts_samples[b_idx].append(text.replace("\n", " "))

    if hits:
        print(f"[COMMENTS] Found {hits} viewer timestamp mentions across {len(ts_buckets)} moments.")
    return ts_buckets, ts_samples


def parse_live_chat_buckets(chat_path: Path, bucket_sec: int = 5) -> Tuple[Dict[int, float], Dict[int, List[str]], float]:
    """Stream-parse YouTube live_chat.json (JSONL) with low memory usage."""
    bucket_scores: Dict[int, float] = defaultdict(float)
    bucket_samples: Dict[int, List[str]] = defaultdict(list)
    max_offset_sec = 0.0
    total_msgs = 0

    with open(chat_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            replay = obj.get("replayChatItemAction")
            if not replay:
                continue
            try:
                offset_sec = float(replay.get("videoOffsetTimeMsec")) / 1000.0
            except (TypeError, ValueError):
                continue
            if offset_sec < 0:
                continue
            max_offset_sec = max(max_offset_sec, offset_sec)

            for act in replay.get("actions", []):
                item = act.get("addChatItemAction", {}).get("item", {})
                renderer = item.get("liveChatTextMessageRenderer")
                is_paid = False
                if not renderer:
                    renderer = item.get("liveChatPaidMessageRenderer")
                    is_paid = renderer is not None
                if not renderer:
                    continue

                runs = renderer.get("message", {}).get("runs", [])
                text = "".join(
                    r.get("text", "") if "text" in r else (r.get("emoji", {}).get("shortcuts", [""])[0])
                    for r in runs
                ).strip()
                if not text:
                    continue

                total_msgs += 1
                w = 3.0 if is_paid else 1.0
                lower = text.lower()
                w += min(2.5, sum(1 for kw in HYPE_KEYWORDS if kw in lower) * 0.8)
                w += min(2.0, sum(text.count(e) for e in HYPE_EMOJIS) * 0.5)
                b_idx = int(offset_sec // bucket_sec)
                bucket_scores[b_idx] += w
                if len(bucket_samples[b_idx]) < 5 and len(text) <= 120:
                    bucket_samples[b_idx].append(text)

    print(f"[CHAT] Parsed {total_msgs:,} live chat messages.")
    return bucket_scores, bucket_samples, max_offset_sec


def parse_heatmap_buckets(info_data: Dict[str, Any], bucket_sec: int = 5) -> Dict[int, float]:
    heatmap_buckets: Dict[int, float] = defaultdict(float)
    for entry in info_data.get("heatmap") or []:
        start = float(entry.get("start_time", 0.0))
        end = float(entry.get("end_time", start + bucket_sec))
        val = float(entry.get("value", 0.0)) * 100.0
        for b in range(int(start // bucket_sec), int(end // bucket_sec) + 1):
            heatmap_buckets[b] = max(heatmap_buckets[b], val)
    return heatmap_buckets


# ---------------------------------------------------------------------------
# Step 2: Mandatory Frame-by-Frame Visual Analyzer (OpenCV + Face Tracking)
# ---------------------------------------------------------------------------

def analyze_video_frame_by_frame(
    video_path: Path,
    bucket_sec: int = 5,
    frame_stride: int = 1,
) -> Dict[str, Any]:
    """
    Reads the video frame-by-frame via OpenCV to extract per-bucket:
      - motion_score (frame-to-frame pixel delta)
      - scene_cut_score (histogram transition spikes)
      - flash_spike_score (sudden luminance jumps — explosions / horror jump scares)
      - face_ratio (% of sampled frames containing faces)
      - face_x_norm (smoothed horizontal center of detected face for smart 9:16 crop)
      - avg_brightness (helps detect dark horror vs bright gaming/podcasts)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"buckets": {}, "avg_face_ratio": 0.0, "avg_motion": 30.0, "avg_brightness": 110.0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Load OpenCV's built-in Haar Cascade face detector (Apache 2.0)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path)) if cascade_path.exists() else None

    bucket_motion: Dict[int, List[float]] = defaultdict(list)
    bucket_cuts: Dict[int, int] = defaultdict(int)
    bucket_flashes: Dict[int, List[float]] = defaultdict(list)
    bucket_faces: Dict[int, List[int]] = defaultdict(list)
    bucket_face_x: Dict[int, List[float]] = defaultdict(list)
    all_brightness: List[float] = []

    prev_gray = None
    prev_hist = None
    prev_luma = None
    frame_idx = 0
    face_check_interval = max(1, int(round(fps * 0.5)))  # Face check 2x/sec for CPU speed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            t_sec = frame_idx / fps
            b_idx = int(t_sec // bucket_sec)

            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            luma = float(np.mean(gray))
            all_brightness.append(luma)

            # Frame-by-frame motion & flash spike
            if prev_gray is not None:
                diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
                bucket_motion[b_idx].append(diff)

                luma_jump = abs(luma - prev_luma) if prev_luma is not None else 0.0
                bucket_flashes[b_idx].append(luma_jump)

            # Scene cut detection via normalized grayscale histogram correlation
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if corr < 0.72:
                    bucket_cuts[b_idx] += 1

            # Periodic face detection & X-center tracking
            if face_cascade is not None and frame_idx % face_check_interval == 0:
                med = cv2.resize(frame, (320, 180))
                med_gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(med_gray, scaleFactor=1.2, minNeighbors=4, minSize=(24, 24))
                if len(faces) > 0:
                    bucket_faces[b_idx].append(1)
                    # Track largest face's horizontal center [0.0..1.0]
                    largest = max(faces, key=lambda box: box[2] * box[3])
                    cx = (largest[0] + largest[2] / 2.0) / 320.0
                    bucket_face_x[b_idx].append(float(cx))
                else:
                    bucket_faces[b_idx].append(0)

            prev_gray = gray
            prev_hist = hist
            prev_luma = luma

        frame_idx += 1

    cap.release()
    print(f"[FRAME-SCAN] Analyzed {frame_idx:,} frames across {len(bucket_motion)} time buckets.")

    buckets: Dict[int, Dict[str, float]] = {}
    all_motion_scores = []
    all_face_ratios = []

    for b_idx in sorted(set(bucket_motion.keys()) | set(bucket_faces.keys())):
        m_avg = float(np.mean(bucket_motion[b_idx])) if bucket_motion[b_idx] else 0.0
        motion_score = min(100.0, m_avg * 3.5)
        cut_score = min(100.0, bucket_cuts.get(b_idx, 0) * 22.0)
        f_max = float(np.max(bucket_flashes[b_idx])) if bucket_flashes[b_idx] else 0.0
        flash_score = min(100.0, f_max * 2.5)
        face_ratio = (float(np.mean(bucket_faces[b_idx])) * 100.0) if bucket_faces[b_idx] else 0.0
        face_x = float(np.median(bucket_face_x[b_idx])) if bucket_face_x[b_idx] else 0.5

        all_motion_scores.append(motion_score)
        all_face_ratios.append(face_ratio)
        buckets[b_idx] = {
            "motion_score": round(motion_score, 1),
            "scene_cut_score": round(cut_score, 1),
            "flash_spike_score": round(flash_score, 1),
            "face_ratio": round(face_ratio, 1),
            "face_x_norm": round(face_x, 3),
        }

    return {
        "buckets": buckets,
        "avg_face_ratio": round(float(np.mean(all_face_ratios)) if all_face_ratios else 0.0, 1),
        "avg_motion": round(float(np.mean(all_motion_scores)) if all_motion_scores else 30.0, 1),
        "avg_brightness": round(float(np.mean(all_brightness)) if all_brightness else 110.0, 1),
    }


def download_lightweight_proxy(youtube_url: str, proxy_path: Path, cookies_file: Optional[str] = None) -> bool:
    """
    For normal videos (<= 90 min), downloads a tiny 360p proxy (~40-150 MB)
    so we can run TRUE frame-by-frame OpenCV analysis across the entire video
    without running out of GitHub Actions disk space!
    """
    cmd = [
        "yt-dlp",
        "-f", "worstvideo[height>=240][ext=mp4]/bestvideo[height<=360][ext=mp4]/worst[ext=mp4]/worst",
        "--no-warnings",
        "-o", str(proxy_path),
    ]
    if cookies_file and Path(cookies_file).exists():
        cmd.extend(["--cookies", cookies_file])
    cmd.append(youtube_url)
    res = run_cmd(cmd, check=False)
    return res.returncode == 0 and proxy_path.exists()


# ---------------------------------------------------------------------------
# Step 3: Audio Energy, Whisper Speech Rate & Kokoro-82M Commercial Voiceover
# ---------------------------------------------------------------------------

def compute_audio_features(clip_path: Path) -> Tuple[float, float]:
    """Return (audio_energy_0_100, audio_punch_0_100) via FFmpeg volumedetect."""
    try:
        res = run_cmd(
            ["ffmpeg", "-i", str(clip_path), "-af", "volumedetect", "-f", "null", "-"],
            check=False,
            capture=True,
        )
        stderr = res.stderr or ""
        mean_m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)
        max_m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", stderr)
        if mean_m and max_m:
            mean_db = float(mean_m.group(1))
            max_db = float(max_m.group(1))
            energy = max(0.0, min(100.0, (mean_db + 38.0) * 3.2))
            punch = max(0.0, min(100.0, (max_db - mean_db) * 4.5))
            return round(energy, 1), round(punch, 1)
    except Exception:
        pass
    return 50.0, 50.0


def transcribe_clip_words(clip_path: Path, model_size: str = "base") -> Tuple[str, List[Dict[str, Any]], float]:
    """Transcribe speech with word-level timestamps & compute speaking rate (0-100)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return "", [], 40.0

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(clip_path), word_timestamps=True, vad_filter=True)
    words: List[Dict[str, Any]] = []
    texts: List[str] = []
    for seg in segments:
        texts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                clean = w.word.strip()
                if clean:
                    words.append({"word": clean, "start": float(w.start), "end": float(w.end)})

    dur = max(1.0, float(getattr(info, "duration", 30.0) or 30.0))
    wps = len(words) / dur
    speech_rate_score = round(min(100.0, wps * 28.0), 1)
    return " ".join(texts).strip(), words, speech_rate_score


def generate_kokoro_voiceover(text: str, out_wav: Path, voice: str = "af_heart") -> bool:
    """
    Generate a 100% commercial-safe AI voiceover hook using Kokoro-82M (Apache 2.0).
    """
    if not text.strip():
        return False
    try:
        from kokoro import KPipeline
        import soundfile as sf

        pipeline = KPipeline(lang_code="a")
        audio_chunks = []
        for _, _, audio in pipeline(text, voice=voice, speed=1.08):
            audio_chunks.append(audio)
        if not audio_chunks:
            return False
        full_audio = np.concatenate(audio_chunks)
        sf.write(str(out_wav), full_audio, 24000)
        print(f"[VOICEOVER] Generated Kokoro-82M hook audio ({len(full_audio)/24000:.1f}s).")
        return out_wav.exists()
    except Exception as exc:
        print(f"[VOICEOVER WARN] Kokoro-82M skipped ({exc}).")
        return False


# ---------------------------------------------------------------------------
# Step 4: Free LLM Viral Hook, Title & Voiceover Script Generator
# ---------------------------------------------------------------------------

def generate_viral_metadata(candidate: ClipCandidate, video_title: str = "") -> None:
    """Use free Groq / OpenRouter / GitHub Models API (or clean offline fallback)."""
    api_key, endpoint, model = None, None, None
    if os.getenv("GROQ_API_KEY"):
        api_key = os.getenv("GROQ_API_KEY")
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    elif os.getenv("OPENROUTER_API_KEY"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        model = os.getenv("LLM_MODEL", "openrouter/free")
    elif os.getenv("GITHUB_TOKEN"):
        api_key = os.getenv("GITHUB_TOKEN")
        endpoint = "https://models.inference.ai.azure.com/chat/completions"
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    genre_label = candidate.genre.upper()
    if not api_key or not endpoint:
        candidate.viral_title = f"INSANE {genre_label} Moment You Won't Believe #{candidate.rank}"
        candidate.hook_text = "WAIT UNTIL THE END..."
        candidate.voiceover_script = "Wait until you see what happens right here."
        candidate.hashtags = [f"#{candidate.genre.capitalize()}", "#Viral", "#Shorts", "#Reels"]
        return

    prompt = (
        f"You are an elite viral Shorts/Reels producer specializing in {genre_label} videos.\n"
        f"Source video title: {video_title}\n"
        f"Transcript: {candidate.transcript or '(visual action / reaction)'}\n"
        f"Audience reactions: {', '.join(candidate.top_chat_samples[:5]) or '(high visual/audio spike)'}\n\n"
        f"Return ONLY valid JSON with:\n"
        f'  "title": viral title under 65 chars,\n'
        f'  "hook": 3-5 word UPPERCASE screen hook,\n'
        f'  "voiceover": 1 punchy 8-12 word spoken intro hook for a 2.5s voiceover,\n'
        f'  "hashtags": list of 5 relevant hashtags.'
    )
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.75,
        "max_tokens": 220,
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            candidate.viral_title = str(parsed.get("title", "")).strip()[:80]
            candidate.hook_text = str(parsed.get("hook", "")).strip()[:40]
            candidate.voiceover_script = str(parsed.get("voiceover", "")).strip()[:120]
            tags = parsed.get("hashtags")
            if isinstance(tags, list) and tags:
                candidate.hashtags = [str(t) for t in tags[:6]]
            return
    except Exception as exc:
        print(f"[LLM WARN] Free API skipped ({exc}); using offline hook.")

    candidate.viral_title = f"CRAZY {genre_label} Moment! #{candidate.rank}"
    candidate.hook_text = "WAIT FOR THIS..."
    candidate.voiceover_script = "You are not ready for what happens next."
    candidate.hashtags = [f"#{candidate.genre.capitalize()}", "#Shorts", "#Reels", "#Viral"]


# ---------------------------------------------------------------------------
# Step 5: Genre-Themed Animated Captions & Advanced FFmpeg 9:16 Renderer
# ---------------------------------------------------------------------------

def write_genre_ass_captions(
    words: List[Dict[str, Any]],
    ass_path: Path,
    genre: str = "entertainment",
    hook_text: str = "",
) -> None:
    profile = GENRE_PROFILES.get(genre, GENRE_PROFILES["entertainment"])
    hl = profile["highlight_color"]
    hook_bg = profile["hook_bg"]

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,DejaVu Sans,74,&H00FFFFFF,{hl},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,420,1
Style: Hook,DejaVu Sans,62,{hl},&H00FFFFFF,&H00000000,{hook_bg},-1,0,0,0,100,100,0,0,3,6,0,8,60,60,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    if hook_text:
        clean_hook = hook_text.replace("\n", " ").replace("{", "").replace("}", "")
        lines.append(f"Dialogue: 1,{format_ass_time(0.15)},{format_ass_time(3.4)},Hook,,0,0,0,,{clean_hook}\n")

    for i in range(0, len(words), 3):
        chunk = words[i : i + 3]
        for active_idx, active_word in enumerate(chunk):
            w_start = active_word["start"]
            w_end = active_word["end"]
            if active_idx + 1 < len(chunk):
                w_end = max(w_end, chunk[active_idx + 1]["start"])
            w_end = max(w_end, w_start + 0.15)
            tokens = []
            for j, tok in enumerate(chunk):
                t = tok["word"].upper().replace("{", "").replace("}", "")
                if j == active_idx:
                    tokens.append(r"{\c" + hl + r"\fscx114\fscy114}" + t + r"{\r}")
                else:
                    tokens.append(t)
            lines.append(
                f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},Caption,,0,0,0,,{' '.join(tokens)}\n"
            )

    ass_path.write_text("".join(lines), encoding="utf-8")


def render_advanced_short(
    input_clip: Path,
    ass_path: Path,
    output_mp4: Path,
    genre: str = "entertainment",
    layout: str = "auto",
    face_x_norm: float = 0.5,
    facecam_corner: str = "bottom_right",
    voiceover_wav: Optional[Path] = None,
) -> bool:
    """
    Render a 1080x1920 (9:16) high-production Short with:
      - Auto genre layout selection (blur_stack / auto_face_track / split_facecam / center_crop)
      - Smooth subtle motion punch-in on hook (first 3s)
      - Optional horror vignette
      - Broadcast audio loudness normalization (-14 LUFS)
      - Optional Kokoro-82M voiceover hook mixed cleanly with audio ducking
    """
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    profile = GENRE_PROFILES.get(genre, GENRE_PROFILES["entertainment"])
    chosen_layout = profile["default_layout"] if layout == "auto" else layout
    ass_escaped = escape_ffmpeg_path(ass_path)

    # Video layout filtergraph
    if chosen_layout == "split_facecam":
        corner_map = {
            "bottom_right": "iw*0.72:ih*0.68",
            "bottom_left": "0:ih*0.68",
            "top_right": "iw*0.72:0",
            "top_left": "0:0",
        }
        xy = corner_map.get(facecam_corner, "iw*0.72:ih*0.68")
        base_vf = (
            f"[0:v]crop=iw*0.28:ih*0.32:{xy},scale=1080:672:force_original_aspect_ratio=increase,"
            f"crop=1080:672[cam];"
            f"[0:v]scale=-2:1248,crop=1080:1248:(iw-1080)/2:0[game];"
            f"[cam][game]vstack=inputs=2[base]"
        )
    elif chosen_layout == "auto_face_track":
        # Center the 9:16 crop dynamically on the detected speaker face_x_norm [0.0..1.0]
        cx = max(0.18, min(0.82, face_x_norm))
        base_vf = (
            f"[0:v]scale=-2:1920,"
            f"crop=1080:1920:max(0\\,min(iw-1080\\,iw*{cx:.3f}-540)):0[base]"
        )
    elif chosen_layout == "center_crop":
        base_vf = "[0:v]scale=-2:1920,crop=1080:1920:(iw-1080)/2:0[base]"
    else:
        # blur_stack (keeps full 16:9 frame + blurred vertical fill)
        base_vf = (
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg];"
            "[0:v]scale=1080:-2[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
        )

    post_fx = "vignette=PI/4," if profile["add_vignette"] else ""
    vf = f"{base_vf};[base]{post_fx}eq=contrast=1.04:saturation=1.08,ass=filename='{ass_escaped}'[v]"

    cmd = ["ffmpeg", "-y", "-i", str(input_clip)]
    if voiceover_wav and voiceover_wav.exists():
        cmd.extend(["-i", str(voiceover_wav)])
        # Duck main video audio cleanly during the opening voiceover hook, then normalize
        af = (
            "[0:a]volume='if(lt(t,3.5),0.32,1.0)':eval=frame[bg_a];"
            "[1:a]aresample=48000,volume=1.35[vo_a];"
            "[bg_a][vo_a]amix=inputs=2:duration=first:dropout_transition=1,loudnorm=I=-14:TP=-1.5:LRA=11[a]"
        )
        cmd.extend([
            "-filter_complex", f"{vf};{af}",
            "-map", "[v]", "-map", "[a]",
        ])
    else:
        cmd.extend([
            "-filter_complex", vf,
            "-map", "[v]", "-map", "0:a?",
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        ])

    cmd.extend([
        "-c:v", "libx264", "-preset", "medium", "-crf", "21",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        str(output_mp4),
    ])
    res = run_cmd(cmd, check=False)
    return res.returncode == 0 and output_mp4.exists()


# ---------------------------------------------------------------------------
# Segment Slicer (1080p HD Slice)
# ---------------------------------------------------------------------------

def slice_hd_clip(
    start_sec: float,
    end_sec: float,
    clip_path: Path,
    youtube_url: Optional[str] = None,
    local_video: Optional[Path] = None,
    cookies_file: Optional[str] = None,
) -> bool:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(5.0, end_sec - start_sec)

    if local_video and local_video.exists():
        res = run_cmd([
            "ffmpeg", "-y", "-ss", f"{start_sec:.2f}", "-i", str(local_video),
            "-t", f"{duration:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-c:a", "aac", "-b:a", "160k", str(clip_path),
        ], check=False)
        return res.returncode == 0 and clip_path.exists()

    if youtube_url:
        cmd = [
            "yt-dlp",
            "--download-sections", f"*{start_sec:.1f}-{end_sec:.1f}",
            "--force-keyframes-at-cuts",
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--no-warnings",
            "-o", str(clip_path),
        ]
        if cookies_file and Path(cookies_file).exists():
            cmd.extend(["--cookies", cookies_file])
        cmd.append(youtube_url)
        res = run_cmd(cmd, check=False)
        return res.returncode == 0 and clip_path.exists()

    return False


# ---------------------------------------------------------------------------
# Offline Self-Test Mode (Verifies Frame-by-Frame + ML Ensemble + Rendering)
# ---------------------------------------------------------------------------

def run_demo_self_test(out_dir: Path) -> None:
    print("[DEMO v2] Running full offline test (Frame-by-Frame + Decision Tree + Neural Net + Render)...")
    demo_dir = out_dir / "_demo_v2"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a 35s synthetic test video with audio
    test_vid = demo_dir / "mock_video.mp4"
    run_cmd([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25:duration=35",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=35",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
        str(test_vid),
    ])

    # 2. Run mandatory frame-by-frame OpenCV analyzer
    scan = analyze_video_frame_by_frame(test_vid, bucket_sec=5, frame_stride=2)
    assert len(scan["buckets"]) >= 5, "Expected >= 5 scanned buckets"

    # 3. Auto-detect genre
    genre = detect_video_genre(
        title="Scary Midnight Ghost Hunting in Abandoned Hospital",
        description="Creepy paranormal jumpscare moments",
        tags=["horror", "scary"],
        avg_face_ratio=scan["avg_face_ratio"],
        avg_motion=scan["avg_motion"],
        avg_brightness=scan["avg_brightness"],
    )
    print(f"[DEMO v2] Auto-detected genre: {genre}")
    assert genre == "horror", f"Expected horror genre, got {genre}"

    # 4. Train & run Hybrid Decision Tree + Neural Network ML Ensemble
    ml = ViralMLEnsemble(
        model_path=demo_dir / "viral_ensemble.joblib",
        feedback_csv=demo_dir / "training_feedback.csv",
    )
    b0 = scan["buckets"][0]
    feats = {
        "motion_score": b0["motion_score"],
        "scene_cut_score": b0["scene_cut_score"],
        "flash_spike_score": b0["flash_spike_score"],
        "face_ratio": b0["face_ratio"],
        "audio_energy": 68.0,
        "audio_punch": 74.0,
        "speech_rate": 52.0,
        "chat_score": 0.0,
        "heatmap_score": 82.0,
        "comment_ts_score": 65.0,
    }
    ens, tree_s, nn_s = ml.predict_viral_score(feats)
    print(f"[DEMO v2] ML Ensemble Score: {ens}/100 (Decision Tree: {tree_s}, Neural Net: {nn_s})")
    assert 0.0 <= ens <= 100.0
    ml.log_clip_for_future_learning("demo123", genre, "short_1.mp4", feats, ens)
    assert (demo_dir / "training_feedback.csv").exists()

    # 5. Render genre-styled 9:16 vertical Short with ASS captions & loudnorm
    ass_path = demo_dir / "short_1.ass"
    write_genre_ass_captions(
        [{"word": "DID", "start": 0.5, "end": 0.8}, {"word": "YOU", "start": 0.8, "end": 1.1}, {"word": "SEE", "start": 1.1, "end": 1.6}, {"word": "THAT?!", "start": 1.6, "end": 2.3}],
        ass_path,
        genre=genre,
        hook_text="WAIT FOR THE JUMPSCARE",
    )
    out_mp4 = demo_dir / "short_1_horror_vertical.mp4"
    slice_vid = demo_dir / "slice_1.mp4"
    assert slice_hd_clip(0.0, 12.0, slice_vid, local_video=test_vid)
    assert render_advanced_short(slice_vid, ass_path, out_mp4, genre=genre, layout="auto")
    print(f"[DEMO v2] SUCCESS! Rendered {out_mp4.name} ({out_mp4.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Multi-Genre AI Viral Clipper v2")
    parser.add_argument("--url", type=str, help="YouTube URL (Livestream VOD OR Normal Video)")
    parser.add_argument("--video-path", type=str, help="Optional local/Google Drive video file")
    parser.add_argument("--out-dir", type=str, default="output_shorts", help="Output directory")
    parser.add_argument("--num-clips", type=int, default=5, help="Number of Shorts to generate")
    parser.add_argument("--clip-duration", type=int, default=45, help="Clip length in seconds (30-50)")
    parser.add_argument("--chat-lag", type=float, default=18.0, help="Livestream chat delay offset in seconds")
    parser.add_argument("--pre-roll", type=float, default=10.0, help="Build-up seconds before peak")
    parser.add_argument(
        "--genre",
        type=str,
        default="auto",
        choices=["auto", "gaming", "podcast", "comedy", "horror", "entertainment"],
        help="Content genre (default: auto-detect)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        choices=["auto", "blur_stack", "auto_face_track", "split_facecam", "center_crop"],
        help="9:16 layout (default: auto-select by genre)",
    )
    parser.add_argument(
        "--facecam-corner",
        type=str,
        default="bottom_right",
        choices=["bottom_right", "bottom_left", "top_right", "top_left"],
    )
    parser.add_argument("--whisper-model", type=str, default="base")
    parser.add_argument("--enable-voiceover", action="store_true", help="Add Kokoro-82M AI voiceover hook")
    parser.add_argument("--cookies", type=str, default=None, help="Optional YouTube cookies.txt")
    parser.add_argument("--feedback-csv", type=str, default="training_feedback.csv")
    parser.add_argument("--model-path", type=str, default="models/viral_ensemble.joblib")
    parser.add_argument("--demo-test", action="store_true", help="Run full offline verification self-test")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo_test:
        run_demo_self_test(out_dir)
        return

    if not args.url and not args.video_path:
        parser.error("Provide --url and/or --video-path.")

    work_dir = out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    bucket_sec = 5

    # 1. Fetch metadata, live chat (if livestream), heatmap, and viewer comment timestamps
    info_data: Dict[str, Any] = {}
    chat_path: Optional[Path] = None
    if args.url:
        _, chat_path, info_data = fetch_metadata_chat_comments(args.url, work_dir, args.cookies)

    local_video = Path(args.video_path) if args.video_path else None
    duration_sec = float(info_data.get("duration") or 0.0)
    if duration_sec <= 0 and local_video and local_video.exists():
        probe = run_cmd(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(local_video)],
            check=False, capture=True
        )
        try:
            duration_sec = float((probe.stdout or "0").strip())
        except ValueError:
            duration_sec = 0.0
    video_id = str(info_data.get("id") or (local_video.stem if local_video else "local_video"))
    video_title = str(info_data.get("title") or (local_video.stem.replace("_", " ") if local_video else "Viral Clip"))

    chat_buckets: Dict[int, float] = {}
    chat_samples: Dict[int, List[str]] = {}
    if chat_path and chat_path.exists():
        chat_buckets, chat_samples, max_chat_sec = parse_live_chat_buckets(chat_path, bucket_sec)
        if duration_sec <= 0:
            duration_sec = max_chat_sec + 30.0

    heatmap_buckets = parse_heatmap_buckets(info_data, bucket_sec)
    comment_buckets, comment_samples = parse_comment_timestamps(info_data, bucket_sec)

    # 2. Mandatory Frame-by-Frame OpenCV Scan:
    #    - For local files OR normal videos <= 90 min: scan full video via lightweight 360p proxy!
    #    - For ultra-long 4-10h livestreams: scan top candidate slices frame-by-frame to stay under disk limits!
    proxy_scan: Dict[str, Any] = {"buckets": {}, "avg_face_ratio": 0.0, "avg_motion": 35.0, "avg_brightness": 110.0}
    if local_video and local_video.exists():
        proxy_scan = analyze_video_frame_by_frame(local_video, bucket_sec=bucket_sec, frame_stride=2)
    elif args.url and 0 < duration_sec <= 5400:
        proxy_path = work_dir / "proxy_360p.mp4"
        if download_lightweight_proxy(args.url, proxy_path, args.cookies):
            proxy_scan = analyze_video_frame_by_frame(proxy_path, bucket_sec=bucket_sec, frame_stride=2)

    # 3. Auto-Detect Content Genre (Gaming / Podcast / Horror / Comedy / Entertainment)
    detected_genre = (
        detect_video_genre(
            title=video_title,
            description=str(info_data.get("description") or ""),
            tags=list(info_data.get("tags") or []),
            avg_face_ratio=proxy_scan["avg_face_ratio"],
            avg_motion=proxy_scan["avg_motion"],
            avg_brightness=proxy_scan["avg_brightness"],
        )
        if args.genre == "auto"
        else args.genre
    )
    print(f"[GENRE] Active content profile: {detected_genre.upper()}")

    # 4. Initialize Self-Learning Decision Tree + Neural Network Ensemble
    ml_engine = ViralMLEnsemble(
        model_path=Path(args.model_path),
        feedback_csv=Path(args.feedback_csv),
    )

    # 5. Build Candidate Pool across the Timeline
    all_b = sorted(
        set(chat_buckets.keys())
        | set(heatmap_buckets.keys())
        | set(comment_buckets.keys())
        | set(proxy_scan["buckets"].keys())
    )
    if not all_b and duration_sec > 0:
        all_b = list(range(0, max(1, int(duration_sec // bucket_sec))))

    max_chat = max(chat_buckets.values()) if chat_buckets else 1.0
    max_comm = max(comment_buckets.values()) if comment_buckets else 1.0
    lag_shift = int(round(args.chat_lag / bucket_sec)) if chat_buckets else 0

    ranked_buckets: List[Tuple[float, int, Dict[str, float]]] = []
    for b in all_b:
        c_norm = min(100.0, (chat_buckets.get(b, 0.0) / max_chat) * 100.0) if chat_buckets else 0.0
        hm_norm = heatmap_buckets.get(max(0, b - lag_shift), 0.0)
        cm_norm = min(100.0, (comment_buckets.get(max(0, b - lag_shift), 0.0) / max_comm) * 100.0) if comment_buckets else 0.0
        vis = proxy_scan["buckets"].get(max(0, b - lag_shift), {})
        feats = {
            "motion_score": vis.get("motion_score", 35.0),
            "scene_cut_score": vis.get("scene_cut_score", 25.0),
            "flash_spike_score": vis.get("flash_spike_score", 20.0),
            "face_ratio": vis.get("face_ratio", 25.0),
            "audio_energy": 50.0,
            "audio_punch": 50.0,
            "speech_rate": 50.0,
            "chat_score": round(c_norm, 1),
            "heatmap_score": round(hm_norm, 1),
            "comment_ts_score": round(cm_norm, 1),
        }
        pre_score, _, _ = ml_engine.predict_viral_score(feats)
        ranked_buckets.append((pre_score, b, feats))

    ranked_buckets.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate non-overlapping candidate windows (pick extra candidates for deep HD verification)
    pool_size = max(args.num_clips, min(args.num_clips + 3, 10))
    min_gap = max(60.0, args.clip_duration * 1.5)
    candidates: List[ClipCandidate] = []

    for pre_score, b_idx, feats in ranked_buckets:
        if len(candidates) >= pool_size:
            break
        peak_sec = b_idx * bucket_sec + (bucket_sec / 2.0)
        shift = args.chat_lag if chat_buckets else 0.0
        start_sec = max(0.0, peak_sec - shift - args.pre_roll)
        end_sec = start_sec + args.clip_duration
        if duration_sec > 0 and end_sec > duration_sec:
            end_sec = duration_sec
            start_sec = max(0.0, end_sec - args.clip_duration)
        if any(abs(start_sec - c.start_sec) < min_gap for c in candidates):
            continue

        samples = (chat_samples.get(b_idx, []) + comment_samples.get(b_idx, []))[:6]
        vis_b = proxy_scan["buckets"].get(max(0, b_idx - lag_shift), {})
        candidates.append(
            ClipCandidate(
                rank=len(candidates) + 1,
                start_sec=round(start_sec, 2),
                end_sec=round(end_sec, 2),
                peak_sec=round(peak_sec, 2),
                genre=detected_genre,
                features=feats,
                face_x_norm=vis_b.get("face_x_norm", 0.5),
                top_chat_samples=samples,
            )
        )

    # 6. Slice Candidate HD Clips -> Deep Frame-by-Frame + Audio + Whisper -> Final ML Ensemble Score
    evaluated: List[Tuple[ClipCandidate, Path, List[Dict[str, Any]]]] = []
    for idx, cand in enumerate(candidates, start=1):
        raw_hd = work_dir / f"hd_slice_{idx:02d}.mp4"
        if not slice_hd_clip(cand.start_sec, cand.end_sec, raw_hd, args.url, local_video, args.cookies):
            continue

        # Run deep frame-by-frame scan on the exact HD clip (guarantees frame-by-frame even on 10h streams!)
        hd_scan = analyze_video_frame_by_frame(raw_hd, bucket_sec=5, frame_stride=2)
        if hd_scan["buckets"]:
            vals = list(hd_scan["buckets"].values())
            cand.features["motion_score"] = round(float(np.mean([v["motion_score"] for v in vals])), 1)
            cand.features["scene_cut_score"] = round(float(np.mean([v["scene_cut_score"] for v in vals])), 1)
            cand.features["flash_spike_score"] = round(float(np.max([v["flash_spike_score"] for v in vals])), 1)
            cand.features["face_ratio"] = round(float(np.mean([v["face_ratio"] for v in vals])), 1)
            cand.face_x_norm = round(float(np.median([v["face_x_norm"] for v in vals])), 3)

        audio_e, audio_p = compute_audio_features(raw_hd)
        cand.features["audio_energy"] = audio_e
        cand.features["audio_punch"] = audio_p

        transcript, words, speech_rate = transcribe_clip_words(raw_hd, model_size=args.whisper_model)
        cand.transcript = transcript
        cand.features["speech_rate"] = speech_rate

        # Final Decision Tree + Neural Network Hybrid Prediction!
        ens_s, tree_s, nn_s = ml_engine.predict_viral_score(cand.features)
        cand.final_score = ens_s
        cand.tree_score = tree_s
        cand.nn_score = nn_s
        evaluated.append((cand, raw_hd, words))

    evaluated.sort(key=lambda item: item[0].final_score, reverse=True)
    winners = evaluated[: args.num_clips]

    # 7. Render Winning 9:16 Shorts (Genre-Adaptive + Optional Kokoro-82M Voiceover)
    finished: List[ClipCandidate] = []
    profile = GENRE_PROFILES.get(detected_genre, GENRE_PROFILES["entertainment"])

    for final_rank, (cand, raw_hd, words) in enumerate(winners, start=1):
        cand.rank = final_rank
        generate_viral_metadata(cand, video_title=video_title)

        vo_wav: Optional[Path] = None
        if args.enable_voiceover and cand.voiceover_script:
            vo_candidate = work_dir / f"vo_{final_rank:02d}.wav"
            if generate_kokoro_voiceover(cand.voiceover_script, vo_candidate, voice=profile["voice"]):
                vo_wav = vo_candidate

        ass_path = work_dir / f"short_{final_rank:02d}.ass"
        write_genre_ass_captions(words, ass_path, genre=cand.genre, hook_text=cand.hook_text)

        safe_slug = re.sub(r"[^a-zA-Z0-9]+", "_", cand.viral_title).strip("_")[:40]
        final_mp4 = out_dir / f"Short_{final_rank:02d}_{cand.genre}_{safe_slug}.mp4"

        if render_advanced_short(
            raw_hd,
            ass_path,
            final_mp4,
            genre=cand.genre,
            layout=args.layout,
            face_x_norm=cand.face_x_norm,
            facecam_corner=args.facecam_corner,
            voiceover_wav=vo_wav,
        ):
            cand.output_video = final_mp4.name
            finished.append(cand)
            ml_engine.log_clip_for_future_learning(
                video_id=video_id,
                genre=cand.genre,
                clip_filename=final_mp4.name,
                features=cand.features,
                predicted_score=cand.final_score,
            )
            print(
                f"[DONE] {final_mp4.name} | Genre: {cand.genre} | "
                f"Ensemble: {cand.final_score} (Tree: {cand.tree_score}, NN: {cand.nn_score})"
            )

    # 8. Export Metadata JSON & Upload CSV
    (out_dir / "shorts_metadata.json").write_text(
        json.dumps([asdict(c) for c in finished], indent=2), encoding="utf-8"
    )
    with open(out_dir / "upload_sheet.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank", "Genre", "Filename", "ML_Viral_Score", "DecisionTree_Score", "NeuralNet_Score",
            "Start", "End", "Title", "Hook", "Voiceover_Script", "Hashtags", "Transcript",
        ])
        for c in finished:
            writer.writerow([
                c.rank, c.genre, c.output_video, c.final_score, c.tree_score, c.nn_score,
                format_timestamp(c.start_sec), format_timestamp(c.end_sec),
                c.viral_title, c.hook_text, c.voiceover_script, " ".join(c.hashtags), c.transcript,
            ])

    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"\n[COMPLETE] Generated {len(finished)} AI-scored 9:16 Shorts in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
