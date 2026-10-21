"""Transcribe the 15.9 meeting recording (Hebrew + English terms) with a Hebrew-tuned Whisper."""
import sys, time
from faster_whisper import WhisperModel

AUDIO = r"D:\Studies\FinalProject\MeetingVideos\GMT20260915-095533_Recording.m4a"
OUT = r"D:\Studies\FinalProject\PythonProject\lca_creativity\meeting\transcript_2026-09-15.txt"

model = WhisperModel("ivrit-ai/whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16")
segments, info = model.transcribe(AUDIO, language="he", beam_size=5, vad_filter=True)
print(f"duration {info.duration/60:.1f} min", flush=True)
t0 = time.time()
with open(OUT, "w", encoding="utf-8") as f:
    for s in segments:
        m, sec = divmod(int(s.start), 60)
        line = f"[{m:03d}:{sec:02d}] {s.text.strip()}"
        f.write(line + "\n"); f.flush()
print(f"done in {time.time()-t0:.0f}s", flush=True)
