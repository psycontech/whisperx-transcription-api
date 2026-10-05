"""Runs inside a dedicated single-worker subprocess (see service.py's
ProcessPoolExecutor), never imported by the main app process directly.

Deliberately kept minimal — no FastAPI, no faster-whisper, no transformers.
When this module is spawned (multiprocessing "spawn" context, required for
CUDA correctness — see service.py), it gets re-imported from scratch in a
brand-new process, so anything imported here is paid for on every subprocess
(re)spawn. Pulling in the rest of the app's dependency graph would only slow
that down for no benefit, since the subprocess only ever does one thing:
diarization + overlapped-speech-detection inference.

The main diarization Pipeline and OSD Pipeline are shared, mutable pyannote
objects that must not cross a process boundary — each subprocess builds and
caches its own copy on first use (see get_diarization_pipeline/get_osd_pipeline
below), lazily, exactly once per process lifetime, and reuses it across calls
for as long as this process lives. No lock is needed around that lazy-init:
a ProcessPoolExecutor(max_workers=1) worker processes one submitted call at a
time, so there is never concurrent access within a single subprocess.
"""

from typing import Optional

import torch
from pyannote.audio import Pipeline  # type: ignore
from pyannote.audio.pipelines import OverlappedSpeechDetection  # type: ignore

from app.whisper.audio_utils import pad_audio

_diarization_pipeline: Optional[Pipeline] = None
_osd_pipeline: Optional[OverlappedSpeechDetection] = None


def get_diarization_pipeline(hf_token: str) -> Pipeline:
    global _diarization_pipeline
    if _diarization_pipeline is None:
        print("[diarization-worker] loading diarization pipeline...")
        if torch.cuda.is_available():
            print("[diarization-worker] CUDA is available")
        else:
            print("[diarization-worker] CUDA not available, using CPU")
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        if torch.cuda.is_available():
            _diarization_pipeline.to(torch.device("cuda"))
    return _diarization_pipeline


def get_osd_pipeline(segmentation_model: str = "pyannote/segmentation-3.0") -> OverlappedSpeechDetection:
    global _osd_pipeline
    if _osd_pipeline is None:
        print("[diarization-worker] loading overlapped speech detection pipeline...")
        _osd_pipeline = OverlappedSpeechDetection(segmentation=segmentation_model)
        _osd_pipeline.instantiate({
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        })
        if torch.cuda.is_available():
            _osd_pipeline.to(torch.device("cuda"))
    return _osd_pipeline


def get_overlap_regions(audio_input: dict) -> list:
    """Run overlapped speech detection once for this file and return (start, end) regions
    where two or more speakers are talking at once.

    Takes the same pre-loaded {"waveform", "sample_rate"} dict the main diarization
    pipeline uses (built by pad_audio) rather than a raw file path — pyannote's Audio
    IO uses "waveform" directly when present, otherwise it falls back to
    torchaudio.load() on the raw path with no format-conversion fallback, which fails
    outright on containers like .webm that soundfile can't parse.
    """
    osd_pipeline = get_osd_pipeline()
    osd_output = osd_pipeline(audio_input)
    return [(seg.start, seg.end) for seg in osd_output.get_timeline().support()]


def diarize_in_subprocess(
    audio_file_path: str,
    hf_token: str,
    clustering_threshold: float,
    min_duration_off: float,
    min_cluster_size: int,
    num_of_speakers: Optional[int],
) -> dict:
    """Entry point submitted to the dedicated diarization ProcessPoolExecutor.

    Must stay a top-level function (picklable by reference) and take only
    plain, cheaply-picklable arguments — no pyannote/torch objects, no open
    file handles. Returns plain (float, float, str) tuples rather than
    pyannote Segment objects for the same reason: the parent process
    reconstructs whatever shape it needs from these.
    """
    audio_input = pad_audio(audio_file_path)
    decoded_duration = audio_input["waveform"].shape[-1] / audio_input["sample_rate"]
    print(
        f"[diarization-worker] decoded audio: shape={tuple(audio_input['waveform'].shape)} "
        f"sample_rate={audio_input['sample_rate']} duration={decoded_duration:.2f}s"
    )

    print("[diarization-worker] running overlapped speech detection...")
    overlap_regions = get_overlap_regions(audio_input)

    diarization_pipeline = get_diarization_pipeline(hf_token)
    diarization_pipeline.instantiate({
        "segmentation": {
            "min_duration_off": min_duration_off,
        },
        "clustering": {
            "threshold": clustering_threshold,
            "method": "centroid",
            "min_cluster_size": min_cluster_size,
        }
    })

    diarization_kwargs = {}
    if num_of_speakers:
        diarization_kwargs["min_speakers"] = num_of_speakers
        diarization_kwargs["max_speakers"] = num_of_speakers

    print("[diarization-worker] running diarization...")
    diarization = diarization_pipeline(audio_input, **diarization_kwargs)

    tracks = [
        (float(turn.start), float(turn.end), speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    print(f"[diarization-worker] done: {len(tracks)} tracks, {len(overlap_regions)} overlap regions")
    return {"tracks": tracks, "overlap_regions": overlap_regions}
