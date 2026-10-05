import os

import torch
import torchaudio  # type: ignore

# torchaudio's MP3 decoding (via its only backend, soundfile/libsndfile) has proven
# unreliable across multiple environments for at least one real recording: on macOS
# it silently truncated a 297s file to 0.08s with no exception at all, and the same
# truncation reproduced on a separate Linux/CUDA production build too — where
# torchaudio.info() ALSO reported the truncated length, so cross-checking the decode
# against torchaudio's own metadata (tried first) didn't catch it either; torchaudio's
# self-reporting can't be trusted here. ffmpeg (via pydub), by contrast, has decoded
# this file correctly on every platform tested. For MP3, skip torchaudio entirely and
# go straight to the reliable path — worth the extra ffmpeg-subprocess overhead given
# the alternative is silently losing most of the audio.
_UNRELIABLE_TORCHAUDIO_EXTENSIONS = {".mp3"}

# Safety net for OTHER formats, in case one of them exhibits the same silent-truncation
# behavior some day: cross-check the decoded length against the file's own header
# metadata (cheap, doesn't require a full decode) and treat anything catastrophically
# short as a failed decode.
_MIN_DECODED_DURATION_FRACTION = 0.5


def _assert_fully_decoded(audio_file_path: str, waveform, sample_rate: int) -> None:
    try:
        expected_frames = torchaudio.info(str(audio_file_path)).num_frames
    except Exception:
        # Can't cross-check — trust the decode rather than block on an unrelated issue.
        return

    if expected_frames <= 0:
        return

    decoded_frames = waveform.shape[-1]
    if decoded_frames < expected_frames * _MIN_DECODED_DURATION_FRACTION:
        raise RuntimeError(
            f"torchaudio.load() appears to have truncated '{audio_file_path}': decoded "
            f"{decoded_frames} frames ({decoded_frames / sample_rate:.1f}s) but the file's "
            f"own metadata reports {expected_frames} frames "
            f"({expected_frames / sample_rate:.1f}s)"
        )


def _load_via_pydub(audio_file_path: str):
    from pydub import AudioSegment
    tmp_wav_path = str(audio_file_path) + "_converted.wav"
    try:
        AudioSegment.from_file(audio_file_path).export(tmp_wav_path, format="wav")
        return torchaudio.load(tmp_wav_path)
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)


def safe_load_audio(audio_file_path: str):
    ext = os.path.splitext(str(audio_file_path))[1].lower()
    if ext in _UNRELIABLE_TORCHAUDIO_EXTENSIONS:
        return _load_via_pydub(audio_file_path)

    try:
        waveform, sample_rate = torchaudio.load(str(audio_file_path))
        _assert_fully_decoded(audio_file_path, waveform, sample_rate)
        return waveform, sample_rate
    except RuntimeError:
        return _load_via_pydub(audio_file_path)


def pad_audio(audio_file_path: str) -> dict:
    waveform, sample_rate = safe_load_audio(audio_file_path)

    chunk_size = 160000
    remainder = waveform.shape[-1] % chunk_size
    if remainder != 0:
        pad_size = chunk_size - remainder
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))

    return {"waveform": waveform, "sample_rate": sample_rate}


def load_audio_for_whisper(audio_file_path: str, sampling_rate: int):
    """Decode audio via safe_load_audio and return a mono float32 numpy array at
    sampling_rate — the same shape faster-whisper's own decode_audio() produces.

    faster-whisper decodes audio itself via PyAV, bypassed entirely by passing a
    numpy array into WhisperModel.transcribe() instead of a file path. This matters
    because PyAV has shown the exact same silent-truncation failure mode as
    torchaudio for at least one real MP3 (confirmed directly: faster_whisper.audio.
    decode_audio() returned 0.08s of a 297s recording, with no exception, same file
    where torchaudio.load() also silently truncated). Whisper should go through our
    one reliable loader instead of trusting either decoder's own file handling.
    """
    import numpy as np

    waveform, sample_rate = safe_load_audio(audio_file_path)
    mono = waveform.mean(dim=0)

    if sample_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sampling_rate)
        mono = resampler(mono.unsqueeze(0)).squeeze(0)

    return mono.numpy().astype(np.float32)
