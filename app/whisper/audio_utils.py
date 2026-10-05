import os

import torch
import torchaudio  # type: ignore

# Some malformed MP3s (an illegal frame header soundfile's MP3 parser can't resync
# past) make torchaudio.load() silently return a tiny truncated fragment instead of
# raising — e.g. one real file decoded to 0.08s of a 297s recording with no exception
# at all. A successful return alone isn't proof the file actually decoded, so the
# decoded length is cross-checked against the file's own header metadata (cheap,
# doesn't require a full decode) and anything catastrophically short is treated as a
# failed decode and routed through the pydub/ffmpeg fallback below, which doesn't
# share this failure mode.
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


def safe_load_audio(audio_file_path: str):
    tmp_wav_path = None
    try:
        waveform, sample_rate = torchaudio.load(str(audio_file_path))
        _assert_fully_decoded(audio_file_path, waveform, sample_rate)
        return waveform, sample_rate
    except RuntimeError:
        from pydub import AudioSegment
        tmp_wav_path = str(audio_file_path) + "_converted.wav"
        AudioSegment.from_file(audio_file_path).export(tmp_wav_path, format="wav")
        waveform, sample_rate = torchaudio.load(tmp_wav_path)
        return waveform, sample_rate
    finally:
        if tmp_wav_path and os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)


def pad_audio(audio_file_path: str) -> dict:
    waveform, sample_rate = safe_load_audio(audio_file_path)

    chunk_size = 160000
    remainder = waveform.shape[-1] % chunk_size
    if remainder != 0:
        pad_size = chunk_size - remainder
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))

    return {"waveform": waveform, "sample_rate": sample_rate}
