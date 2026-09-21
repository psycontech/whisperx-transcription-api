import os

import torch
import torchaudio  # type: ignore


def safe_load_audio(audio_file_path: str):
    tmp_wav_path = None
    try:
        return torchaudio.load(str(audio_file_path))
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
