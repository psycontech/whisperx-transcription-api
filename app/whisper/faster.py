import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from faster_whisper import WhisperModel # type: ignore
from pyannote.audio import Pipeline # type: ignore

device = "cpu"
audio_file = "test_data/data.mp3"
HF_TOKEN = "..."


# 1. Transcribe with word timestamps
print("Transcribing...")
model = WhisperModel("small", device=device, compute_type="int8")
segments, info = model.transcribe(audio_file, beam_size=1, word_timestamps=True)

print("\nDetected Language: ", info.language, "\n")

result_segments = []
for segment in segments:
    result_segments.append({
        'start': segment.start,
        'end': segment.end,
        'text': segment.text,
        'words': segment.words
    })

# 2. Diarize
print("Loading diarization model...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

print("Diarizing audio...")
diarization = diarization_pipeline(audio_file, min_speakers=1, max_speakers=1)

print("\nSpeaker segments:")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")

# 3. Assign words to speakers
def assign_word_speakers(diarization, transcription_segments):
    words_with_speakers = []
    
    for segment in transcription_segments:
        if segment['words'] is None:
            continue
            
        for word in segment['words']:
            assigned_speaker = None
            max_overlap = 0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap_start = max(word.start, turn.start)
                overlap_end = min(word.end, turn.end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_speaker = speaker
            
            words_with_speakers.append({
                'word': word.word,
                'start': word.start,
                'end': word.end,
                'speaker': assigned_speaker
            })
    
    return words_with_speakers

# 4. Group by speaker turns AND FILTER NONE
def group_by_speaker_turns(words_with_speakers):
    """Group consecutive words by the same speaker, skip None"""
    if not words_with_speakers:
        return []
    
    # FILTER OUT NONE SPEAKERS FIRST
    valid_words = [w for w in words_with_speakers if w['speaker'] is not None]
    
    if not valid_words:
        return []
    
    turns = []
    current_speaker = valid_words[0]['speaker']
    current_words = [valid_words[0]['word']]
    current_start = valid_words[0]['start']
    current_end = valid_words[0]['end']
    
    for word_info in valid_words[1:]:
        if word_info['speaker'] == current_speaker:
            current_words.append(word_info['word'])
            current_end = word_info['end']
        else:
            turns.append({
                'speaker': current_speaker,
                'start': current_start,
                'end': current_end,
                'text': ''.join(current_words).strip()
            })
            current_speaker = word_info['speaker']
            current_words = [word_info['word']]
            current_start = word_info['start']
            current_end = word_info['end']
    
    # Add the last turn
    turns.append({
        'speaker': current_speaker,
        'start': current_start,
        'end': current_end,
        'text': ''.join(current_words).strip()
    })
    
    return turns

words_with_speakers = assign_word_speakers(diarization, result_segments)
speaker_turns = group_by_speaker_turns(words_with_speakers)

print("\n=== Speaker Turns (None filtered) ===")
for turn in speaker_turns:
    print(f"[{turn['start']:.2f}s - {turn['end']:.2f}s] "
          f"Speaker {turn['speaker']}: {turn['text']}")