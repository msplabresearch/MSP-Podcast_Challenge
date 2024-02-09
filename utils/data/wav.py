import os
import librosa
from tqdm import tqdm
from multiprocessing import Pool

# Load audio
def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=16000)
    return raw_wav
def load_audio(audio_path, utts, nj=24):
    # Audio path: directory of audio files
    # utts: list of utterance names with .wav extension
    wav_paths = [os.path.join(audio_path, utt) for utt in utts]
    with Pool(nj) as p:
        wavs = list(tqdm(p.imap(extract_wav, wav_paths), total=len(wav_paths)))
    return wavs