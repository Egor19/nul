import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    
    in_dir = '/content/drive/MyDrive/RUSLAN'
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "RUSLAN"
    with open(os.path.join(in_dir, "metadata_RUSLAN.csv"), encoding="utf-8", newline='\n') as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            
            text = parts[1]
            text = _clean_text(text, cleaners)

            wav_path = os.path.join(in_dir, "RUSLAN", "{}.wav".format(base_name))
            #print(wav_path)
            if os.path.exists(wav_path):
                #print('--------------')
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sr=sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)