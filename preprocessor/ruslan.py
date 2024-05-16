import os
import json
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text

def prepare_align(config):
    
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "RUSLAN"

    with open(os.path.join(in_dir, "manifest2.json"), encoding="utf-8") as f:
        data = json.load(f)
        
        for entry in tqdm(data):
            base_name = entry["audio_filepath"].split("/")[-1].split(".")[0]
            text = entry["text"]
            text = _clean_text(text, cleaners)

            wav_path = os.path.join(in_dir, "audio", "{}.wav".format(base_name))
            
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sr=sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                
                # Проверка, что путь для lab файла корректно сформирован
                lab_file_path = os.path.join(out_dir, speaker, "{}.lab".format(base_name))
                print("Путь к lab файлу:", lab_file_path)
                
                # Запись текста в lab файл
                with open(lab_file_path, "w", encoding="utf-8") as f1:
                    f1.write(text)
                    print("Текст успешно записан в lab файл:", text)
