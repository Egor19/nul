import torch
from model.fastspeech2 import FastSpeech2
from waveglow.denoiser import Denoiser
from waveglow.model import WaveGlow

from text import text_to_sequence
#from hparams import hparams as hp
import numpy as np
import librosa
import soundfile as sf
import yaml

# Загрузка конфигурационных файлов preprocess_config и model_config
with open('/content/drive/MyDrive/FastSpeech2-ru/config/RUSLAN/preprocess.yaml') as f:
    preprocess_config = yaml.load(f, Loader=yaml.FullLoader)

with open('/content/drive/MyDrive/FastSpeech2-ru/config/RUSLAN/model.yaml') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

# Загрузка моделей
fastspeech2_checkpoint = torch.load('/content/drive/MyDrive/FastSpeech2-ru/output/ckpt/RUSLAN/10000.pth.tar')
fastspeech2 = FastSpeech2(preprocess_config, model_config)
#print(fastspeech2_checkpoint.keys())
fastspeech2.load_state_dict(fastspeech2_checkpoint['model'])
fastspeech2.eval()

waveglow_checkpoint = torch.load('/content/drive/MyDrive/FastSpeech2-ru/waveglow/checkpoint_WaveGlow_550.pt')
print(waveglow_checkpoint.keys())
waveglow = WaveGlow() 
waveglow = waveglow_checkpoint['state_dict']
denoiser = Denoiser(waveglow)
waveglow.cuda().eval()
denoiser.cuda().eval()

# Функция синтеза аудио из текста
def synthesize(text):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    with torch.no_grad():
        mel_outputs, mel_lengths, alignments = fastspeech2(sequence)

        with torch.autograd.no_grad():
            audio = waveglow.infer(mel_outputs, sigma=0.666)
            audio = denoiser(audio, strength=0.01)[:, 0]

    return audio.cpu().numpy()

# Пример использования
text = "Привет, как дела?"
audio = synthesize(text)

# Сохранение аудио в файл
sf.write('output.wav', audio, hp.sample_rate)