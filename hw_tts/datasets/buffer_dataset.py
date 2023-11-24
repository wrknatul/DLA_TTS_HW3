import time
import torch
import os
import librosa
import numpy as np
import pyworld as pw

from torch.utils.data import Dataset
from tqdm import tqdm

from hw_tts.text import text_to_sequence
import hw_tts.audio.hparams_audio as hparams_audio

def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt

def get_data_to_buffer(data_path, mel_ground_truth, alignment_path, text_cleaners):
    buffer = list()
    wav_files = []
    text = process_text(data_path)
    start = time.perf_counter()
    wav_files = sorted([os.path.join("data/LJSpeech-1.1/wavs/", f) for f in os.listdir("data/LJSpeech-1.1/wavs/") \
            if os.path.isfile(os.path.join("data/LJSpeech-1.1/wavs/", f))])
    for i in tqdm(range(min(10, len(text)))):
        mel_gt_name = os.path.join(
            mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            alignment_path, str(i) + ".npy"))
        character = text[i][0:len(text[i]) - 1]
        character = np.array(
            text_to_sequence(character, text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        wav_path = wav_files[i]
        wav, _ = librosa.load(wav_path)
        pitch, t = pw.dio(
            wav.astype(np.float64),
            hparams_audio.sampling_rate,
            frame_period=hparams_audio.hop_length / hparams_audio.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, 22050)
        
        pitch = torch.tensor(pitch[: sum(duration)])

        energy = torch.stft(torch.tensor(wav),
                            n_fft=1024,
                            hop_length=256,
                            win_length=1024,
                            return_complex=True
                            ).transpose(0, 1)

        buffer.append({"text": character, "duration": duration, "mel_target": mel_gt_target,
                "pitch": pitch, "energy": energy})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end - start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, data_path, mel_ground_truth, alignment_path, text_cleaners):
        self.buffer = get_data_to_buffer(data_path, mel_ground_truth, alignment_path,
                                        text_cleaners)
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
