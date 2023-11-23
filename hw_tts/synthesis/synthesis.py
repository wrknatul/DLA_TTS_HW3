import numpy as np
import torch
import hw_tts.waveglow.inference


def synthesis(model, device, waveglow_model, text, alpha=1.0, alpha_p=1.0, alpha_e=1.0, path=None):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i + 1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)

    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, alpha_p=alpha_p, alpha_e=alpha_e)

    mel_cpu, mel_cuda = mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)
    if path is None:
        return hw_tts.waveglow.inference.get_wav(mel_cuda, waveglow_model)
    else:
        hw_tts.waveglow.inference.inference(mel_cuda, waveglow_model, path)
