import torch
import torch.nn as nn
import numpy as np

def collate_fn_wav_lab_mask(batch):
    total_wav = []
    total_lab = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        lab = wav_data[1]
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_dur.append(dur)
        total_utt.append(wav_data[2])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_lab = torch.Tensor(np.array(total_lab))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_lab, attention_mask, total_utt


def collate_fn_wav_test3(batch):
    total_wav = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        total_wav.append(torch.Tensor(wav))
        total_dur.append(dur)
        total_utt.append(wav_data[1])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, attention_mask, total_utt
