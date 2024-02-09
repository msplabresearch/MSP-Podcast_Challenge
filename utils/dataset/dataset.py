import numpy as np
import pickle as pk
import torch.utils as torch_utils
from . import normalizer

"""
All dataset should have the same order based on the utt_list
"""
def load_norm_stat(norm_stat_file):
    with open(norm_stat_file, 'rb') as f:
        wav_mean, wav_std = pk.load(f)
    return wav_mean, wav_std


class CombinedSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(CombinedSet, self).__init__()
        self.datasets = kwargs.get("datasets", args[0]) 
        self.data_len = len(self.datasets[0])
        for cur_dataset in self.datasets:
            assert len(cur_dataset) == self.data_len, "All dataset should have the same order based on the utt_list"
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        result = []
        for cur_dataset in self.datasets:
            result.append(cur_dataset[idx])
        return result


class WavSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(WavSet, self).__init__()
        self.wav_list = kwargs.get("wav_list", args[0]) # (N, D, T)

        self.wav_mean = kwargs.get("wav_mean", None)
        self.wav_std = kwargs.get("wav_std", None)

        self.upper_bound_max_dur = kwargs.get("max_dur", 12)
        self.sampling_rate = kwargs.get("sr", 16000)

        # check max duration
        self.max_dur = np.min([np.max([len(cur_wav) for cur_wav in self.wav_list]), self.upper_bound_max_dur*self.sampling_rate])
        if self.wav_mean is None or self.wav_std is None:
            self.wav_mean, self.wav_std = normalizer. get_norm_stat_for_wav(self.wav_list)
    
    def save_norm_stat(self, norm_stat_file):
        with open(norm_stat_file, 'wb') as f:
            pk.dump((self.wav_mean, self.wav_std), f)
            
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_wav = self.wav_list[idx][:self.max_dur]
        cur_dur = len(cur_wav)
        cur_wav = (cur_wav - self.wav_mean) / (self.wav_std+0.000001)
        
        result = (cur_wav, cur_dur)
        return result

class ADV_EmoSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(ADV_EmoSet, self).__init__()
        self.lab_list = kwargs.get("lab_list", args[0])
        self.max_score = kwargs.get("max_score", 7)
        self.min_score = kwargs.get("min_score", 1)
    
    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        cur_lab = self.lab_list[idx]
        cur_lab = (cur_lab - self.min_score) / (self.max_score-self.min_score)
        result = cur_lab
        return result
    
class CAT_EmoSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(CAT_EmoSet, self).__init__()
        self.lab_list = kwargs.get("lab_list", args[0])
    
    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        cur_lab = self.lab_list[idx]
        result = cur_lab
        return result

class SpkSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(SpkSet, self).__init__()
        self.spk_list = kwargs.get("spk_list", args[0])
    
    def __len__(self):
        return len(self.spk_list)

    def __getitem__(self, idx):
        cur_lab = self.spk_list[idx]
        result = cur_lab
        return result    

class UttSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(UttSet, self).__init__()
        self.utt_list = kwargs.get("utt_list", args[0])
    
    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        cur_lab = self.utt_list[idx]
        result = cur_lab
        return result
