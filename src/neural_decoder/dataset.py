import torch
import numpy as np
from torch.utils.data import Dataset


def apply_time_mask(x_np, p=0.10, max_width=20, max_masks=1):
    """
    Lightweight time masking (SpecAugment-lite): zeros short random time spans.
    
    Args:
        x_np: numpy array of shape [T, F] (time, features)
        p: probability of applying each mask (default 0.10)
        max_width: maximum width of each mask in time steps (default 20)
        max_masks: maximum number of masks to apply (default 1)
    
    Returns:
        Masked array (same shape as x_np)
    """
    T, F = x_np.shape
    if T <= 1:
        return x_np
    
    x_masked = x_np.copy()
    
    # Apply up to max_masks masks
    for _ in range(max_masks):
        if np.random.rand() > p:
            continue  # Skip this mask, but try next one
        
        width = np.random.randint(1, min(max_width + 1, T))
        start = np.random.randint(0, T - width + 1)
        if start + width > T:
            width = T - start
        x_masked[start:start+width, :] = 0.0
    
    return x_masked


def apply_frequency_mask(x_np, p=0.10, max_width=12, max_masks=2):
    """
    Frequency masking (SpecAugment): zeros random frequency bands.
    
    Args:
        x_np: numpy array of shape [T, F] (time, features)
        p: probability of applying each mask (default 0.10)
        max_width: maximum width of each mask in frequency bins (default 12)
        max_masks: maximum number of masks to apply (default 2)
    
    Returns:
        Masked array (same shape as x_np)
    """
    T, F = x_np.shape
    if F <= 1:
        return x_np
    
    x_masked = x_np.copy()
    
    # Apply up to max_masks masks
    for _ in range(max_masks):
        if np.random.rand() > p:
            continue  # Skip this mask, but try next one
            
        width = np.random.randint(1, min(max_width + 1, F))
        start = np.random.randint(0, F - width + 1)
        x_masked[:, start:start+width] = 0.0
    
    return x_masked


class SpeechDataset(Dataset):
    def __init__(self, data, transform=None, split=None, time_mask_prob=0.10, time_mask_width=20, time_mask_max_masks=1,
                 freq_mask_prob=0.10, freq_mask_width=12, freq_mask_max_masks=2):
        self.data = data
        self.transform = transform
        self.split = split
        self.time_mask_prob = time_mask_prob
        self.time_mask_width = time_mask_width
        self.time_mask_max_masks = time_mask_max_masks
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_width = freq_mask_width
        self.freq_mask_max_masks = freq_mask_max_masks
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = self.neural_feats[idx]  # numpy array [T, F]
        
        # Apply SpecAugment (time + frequency masking) for training only
        if self.split == "train":
            # Apply time masking
            neural_feats = apply_time_mask(
                neural_feats, 
                p=self.time_mask_prob,
                max_width=self.time_mask_width,
                max_masks=self.time_mask_max_masks
            )
            # Apply frequency masking
            neural_feats = apply_frequency_mask(
                neural_feats,
                p=self.freq_mask_prob,
                max_width=self.freq_mask_width,
                max_masks=self.freq_mask_max_masks
            )
        
        neural_feats = torch.tensor(neural_feats, dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )
