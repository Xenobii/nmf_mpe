import os
import torch
import torchaudio
import numpy as np
import pretty_midi
import torchaudio.transforms as T
from typing import Dict
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm

pretty_midi.pretty_midi.MAX_TICK = 1e10



class MAPSDataset(Dataset):
    """
    MAPS dataset class
    """
    def __init__(
        self,
        root: str,
        data_sr: int,
        sr: int,
        hop_length: int,
        T_frames: int
    ):
        super().__init__()

        self.data_sr = data_sr  # 44100
        self.target_sr = sr
        self.hop_length = hop_length
        self.T_frames = T_frames
        self.T_wav = T_frames * hop_length

        self.files = [
            y.replace(".wav", "")
            for x in os.walk(root)
            for y in glob(os.path.join(x[0], "*.wav"))
        ]

        self.index = []

        for file_idx, stem in tqdm(enumerate(self.files), total=len(self.files), desc="Initializing dataset"):
            wav_path = stem + ".wav"
            wav_info = torchaudio.info(wav_path)
            num_samples = wav_info.num_frames

            K = (num_samples + self.T_wav - 1) // self.T_wav
            for k in range(K):
                self.index.append((file_idx, k))

        self.resampler = T.Resample(self.data_sr, self.target_sr)

    def __len__(self):
        return len(self.index)
    
    def load_wav(self, wav_file: str) -> torch.Tensor:
        """
        Waveform loader and processor
        """
        wav, sr = torchaudio.load(wav_file)

        assert sr == self.data_sr, \
            f"Expected sr = {self.data_sr}, got sr = {sr}."
        
        # Convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0)
        else: 
            wav = wav.squeeze(0)

        # Resample
        wav = self.resampler(wav)

        return wav
    
    def load_mid(self, mid_file: str) -> torch.Tensor:
        """
        MIDI loader and processor
        """
        mid = pretty_midi.PrettyMIDI(mid_file)

        fs = self.target_sr / self.hop_length
        
        # Piano roll
        roll = mid.get_piano_roll(fs=fs)
        roll = (roll > 0).astype(np.float32)

        return torch.from_numpy(roll)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        file_idx, chunk_idx = self.index[idx]
        stem = self.files[file_idx]

        wave = self.load_wav(stem + ".wav")
        roll = self.load_mid(stem + ".mid")

        wav_start = chunk_idx * self.T_wav
        wav_end   = wav_start + self.T_wav

        frame_start = chunk_idx * self.T_frames
        frame_end   = frame_start + self.T_frames
        
        wav_chunk = wave[wav_start:wav_end]
        if wav_chunk.numel() < self.T_wav:
            wav_chunk = torch.nn.functional.pad(
                wav_chunk, (0, self.T_wav - wav_chunk.numel())
            )

        roll_chunk = roll[:, frame_start:frame_end]
        if roll_chunk.shape[1] < self.T_frames:
            roll_chunk = torch.nn.functional.pad(
                roll_chunk, (0, self.T_frames - roll_chunk.shape[1])
            )

        return {
            "wave": wav_chunk,    # [T_wav]
            "roll": roll_chunk,   # [128, T_frames]
        }
        


class MAPSDataloader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=True,
        )