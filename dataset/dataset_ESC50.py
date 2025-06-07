import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import os
import sys
from functools import partial
import numpy as np

import torchaudio
from torchaudio.functional import compute_deltas
from torchaudio.transforms import FrequencyMasking, TimeMasking
import config
from . import transforms


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_extract_zip(url: str, file_path: str):
    #import wget
    import zipfile
    root = os.path.dirname(file_path)
    # wget.download(url, out=file_path, bar=download_progress)
    download_file(url=url, fname=file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(root)


# create this bar_progress method which is invoked automatically from wget
def download_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


class ESC50(data.Dataset):

    def __init__(self, root, test_folds=frozenset((1,)), subset="train", compute_stats: bool=False,  download=False):
        audio = 'ESC-50-master/audio'
        root = os.path.normpath(root)
        audio = os.path.join(root, audio)
        if subset in {"train", "test", "val"}:
            self.subset = subset
        else:
            raise ValueError
        # path = path.split(os.sep)
        if not os.path.exists(audio) and download:
            os.makedirs(root, exist_ok=True)
            file_name = 'master.zip'
            file_path = os.path.join(root, file_name)
            url = f'https://github.com/karoldvl/ESC-50/archive/{file_name}'
            download_extract_zip(url, file_path)

        self.root = audio
        # getting name of all files inside the all the train_folds
        temp = sorted(os.listdir(self.root))
        folds = {int(v.split('-')[0]) for v in temp}
        self.test_folds = set(test_folds)
        self.train_folds = folds - test_folds
        train_files = [f for f in temp if int(f.split('-')[0]) in self.train_folds]
        test_files = [f for f in temp if int(f.split('-')[0]) in test_folds]
        # sanity check
        assert set(temp) == (set(train_files) | set(test_files))
        if subset == "test":
            self.file_names = test_files
        else:
            if config.val_size:
                train_files, val_files = train_test_split(train_files, test_size=config.val_size, random_state=0)
            if subset == "train":
                self.file_names = train_files
            else:
                self.file_names = val_files
        # the number of samples in the wave (=length) required for spectrogram
        out_len = int(((config.sr * 5) // config.hop_length) * config.hop_length)
        train = self.subset == "train"
        if train:
            # augment training data with transformations that include randomness
            # transforms can be applied on wave and spectral representation
            self.wave_transforms = transforms.Compose(
                torch.Tensor,
                # you can still add RandomScale or other wave-domain augs here
                transforms.RandomPadding(out_len=out_len),
                transforms.RandomCrop(out_len=out_len),
                transforms.RandomScale(max_scale=1.2),
                transforms.RandomNoise(min_noise=0.002, max_noise=0.02)
            )

            # === spec_transforms for training: add SpecAugment ===
            self.spec_transforms = transforms.Compose(
                # (1) to Tensor, (2) add channel dim
                torch.Tensor,
                partial(torch.unsqueeze, dim=0),
                # (3) zero-out up to 15 random mel bins
                FrequencyMasking(freq_mask_param=15),
                # (4) zero-out up to 35 random time frames
                TimeMasking(time_mask_param=35),
            )

        else:
            # for testing transforms are applied deterministically to support reproducible scores
            self.wave_transforms = transforms.Compose(
                torch.Tensor,
                # disable randomness
                transforms.RandomPadding(out_len=out_len, train=False),
                transforms.RandomCrop(out_len=out_len, train=False)
            )

            self.spec_transforms = transforms.Compose(
                torch.Tensor,
                partial(torch.unsqueeze, dim=0),
            )
        self.global_mean = None
        self.global_std = None

        if compute_stats:
            self.global_mean, self.global_std = self._compute_global_stats()

        self.n_mfcc = config.n_mfcc if hasattr(config, "n_mfcc") else None

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        path = os.path.join(self.root, file_name)

        # --- (1) Load audio with torchaudio ---
        # torchaudio.load returns a Tensor [channels, time] and its sample rate.
        waveform, orig_sr = torchaudio.load(path)  # shape: [channels, time]

        # If the audio was recorded at a different sample rate, resample to config.sr:
        if orig_sr != config.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=config.sr)
            waveform = resampler(waveform)  # still shape [channels, time]

        # --- (2) Convert to mono if multi-channel ---
        # If there is more than 1 channel, just take the mean of channels.
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # now shape [1, time]

        # At this point: waveform is a FloatTensor in [-1.0, +1.0], shape [1, time].
        waveform = waveform * 32768.0  # now roughly integer‐scale as before.

        # --- (3) Remove any leading/trailing silence exactly as before ---
        # Find indices of non-zero samples (very naive VAD).
        nonzero_indices = (waveform != 0.0).nonzero(as_tuple=False)
        if nonzero_indices.numel() > 0:
            time_min = torch.min(nonzero_indices[:, 1])
            time_max = torch.max(nonzero_indices[:, 1])
            waveform = waveform[:, time_min: time_max + 1]
        # If everything was zero, we just keep waveform as is.

        # --- (4) Apply existing "wave_transforms" pipeline ---
        wave_copy = waveform.clone().squeeze(0)  # shape [time]
        wave_copy = self.wave_transforms(wave_copy)  # crop/pad (returns Tensor [time] or shorter/longer)
        wave_copy = wave_copy.unsqueeze(0)  # restore shape [1, time]

        # --- (5) Label extraction: exactly as before ---
        base = file_name.split('.')[0]
        class_id = int(base.split('-')[-1])

        # --- (6) Feature extraction using torchaudio ---

        # First, sanity‐check that we never request more mel filters
        # than there are FFT frequency bins (n_fft//2 + 1 = 513).
        # warning still persists at first 1-2 episodes of training
        assert config.n_mels <= (1024 // 2 + 1), \
            f"config.n_mels={config.n_mels} must be ≤ (n_fft//2 + 1)=513"

        if self.n_mfcc:
            # (6a) MFCC path
            # torchaudio.transforms.MFCC outputs shape [channel, n_mfcc, time]
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=config.sr,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    'n_mels': config.n_mels,
                    'n_fft': config.n_fft,
                    'hop_length': config.hop_length,
                    'f_min': 0.0,
                    'f_max': config.sr / 2.0
                }
            )
            feat = mfcc_transform(wave_copy)  # [1, n_mfcc, T]
        else:
            # (6b) MelSpectrogram + AmplitudeToDB path
            melspec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sr,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                f_min=0.0,
                f_max=(config.sr / 2.0),  # ensure we only build filters up to Nyquist
            )
            mel_power = melspec_transform(wave_copy)  # [1, n_mels, T]
            db_transform = torchaudio.transforms.AmplitudeToDB(stype='power')
            feat = db_transform(mel_power)  # [1, n_mels, T]


        # --- (7) Normalize with the global mean/std as before ---
        # Originally, "feat" was a NumPy array or a Torch Tensor created via torch.Tensor(log_s).
        # Now `feat` is already a FloatTensor with shape [1, n_mels, T] (or [1, n_mfcc, T]).
        if self.global_mean is not None:
            feat = (feat - self.global_mean) / self.global_std

        # --- (8) Compute first‐ and second‐order deltas ---
        # feat: [1, n_mels, T] → delta1: same shape
        delta1 = compute_deltas(feat)
        # then delta2 from delta1
        delta2 = compute_deltas(delta1)

        # --- (9) Stack into 3 channels: [3, n_mels, T] ---
        # static, Δ, and Δ²
        feat = torch.cat([feat, delta1, delta2], dim=0)

        return file_name, feat, class_id

    def _compute_global_stats(self):
        """
        Compute mean & std of the *static* log-Mel feature over all
        *training* files in this fold (before delta stacking).
        Returns two tensors of shape [1,1,1] that broadcast over [1, n_mels, T].
        """
        # Prepare your GPU transforms once:
        melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=0.0,
            f_max=config.sr / 2.0,
        ).to(device)
        db_transform = torchaudio.transforms.AmplitudeToDB(stype='power').to(device)

        sum_ = 0.0
        sum_sq = 0.0
        count = 0

        for fname in tqdm(self.file_names, desc="Calc mean/std"):
            path = os.path.join(self.root, fname)

            # 1) LOAD on CPU & cast to float32
            waveform, sr = torchaudio.load(path)  # -> CPU DoubleTensor
            waveform = waveform.to(dtype=torch.float32)  # -> CPU FloatTensor

            # 2) RESAMPLE on CPU if needed
            if sr != config.sr:
                resampler = torchaudio.transforms.Resample(sr, config.sr)
                waveform = resampler(waveform)  # still CPU FloatTensor

            # 3) MONO mix-down on CPU
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # still CPU

            # 4) APPLY wave_transforms on CPU
            wave_cpu = self.wave_transforms(waveform.squeeze(0))  # -> CPU FloatTensor [time]
            wave_cpu = wave_cpu.unsqueeze(0)  # -> CPU [1, time]

            # 5) MOVE to GPU for spectrogram
            wave = wave_cpu.to(device)  # -> GPU FloatTensor

            # 6) MEL + DB on GPU
            melspec = melspec_transform(wave)  # -> GPU [1, n_mels, T]
            feat = db_transform(melspec)  # -> GPU [1, n_mels, T]

            # 7) ACCUMULATE stats (back on CPU via .item())
            sum_ += feat.sum().item()
            sum_sq += (feat ** 2).sum().item()
            count += feat.numel()

        # 8) FINALIZE mean/std and return as GPU tensors
        mean = sum_ / count
        var = sum_sq / count - mean ** 2
        std = float(var ** 0.5)

        return (
            torch.tensor(mean, device=device).view(1, 1, 1),
            torch.tensor(std, device=device).view(1, 1, 1)
        )

def get_global_stats(data_path):
    res = []
    for i in range(1, 6):
        train_set = ESC50(subset="train", test_folds={i}, root=data_path, download=True)
        a = torch.concatenate([v[1] for v in tqdm(train_set)])
        res.append((a.mean(), a.std()))
    return np.array(res)
