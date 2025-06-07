import numpy as np
import torch
import random
import torch.nn.functional as F


# Composes several transforms together.
class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value


class RandomNoise:
    """
    Adds Gaussian noise directly to the waveform.
    Noise is generated on the same device & dtype as the input tensor.
    """
    def __init__(self, min_noise: float = 0.002, max_noise: float = 0.02, p: float = 1.0):
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.p = p

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        # wave: Tensor of shape [time] (or [1, time]) on some device
        if random.random() > self.p:
            return wave
        noise_amp = random.uniform(self.min_noise, self.max_noise)
        noise = torch.randn_like(wave) * noise_amp
        return wave + noise


class RandomScale:
    """
    Randomly speeds up or slows down the waveform by linear interpolation.
    Operates on the same device & dtype as the input tensor.
    """
    def __init__(self, min_scale: float = 0.8, max_scale: float = 1.2, p: float = 1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        # wave: Tensor of shape [time]; we’ll interpolate in batch/channel dims
        if random.random() > self.p:
            return wave
        scale = random.uniform(self.min_scale, self.max_scale)
        # add batch & channel dims for interpolate
        w = wave.unsqueeze(0).unsqueeze(0)  # shape [1,1,time]
        new_len = int(w.size(-1) * scale)
        w2 = F.interpolate(
            w,
            size=new_len,
            mode="linear",
            align_corners=False
        )
        return w2.squeeze()  # back to [time]


class RandomCrop:

    def __init__(self, out_len: int = 44100, train: bool = True):
        super(RandomCrop, self).__init__()

        self.out_len = out_len
        self.train = train

    def random_crop(self, signal: torch.Tensor) -> torch.Tensor:
        if self.train:
            left = np.random.randint(0, signal.shape[-1] - self.out_len)
        else:
            left = int(round(0.5 * (signal.shape[-1] - self.out_len)))

        orig_std = signal.float().std() * 0.5
        output = signal[..., left:left + self.out_len]

        out_std = output.float().std()
        if out_std < orig_std:
            output = signal[..., :self.out_len]

        new_out_std = output.float().std()
        if orig_std > new_out_std > out_std:
            output = signal[..., -self.out_len:]

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_crop(x) if x.shape[-1] > self.out_len else x


class RandomPadding:

    def __init__(self, out_len: int = 88200, train: bool = True):
        super(RandomPadding, self).__init__()

        self.out_len = out_len
        self.train = train

    def random_pad(self, signal: torch.Tensor) -> torch.Tensor:

        if self.train:
            left = np.random.randint(0, self.out_len - signal.shape[-1])
        else:
            left = int(round(0.5 * (self.out_len - signal.shape[-1])))

        right = self.out_len - (left + signal.shape[-1])

        pad_value_left = signal[..., 0].float().mean().to(signal.dtype)
        pad_value_right = signal[..., -1].float().mean().to(signal.dtype)
        output = torch.cat((
            torch.zeros(signal.shape[:-1] + (left,), dtype=signal.dtype, device=signal.device).fill_(pad_value_left),
            signal,
            torch.zeros(signal.shape[:-1] + (right,), dtype=signal.dtype, device=signal.device).fill_(pad_value_right)
        ), dim=-1)

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_pad(x) if x.shape[-1] < self.out_len else x


class FrequencyMask():
    def __init__(self, max_width, numbers):
        super(FrequencyMask, self).__init__()

        self.max_width = max_width
        self.numbers = numbers

    def addFreqMask(self, wave):
        # print(wave.shape)
        for _ in range(self.numbers):
            # choose the length of mask
            mask_len = random.randint(0, self.max_width)
            start = random.randint(0, wave.shape[1] - mask_len)  # start of the mask
            end = start + mask_len
            wave[:, start:end, :] = 0

        return wave

    def __call__(self, wave):
        return self.addFreqMask(wave)


class TimeMask():
    def __init__(self, max_width, numbers):
        super(TimeMask, self).__init__()

        self.max_width = max_width
        self.numbers = numbers

    def addTimeMask(self, wave):
        for _ in range(self.numbers):
            # choose the length of mask
            mask_len = random.randint(0, self.max_width)
            start = random.randint(0, wave.shape[2] - mask_len)  # start of the mask
            end = start + mask_len
            wave[:, :, start:end] = 0

        return wave

    def __call__(self, wave):
        return self.addTimeMask(wave)