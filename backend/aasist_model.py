import torch
import torch.nn as nn
import torch.nn.functional as F

class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * torch.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1):
        super(SincConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Initialize filterbanks
        low_hz = 30
        high_hz = self.sample_rate / 2 - (10 + low_hz)
        mel = torch.linspace(self.to_mel(torch.tensor(float(low_hz))),
                             self.to_mel(torch.tensor(float(high_hz))),
                             out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(hz[:-1].view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.diff(hz).view(-1, 1))

        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * torch.pi * n_lin / self.kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * torch.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms):
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.low_hz_
        high = torch.clamp(low + self.band_hz_, 50, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, filters, stride=10, padding=self.kernel_size // 2)

class AASISTClassifier(nn.Module):
    def __init__(self):
        super(AASISTClassifier, self).__init__()
        self.sinc = SincConv(out_channels=70, kernel_size=128)
        self.conv1 = nn.Conv1d(70, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.fc1 = nn.Linear(32 * 64, 128)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, 2) # [Authentic, Synthetic]

    def forward(self, x):
        # Expected input shape: (Batch, 1, 64600)
        x = F.max_pool1d(torch.abs(self.sinc(x)), 3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.out(x)
        return logits