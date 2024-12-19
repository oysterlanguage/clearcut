import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer
from intervaltree import Interval, IntervalTree
import soundfile as sf
import os
from collections import defaultdict
import tqdm
import pathlib
from models.uroman_align import find_maxima_in_range
import logging

logger = logging.getLogger(__name__)


def zcr_extractor(wav, win_length, hop_length):
    pad_length = win_length // 2
    wav = np.pad(wav, (pad_length, pad_length), "constant")
    num_frames = 1 + (wav.shape[0] - win_length) // hop_length
    zcrs = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        zcr = np.abs(np.sign(wav[start + 1 : end]) - np.sign(wav[start : end - 1]))
        zcr = np.sum(zcr) * 0.5 / win_length
        zcrs[i] = zcr
    return zcrs.astype(np.float32)


def feature_extractor(wav, sr=16000):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=128
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    zcr = zcr_extractor(wav, win_length=int(sr * 0.025), hop_length=int(sr * 0.01))
    vms = np.var(mel, axis=0)

    mel = torch.tensor(mel).unsqueeze(0)
    zcr = torch.tensor(zcr).unsqueeze(0)
    vms = torch.tensor(vms).unsqueeze(0)

    zcr = zcr.unsqueeze(1).expand(-1, 128, -1)
    vms = torch.var(mel, dim=1).unsqueeze(1).expand(-1, mel.shape[1], -1)

    feature = torch.stack((mel, vms, zcr), dim=1)
    length = torch.tensor([zcr.shape[-1]])
    return feature, length


class Conv2dDownsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dDownsampling, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x, length):
        keep_dim_padding = 1 - x.shape[-1] % 2  # odd: 0; even: 1
        x = F.pad(x, (0, keep_dim_padding, 0, 0))
        x = self.conv1(x)
        length = (length - 3 + keep_dim_padding) // 2 + 1

        keep_dim_padding = 1 - x.shape[-1] % 2
        x = F.pad(x, (0, keep_dim_padding, 0, 0))
        x = self.conv2(x)
        length = (length - 3 + keep_dim_padding) // 2 + 1
        return x, length


class Conv1dUpsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv1dUpsampling, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.deconv(x)
        return x


class DetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsampling = Conv2dDownsampling(3, 1)
        self.upsampling = Conv1dUpsampling(128, 128)
        self.linear = nn.Linear(31, 128)
        self.dropout = nn.Dropout(0.1)

        # Conformer
        self.conformer = Conformer(
            input_dim=128,
            num_heads=4,
            ffn_dim=256,
            num_layers=8,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=256, bidirectional=True, batch_first=True
        )

        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        sequence = x.shape[-1]

        # Ensure x is contiguous before downsampling
        x = x.contiguous()
        x, length = self.downsampling(x, length)  # x.shape: [B, 1, 31, S]

        # Squeeze and transpose while ensuring it's contiguous
        x = x.squeeze(1).transpose(1, 2).contiguous()  # x.shape: [B, S, 31]

        # Apply the linear layer and dropout
        x = self.linear(x)  # x.shape: [B, S, 128]
        x = self.dropout(x)

        # Ensure x is contiguous before passing to Conformer
        x = x.contiguous()
        x = self.conformer(x, length)[0]

        # Transpose for the upsampling while ensuring x is contiguous
        x = x.transpose(1, 2).contiguous()
        x = self.upsampling(x)

        # Transpose and check shape before LSTM
        x = x.transpose(1, 2).contiguous()

        # Ensure x is contiguous before passing into LSTM and check tensor dtype
        x = x.contiguous()
        if not x.is_floating_point():
            x = x.float()  # Ensure it's in float32
        x = self.lstm(x)[0]

        # Pass through fully connected and sigmoid layers
        x = self.fc(x)
        x = self.sigmoid(x.squeeze(-1))

        # Ensure output shape matches the input sequence length
        return x[:, :sequence]


class BreathDetector:
    def __init__(self, model_path, device=None, threshold=0.064, min_length=10):
        super().__init__()
        self.model = None
        self.threshold = threshold
        self.min_length = min_length
        self.model_path = model_path
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def load_model(self):
        model = DetectionNet().to(self.device)
        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model = model

    def unload_model(self):
        self.model = None

    def get_breath_timestamps(self, audio_data, valleys):
        tree = IntervalTree()
        chunk_size_frames = 9_600_000  # 600 seconds
        offset_frames = 0
        overlap_frames = 80_000  # 5 seconds
        while offset_frames < len(audio_data):
            audio = audio_data[
                offset_frames : min(offset_frames + chunk_size_frames, len(audio_data))
            ]
            feature, length = feature_extractor(audio, 16_000)
            feature, length = feature.to(self.device), length.to(self.device)
            output = self.model(feature, length)

            prediction = (output[0] > self.threshold).nonzero().squeeze().tolist()

            if isinstance(prediction, list) and len(prediction) > 1:
                diffs = np.diff(prediction)
                splits = np.where(diffs != 1)[0] + 1
                splits = np.split(prediction, splits)
                splits = list(
                    filter(lambda split: len(split) > self.min_length, splits)
                )

                for split in splits:
                    if split[-1] * 0.01 > split[0] * 0.01:
                        start = round(split[0] * 0.01, 3) + (offset_frames / 16000)
                        end = round(split[-1] * 0.01, 3) + (offset_frames / 16000)
                        peak = find_maxima_in_range(valleys=valleys, start_time=start, end_time=end)
                        tree.add(
                            Interval(
                                start,
                                end,
                                data=peak
                            )
                        )
            offset_frames += chunk_size_frames - overlap_frames
        tree.merge_overlaps(strict=False)
        return tree

    def remove_breath(self, audio_data, sr):
        audio = librosa.resample(audio_data, orig_sr=sr, target_sr=16_000)
        tree = self.get_breath_timestamps(audio, valleys={})
        breath_frames = []
        for interval in tree:
            breath_frames.append((int(interval.begin * sr), int(interval.end * sr)))
        new_audio = self.apply_ducking(
            audio_data, sr, breath_frames, fade_duration=0.02
        )
        return new_audio

    def get_breath_intervals(self, audio_data, sr):
        if sr != 16_000:
            audio = librosa.resample(audio_data, orig_sr=sr, target_sr=16_000)
        else:
            audio = audio_data
        _, significate_timestaps_tree, _, _, _ = calculate_significant_timestamps(
            audio, 16_000, 0
        )
        tree = self.get_breath_timestamps(audio, valleys={})

        final_intervals = IntervalTree()
        for interval in sorted(tree):
            timestamps = sorted(
                significate_timestaps_tree.overlap(interval.begin, interval.end)
            )
            left_time = interval.begin
            right_time = interval.end
            if len(timestamps) == 0:
                logger.info(f"No timestamps found {interval.begin:.3f} - {interval.end:.3f}")

            elif len(timestamps) == 1:
                left_time = timestamps[0].begin
                right_time = timestamps[0].end
            elif len(timestamps) > 1:
                if timestamps[0].begin < interval.begin:
                    (
                        start_db,
                        end_db,
                        interval_type,
                        avg_db,
                        slide,
                        direction,
                    ) = timestamps[0].data
                    if slide and direction == -1:
                        left_time = timestamps[0].begin
                    else:
                        left_time = timestamps[0].end
                if timestamps[-1].end > interval.end:
                    (
                        start_db,
                        end_db,
                        interval_type,
                        avg_db,
                        slide,
                        direction,
                    ) = timestamps[-1].data
                    if slide and direction == 1:
                        right_time = timestamps[-1].end
                    else:
                        right_time = timestamps[-1].begin

            # close, timestamps, found = vad.significant_timestamp_minima_slide(
            #     audio_data,
            #     sampling_rate=sr,
            #     timestamp_s=(interval.end + interval.begin / 2.0),
            #     window_size_s=0.2,
            #     left_boundary_s=interval.begin,
            #     right_boundary_s=interval.end,
            # )
            # tmp_timestamps = []
            # for timestamp in timestamps:
            #     if interval.begin - 0.02 <= timestamp[0] <= interval.end + 0.02:
            #         tmp_timestamps.append(timestamp)
            # timestamps = tmp_timestamps
            # if len(timestamps) < 2:
            #     new_audio_data = audio_data[
            #         max(0, int((interval.begin - 0.2) * 16_000)) : min(
            #             len(audio_data), int((interval.end + 0.2) * 16_000)
            #         )
            #     ]
            #     timestamps = vad.find_significant_timestamps(
            #         audio_data=new_audio_data,
            #         sr=16_000,
            #         start_time_s=max(0, interval.begin - 0.2),
            #         samples_per_second=1600,
            #     )
            #     pass
            # logger.info(
            #     f"Breath interval: {interval.begin:.3f} - {interval.end:.3f} {left_time:.3f} - {right_time:.3f} timestamp: {timestamps}"
            # )
            if left_time == right_time:
                best_start = np.inf
                best_end = np.inf
                for timestamp in timestamps:
                    if abs(interval.begin - timestamp.begin) < abs(
                        interval.begin - best_start
                    ):
                        best_start = timestamp.begin
                    if abs(interval.begin - timestamp.end) < abs(
                        interval.begin - best_start
                    ):
                        best_start = timestamp.end
                    if abs(interval.end - timestamp.begin) < abs(
                        interval.end - best_end
                    ):
                        best_end = timestamp.begin
                    if abs(interval.end - timestamp.end) < abs(interval.end - best_end):
                        best_end = timestamp.end
                if best_start == best_end:
                    left_time = timestamps[0].begin
                    right_time = timestamps[0].end
                else:
                    left_time = best_start
                    right_time = best_end

            final_intervals.add(Interval(left_time, right_time, "<breath>"))
        return sorted(final_intervals)

        # def get_breath_intervals(self, audio_data, sr):
        # make sure we don't go out of bounds
        new_audio_data = audio_data[
            max(0, int((left_boundary_s - window_size_s) * sampling_rate)) : min(
                len(audio_data), int((right_boundary_s + window_size_s) * sampling_rate)
            )
        ]
        if sampling_rate != 16000:
            new_audio_data = librosa.resample(
                new_audio_data, orig_sr=sampling_rate, target_sr=16000
            )
            sampling_rate = 16000
        timestamps = self.find_significant_timestamps(
            new_audio_data,
            sampling_rate,
            start_time_s=max(left_boundary_s - window_size_s, 0),
        )

    def get_breath_intervals_directory(self, directory):
        files = sorted(
            [str(tmp_path) for tmp_path in pathlib.Path(directory).rglob("*.wav")]
        )
        breath_intervals = dict()
        for index, file in enumerate(tqdm.tqdm(files, desc="Breath Detection")):
            audio_data, sr = librosa.load(file, sr=None)
            if sr != 16_000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16_000)
            tree = self.get_breath_timestamps(audio_data, valleys={})
            breaths = []
            for interval in sorted(tree):
                breaths.append({"start": interval.begin, "end": interval.end})
            breath_intervals[file] = breaths
        return breath_intervals

    @staticmethod
    def apply_ducking(
        bg_audio, sr: int, breath_intervals: IntervalTree, fade_duration: float = 0.5
    ):
        """
        Apply ease-in-ease-out ducking to specified sections of the background audio.

        Parameters:
        - bg_audio: np.ndarray
            The background audio signal (1D numpy array).
        - sr: int
            The sample rate of the audio signal.
        - silence_frames: list of tuples
            List of (start_frame, end_frame) tuples specifying the sections to duck.
        - fade_duration: float
            The duration of the fade-in and fade-out in seconds (default is 0.5 seconds).

        Returns:
        - modified_audio: np.ndarray
            The modified audio signal with ducking applied.
        """

        def ease_in_out_curve(n):
            """Generate an ease-in-ease-out curve."""
            t = np.linspace(0, 1, n)
            return 3 * t**2 - 2 * t**3

        fade_length = int(fade_duration * sr)
        ease_in_curve = ease_in_out_curve(fade_length)
        ease_out_curve = ease_in_out_curve(fade_length)[::-1]

        modified_audio = np.copy(bg_audio)

        for interval in breath_intervals:
            start_frame = int(interval.begin * sr)
            end_frame = int(interval.end * sr)
            # Apply ease-in
            modified_audio[start_frame : start_frame + fade_length] *= ease_in_curve

            # Silence the middle section
            modified_audio[start_frame + fade_length : end_frame - fade_length] *= 0.0

            # Apply ease-out
            modified_audio[end_frame - fade_length : end_frame] *= ease_out_curve

        return modified_audio

    @staticmethod
    def apply_muting(bg_audio, sr: int, breath_intervals: IntervalTree):
        """
        Mute specified sections of the background audio.

        Parameters:
        - bg_audio: np.ndarray
            The background audio signal (1D numpy array).
        - sr: int
            The sample rate of the audio signal.
        - breath_intervals: IntervalTree
            Tree containing intervals (in seconds) for sections to mute.

        Returns:
        - modified_audio: np.ndarray
            The modified audio signal with the specified sections muted.
        """

        modified_audio = np.copy(bg_audio)

        for interval in breath_intervals:
            start_frame = int(interval.begin * sr)
            end_frame = int(interval.end * sr)

            # Mute the section by setting it to zero
            modified_audio[start_frame:end_frame] = 0.0

        return modified_audio
