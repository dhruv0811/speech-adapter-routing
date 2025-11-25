"""Audio preprocessing utilities."""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing pipeline for ASR."""
    
    def __init__(
        self,
        target_sr: int = 16000,
        normalize: bool = True,
        trim_silence: bool = False,
        trim_db: float = 30.0,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
    ):
        """
        Args:
            target_sr: Target sampling rate
            normalize: Whether to normalize audio amplitude
            trim_silence: Whether to trim leading/trailing silence
            trim_db: Threshold in dB for silence trimming
            max_duration: Maximum duration in seconds (truncate longer)
            min_duration: Minimum duration in seconds (pad shorter or skip)
        """
        self.target_sr = target_sr
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.trim_db = trim_db
        self.max_duration = max_duration
        self.min_duration = min_duration
        
    def __call__(
        self, 
        audio: Union[np.ndarray, torch.Tensor],
        sr: int = None,
    ) -> Tuple[np.ndarray, int]:
        """Process audio array.
        
        Args:
            audio: Audio waveform (1D array)
            sr: Source sampling rate
            
        Returns:
            Processed audio array and sampling rate
        """
        sr = sr or self.target_sr
        
        # Convert to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure 1D
        if audio.dim() > 1:
            audio = audio.mean(dim=0)  # Convert to mono
            
        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            audio = resampler(audio)
            sr = self.target_sr
            
        # Normalize amplitude
        if self.normalize:
            audio = self._normalize(audio)
            
        # Trim silence
        if self.trim_silence:
            audio = self._trim_silence(audio, sr)
            
        # Handle duration constraints
        duration = len(audio) / sr
        
        if self.max_duration and duration > self.max_duration:
            max_samples = int(self.max_duration * sr)
            audio = audio[:max_samples]
            
        if self.min_duration and duration < self.min_duration:
            # Pad with zeros
            min_samples = int(self.min_duration * sr)
            padding = min_samples - len(audio)
            audio = torch.nn.functional.pad(audio, (0, padding))
            
        return audio.numpy(), sr
    
    def _normalize(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def _trim_silence(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Trim leading and trailing silence."""
        # Use energy-based VAD
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Compute frame energies
        energy = audio.unfold(0, frame_length, hop_length).pow(2).mean(dim=1)
        energy_db = 10 * torch.log10(energy + 1e-10)
        
        # Find threshold
        threshold = energy_db.max() - self.trim_db
        
        # Find start and end of speech
        above_thresh = energy_db > threshold
        if not above_thresh.any():
            return audio
            
        start_frame = above_thresh.nonzero()[0].item()
        end_frame = above_thresh.nonzero()[-1].item()
        
        start_sample = start_frame * hop_length
        end_sample = min((end_frame + 1) * hop_length + frame_length, len(audio))
        
        return audio[start_sample:end_sample]


class SpecAugment:
    """SpecAugment data augmentation for spectrograms.
    
    Applies frequency and time masking as described in:
    "SpecAugment: A Simple Data Augmentation Method for ASR"
    https://arxiv.org/abs/1904.08779
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        mask_value: float = 0.0,
    ):
        """
        Args:
            freq_mask_param: Maximum frequency mask width (F)
            time_mask_param: Maximum time mask width (T)
            num_freq_masks: Number of frequency masks
            num_time_masks: Number of time masks
            mask_value: Value to fill masked regions
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
        
        # Create torchaudio transforms
        self.freq_masking = T.FrequencyMasking(freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param)
        
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Log-mel spectrogram (batch, n_mels, time) or (n_mels, time)
            
        Returns:
            Augmented spectrogram
        """
        # Add batch dimension if needed
        squeeze = False
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
            squeeze = True
            
        # Apply frequency masks
        for _ in range(self.num_freq_masks):
            spectrogram = self.freq_masking(spectrogram)
            
        # Apply time masks
        for _ in range(self.num_time_masks):
            spectrogram = self.time_masking(spectrogram)
            
        if squeeze:
            spectrogram = spectrogram.squeeze(0)
            
        return spectrogram


def load_audio(
    path: str,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load audio file.
    
    Args:
        path: Path to audio file
        target_sr: Target sampling rate
        mono: Whether to convert to mono
        
    Returns:
        Audio tensor and sampling rate
    """
    audio, sr = torchaudio.load(path)
    
    # Convert to mono
    if mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze(0)
    
    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        audio = resampler(audio)
        sr = target_sr
        
    return audio, sr
