"""Data augmentation for piano chord classifier.

Simulates real-world recording variations to make the model robust
against noise, reverb, out-of-tune pianos, and pedal effects.

Applied online during training (not saved to disk).
"""

import numpy as np
import torch


class PianoAugmentation:
    """Feature-space augmentation for 36-dim piano vectors.

    [0:12]  CREPE chroma → pitch shift, tuning noise
    [12:24] CQT chroma   → same augmentations, slightly different
    [24:36] Spectral      → gaussian noise, dropout
    """

    def __init__(self, p_pitch_shift: float = 0.3, p_noise: float = 0.4,
                 p_dropout: float = 0.2, p_detune: float = 0.2):
        self.p_pitch_shift = p_pitch_shift
        self.p_noise = p_noise
        self.p_dropout = p_dropout
        self.p_detune = p_detune

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        feat = features.numpy().copy()

        # 1. Pitch shift (±1 semitone) — simulate transposition / tuning
        if np.random.random() < self.p_pitch_shift:
            shift = np.random.choice([-1, 1])
            feat[:12] = np.roll(feat[:12], shift)    # CREPE chroma
            feat[12:24] = np.roll(feat[12:24], shift)  # CQT chroma

        # 2. Gaussian noise — simulate recording noise, reverb tail
        if np.random.random() < self.p_noise:
            noise_level = np.random.uniform(0.01, 0.05)
            feat += np.random.normal(0, noise_level, feat.shape).astype(np.float32)
            feat = np.clip(feat, 0, 1)

        # 3. Feature dropout — simulate missing frequency info
        if np.random.random() < self.p_dropout:
            drop_rate = np.random.uniform(0.05, 0.15)
            mask = np.random.random(feat.shape) > drop_rate
            feat *= mask

        # 4. Chroma detuning — simulate slightly out-of-tune piano
        #    Bleed energy into adjacent semitones
        if np.random.random() < self.p_detune:
            alpha = np.random.uniform(0.05, 0.15)
            for section in [slice(0, 12), slice(12, 24)]:
                original = feat[section].copy()
                shifted_up = np.roll(original, 1)
                shifted_down = np.roll(original, -1)
                feat[section] = (1 - alpha) * original + (alpha / 2) * (shifted_up + shifted_down)

        return torch.tensor(feat, dtype=torch.float32)


class MixupAugmentation:
    """Mixup: blend two samples to create virtual training examples.

    Particularly effective for chord classification since many chords
    share common tones (e.g., C major and Am share C and E).
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, x1: torch.Tensor, y1: int,
                 x2: torch.Tensor, y2: int) -> tuple[torch.Tensor, torch.Tensor]:
        lam = np.random.beta(self.alpha, self.alpha)

        x_mixed = lam * x1 + (1 - lam) * x2

        # Soft labels
        n_classes = 96
        y_soft = torch.zeros(n_classes)
        y_soft[y1] = lam
        y_soft[y2] = 1 - lam

        return x_mixed, y_soft
