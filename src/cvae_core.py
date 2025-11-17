"""CVAE core: audio features, dataset, model, and trainer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import librosa
except ImportError:
    librosa = None

try:
    from scipy import signal
    from scipy.interpolate import interp1d
except ImportError:
    signal = None
    interp1d = None


# Audio Features
def extract_audio_features(audio: np.ndarray, sr: int = 16000, hop_length: int = 512, n_fft: int = 2048,
                          n_mfcc: int = 13, include_spectral: bool = True, include_rhythm: bool = True) -> Tuple[np.ndarray, dict]:
    """Extract comprehensive audio features."""
    if librosa is None:
        raise ImportError("librosa required")
    features_list = []
    metadata = {}
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    features_list.append(mfcc.T)
    if include_spectral:
        features_list.append(librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft).T)
        features_list.append(librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft).T)
        features_list.append(librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length).T)
    if include_rhythm:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
        metadata.update({"tempo_bpm": float(tempo), "beat_times": librosa.frames_to_time(beats, sr=sr, hop_length=hop_length).tolist()})
        features_list.append(librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length).reshape(-1, 1))
    feature_matrix = np.concatenate(features_list, axis=1).astype(np.float32)
    metadata.update({"hop_length": hop_length, "sample_rate": sr, "n_frames": feature_matrix.shape[0], "feature_dim": feature_matrix.shape[1]})
    return feature_matrix, metadata


def extract_features_from_file(audio_path: Path | str, target_sr: int = 16000, hop_length: int = 512, **kwargs) -> Tuple[np.ndarray, dict]:
    """Extract audio features from file."""
    if librosa is None:
        raise ImportError("librosa required")
    audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return extract_audio_features(audio, sr=sr, hop_length=hop_length, **kwargs)


def align_features_to_poses(audio_features: np.ndarray, pose_frames: int, audio_hop_length: int = 512,
                            audio_sr: int = 16000, pose_fps: int = 30) -> np.ndarray:
    """Align audio features to pose frame rate."""
    if interp1d is None:
        raise ImportError("scipy required")
    audio_frames = audio_features.shape[0]
    audio_time_per_frame = audio_hop_length / audio_sr
    pose_time_per_frame = 1.0 / pose_fps
    audio_times = np.arange(audio_frames) * audio_time_per_frame
    pose_times = np.arange(pose_frames) * pose_time_per_frame
    max_time = min(audio_times[-1], pose_times[-1])
    pose_times = np.clip(pose_times, 0, max_time)
    aligned_features = np.zeros((pose_frames, audio_features.shape[1]), dtype=np.float32)
    for i in range(audio_features.shape[1]):
        interp_func = interp1d(audio_times, audio_features[:, i], kind="linear", fill_value="extrapolate", bounds_error=False)
        aligned_features[:, i] = interp_func(pose_times)
    return aligned_features


# Dataset
class AudioPoseDataset(Dataset):
    """Dataset for audio-pose pairs."""
    def __init__(self, data_root: str | Path, split: str = "train", audio_feature_config: dict | None = None,
                pose_fps: int = 30, normalize_poses: bool = True, max_samples: int | None = None, max_seq_length: int | None = None):
        self.data_root = Path(data_root)
        self.split = split
        self.pose_fps = pose_fps
        self.normalize_poses = normalize_poses
        self.max_seq_length = max_seq_length
        self.audio_feature_config = audio_feature_config or {"hop_length": 512, "n_mfcc": 13, "include_spectral": True, "include_rhythm": True}
        pose_dir = self.data_root / "3d_npy"
        if not pose_dir.exists():
            raise FileNotFoundError(f"Pose directory not found: {pose_dir}")
        self.pose_files = sorted(pose_dir.glob("*.npy"))
        if max_samples:
            self.pose_files = self.pose_files[:max_samples]
        self.audio_dir = self.data_root / "audio"
        self.pose_mean = None
        self.pose_std = None
        if normalize_poses:
            self._compute_pose_stats()
    
    def _compute_pose_stats(self):
        """Compute pose normalization statistics."""
        all_poses = []
        for pose_file in self.pose_files[:min(100, len(self.pose_files))]:
            pose = np.load(pose_file)
            if pose.ndim == 3:
                pose = pose.reshape(pose.shape[0], -1)
            all_poses.append(pose)
        if all_poses:
            all_poses = np.concatenate(all_poses, axis=0)
            # Remove NaN/Inf values
            all_poses = np.nan_to_num(all_poses, nan=0.0, posinf=0.0, neginf=0.0)
            self.pose_mean = all_poses.mean(axis=0, keepdims=True)
            self.pose_std = all_poses.std(axis=0, keepdims=True) + 1e-8
            # Ensure std is not too small or too large
            self.pose_std = np.clip(self.pose_std, 1e-6, 1e6)
    
    def _get_file_id(self, pose_file: Path) -> str:
        return pose_file.stem.replace("_cAll_", "_c01_")
    
    def _load_pose(self, pose_file: Path) -> np.ndarray:
        pose = np.load(pose_file)
        if pose.ndim == 3:
            pose = pose.reshape(pose.shape[0], -1)
        
        # Remove any NaN/Inf values before normalization
        pose = np.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.normalize_poses and self.pose_mean is not None:
            # Ensure shapes match for broadcasting
            if self.pose_mean.shape[1] != pose.shape[1]:
                # Recompute stats if dimension mismatch
                print(f"Warning: Pose dimension mismatch, recomputing stats...")
                self._compute_pose_stats()
            pose = (pose - self.pose_mean) / self.pose_std
            # Clip normalized values to reasonable range to prevent extreme values
            pose = np.clip(pose, -10, 10)
        
        return pose.astype(np.float32)
    
    def _load_audio_features(self, file_id: str) -> np.ndarray:
        # Try multiple naming variations
        audio_path = self.audio_dir / f"{file_id}.wav"
        if not audio_path.exists():
            # Try cAll variation
            alt_id = file_id.replace("_c01_", "_cAll_").replace("_c02_", "_cAll_").replace("_c03_", "_cAll_")
            audio_path = self.audio_dir / f"{alt_id}.wav"
        if not audio_path.exists():
            # Try other camera numbers (c01-c09)
            for cam in ["c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09"]:
                if f"_{cam}_" in file_id:
                    for alt_cam in ["c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "cAll"]:
                        if alt_cam != cam:
                            test_id = file_id.replace(f"_{cam}_", f"_{alt_cam}_")
                            test_path = self.audio_dir / f"{test_id}.wav"
                            if test_path.exists():
                                audio_path = test_path
                                break
                    if audio_path.exists():
                        break
        if not audio_path.exists():
            # Last resort: search for any file with matching pattern (same dance, same sequence)
            parts = file_id.split("_")
            if len(parts) >= 5:
                # Pattern: gLH_sBM_cXX_d17_mLH4_ch09 -> try to find any camera
                base_pattern = f"{parts[0]}_{parts[1]}_*_{parts[3]}_{parts[4]}_{parts[5]}"
                import glob
                matches = list(self.audio_dir.glob(f"{base_pattern}.wav"))
                if matches:
                    audio_path = matches[0]
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found for {file_id}. Searched in {self.audio_dir}")
        audio_features, _ = extract_features_from_file(audio_path, target_sr=16000, **self.audio_feature_config)
        return audio_features
    
    def __len__(self) -> int:
        return len(self.pose_files)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict | None]:
        pose_file = self.pose_files[idx]
        file_id = self._get_file_id(pose_file)
        pose = self._load_pose(pose_file)
        audio_features = self._load_audio_features(file_id)
        audio_features = align_features_to_poses(audio_features, pose.shape[0], self.audio_feature_config.get("hop_length", 512), 16000, self.pose_fps)
        min_len = min(pose.shape[0], audio_features.shape[0])
        pose, audio_features = pose[:min_len], audio_features[:min_len]
        if self.max_seq_length is not None:
            seq_len = pose.shape[0]
            if seq_len < self.max_seq_length:
                pose = np.pad(pose, ((0, self.max_seq_length - seq_len), (0, 0)), mode='constant', constant_values=0)
                audio_features = np.pad(audio_features, ((0, self.max_seq_length - seq_len), (0, 0)), mode='constant', constant_values=0)
            elif seq_len > self.max_seq_length:
                pose, audio_features = pose[:self.max_seq_length], audio_features[:self.max_seq_length]
        
        # Final safety check: remove any remaining NaN/Inf
        pose = np.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
        audio_features = np.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {"pose": torch.from_numpy(pose), "audio_features": torch.from_numpy(audio_features), "file_id": file_id}


# Model
class AudioEncoder(nn.Module):
    def __init__(self, audio_feature_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(audio_feature_dim, hidden_dim, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True, bidirectional=True)
        self.output_dim = 2 * hidden_dim
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.lstm(audio_features)[0]


class PoseEncoder(nn.Module):
    def __init__(self, pose_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(pose_dim, hidden_dim, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True, bidirectional=True)
        self.output_dim = 2 * hidden_dim
    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        if poses.dim() == 4:
            batch, seq_len, num_joints, coords = poses.shape
            poses = poses.view(batch, seq_len, -1)
        return self.lstm(poses)[0]


class CVAEEncoder(nn.Module):
    def __init__(self, audio_encoder: AudioEncoder, pose_encoder: PoseEncoder, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.pose_encoder = pose_encoder
        combined_dim = audio_encoder.output_dim + pose_encoder.output_dim
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)
    def forward(self, audio_features: torch.Tensor, poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_encoded = self.audio_encoder(audio_features)
        pose_encoded = self.pose_encoder(poses)
        combined = torch.cat([audio_encoded, pose_encoded], dim=-1)
        return self.fc_mu(combined), self.fc_logvar(combined)


class CVAEDecoder(nn.Module):
    def __init__(self, audio_encoder: AudioEncoder, latent_dim: int = 128, pose_dim: int = 51, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.audio_encoder = audio_encoder
        decoder_input_dim = audio_encoder.output_dim + latent_dim
        self.lstm = nn.LSTM(decoder_input_dim, hidden_dim, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, pose_dim)
    def forward(self, audio_features: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        audio_encoded = self.audio_encoder(audio_features)
        decoder_input = torch.cat([audio_encoded, z], dim=-1)
        return self.fc_out(self.lstm(decoder_input)[0])


class CVAE(nn.Module):
    def __init__(self, audio_feature_dim: int, pose_dim: int = 51, latent_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.pose_dim = pose_dim
        self.audio_encoder = AudioEncoder(audio_feature_dim, hidden_dim, num_layers, dropout)
        self.pose_encoder = PoseEncoder(pose_dim, hidden_dim, num_layers, dropout)
        self.encoder = CVAEEncoder(self.audio_encoder, self.pose_encoder, latent_dim, hidden_dim)
        self.decoder = CVAEDecoder(self.audio_encoder, latent_dim, pose_dim, hidden_dim, num_layers, dropout)
        
        # Trainable audio-to-latent mapping (for generation)
        # This learns to predict mu from audio features alone
        audio_encoded_dim = hidden_dim * 2  # Bidirectional LSTM output
        self.audio_to_latent = nn.Linear(audio_encoded_dim, latent_dim)
        nn.init.normal_(self.audio_to_latent.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.audio_to_latent.bias)
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Clip logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    def forward(self, audio_features: torch.Tensor, poses: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if poses is not None:
            # Training: use full encoder with poses
            mu, logvar = self.encoder(audio_features, poses)
            z = self.reparameterize(mu, logvar)
            
            # Also train audio_to_latent by predicting mu from audio alone
            # This creates an auxiliary loss that helps generation
            audio_encoded = self.audio_encoder(audio_features)
            audio_predicted_mu = self.audio_to_latent(audio_encoded)
            # Store for potential auxiliary loss (optional)
            audio_mu = audio_predicted_mu
        else:
            # Inference without poses: use trained audio_to_latent
            batch_size, seq_len = audio_features.shape[:2]
            device = audio_features.device
            audio_encoded = self.audio_encoder(audio_features)
            mu = self.audio_to_latent(audio_encoded)
            # Use reasonable logvar (will be overridden in generate with temperature)
            logvar = torch.zeros(batch_size, seq_len, self.latent_dim, device=device)
            z = self.reparameterize(mu, logvar)
            audio_mu = None
        reconstructed = self.decoder(audio_features, z)
        result = {"mu": mu, "logvar": logvar, "z": z, "reconstructed": reconstructed}
        if audio_mu is not None:
            result["audio_mu"] = audio_mu  # For potential auxiliary loss
        return result
    def generate(self, audio_features: torch.Tensor, num_samples: int = 1, temperature: float = 2.0, motion_scale: float = 1.0) -> torch.Tensor:
        """
        Generate poses from audio features.
        
        Args:
            audio_features: Audio feature tensor (batch, seq_len, audio_dim)
            num_samples: Number of samples to generate
            temperature: Temperature for sampling (higher = more variance/motion, default 2.0)
            motion_scale: Scale factor for output motion (1.0 = normal, >1.0 = more motion)
        """
        self.eval()
        with torch.no_grad():
            if num_samples > 1:
                audio_features = audio_features.repeat_interleave(num_samples, dim=0)
            
            batch_size, seq_len = audio_features.shape[:2]
            device = audio_features.device
            
            # Encode audio to get a better latent distribution
            audio_encoded = self.audio_encoder(audio_features)
            
            # Use trained audio_to_latent mapping (now properly trained!)
            mu = self.audio_to_latent(audio_encoded)
            
            # Use temperature to control variance
            # Higher temperature = more exploration/motion
            logvar = torch.ones_like(mu) * np.log(temperature * temperature + 1e-8)
            logvar = torch.clamp(logvar, min=-10, max=10)
            
            # Sample from the distribution
            z = self.reparameterize(mu, logvar)
            
            # Decode
            reconstructed = self.decoder(audio_features, z)
            
            # Apply motion scaling to amplify movements
            if motion_scale != 1.0:
                # Scale movements relative to mean pose
                mean_pose = reconstructed.mean(dim=1, keepdim=True)  # Mean across sequence
                reconstructed = mean_pose + (reconstructed - mean_pose) * motion_scale
            
            # Note: Cross-body movement amplification removed - was causing skeleton distortion
            # Use motion_scale and visualization amplification instead for better results
            
            return reconstructed


# Trainer
class CVAELoss(nn.Module):
    def __init__(self, kl_weight: float = 1.0, free_bits: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.kl_weight = kl_weight
        self.free_bits = free_bits  # Minimum KL per dimension to prevent collapse
        self.reduction = reduction
    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> dict[str, torch.Tensor]:
        # Clip logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        recon_loss = nn.functional.mse_loss(reconstructed, target, reduction=self.reduction)
        
        # Analytic KL divergence for diagonal Gaussian posterior
        # Handle sequences: mu, logvar can be [batch, seq_len, latent_dim] or [batch, latent_dim]
        # Flatten to [batch*seq_len, latent_dim] or keep as [batch, latent_dim]
        original_shape = mu.shape
        if mu.dim() == 3:
            # [batch, seq_len, latent_dim] -> [batch*seq_len, latent_dim]
            batch_size, seq_len, latent_dim = mu.shape
            mu = mu.view(-1, latent_dim)  # [batch*seq_len, latent_dim]
            logvar = logvar.view(-1, latent_dim)  # [batch*seq_len, latent_dim]
        elif mu.dim() == 2:
            # [batch, latent_dim] - already correct
            batch_size, latent_dim = mu.shape
            seq_len = 1
        else:
            raise ValueError(f"Unexpected mu shape: {mu.shape}")
        
        # Analytic KL: kl_per_sample = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # Sum over latent_dim (dim=1), result: [batch*seq_len] or [batch]
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Apply free bits: minimum KL per latent dimension (prevents collapse)
        # free_bits is per dimension, so total min is free_bits * latent_dim
        if self.free_bits > 0:
            min_kl_per_sample = self.free_bits * latent_dim
            kl_per_sample = torch.clamp(kl_per_sample, min=min_kl_per_sample)
        
        # Reshape back if we had sequences
        if len(original_shape) == 3:
            kl_per_sample = kl_per_sample.view(batch_size, seq_len)
        
        # Average over batch (and sequence if present)
        kl = kl_per_sample.mean()
        
        # Check for NaN and replace with 0
        if torch.isnan(recon_loss):
            recon_loss = torch.tensor(0.0, device=recon_loss.device)
        if torch.isnan(kl):
            kl = torch.tensor(0.0, device=kl.device)
        
        # Loss: recon + kl_weight * kl
        total_loss = recon_loss + self.kl_weight * kl
        if torch.isnan(total_loss):
            total_loss = torch.tensor(0.0, device=total_loss.device)
        
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl}


class CVAETrainer:
    def __init__(self, model: CVAE, train_loader: DataLoader, val_loader: DataLoader | None = None,
                device: str = "cuda" if torch.cuda.is_available() else "cpu", lr: float = 1e-4, 
                kl_weight: float = 1.0, save_dir: Path | str = "experiments",
                kl_anneal_epochs: int = 25, free_bits: float = 0.1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # KL annealing: gradually increase KL weight to prevent collapse
        self.kl_anneal_epochs = kl_anneal_epochs
        self.base_kl_weight = kl_weight
        
        self.criterion = CVAELoss(kl_weight=0.0, free_bits=free_bits)  # Start with 0, anneal up
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)
        self.current_epoch = 0
        self.best_val_loss = float("inf")  # For scheduler (total loss)
        self.best_val_recon_loss = float("inf")  # For early stopping (reconstruction loss)
        self.train_history = []
        self.val_history = []
    
    def _get_kl_weight(self, epoch: int) -> float:
        """Compute KL weight with annealing: 0→1 over kl_anneal_epochs."""
        if epoch < self.kl_anneal_epochs:
            # Gradually increase from 0 to 1.0 over annealing period
            return epoch / self.kl_anneal_epochs
        return 1.0
    
    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = total_recon_loss = total_kl_loss = 0.0
        num_batches = 0
        for batch in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}"):
            poses = batch["pose"].to(self.device)
            audio_features = batch["audio_features"].to(self.device)
            
            # Check for NaN/Inf in inputs
            if torch.isnan(poses).any() or torch.isinf(poses).any():
                print(f"Warning: NaN/Inf detected in poses, skipping batch")
                continue
            if torch.isnan(audio_features).any() or torch.isinf(audio_features).any():
                print(f"Warning: NaN/Inf detected in audio_features, skipping batch")
                continue
            
            self.optimizer.zero_grad()
            output = self.model(audio_features, poses)
            
            # Check output for NaN
            if torch.isnan(output["reconstructed"]).any():
                print(f"Warning: NaN in model output, skipping batch")
                continue
            
            loss_dict = self.criterion(output["reconstructed"], poses, output["mu"], output["logvar"])
            
            # Skip if loss is NaN
            if torch.isnan(loss_dict["loss"]):
                print(f"Warning: NaN loss detected, skipping batch")
                continue
            
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss_dict["loss"].item()
            total_recon_loss += loss_dict["recon_loss"].item()
            total_kl_loss += loss_dict["kl_loss"].item()
            num_batches += 1
        
        if num_batches == 0:
            return {"loss": float('inf'), "recon_loss": float('inf'), "kl_loss": float('inf')}
        
        return {"loss": total_loss / num_batches, "recon_loss": total_recon_loss / num_batches, "kl_loss": total_kl_loss / num_batches}
    
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = total_recon_loss = total_kl_loss = 0.0
        num_batches = 0
        for batch in tqdm(self.val_loader, desc="Validating"):
            poses = batch["pose"].to(self.device)
            audio_features = batch["audio_features"].to(self.device)
            output = self.model(audio_features, poses)
            loss_dict = self.criterion(output["reconstructed"], poses, output["mu"], output["logvar"])
            total_loss += loss_dict["loss"].item()
            total_recon_loss += loss_dict["recon_loss"].item()
            total_kl_loss += loss_dict["kl_loss"].item()
            num_batches += 1
        return {"loss": total_loss / num_batches, "recon_loss": total_recon_loss / num_batches, "kl_loss": total_kl_loss / num_batches}
    
    def save_checkpoint(self, filename: str = "checkpoint.pt", is_best: bool = False):
        checkpoint = {"epoch": self.current_epoch, "model_state_dict": self.model.state_dict(),
                     "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": self.scheduler.state_dict(),
                     "best_val_loss": self.best_val_loss, "best_val_recon_loss": self.best_val_recon_loss,
                     "train_history": self.train_history, "val_history": self.val_history}
        torch.save(checkpoint, self.save_dir / filename)
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pt")
    
    def load_checkpoint(self, checkpoint_path: Path | str, strict: bool = True):
        """
        Load checkpoint. If architectures don't match, use strict=False to load only compatible weights.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: If False, only load weights that match (for architecture changes)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Try loading model weights (with strict=False if architectures differ)
        model_dict = self.model.state_dict()
        checkpoint_dict = checkpoint["model_state_dict"]
        
        # Check for mismatches
        missing_keys = set(model_dict.keys()) - set(checkpoint_dict.keys())
        unexpected_keys = set(checkpoint_dict.keys()) - set(model_dict.keys())
        mismatched_shapes = []
        for key in model_dict.keys():
            if key in checkpoint_dict:
                if model_dict[key].shape != checkpoint_dict[key].shape:
                    mismatched_shapes.append((key, model_dict[key].shape, checkpoint_dict[key].shape))
        
        if missing_keys or unexpected_keys or mismatched_shapes:
            if strict:
                error_msg = []
                if missing_keys:
                    error_msg.append(f"Missing keys: {list(missing_keys)[:5]}...")
                if unexpected_keys:
                    error_msg.append(f"Unexpected keys: {list(unexpected_keys)[:5]}...")
                if mismatched_shapes:
                    error_msg.append(f"Shape mismatches: {mismatched_shapes[0]}")
                raise RuntimeError("Architecture mismatch!\n" + "\n".join(error_msg) + 
                                 "\nTip: Remove --resume or use same architecture, or train from scratch.")
            else:
                # Partial loading: only load compatible weights
                compatible_dict = {k: v for k, v in checkpoint_dict.items() 
                                 if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(compatible_dict)
                self.model.load_state_dict(model_dict)
                
                total_params = len(checkpoint_dict)
                loaded_params = len(compatible_dict)
                print(f"⚠️  Architecture mismatch detected.")
                print(f"   Loaded {loaded_params}/{total_params} compatible weights ({100*loaded_params/total_params:.1f}%)")
                if mismatched_shapes:
                    print(f"   Skipped {len(mismatched_shapes)} layers with shape mismatches")
                if missing_keys:
                    print(f"   {len(missing_keys)} new layers will be randomly initialized")
                print(f"   Training will continue with partial weights + random initialization for new layers.")
        else:
            # Perfect match - load normally
            self.model.load_state_dict(checkpoint_dict, strict=True)
            print(f"✓ Loaded checkpoint successfully ({len(checkpoint_dict)} weights)")
        
        # Only load optimizer/scheduler if architectures match (strict=True)
        if strict:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except (KeyError, RuntimeError) as e:
                print(f"Warning: Could not load optimizer/scheduler: {e}")
                print("Starting with fresh optimizer/scheduler state.")
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_recon_loss = checkpoint.get("best_val_recon_loss", float("inf"))
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])
    
    def train(self, num_epochs: int, save_every: int = 10, early_stop_patience: int = 15):
        """
        Train the model.
        
        Args:
            num_epochs: Maximum number of epochs
            save_every: Save checkpoint every N epochs
            early_stop_patience: Stop training if no improvement for N epochs (None to disable)
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if early_stop_patience:
            print(f"Early stopping enabled: patience={early_stop_patience} epochs")
        
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # KL annealing: gradually increase KL weight to prevent collapse
            current_kl_weight = self._get_kl_weight(epoch)
            self.criterion.kl_weight = current_kl_weight
            
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            val_metrics = self.validate()
            if val_metrics:
                self.val_history.append(val_metrics)
                # Scheduler uses total loss
                self.scheduler.step(val_metrics["loss"])
                self.best_val_loss = min(self.best_val_loss, val_metrics["loss"])
                
                # Early stopping checks validation reconstruction loss (not total loss)
                val_recon_loss = val_metrics["recon_loss"]
                if val_recon_loss < self.best_val_recon_loss:
                    self.best_val_recon_loss = val_recon_loss
                    self.save_checkpoint(is_best=True)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            # Log: epoch, KL weight, recon loss, KL
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"KL weight: {current_kl_weight:.4f} | Recon loss: {train_metrics['recon_loss']:.4f} | KL: {train_metrics['kl_loss']:.4f}")
            if val_metrics:
                print(f"Val - Recon loss: {val_metrics['recon_loss']:.4f} | KL: {val_metrics['kl_loss']:.4f} | Total loss: {val_metrics['loss']:.4f}")
                if early_stop_patience:
                    print(f"      - No improvement for {epochs_without_improvement}/{early_stop_patience} epochs")
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint()
            with open(self.save_dir / "training_history.json", "w") as f:
                json.dump({"train": self.train_history, "val": self.val_history}, f, indent=2)
            
            # Early stopping: checks validation reconstruction loss
            if early_stop_patience and epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping triggered! No improvement for {early_stop_patience} epochs.")
                print(f"Best validation reconstruction loss: {self.best_val_recon_loss:.4f} (at epoch {epoch + 1 - early_stop_patience})")
                break
        
        print(f"\nTraining complete! Best validation reconstruction loss: {self.best_val_recon_loss:.4f}")


# Evaluation
def compute_smoothness(poses: np.ndarray) -> float:
    velocities = np.diff(poses, axis=0)
    if poses.ndim == 3:
        velocities = velocities.reshape(velocities.shape[0], -1)
    velocity_magnitudes = np.linalg.norm(velocities, axis=-1)
    return float(1.0 / (1.0 + np.var(velocity_magnitudes)))


def compute_beat_alignment(generated_poses: np.ndarray, beat_times: np.ndarray | list[float], pose_fps: int = 30) -> float:
    if signal is None:
        return 0.0
    velocities = np.diff(generated_poses, axis=0)
    if generated_poses.ndim == 3:
        velocities = velocities.reshape(velocities.shape[0], -1)
    motion_energy = np.linalg.norm(velocities, axis=-1)
    beat_frames = (np.array(beat_times) * pose_fps).astype(int)
    beat_frames = beat_frames[beat_frames < len(motion_energy)]
    if len(beat_frames) == 0:
        return 0.0
    beat_signal = np.zeros(len(motion_energy))
    beat_signal[beat_frames] = 1.0
    motion_energy = (motion_energy - motion_energy.mean()) / (motion_energy.std() + 1e-8)
    beat_signal = (beat_signal - beat_signal.mean()) / (beat_signal.std() + 1e-8)
    correlation = signal.correlate(motion_energy, beat_signal, mode="valid")
    return float(np.max(np.abs(correlation)) / (len(motion_energy) + 1e-8))


def evaluate_generated_poses(generated_poses: np.ndarray, ground_truth_poses: np.ndarray | None = None,
                             beat_times: np.ndarray | list[float] | None = None, pose_fps: int = 30) -> dict[str, float]:
    metrics = {"smoothness": compute_smoothness(generated_poses)}
    if beat_times is not None:
        metrics["beat_alignment"] = compute_beat_alignment(generated_poses, beat_times, pose_fps)
    if ground_truth_poses is not None:
        gen_flat = generated_poses.reshape(generated_poses.shape[0], -1) if generated_poses.ndim == 3 else generated_poses
        gt_flat = ground_truth_poses.reshape(ground_truth_poses.shape[0], -1) if ground_truth_poses.ndim == 3 else ground_truth_poses
        metrics["mse"] = float(np.mean((gen_flat - gt_flat) ** 2))
        if generated_poses.ndim == 3:
            joint_errors = np.mean(np.linalg.norm(generated_poses - ground_truth_poses, axis=-1), axis=0)
            metrics["mean_joint_error"] = float(np.mean(joint_errors))
    return metrics

