import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from pytorch_wavelets import DWTForward  # 使用 pytorch_wavelets


class WaveletVisualizationHook:

    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    def __init__(self, save_dir=f'./wavelet_visualizations/{current_time}', max_samples=10, wavelet='db1'):
        self.base_dir  = Path(save_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.wavelet = wavelet
        self.current_epoch = 0
        self.sample_count = 0
        self.dwt = DWTForward(wave=self.wavelet, J=1, mode='periodization')
        self._update_save_dir()

    def _update_save_dir(self):
        self.save_dir = self.base_dir / f'epoch_{self.current_epoch:03d}'
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def set_epoch(self, epoch):
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            self.sample_count = 0 
            self._update_save_dir()
            print(f" Wavelet visualization: Starting epoch {epoch}, saving to {self.save_dir}")

    def should_visualize(self):
        return self.sample_count < self.max_samples

    def visualize_augmentation(self, frames_before, frames_after, index):

        if not self.should_visualize():
            return

        if len(frames_before) == 0:
            return

        orig_tensor = frames_before[0][0]

        if isinstance(orig_tensor, torch.Tensor):
            img = orig_tensor.detach().cpu().numpy()
        else:
            img = orig_tensor

        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0, 1)

        if img.ndim == 3 and img.shape[2] == 3:
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img.squeeze()

        gray_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)

        yl, yh = self.dwt(gray_tensor)

        LL = yl[0, 0].cpu().numpy()  # [H/2, W/2]
        LH = yh[0][0, 0, 0].cpu().numpy()  # [H/2, W/2]
        HL = yh[0][0, 0, 1].cpu().numpy()  # [H/2, W/2]
        HH = yh[0][0, 0, 2].cpu().numpy()  # [H/2, W/2]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f'Wavelet Subbands (Sample {index}) - {self.wavelet.upper()}',
                     fontsize=14, fontweight='bold')

        im1 = axes[0, 0].imshow(LL, cmap='gray')
        axes[0, 0].set_title('LL (Approximation)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

        im2 = axes[0, 1].imshow(LH, cmap='RdBu_r')
        axes[0, 1].set_title('LH (Horizontal)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

        im3 = axes[1, 0].imshow(HL, cmap='RdBu_r')
        axes[1, 0].set_title('HL (Vertical)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

        im4 = axes[1, 1].imshow(HH, cmap='RdBu_r')
        axes[1, 1].set_title('HH (Diagonal)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

        plt.tight_layout()

        save_path = self.save_dir / f'subbands_{index:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.sample_count += 1
        print(f"Saved wavelet subbands: {save_path.name}")

    def visualize_detailed_comparison(self, frames_before, frames_after, index):

        if not self.should_visualize():
            return

        views_to_visualize = []

        if len(frames_before) > 0:
            orig_tensor = frames_before[0][0]  # [H, W, C]
            views_to_visualize.append(("Original", orig_tensor))

        if len(frames_after) > 0:
            global_frame = frames_after[0][:, 0, :, :]  # [C, H, W]
            views_to_visualize.append(("Global", global_frame))

        if len(frames_after) > 2:
            local_frame = frames_after[2][:, 0, :, :]  # [C, H, W]
            views_to_visualize.append(("Local", local_frame))

        num_views = len(views_to_visualize)
        if num_views == 0:
            return

        fig, axes = plt.subplots(num_views, 4, figsize=(16, 4 * num_views))
        if num_views == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Wavelet Subbands Comparison (Sample {index}) - {self.wavelet.upper()}',
                     fontsize=15, fontweight='bold')

        for row_idx, (view_name, tensor) in enumerate(views_to_visualize):
            if isinstance(tensor, torch.Tensor):
                img = tensor.detach().cpu().numpy()
                if img.ndim == 3 and img.shape[0] == 3:  # [C, H, W]
                    img = np.transpose(img, (1, 2, 0))  # [H, W, C]
            else:
                img = tensor

            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0, 1)

            if img.ndim == 3 and img.shape[2] == 3:
                gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img.squeeze()

            gray_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)

            yl, yh = self.dwt(gray_tensor)

            LL = yl[0, 0].cpu().numpy()
            LH = yh[0][0, 0, 0].cpu().numpy()
            HL = yh[0][0, 0, 1].cpu().numpy()
            HH = yh[0][0, 0, 2].cpu().numpy()

            im1 = axes[row_idx, 0].imshow(LL, cmap='gray')
            axes[row_idx, 0].set_title(f'{view_name} - LL', fontsize=11, fontweight='bold')
            axes[row_idx, 0].axis('off')
            plt.colorbar(im1, ax=axes[row_idx, 0], fraction=0.046)

            im2 = axes[row_idx, 1].imshow(LH, cmap='RdBu_r')
            axes[row_idx, 1].set_title(f'{view_name} - LH', fontsize=11, fontweight='bold')
            axes[row_idx, 1].axis('off')
            plt.colorbar(im2, ax=axes[row_idx, 1], fraction=0.046)

            im3 = axes[row_idx, 2].imshow(HL, cmap='RdBu_r')
            axes[row_idx, 2].set_title(f'{view_name} - HL', fontsize=11, fontweight='bold')
            axes[row_idx, 2].axis('off')
            plt.colorbar(im3, ax=axes[row_idx, 2], fraction=0.046)

            im4 = axes[row_idx, 3].imshow(HH, cmap='RdBu_r')
            axes[row_idx, 3].set_title(f'{view_name} - HH', fontsize=11, fontweight='bold')
            axes[row_idx, 3].axis('off')
            plt.colorbar(im4, ax=axes[row_idx, 3], fraction=0.046)

        plt.tight_layout()

        save_path = self.save_dir / f'subbands_comparison_{index:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"✓ Saved comparison: {save_path.name}")