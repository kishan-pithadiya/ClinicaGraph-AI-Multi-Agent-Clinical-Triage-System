"""
ClinicaGraph AI - Dermoscopic Skin Lesion Segmentation & Analysis
Provides U-Net deep learning segmentation with graceful computer vision fallback.
"""

import os
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("ClinicaGraph.SkinLesionAgent")


class SkinLesionSegmentation:
    """Handles skin lesion segmentation using trained U-Net or dermoscopic CV fallback."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Attempts to load PyTorch U-Net checkpoint safely."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            if os.path.exists(self.model_path):
                # Try loading state dict
                checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
                from .skin_lesion_inference import UNet
                model = UNet(n_channels=3, n_classes=1)
                state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                model.load_state_dict(state, strict=False)
                model.eval()
                logger.info(f"Loaded Skin Lesion U-Net model from {self.model_path}")
                return model
        except Exception as e:
            logger.warning(f"PyTorch Skin Lesion checkpoint unavailable: {e}. Active CV fallback enabled.")
        return None

    def _cv_segmentation_fallback(self, image_path: str, output_path: str) -> bool:
        """
        Dermoscopic lesion boundary extraction using adaptive color thresholding & morphological filtering.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False

            h, w = img.shape[:2]
            # Convert to LAB color space (lesions contrast strongly in L and B channels)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Invert L channel (lesions are typically darker)
            inv_l = cv2.bitwise_not(l_channel)
            blurred = cv2.GaussianBlur(inv_l, (9, 9), 0)

            # Otsu thresholding
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Circular vignette mask to avoid camera perimeter edges
            center = (w // 2, h // 2)
            radius = int(min(w, h) * 0.45)
            vignette_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(vignette_mask, center, radius, 255, -1)
            mask = cv2.bitwise_and(mask, vignette_mask)

            # Morphological closing to seal lesion interior
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Create visualization overlay
            overlay = img.copy()
            # Draw semi-transparent cyan highlight
            color_mask = np.zeros_like(img)
            color_mask[:, :] = (255, 180, 0)  # BGR Cyan/Amber
            lesion_area = closed_mask > 0
            overlay[lesion_area] = cv2.addWeighted(overlay[lesion_area], 0.6, color_mask[lesion_area], 0.4, 0)

            # Draw outer contour line
            contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                cv2.drawContours(overlay, [largest_c], -1, (0, 255, 0), 2)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, overlay)
            logger.info(f"Saved dermoscopic segmentation overlay to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Fallback segmentation failed: {e}")
            return False

    def predict(self, image_path: str, output_path: str) -> bool:
        """Executes segmentation and writes visualization overlay to output_path."""
        if self.model is not None:
            try:
                import torch
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                img_resized = cv2.resize(img_rgb, (256, 256))
                img_tensor = torch.Tensor(img_resized).unsqueeze(0).permute(0, 3, 1, 2)

                with torch.no_grad():
                    generated_mask = self.model(img_tensor).squeeze().cpu().numpy()

                generated_mask_resized = cv2.resize(generated_mask, (img.shape[1], img.shape[0]))
                
                # Plot
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.axis("off")
                ax.imshow(img_rgb)
                ax.imshow(generated_mask_resized, alpha=0.4, cmap='jet')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, bbox_inches="tight")
                plt.close(fig)
                return True
            except Exception as e:
                logger.warning(f"U-Net inference error: {e}. Executing CV fallback.")

        return self._cv_segmentation_fallback(image_path, output_path)
