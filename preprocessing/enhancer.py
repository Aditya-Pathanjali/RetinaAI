
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class RetinalEnhancer:
    
    def __init__(self, config: Dict[str, Any]):
        
        self.image_size = config.get("image_size", 512)

        # CLAHE settings
        self.apply_clahe = config.get("apply_clahe", True)
        self.clahe_clip_limit = config.get("clahe_clip_limit", 2.0)
        self.clahe_tile_grid = tuple(config.get("clahe_tile_grid", [8, 8]))

        # Other toggles
        self.apply_bg_crop = config.get("apply_bg_crop", True)
        self.apply_green_enhance = config.get("apply_green_enhance", True)
        self.apply_illum_correct = config.get("apply_illum_correct", True)
        self.apply_denoise = config.get("apply_denoise", False)
        self.denoise_kernel = config.get("denoise_kernel", 3)

   #public api
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.process(image)

    def process(self, image: np.ndarray) -> np.ndarray:
        # Step 1: Remove black borders (must come before resizing)
        if self.apply_bg_crop:
            image = self.crop_background(image)

        # Step 2: Resize to target resolution
        image = cv2.resize(
            image, (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,  # Best for downsampling
        )

        # Step 3: Illumination correction
        if self.apply_illum_correct:
            image = self.correct_illumination(image)

        # Step 4: CLAHE on Luminance channel
        if self.apply_clahe:
            image = self.apply_clahe_enhancement(image)

        # Step 5: Green channel enhancement
        if self.apply_green_enhance:
            image = self.enhance_green_channel(image)

        # Step 6: Optional denoising
        if self.apply_denoise:
            image = self.denoise(image)

        return image

    #individual enhancement steps
    @staticmethod
    def crop_background(image: np.ndarray, threshold: int = 10) -> np.ndarray:
        """Crop black borders. Returns cropped image only (legacy API)."""
        cropped, _ = RetinalEnhancer.crop_background_with_box(image, threshold)
        return cropped

    @staticmethod
    def crop_background_with_box(
        image: np.ndarray, threshold: int = 10
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """Crop black borders and return (cropped_image, (x, y, w, h))."""
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        coords = cv2.findNonZero(mask)
        if coords is None:
            return image, None

        x, y, w, h = cv2.boundingRect(coords)

        pad_x = max(int(w * 0.02), 2)
        pad_y = max(int(h * 0.02), 2)
        x = max(x - pad_x, 0)
        y = max(y - pad_y, 0)
        w = min(w + 2 * pad_x, image.shape[1] - x)
        h = min(h + 2 * pad_y, image.shape[0] - y)

        return image[y:y + h, x:x + w], (x, y, w, h)

    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid,
        )
        l_channel = clahe.apply(l_channel)

        lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def enhance_green_channel(image: np.ndarray, weight: float = 0.3) -> np.ndarray:
        
        b, g, r = cv2.split(image)

        # CLAHE-enhanced green channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_enhanced = clahe.apply(g)

        
        enhanced = image.copy().astype(np.float32)
        green_boost = cv2.merge([g_enhanced, g_enhanced, g_enhanced]).astype(np.float32)
        blended = cv2.addWeighted(
            enhanced, 1.0 - weight, green_boost, weight, 0.0,
        )

        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def correct_illumination(image: np.ndarray, kernel_size: int = 67) -> np.ndarray:
        
        if kernel_size % 2 == 0:
            kernel_size += 1

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        # Estimate background illumination with large Gaussian
        background = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)

        # Subtract background and rescale
        corrected = l_channel - background + 128.0  # Centre at mid-grey
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        lab[:, :, 0] = corrected
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        
        k = self.denoise_kernel
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(image, (k, k), 0)

    #Mask Preprocessing
    def process_mask(self, mask: np.ndarray, crop_box: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Resize mask (and optionally crop with the same box used for the image)."""
        # Apply the same crop that was applied to the image
        if crop_box is not None:
            x, y, w, h = crop_box
            mask = mask[y:y + h, x:x + w]

        if mask.ndim == 3:
            resized_channels = []
            for c in range(mask.shape[2]):
                resized = cv2.resize(
                    mask[:, :, c],
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                resized_channels.append(resized)
            return np.stack(resized_channels, axis=2)
        else:
            return cv2.resize(
                mask, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

    def process_pair(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process image AND mask together, ensuring identical spatial transforms.
        This is the CORRECT method to use during training/evaluation.
        """
        crop_box = None

        # Step 1: Background crop (SAME box for image and mask)
        if self.apply_bg_crop:
            image, crop_box = self.crop_background_with_box(image)
            if crop_box is not None:
                x, y, w, h = crop_box
                mask = mask[y:y + h, x:x + w]

        # Step 2: Resize both to target resolution
        image = cv2.resize(
            image, (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        mask = self._resize_mask(mask)

        # Step 3–6: Image-only enhancements (do NOT touch the mask)
        if self.apply_illum_correct:
            image = self.correct_illumination(image)
        if self.apply_clahe:
            image = self.apply_clahe_enhancement(image)
        if self.apply_green_enhance:
            image = self.enhance_green_channel(image)
        if self.apply_denoise:
            image = self.denoise(image)

        return image, mask

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize a mask to self.image_size using nearest-neighbour."""
        if mask.ndim == 3:
            resized_channels = []
            for c in range(mask.shape[2]):
                resized = cv2.resize(
                    mask[:, :, c],
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                resized_channels.append(resized)
            return np.stack(resized_channels, axis=2)
        else:
            return cv2.resize(
                mask, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

    #Visualization Helpers
    def get_intermediate_results(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        
        results = {"0_original": image.copy()}

        if self.apply_bg_crop:
            image = self.crop_background(image)
            results["1_cropped"] = image.copy()

        image = cv2.resize(
            image, (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        results["2_resized"] = image.copy()

        if self.apply_illum_correct:
            image = self.correct_illumination(image)
            results["3_illum_corrected"] = image.copy()

        if self.apply_clahe:
            image = self.apply_clahe_enhancement(image)
            results["4_clahe"] = image.copy()

        if self.apply_green_enhance:
            image = self.enhance_green_channel(image)
            results["5_green_enhanced"] = image.copy()

        if self.apply_denoise:
            image = self.denoise(image)
            results["6_denoised"] = image.copy()

        results["final"] = image
        return results
