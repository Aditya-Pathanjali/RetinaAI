

import sys
import os
import gc
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
torch.set_num_threads(2)  # Limit CPU parallelism to prevent RAM spikes
from PIL import Image, ImageTk

import time
import traceback

IS_FROZEN = getattr(sys, 'frozen', False)
if IS_FROZEN:
    RESOURCE_ROOT = Path(sys._MEIPASS)
    OUTPUT_ROOT = Path(sys.executable).resolve().parent
else:
    RESOURCE_ROOT = Path(__file__).resolve().parent
    OUTPUT_ROOT = RESOURCE_ROOT

sys.path.insert(0, str(RESOURCE_ROOT))

def log_exception(exc_type, exc_value, exc_traceback):
    try:
        log_file = OUTPUT_ROOT / "retinaai_crash.log"
        with open(log_file, "a") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Crash Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Executable Path: {sys.executable}\n")
            f.write(f"Frozen: {IS_FROZEN}\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    except:
        pass

sys.excepthook = log_exception

from utils.helpers import load_config
from preprocessing.enhancer import RetinalEnhancer
from preprocessing.transforms import get_val_transforms
from models.attention_unet import build_model
from models.hybrid_classifier import build_classifier
from utils.lesion_counter import count_lesions_batch

# Palette
BG_DARK = "#121824"
BG_PANEL = "#1a2333"
TEXT_COLOR = "#ffffff"
TEXT_MUTED = "#8a9bb8"
ACCENT_GREEN = "#2ecc71"
ACCENT_BLUE = "#3498db"
ACCENT_RED = "#e74c3c"
FONT_FAMILY = "Segoe UI"


class RetinaAIApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RetinaAI — Automated Diabetic Retinopathy Screening System")
        self.root.geometry("1050x700")
        self.root.minsize(950, 650)
        self.root.configure(bg=BG_DARK)
        
        # config
        self.config_path = RESOURCE_ROOT / "configs" / "config.yaml"
        self.config = load_config(str(self.config_path))
        
        # Session states
        self.current_image_path: Optional[str] = None
        self.seg_overlay_highres: Optional[np.ndarray] = None
        self.class_names = self.config["dataset"]["class_names"]
        
        # Resolve thresholds
        eval_cfg = self.config.get("evaluation", {})
        config_thresholds = eval_cfg.get("threshold", 0.5)
        config_min_areas = eval_cfg.get("min_area", 0)
        
        if isinstance(config_thresholds, dict):
            self.thresholds = {cls: float(config_thresholds.get(cls, 0.5)) for cls in self.class_names}
        else:
            self.thresholds = {cls: float(config_thresholds) for cls in self.class_names}
            
        if isinstance(config_min_areas, dict):
            self.min_areas = {cls: int(config_min_areas.get(cls, 0)) for cls in self.class_names}
        else:
            self.min_areas = {cls: int(config_min_areas) for cls in self.class_names}
            
        # UI Setup
        self._setup_style()
        self._build_ui()
        
        
        self.device = torch.device("cpu")
        self.seg_model = None
        self.classifier = None
        self.status_text.config(
            text="System ready. Upload a retinal fundus scan to begin.",
            fg=ACCENT_GREEN
        )

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", bg=BG_DARK, fg=TEXT_COLOR)
        style.configure("TFrame", background=BG_DARK)
        style.configure("Panel.TFrame", background=BG_PANEL, relief="flat")
        style.configure("Status.TLabel", background=BG_PANEL, foreground=TEXT_MUTED, font=(FONT_FAMILY, 9))

    def _build_ui(self):
        # Top Header Area
        header_frame = tk.Frame(self.root, bg=BG_DARK, height=80)
        header_frame.pack(fill="x", padx=20, pady=10)
        
        title_label = tk.Label(
            header_frame,
            text="RetinaAI",
            font=(FONT_FAMILY, 22, "bold"),
            bg=BG_DARK,
            fg=ACCENT_BLUE
        )
        title_label.pack(side="left")
        
        subtitle_label = tk.Label(
            header_frame,
            text=" | Automated Diabetic Retinopathy Diagnostic GUI",
            font=(FONT_FAMILY, 11, "italic"),
            bg=BG_DARK,
            fg=TEXT_MUTED
        )
        subtitle_label.pack(side="left", pady=10)
        
        # Main splits
        main_pane = tk.Frame(self.root, bg=BG_DARK)
        main_pane.pack(fill="both", expand=True, padx=20, pady=5)
        
        # Left pane (Input and controls)
        left_pane = tk.Frame(main_pane, bg=BG_DARK, width=450)
        left_pane.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
       
        # Custom frame with padding
        file_frame = tk.Frame(left_pane, bg=BG_PANEL)
        file_frame.pack(fill="x", pady=(0, 10), ipady=5)
        
        self.file_btn = tk.Button(
            file_frame,
            text="Upload Retinal Scan",
            font=(FONT_FAMILY, 10, "bold"),
            bg=ACCENT_BLUE,
            fg=TEXT_COLOR,
            activebackground="#2980b9",
            activeforeground=TEXT_COLOR,
            relief="flat",
            padx=10,
            command=self._upload_image
        )
        self.file_btn.pack(side="left", padx=10, pady=10)
        
        self.file_path_label = tk.Label(
            file_frame,
            text="No scan uploaded.",
            font=(FONT_FAMILY, 9),
            bg=BG_PANEL,
            fg=TEXT_MUTED,
            anchor="w"
        )
        self.file_path_label.pack(side="left", fill="x", expand=True, padx=(5, 10))
        
        # Input image view frame
        self.input_frame = tk.Frame(left_pane, bg=BG_PANEL)
        self.input_frame.pack(fill="both", expand=True)
        
        self.input_label = tk.Label(
            self.input_frame,
            text="[ Retinal Scan Input View ]",
            font=(FONT_FAMILY, 11),
            bg=BG_PANEL,
            fg=TEXT_MUTED
        )
        self.input_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Right pane (Segmentation Overlay display)
        right_pane = tk.Frame(main_pane, bg=BG_DARK, width=450)
        right_pane.pack(side="left", fill="both", expand=True, padx=(10, 0))
        
        # Action Buttons frame
        actions_frame = tk.Frame(right_pane, bg=BG_PANEL)
        actions_frame.pack(fill="x", pady=(0, 10), ipady=5)
        
        self.btn_seg = tk.Button(
            actions_frame,
            text="A. Run Segmentation",
            font=(FONT_FAMILY, 10, "bold"),
            bg=ACCENT_GREEN,
            fg=TEXT_COLOR,
            activebackground="#27ae60",
            activeforeground=TEXT_COLOR,
            relief="flat",
            state="disabled",
            padx=10,
            command=self._run_segmentation
        )
        self.btn_seg.pack(side="left", padx=10, pady=10)
        
        self.btn_cls = tk.Button(
            actions_frame,
            text="B. Run Classification",
            font=(FONT_FAMILY, 10, "bold"),
            bg=ACCENT_BLUE,
            fg=TEXT_COLOR,
            activebackground="#2980b9",
            activeforeground=TEXT_COLOR,
            relief="flat",
            state="disabled",
            padx=10,
            command=self._run_classification
        )
        self.btn_cls.pack(side="left", padx=10, pady=10)
        
        self.btn_dl = tk.Button(
            actions_frame,
            text="Download Overlay",
            font=(FONT_FAMILY, 10, "bold"),
            bg="#f39c12",
            fg=TEXT_COLOR,
            activebackground="#d35400",
            activeforeground=TEXT_COLOR,
            relief="flat",
            state="disabled",
            padx=10,
            command=self._download_overlay
        )
        self.btn_dl.pack(side="right", padx=10, pady=10)
        
        # Output image view frame
        self.output_frame = tk.Frame(right_pane, bg=BG_PANEL)
        self.output_frame.pack(fill="both", expand=True)
        
        self.output_label = tk.Label(
            self.output_frame,
            text="[ Segmentation Overlay View ]",
            font=(FONT_FAMILY, 11),
            bg=BG_PANEL,
            fg=TEXT_MUTED
        )
        self.output_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Legend and Analysis frame
        self.analysis_frame = tk.LabelFrame(
            right_pane,
            text=" Segmentation Analysis & Clinical Legend ",
            font=(FONT_FAMILY, 10, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_BLUE,
            relief="solid",
            bd=1
        )
        self.analysis_frame.pack(fill="x", side="bottom", pady=(10, 0), ipady=5)
        
        self.placeholder_label = tk.Label(
            self.analysis_frame,
            text="Upload a retinal scan and click 'A. Run Segmentation' to view lesion analysis and model accuracy.",
            font=(FONT_FAMILY, 9, "italic"),
            bg=BG_PANEL,
            fg=TEXT_MUTED,
            wraplength=420
        )
        self.placeholder_label.pack(fill="both", expand=True, padx=15, pady=20)
        
        self.analysis_text = tk.Text(
            self.analysis_frame,
            bg=BG_PANEL,
            fg=TEXT_COLOR,
            font=(FONT_FAMILY, 9),
            wrap="word",
            state="disabled",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=10,
            pady=10,
            height=6
        )
        
        self.scrollbar = tk.Scrollbar(self.analysis_frame, command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=self.scrollbar.set)
        
        # Tags for text coloring and formatting
        self.analysis_text.tag_configure("red", foreground="#ff5252", font=(FONT_FAMILY, 9, "bold"))
        self.analysis_text.tag_configure("blue", foreground="#52a0ff", font=(FONT_FAMILY, 9, "bold"))
        self.analysis_text.tag_configure("yellow", foreground="#ffd200", font=(FONT_FAMILY, 9, "bold"))
        self.analysis_text.tag_configure("green", foreground="#4be64b", font=(FONT_FAMILY, 9, "bold"))
        self.analysis_text.tag_configure("bold", font=(FONT_FAMILY, 9, "bold"))
        self.analysis_text.tag_configure("header", font=(FONT_FAMILY, 10, "bold"), foreground=ACCENT_GREEN)
        
        # Bottom Status / Classification Result Bar
        self.status_bar = tk.Frame(self.root, bg=BG_PANEL, height=40)
        self.status_bar.pack(fill="x", side="bottom", padx=20, pady=15)
        
        self.status_text = tk.Label(
            self.status_bar,
            text="Loading AI models on CPU... Please wait.",
            font=(FONT_FAMILY, 10, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_BLUE,
            anchor="w",
            padx=15
        )
        self.status_text.pack(fill="x", pady=8)

    def _ensure_seg_model(self):
        if self.seg_model is not None:
            return True

        self.status_text.config(text="Loading segmentation model... Please wait.", fg=ACCENT_BLUE)
        self.root.update()

        try:
            if IS_FROZEN:
                output_root = RESOURCE_ROOT / "experiments"
            else:
                output_root = Path(self.config["experiment"]["output_root"])

            seg_ckpt = output_root / self.config["experiment"]["name"] / "checkpoints" / "best.pth"
            if not seg_ckpt.exists():
                seg_ckpt = output_root / "exp_03_optimized" / "checkpoints" / "best.pth"

            if not seg_ckpt.exists():
                raise FileNotFoundError(f"Segmentation weights not found at {seg_ckpt}")

            if "model" in self.config:
                self.config["model"]["encoder_weights"] = None

            self.seg_model = build_model(self.config)
            ckpt = torch.load(seg_ckpt, map_location=self.device, weights_only=False)
            self.seg_model.load_state_dict(ckpt["model_state_dict"])
            del ckpt  # Free checkpoint dict from memory immediately
            gc.collect()

            self.seg_model.eval()
            for p in self.seg_model.parameters():
                p.requires_grad = False

            return True

        except Exception as e:
            log_exception(*sys.exc_info())
            self.seg_model = None
            self.status_text.config(text=f"ERROR: {str(e)}", fg=ACCENT_RED)
            messagebox.showerror("Model Loading Error", f"Failed to load segmentation model:\n\n{str(e)}")
            return False

    def _ensure_cls_model(self):
        
        if self.classifier is not None:
            return True

        self.status_text.config(text="Loading classification model... Please wait.", fg=ACCENT_BLUE)
        self.root.update()

        try:
            if IS_FROZEN:
                output_root = RESOURCE_ROOT / "experiments"
            else:
                output_root = Path(self.config["experiment"]["output_root"])

            cls_ckpt = output_root / f"{self.config['experiment']['name']}_cls_hybrid" / "checkpoints" / "best.pth"
            if not cls_ckpt.exists():
                cls_ckpt = output_root / "exp_04_cls_hybrid" / "checkpoints" / "best.pth"

            if not cls_ckpt.exists():
                raise FileNotFoundError(f"Classification weights not found at {cls_ckpt}")

            self.config["classification_model"]["lesion_feature_dim"] = len(self.class_names)
            if "classification_model" in self.config:
                self.config["classification_model"]["pretrained"] = False

            self.classifier = build_classifier(self.config)
            ckpt_cls = torch.load(cls_ckpt, map_location=self.device, weights_only=False)
            self.classifier.load_state_dict(ckpt_cls["model_state_dict"])
            del ckpt_cls  # Free checkpoint dict from memory immediately
            gc.collect()

            self.classifier.eval()
            for p in self.classifier.parameters():
                p.requires_grad = False

            return True

        except Exception as e:
            log_exception(*sys.exc_info())
            self.classifier = None
            self.status_text.config(text=f"ERROR: {str(e)}", fg=ACCENT_RED)
            messagebox.showerror("Model Loading Error", f"Failed to load classification model:\n\n{str(e)}")
            return False

    def _validate_retinal_scan(self, path: str) -> Tuple[bool, str]:
        """
        Runs mathematical/color profile heuristics on input images.
        Rejects non-retinal photographs.
        """
        img = cv2.imread(path)
        if img is None:
            return False, "Unable to read the image file."
            
        h, w = img.shape[:2]
        if h < 120 or w < 120:
            return False, "Resolution is too low to extract features."
            
        # Convert to grayscale and check corners for the dark circular border
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cw, ch = int(w * 0.05), int(h * 0.05)
        
        corners = [
            gray[:ch, :cw],
            gray[:ch, -cw:],
            gray[-ch:, :cw],
            gray[-ch:, -cw:]
        ]
        avg_corner = np.mean([c.mean() for c in corners])
        
        # Check center region brightness (retina is illuminated, corners are dark background)
        center_gray = gray[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
        center_mean = center_gray.mean()
        
        # Color channel analysis (RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r_mean = img_rgb[:, :, 0].mean()
        g_mean = img_rgb[:, :, 1].mean()
        b_mean = img_rgb[:, :, 2].mean()
        
        # 1. Reject grayscale/monochrome images (like chest X-rays or brain MRIs)
        channel_means_std = np.std([r_mean, g_mean, b_mean])
        if channel_means_std < 2.0:
            return False, "Grayscale/monochrome image detected. Retinal scans must be color images."
            
        # 2. Verify black corners (all typical retinal fundus scans have a circular field of view on black background)
        # We allow up to 85 to accommodate camera text, light artifacts, or noise in corners.
        if avg_corner > 85:
            return False, f"Corners must be dark (avg={avg_corner:.1f}). Retinal scans must have a dark circular background border."
            
        # 3. Verify that the center region is significantly brighter than the corners
        if center_mean < avg_corner + 15:
            return False, f"Center region must be brighter than corners (center={center_mean:.1f}, corners={avg_corner:.1f})."
            
        # 4. Color profile check: Blue is never the dominant color in a retinal scan.
        # Either Red or Green must be strictly greater than Blue.
        if b_mean >= r_mean and b_mean >= g_mean:
            return False, f"Invalid color profile. Blue cannot be the dominant channel (R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f})."
            
        # 5. Ensure the image is not completely dark
        if max(r_mean, g_mean) < 15:
            return False, "Image is too dark to be a valid scan."
            
        return True, "Valid Retinal Scan"

    def _upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff")]
        )
        if not file_path:
            return
            
        # Validate scan
        valid, msg = self._validate_retinal_scan(file_path)
        if not valid:
            messagebox.showerror(
                "Invalid Image Uploaded",
                f"The uploaded file is not a valid retinal fundus scan:\n\n→ {msg}"
            )
            return
            
        self.current_image_path = file_path
        self.file_path_label.config(text=Path(file_path).name, fg=TEXT_COLOR)
        
        # Display image
        self._display_image(file_path, self.input_label)
        
        # Reset output views & activate buttons
        self.output_label.config(image="", text="[ Ready to Process ]")
        self.btn_seg.config(state="normal")
        self.btn_cls.config(state="disabled")  # Classification is kept disabled per user request
        self.btn_dl.config(state="disabled")
        self.seg_overlay_highres = None
        self.status_text.config(text="Scan uploaded successfully.", fg=TEXT_COLOR)
        
        # Reset analysis legend view to placeholder
        self.analysis_text.pack_forget()
        self.scrollbar.pack_forget()
        self.placeholder_label.pack(fill="both", expand=True, padx=15, pady=20)

    def _display_image(self, path: str, widget: tk.Label):
        img = Image.open(path)
        # Preserve ratio using Pillow thumbnail
        img.thumbnail((350, 350))
        photo = ImageTk.PhotoImage(img)
        widget.config(image=photo, text="")
        widget.image = photo

    def _run_segmentation(self):
        if not self.current_image_path:
            return

        # Lazy-load segmentation model on first use
        if not self._ensure_seg_model():
            return

        self.status_text.config(text="Running lesion segmentation... Please wait.", fg=ACCENT_BLUE)
        self.root.update()
        
        try:
            # Preprocessing
            enhancer = RetinalEnhancer(self.config["preprocessing"])
            transform = get_val_transforms(self.config)
            
            image_raw = cv2.imread(self.current_image_path)
            orig_h, orig_w = image_raw.shape[:2]
            
            enhanced = enhancer.process(image_raw)
            image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
            transformed = transform(image=image_rgb)
            input_tensor = transformed["image"].unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.seg_model(input_tensor)
                probs_tensor = torch.sigmoid(logits)
                probs = probs_tensor[0].numpy()  # (C, H, W)

            # Class-specific threshold maps
            preds = np.zeros_like(probs)
            for c, cls in enumerate(self.class_names):
                thresh = self.thresholds.get(cls, 0.5)
                preds[c] = (probs[c] >= thresh).astype(np.float32)

            # Run connected component counting to find lesion instances
            binary_masks = torch.zeros_like(probs_tensor)
            for c, cls in enumerate(self.class_names):
                thresh = self.thresholds.get(cls, 0.5)
                binary_masks[:, c] = (probs_tensor[:, c] >= thresh).float()
                
            counts_dicts = count_lesions_batch(
                binary_masks,
                class_names=self.class_names,
                min_areas=self.min_areas,
                connectivity=8
            )
            counts = counts_dicts[0]
            
            del input_tensor, logits, probs_tensor  # Free inference tensors

            # Create color overlay (on enhanced RGB resolution)
            # Resize predicted masks back to original dimensions for high-res save
            overlay_h, overlay_w = orig_h, orig_w
            overlay = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            # Colors corresponding to: Red (MA), Blue (HE), Yellow (EX), Green (SE)
            colors = {
                "MA": (255, 50, 50),     # Red
                "HE": (50, 150, 255),    # Blue
                "EX": (255, 220, 0),     # Yellow
                "SE": (75, 230, 75),     # Green
            }
            
            for c, cls in enumerate(self.class_names):
                color = colors[cls]
                mask_resized = cv2.resize(preds[c], (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST) > 0.5
                
                for ch in range(3):
                    overlay[:, :, ch] = np.where(
                        mask_resized,
                        overlay[:, :, ch] * 0.45 + color[ch] * 0.55,
                        overlay[:, :, ch]
                    )
            
            self.seg_overlay_highres = np.clip(overlay, 0, 255).astype(np.uint8)
            
            # Save temporary file to display in Tkinter
            temp_path = OUTPUT_ROOT / "outputs" / "temp_preview.png"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            Image.fromarray(self.seg_overlay_highres).save(temp_path)
            
            self._display_image(str(temp_path), self.output_label)
            self.btn_dl.config(state="normal")
            
            # Update Legend & Analysis Panel
            self.placeholder_label.pack_forget()
            self.scrollbar.pack(side="right", fill="y")
            self.analysis_text.pack(side="left", fill="both", expand=True)
            self._update_analysis_text(counts)
            
            self.status_text.config(text="Lesion segmentation complete.", fg=ACCENT_GREEN)
            
        except Exception as e:
            self.status_text.config(text=f"ERROR: {str(e)}", fg=ACCENT_RED)
            messagebox.showerror("Segmentation Error", f"Failed to run segmentation:\n\n{str(e)}")
        finally:
            gc.collect()

    def _update_analysis_text(self, counts: Dict[str, int]):
        self.analysis_text.config(state="normal")
        self.analysis_text.delete("1.0", tk.END)
        
        # Predetermined best model metrics on testing dataset (exp_04)
        metrics = {
            "MA": {
                "name": "Microaneurysms",
                "color": "red",
                "emoji": "🔴",
                "desc": "The earliest clinically detectable sign of diabetic retinopathy. They represent tiny, round, focal dilations of retinal capillaries that result from the loss of pericytes (supporting cells).",
                "dice": "1.09%",
                "iou": "0.55%",
                "recall": "75.84%"
            },
            "HE": {
                "name": "Haemorrhages",
                "color": "blue",
                "emoji": "🔵",
                "desc": "Bleeding within the retinal layers. They occur when microaneurysms or fragile damaged capillaries rupture. Depending on the depth in the retina, they appear as 'dot-and-blot' or 'flame-shaped' blood pools.",
                "dice": "46.00%",
                "iou": "29.87%",
                "recall": "59.40%"
            },
            "EX": {
                "name": "Hard Exudates",
                "color": "yellow",
                "emoji": "🟡",
                "desc": "Yellowish-white lipid and lipoprotein deposits with distinct, sharp borders. They are left behind in the retina when fluid leaks from hyperpermeable capillaries and gets resorbed, leaving behind dense lipid structures.",
                "dice": "69.99%",
                "iou": "53.83%",
                "recall": "72.27%"
            },
            "SE": {
                "name": "Soft Exudates (Cotton Wool Spots)",
                "color": "green",
                "emoji": "🟢",
                "desc": "White, fluffy lesions with indistinct borders. They represent areas of localized micro-infarctions in the nerve fiber layer, caused by the occlusion of precapillary arterioles (severe ischemia).",
                "dice": "69.49%",
                "iou": "53.24%",
                "recall": "89.82%"
            }
        }
        
        self.analysis_text.insert(tk.END, "DETECTION QUANTIFICATION & BENCHMARK ACCURACY\n", "header")
        self.analysis_text.insert(tk.END, "=" * 62 + "\n\n")
        
        for cls in ["MA", "HE", "EX", "SE"]:
            meta = metrics[cls]
            count = counts.get(cls, 0)
            
            # Line 1: Emoji, Name (Abbr) and count detected
            self.analysis_text.insert(tk.END, f"{meta['emoji']} ")
            self.analysis_text.insert(tk.END, f"{meta['name']} ({cls})", meta['color'])
            self.analysis_text.insert(tk.END, f" — {count} instances detected\n", "bold")
            
            # Line 2: Accuracy metrics on test set
            self.analysis_text.insert(tk.END, "  • Expected Test Set Accuracy: ")
            self.analysis_text.insert(tk.END, f"Dice Coeff: {meta['dice']} | IoU: {meta['iou']} | Sensitivity/Recall: {meta['recall']}\n", "bold")
            
            # Line 3: Clinical description
            self.analysis_text.insert(tk.END, f"  • Clinical Meaning: {meta['desc']}\n\n")
            
        self.analysis_text.config(state="disabled")

    def _run_classification(self):
        if not self.current_image_path:
            return

        # Lazy-load both models (classification needs segmentation for lesion counts)
        if not self._ensure_seg_model():
            return
        if not self._ensure_cls_model():
            return

        self.status_text.config(text="Running severity classification...", fg=ACCENT_BLUE)
        self.root.update()
        
        try:
            # 1. Run U-Net to count lesions
            enhancer = RetinalEnhancer(self.config["preprocessing"])
            transform = get_val_transforms(self.config)
            
            image_raw = cv2.imread(self.current_image_path)
            enhanced = enhancer.process(image_raw)
            image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image_rgb)
            input_tensor = transformed["image"].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.seg_model(input_tensor)
                probs = torch.sigmoid(logits)
            
            # Binary masks for counting
            binary_masks = torch.zeros_like(probs)
            for c, cls in enumerate(self.class_names):
                thresh = self.thresholds.get(cls, 0.5)
                binary_masks[:, c] = (probs[:, c] >= thresh).float()
                
            # connected component counting
            counts_dicts = count_lesions_batch(
                binary_masks,
                class_names=self.class_names,
                min_areas=self.min_areas,
                connectivity=8
            )
            
            counts_tensor = torch.zeros((1, len(self.class_names)), dtype=torch.float32)
            for c, cls in enumerate(self.class_names):
                counts_tensor[0, c] = counts_dicts[0].get(cls, 0)
                
            # 2. Run Classifier
            with torch.no_grad():
                outputs = self.classifier(input_tensor, counts_tensor)
                pred_class = torch.argmax(outputs, dim=1).item()
                
            # Diagnostic mapping
            diagnoses = {
                0: "Grade 0: No Diabetic Retinopathy (Normal)",
                1: "Grade 1: Mild Non-Proliferative Diabetic Retinopathy",
                2: "Grade 2: Moderate Non-Proliferative Diabetic Retinopathy",
                3: "Grade 3: Severe Non-Proliferative Diabetic Retinopathy",
                4: "Grade 4: Proliferative Diabetic Retinopathy (Active Leakages)"
            }
            
            result_str = diagnoses.get(pred_class, f"Grade {pred_class}: Unidentified Severity Rating")
            self.status_text.config(text=f"CLASSIFICATION RESULT: {result_str}", fg=ACCENT_GREEN)
            
            # Print lesion count analysis in pop-up
            counts_str = "\n".join([f"  • {cls}: {int(counts_tensor[0, c])} instances detected" 
                                    for c, cls in enumerate(self.class_names)])
            messagebox.showinfo(
                "Diagnostic Report",
                f"RetinaAI Severity Grading:\n\n{result_str}\n\nQuantified Lesion Analysis:\n{counts_str}"
            )
            
        except Exception as e:
            self.status_text.config(text=f"ERROR: {str(e)}", fg=ACCENT_RED)
            messagebox.showerror("Classification Error", f"Failed to run classification:\n\n{str(e)}")
        finally:
            gc.collect()

    def _download_overlay(self):
        if self.seg_overlay_highres is None:
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        if not save_path:
            return
            
        try:
            # Convert RGB back to BGR for saving via OpenCV
            save_bgr = cv2.cvtColor(self.seg_overlay_highres, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_bgr)
            messagebox.showinfo("Success", f"Segmentation overlay successfully saved to:\n\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image file:\n\n{str(e)}")


def main():
    root = tk.Tk()
    app = RetinaAIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
