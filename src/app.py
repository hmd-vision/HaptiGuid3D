import threading
import time
from pathlib import Path
from typing import List, Optional

import customtkinter as ctk
import cv2
from PIL import Image, ImageDraw, ImageFilter
from tkinter import filedialog

from .audio_manager import AudioManager
from .config import (
    APP_TITLE,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEPTH_EVERY_N_FRAMES,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    INFER_EVERY_N_FRAMES,
)
from .models import DetectedObstacle
from .pipeline import HaptiGuidePipeline
from .hri_gesture_control import HRIGestureControl

class HaptiGuideApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1460x900")
        self.minsize(1200, 780)

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.pipeline: Optional[HaptiGuidePipeline] = None
        self.hri_control = HRIGestureControl(stable_frames=3, cooldown_frames=10)
        self.hri_active = False
        self.focus_mode = False
        self.focus_zone = None
        self.audio_muted = False
        self.emergency_paused = False
        self.audio = AudioManager()

        self.current_image_ctk = None
        self.camera_running = False
        self.camera_thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self.last_depth_map = None
        self.last_detections = None
        self.frame_counter = 0

        self.status_var = ctk.StringVar(value="Loading models...")
        self.file_var = ctk.StringVar(value="No image selected")

        self.priority_var = ctk.StringVar(value="No active alert")
        self.zone_var = ctk.StringVar(value="Zone: --")
        self.direction_var = ctk.StringVar(value="Direction: --")
        self.score_var = ctk.StringVar(value="Score: --")
        self.audio_var = ctk.StringVar(value="Audio: --")

        self.hri_state_var = ctk.StringVar(value="LOW POWER")
        self.region_state_var = ctk.StringVar(value="Region: ALL")
        self.gesture_state_var = ctk.StringVar(value="Gesture: --")
        self.mode_state_var = ctk.StringVar(value="Mode: Waiting")
        self.mute_state_var = ctk.StringVar(value="Audio: ON")
        self.object_state_var = ctk.StringVar(value="No object selected")

        self._build_ui()
        self._load_models_in_background()

    def _build_ui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.configure(fg_color="#07111F")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Compact deluxe header
        header = ctk.CTkFrame(self, corner_radius=22, fg_color="#0B1628", border_width=1, border_color="#1E3A5F")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        title_block = ctk.CTkFrame(header, fg_color="transparent")
        title_block.grid(row=0, column=0, sticky="w", padx=18, pady=12)

        title = ctk.CTkLabel(
            title_block,
            text="HaptiGuide3D Pro",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#EAF6FF",
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            title_block,
            text="Gesture-controlled assistive perception",
            font=ctk.CTkFont(size=13),
            text_color="#7DD3FC",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(1, 0))

        self.active_pill = ctk.CTkLabel(
            header,
            textvariable=self.hri_state_var,
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#FFFFFF",
            fg_color="#991B1B",
            corner_radius=18,
            width=150,
            height=38,
        )
        self.active_pill.grid(row=0, column=1, sticky="e", padx=(8, 18), pady=12)

        # Compact control bar
        controls = ctk.CTkFrame(self, corner_radius=18, fg_color="#0E1B2F", border_width=1, border_color="#183456")
        controls.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        controls.grid_columnconfigure(4, weight=1)

        self.choose_btn = ctk.CTkButton(
            controls,
            text="Choose Photo",
            command=self.choose_photo,
            state="disabled",
            width=135,
            height=36,
            corner_radius=14,
            fg_color="#2563EB",
            hover_color="#1D4ED8",
        )
        self.choose_btn.grid(row=0, column=0, padx=(14, 8), pady=10)

        self.camera_btn = ctk.CTkButton(
            controls,
            text="Start Camera",
            command=self.toggle_camera,
            state="disabled",
            width=135,
            height=36,
            corner_radius=14,
            fg_color="#0F766E",
            hover_color="#115E59",
        )
        self.camera_btn.grid(row=0, column=1, padx=8, pady=10)

        status_chip = ctk.CTkLabel(
            controls,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#BAE6FD",
            fg_color="#10243D",
            corner_radius=12,
            width=150,
            height=30,
        )
        status_chip.grid(row=0, column=2, padx=8, pady=10)

        file_label = ctk.CTkLabel(
            controls,
            textvariable=self.file_var,
            font=ctk.CTkFont(size=12),
            text_color="#94A3B8",
        )
        file_label.grid(row=0, column=4, sticky="w", padx=12)

        # Main compact layout
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))
        main.grid_columnconfigure(0, weight=4)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        left_panel = ctk.CTkFrame(main, corner_radius=24, fg_color="#0A1324", border_width=1, border_color="#1E3A5F")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_rowconfigure(1, weight=1)

        top_live = ctk.CTkFrame(left_panel, fg_color="transparent")
        top_live.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        top_live.grid_columnconfigure(0, weight=1)

        live_title = ctk.CTkLabel(
            top_live,
            text="Smart Goggle View",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#E0F2FE",
        )
        live_title.grid(row=0, column=0, sticky="w")

        self.region_badge = ctk.CTkLabel(
            top_live,
            textvariable=self.region_state_var,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#E0F2FE",
            fg_color="#123B5D",
            corner_radius=12,
            height=28,
            width=120,
        )
        self.region_badge.grid(row=0, column=1, sticky="e")

        self.image_label = ctk.CTkLabel(left_panel, text="")
        self.image_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=(4, 10))

        command_strip = ctk.CTkFrame(left_panel, corner_radius=18, fg_color="#08111F", border_width=1, border_color="#16324F")
        command_strip.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        for i in range(5):
            command_strip.grid_columnconfigure(i, weight=1)

        command_items = [
            ("🖐", "All regions"),
            ("☝", "Region focus"),
            ("✌", "Mute"),
            ("✊", "Emergency"),
            ("👍", "Stop"),
        ]
        for idx, (icon, text) in enumerate(command_items):
            item = ctk.CTkLabel(
                command_strip,
                text=f"{icon}  {text}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#A7F3D0" if idx == 0 else "#CBD5E1",
            )
            item.grid(row=0, column=idx, padx=6, pady=8)

        right_panel = ctk.CTkFrame(main, corner_radius=24, fg_color="#0A1324", border_width=1, border_color="#1E3A5F")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(3, weight=1)

        right_title = ctk.CTkLabel(
            right_panel,
            text="HRI Control Deck",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#E0F2FE",
        )
        right_title.grid(row=0, column=0, sticky="w", padx=16, pady=(12, 8))

        state_card = ctk.CTkFrame(right_panel, corner_radius=18, fg_color="#0E1B2F", border_width=1, border_color="#1D4E73")
        state_card.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        state_card.grid_columnconfigure(0, weight=1)

        self.mode_label = ctk.CTkLabel(
            state_card,
            textvariable=self.mode_state_var,
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#F8FAFC",
        )
        self.mode_label.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 2))

        self.gesture_label = ctk.CTkLabel(
            state_card,
            textvariable=self.gesture_state_var,
            font=ctk.CTkFont(size=12),
            text_color="#93C5FD",
        )
        self.gesture_label.grid(row=1, column=0, sticky="w", padx=14, pady=2)

        self.mute_label = ctk.CTkLabel(
            state_card,
            textvariable=self.mute_state_var,
            font=ctk.CTkFont(size=12),
            text_color="#C4B5FD",
        )
        self.mute_label.grid(row=2, column=0, sticky="w", padx=14, pady=(2, 12))

        alert_card = ctk.CTkFrame(right_panel, corner_radius=18, fg_color="#0E1B2F", border_width=1, border_color="#1D4E73")
        alert_card.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        alert_card.grid_columnconfigure(0, weight=1)

        priority_title = ctk.CTkLabel(alert_card, text="Priority Alert", font=ctk.CTkFont(size=14, weight="bold"), text_color="#7DD3FC")
        priority_title.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 2))

        self.priority_label = ctk.CTkLabel(
            alert_card,
            textvariable=self.priority_var,
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#F8FAFC",
        )
        self.priority_label.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 4))

        meta_frame = ctk.CTkFrame(alert_card, fg_color="transparent")
        meta_frame.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))
        meta_frame.grid_columnconfigure((0, 1), weight=1)

        self.zone_chip = ctk.CTkLabel(meta_frame, textvariable=self.zone_var, font=ctk.CTkFont(size=11), text_color="#CFFAFE", fg_color="#123B5D", corner_radius=10, height=26)
        self.zone_chip.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=3)

        self.direction_chip = ctk.CTkLabel(meta_frame, textvariable=self.direction_var, font=ctk.CTkFont(size=11), text_color="#CFFAFE", fg_color="#123B5D", corner_radius=10, height=26)
        self.direction_chip.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=3)

        self.score_chip = ctk.CTkLabel(meta_frame, textvariable=self.score_var, font=ctk.CTkFont(size=11), text_color="#CFFAFE", fg_color="#123B5D", corner_radius=10, height=26)
        self.score_chip.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=3)

        self.audio_chip = ctk.CTkLabel(meta_frame, textvariable=self.audio_var, font=ctk.CTkFont(size=11), text_color="#CFFAFE", fg_color="#123B5D", corner_radius=10, height=26)
        self.audio_chip.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=3)

        self.results_box = ctk.CTkTextbox(
            right_panel,
            font=("Consolas", 12),
            corner_radius=18,
            fg_color="#08111F",
            text_color="#D7EAFE",
            border_width=1,
            border_color="#16324F",
        )
        self.results_box.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.results_box.insert("1.0", "Start camera and use gestures to control perception.\n")
        self.results_box.configure(state="disabled")
        self._refresh_hri_widgets()

    def _refresh_hri_widgets(self, gesture_text: str = "--"):
        if self.emergency_paused:
            state = "EMERGENCY"
            color = "#B91C1C"
            mode = "Mode: Emergency pause"
        elif self.hri_active:
            state = "ACTIVE"
            color = "#15803D"
            region = self.focus_zone.upper() if self.focus_mode and self.focus_zone else "ALL"
            mode = f"Mode: Scanning {region}"
        else:
            state = "LOW POWER"
            color = "#991B1B"
            mode = "Mode: Low power"

        self.hri_state_var.set(state)
        self.active_pill.configure(fg_color=color)

        region_text = self.focus_zone.upper() if self.focus_mode and self.focus_zone else "ALL"
        self.region_state_var.set(f"Region: {region_text}")
        self.gesture_state_var.set(f"Gesture: {gesture_text}")
        self.mode_state_var.set(mode)
        self.mute_state_var.set("Audio: MUTED" if self.audio_muted else "Audio: ON")

    @staticmethod
    def _rounded_poly_mask(size):
        w, h = size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        pad_x = int(w * 0.035)
        pad_y = int(h * 0.12)
        gap = int(w * 0.045)
        lens_w = (w - 2 * pad_x - gap) // 2
        lens_h = int(h * 0.72)
        y1 = pad_y
        y2 = y1 + lens_h

        left = (pad_x, y1, pad_x + lens_w, y2)
        right = (pad_x + lens_w + gap, y1, pad_x + 2 * lens_w + gap, y2)
        r = int(lens_h * 0.34)

        draw.rounded_rectangle(left, radius=r, fill=255)
        draw.rounded_rectangle(right, radius=r, fill=255)

        # bridge around nose
        cx = w // 2
        bridge = [
            (cx - gap, int(h * 0.40)),
            (cx + gap, int(h * 0.40)),
            (cx + int(gap * 0.75), int(h * 0.58)),
            (cx - int(gap * 0.75), int(h * 0.58)),
        ]
        draw.polygon(bridge, fill=255)

        return mask.filter(ImageFilter.GaussianBlur(0.6))

    def _make_goggle_view(self, pil_img: Image.Image, size=(930, 500)) -> Image.Image:
        w, h = size
        bg = Image.new("RGB", size, "#050B14")
        bg_draw = ImageDraw.Draw(bg)

        # subtle tech texture
        for x in range(0, w, 34):
            bg_draw.line((x, 0, x, h), fill="#071C2E", width=1)
        for y in range(0, h, 34):
            bg_draw.line((0, y, w, y), fill="#071C2E", width=1)

        img = pil_img.convert("RGB")
        img_ratio = img.width / img.height
        target_ratio = w / h
        if img_ratio > target_ratio:
            new_h = h
            new_w = int(h * img_ratio)
        else:
            new_w = w
            new_h = int(w / img_ratio)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        crop_x = max(0, (new_w - w) // 2)
        crop_y = max(0, (new_h - h) // 2)
        img = img.crop((crop_x, crop_y, crop_x + w, crop_y + h))

        # mild glass tint
        tint = Image.new("RGB", size, "#0EA5E9")
        img = Image.blend(img, tint, 0.08)

        mask = self._rounded_poly_mask(size)
        bg.paste(img, (0, 0), mask)

        outline = ImageDraw.Draw(bg)
        # neon outline by drawing mask edges
        edge = mask.filter(ImageFilter.FIND_EDGES)
        cyan = Image.new("RGB", size, "#38BDF8")
        bg = Image.composite(cyan, bg, edge.point(lambda p: 210 if p > 20 else 0))

        outline = ImageDraw.Draw(bg)
        # outer decorative frame
        outline.rounded_rectangle((8, 8, w - 8, h - 8), radius=44, outline="#0EA5E9", width=2)
        outline.rounded_rectangle((15, 15, w - 15, h - 15), radius=38, outline="#155E75", width=1)

        # center nose arc
        outline.arc((w//2 - 70, h//2 - 10, w//2 + 70, h//2 + 110), start=190, end=350, fill="#67E8F9", width=2)

        # HUD corner ticks
        tick = 34
        for sx in (24, w - 24):
            for sy in (24, h - 24):
                if sx < w/2:
                    outline.line((sx, sy, sx + tick, sy), fill="#7DD3FC", width=2)
                else:
                    outline.line((sx, sy, sx - tick, sy), fill="#7DD3FC", width=2)
                if sy < h/2:
                    outline.line((sx, sy, sx, sy + tick), fill="#7DD3FC", width=2)
                else:
                    outline.line((sx, sy, sx, sy - tick), fill="#7DD3FC", width=2)

        return bg

    def _load_models_in_background(self):
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            self.pipeline = HaptiGuidePipeline()
            self.after(0, self._on_models_ready)
        except Exception as e:
            self.after(0, lambda: self._on_models_error(str(e)))

    def _on_models_ready(self):
        self.status_var.set("Models ready.")
        self.choose_btn.configure(state="normal")
        self.camera_btn.configure(state="normal")

    def _on_models_error(self, error_message: str):
        self.status_var.set("Model loading failed.")
        self._set_results(f"Model loading failed:\n{error_message}")

    def _set_results(self, text: str):
        self.results_box.configure(state="normal")
        self.results_box.delete("1.0", "end")
        self.results_box.insert("1.0", text)
        self.results_box.configure(state="disabled")

    def _update_priority_card(self, top: Optional[DetectedObstacle], audio_text: str = "--"):
        if top is None:
            self.priority_var.set("No active alert")
            self.zone_var.set("Zone: --")
            self.direction_var.set("Direction: --")
            self.score_var.set("Score: --")
            self.audio_var.set("Audio: --")
            return

        self.priority_var.set(f"{top.label.title()}")
        self.zone_var.set(f"Zone: {top.distance_zone}")
        self.direction_var.set(f"Direction: {top.direction}")
        self.score_var.set(f"Score: {top.fused_score:.2f}")
        self.audio_var.set(f"Audio: {audio_text}")

    def _play_top_obstacle_audio(self, top: DetectedObstacle) -> str:
        if top.distance_zone == "near":
            self.audio.play_level3_voice(top.label, top.direction)
            return "context level-3 voice"
        else:
            self.audio.play_level_sound(top.sound_level)
            return str(top.sound_level)

    def choose_photo(self):
        if self.pipeline is None:
            return

        file_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        self.file_var.set(file_path)
        self.status_var.set("Analyzing selected photo...")
        self.choose_btn.configure(state="disabled")
        self.camera_btn.configure(state="disabled")
        self._set_results("Running object detection and depth estimation...\n")

        threading.Thread(target=self._analyze_photo_worker, args=(file_path,), daemon=True).start()

    def _analyze_photo_worker(self, file_path: str):
        try:
            obstacles, annotated_path, proximity_path = self.pipeline.analyze_image(file_path)
            self.after(0, lambda: self._show_photo_results(file_path, obstacles, annotated_path, proximity_path))
        except Exception as e:
            self.after(0, lambda: self._show_error(str(e)))

    def _show_photo_results(self, file_path: str, obstacles, annotated_path: Path, proximity_path: Path):
        self.status_var.set("Photo analysis complete.")
        self.choose_btn.configure(state="normal")
        self.camera_btn.configure(state="normal")

        self._display_image(str(annotated_path))

        lines = [
            f"Input image: {file_path}",
            f"Annotated image saved to: {annotated_path}",
            f"Proximity map saved to: {proximity_path}",
            "",
        ]

        if not obstacles:
            lines.append("No chairs or tables were detected.")
            self.audio.stop_audio()
            self._update_priority_card(None)
        else:
            lines.append("Detected obstacles (closest first):")
            lines.append("")

            for i, obs in enumerate(obstacles, start=1):
                lines.append(f"{i}. Object: {obs.label}")
                lines.append(f"   Confidence: {obs.confidence:.2f}")
                lines.append(f"   Direction: {obs.direction}")
                lines.append(f"   Distance zone: {obs.distance_zone}")
                lines.append(f"   Depth score: {obs.depth_score:.2f}")
                lines.append(f"   Area score: {obs.area_score:.2f}")
                lines.append(f"   Fused score: {obs.fused_score:.2f}")
                lines.append(f"   Reaction: {obs.reaction_text}")
                lines.append("")

            top = obstacles[0]
            audio_text = self._play_top_obstacle_audio(top)
            self._update_priority_card(top, audio_text)

            lines.append("Top priority guidance:")
            lines.append(f"Nearest object: {top.label} | {top.direction} | {top.distance_zone}")
            lines.append(f"Audio: {audio_text}")

        self._set_results("\n".join(lines))

    def toggle_camera(self):
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.pipeline is None or self.camera_running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            self._show_error("Cannot access webcam.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.camera_running = True
        self.frame_counter = 0
        self.last_depth_map = None
        self.last_detections = None
        self.hri_control.reset()
        self.hri_active = False

        self.camera_btn.configure(text="Stop Camera", fg_color="#B91C1C", hover_color="#7F1D1D")
        self.status_var.set("Camera running...")
        self.file_var.set("Live webcam mode")

        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def stop_camera(self):
        self.camera_running = False
        self.audio.stop_audio()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.camera_btn.configure(text="Start Camera", fg_color="#0F766E", hover_color="#115E59")
        self._refresh_hri_widgets()
        if self.pipeline:
            self.status_var.set("Models ready.")
        else:
            self.status_var.set("Camera stopped.")

    def _camera_loop(self):
        assert self.cap is not None

        while self.camera_running and self.cap.isOpened():
            for _ in range(2):
                self.cap.grab()

            ret, frame = self.cap.read()
            if not ret:
                continue

            self.frame_counter += 1
            self.hri_active, triggered_gesture, gesture_debug = self.hri_control.update(frame)

            if triggered_gesture == "ONE_FINGER_FOCUS":
                self.focus_mode = True
                self.focus_zone = gesture_debug.focus_zone or "center"
            elif triggered_gesture == "VICTORY_MUTE":
                self.audio_muted = not self.audio_muted
                if self.audio_muted:
                    self.audio.stop_audio()
            elif triggered_gesture == "FIST_EMERGENCY_STOP":
                self.emergency_paused = True
                self.hri_active = False
                self.audio.stop_audio()
            elif triggered_gesture == "OPEN_PALM":
                self.emergency_paused = False
                self.focus_mode = False
                self.focus_zone = None
            elif triggered_gesture == "THUMBS_UP":
                self.audio.stop_audio()

            frame_small = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

            try:
                if self.hri_active and not self.emergency_paused:
                    if self.last_depth_map is None or self.frame_counter % DEPTH_EVERY_N_FRAMES == 0:
                        self.last_depth_map = self.pipeline.estimate_depth(frame_small)

                    if self.last_detections is None or self.frame_counter % INFER_EVERY_N_FRAMES == 0:
                        self.last_detections = self.pipeline.detect_objects(frame_small)

                    if self.last_depth_map is not None and self.last_detections is not None:
                        obstacles = self.pipeline.fuse(frame_small, self.last_depth_map, self.last_detections)

                        if self.focus_mode and self.focus_zone:
                            obstacles = [obs for obs in obstacles if obs.direction == self.focus_zone]

                        annotated = self.pipeline.annotate(frame_small, obstacles)
                    else:
                        obstacles = []
                        annotated = frame_small.copy()
                else:
                    self.last_depth_map = None
                    self.last_detections = None
                    obstacles = []
                    annotated = frame_small.copy()
                    self.audio.stop_audio()

                annotated = self.hri_control.draw_hand_overlay(annotated)

                h_vis, w_vis = annotated.shape[:2]
                cv2.line(annotated, (w_vis // 3, 0), (w_vis // 3, h_vis), (80, 80, 80), 1)
                cv2.line(annotated, (2 * w_vis // 3, 0), (2 * w_vis // 3, h_vis), (80, 80, 80), 1)

                if self.focus_mode and self.focus_zone:
                    zone_ranges = {
                        "left": (0, w_vis // 3),
                        "center": (w_vis // 3, 2 * w_vis // 3),
                        "right": (2 * w_vis // 3, w_vis),
                    }
                    zx1, zx2 = zone_ranges.get(self.focus_zone, (w_vis // 3, 2 * w_vis // 3))
                    cv2.rectangle(annotated, (zx1, 0), (zx2, h_vis - 1), (255, 255, 0), 2)

                if self.emergency_paused:
                    status_text = "EMERGENCY PAUSE - open palm to resume all regions"
                    status_color = (0, 0, 255)
                elif self.hri_active:
                    mode_bits = []
                    if self.focus_mode and self.focus_zone:
                        mode_bits.append(f"ONLY {self.focus_zone.upper()}")
                    if self.audio_muted:
                        mode_bits.append("MUTED")
                    suffix = " | " + " | ".join(mode_bits) if mode_bits else " | ALL REGIONS"
                    status_text = "HRI ACTIVE" + suffix
                    status_color = (0, 200, 0)
                else:
                    status_text = "HRI STOPPED - show OPEN PALM"
                    status_color = (0, 0, 255)

                seen_text = gesture_debug.gesture if gesture_debug.gesture else "none"

                active_region = self.focus_zone if self.focus_mode and self.focus_zone else "all"

                if self.hri_active and not self.emergency_paused and len(obstacles) > 0:
                    top = obstacles[0]
                    if self.audio_muted:
                        self.audio.stop_audio()
                        audio_text = "muted by victory gesture"
                    else:
                        audio_text = self._play_top_obstacle_audio(top)

                    result_text = (
                        f"Live HRI detection: ACTIVE\n\n"
                        f"Active region: {active_region}\n"
                        f"Command gesture: {triggered_gesture or 'none'}\n"
                        f"Gesture seen: {seen_text}\n"
                        f"Gesture note: {gesture_debug.note}\n"
                        f"Extended fingers: {gesture_debug.extended_count}\n"
                        f"Focus mode: {self.focus_mode}\n"
                        f"Focus zone: {self.focus_zone or '--'}\n"
                        f"Audio muted: {self.audio_muted}\n"
                        f"Emergency paused: {self.emergency_paused}\n\n"
                        f"Top object: {top.label}\n"
                        f"Direction: {top.direction}\n"
                        f"Distance zone: {top.distance_zone}\n"
                        f"Depth score: {top.depth_score:.2f}\n"
                        f"Area score: {top.area_score:.2f}\n"
                        f"Fused score: {top.fused_score:.2f}\n"
                        f"Audio: {audio_text}\n\n"
                        f"Controls:\n"
                        f"Open palm = start/resume + all regions\n"
                        f"Thumbs up = stop\n"
                        f"One finger = activate only selected region\n"
                        f"Victory = mute/unmute\n"
                        f"Fist = emergency pause\n"
                    )
                    self.after(0, lambda t=top, a=audio_text: self._update_priority_card(t, a))
                elif self.hri_active and not self.emergency_paused:
                    self.audio.stop_audio()
                    result_text = (
                        f"Live HRI detection: ACTIVE\n\n"
                        f"No chairs or tables detected in active region.\n"
                        f"Active region: {active_region}\n"
                        f"Gesture seen: {seen_text}\n"
                        f"Gesture note: {gesture_debug.note}\n"
                        f"Extended fingers: {gesture_debug.extended_count}\n"
                        f"Focus mode: {self.focus_mode}\n"
                        f"Focus zone: {self.focus_zone or '--'}\n"
                        f"Audio muted: {self.audio_muted}\n\n"
                        f"Open palm = all regions again.\n"
                    )
                    self.after(0, lambda: self._update_priority_card(None))
                else:
                    result_text = (
                        f"Live HRI detection: STOPPED\n\n"
                        f"Object recognition is paused.\n"
                        f"Gesture seen: {seen_text}\n"
                        f"Gesture note: {gesture_debug.note}\n"
                        f"Extended fingers: {gesture_debug.extended_count}\n"
                        f"Focus mode: {self.focus_mode}\n"
                        f"Focus zone: {self.focus_zone or '--'}\n"
                        f"Audio muted: {self.audio_muted}\n"
                        f"Emergency paused: {self.emergency_paused}\n\n"
                        f"Show OPEN PALM to start/resume scanning and reactivate all regions.\n"
                    )
                    self.after(0, lambda: self._update_priority_card(None))

                self.after(0, lambda g=seen_text: self._refresh_hri_widgets(g))
                self.after(0, lambda img=annotated: self._display_cv_image(img))
                self.after(0, lambda txt=result_text: self._set_results(txt))

            except Exception as e:
                error_message = str(e)
                self.after(0, lambda msg=error_message: self._show_error(msg))
                break

            time.sleep(0.001)

        self.after(0, self.stop_camera)

    def _display_image(self, image_path: str):
        pil_img = Image.open(image_path).convert("RGB")
        goggle_img = self._make_goggle_view(pil_img)
        self.current_image_ctk = ctk.CTkImage(light_image=goggle_img, dark_image=goggle_img, size=goggle_img.size)
        self.image_label.configure(image=self.current_image_ctk, text="")

    def _display_cv_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        goggle_img = self._make_goggle_view(pil_img)
        self.current_image_ctk = ctk.CTkImage(light_image=goggle_img, dark_image=goggle_img, size=goggle_img.size)
        self.image_label.configure(image=self.current_image_ctk, text="")

    def clear_all(self):
        pass

    def _show_error(self, error_message: str):
        self.status_var.set("An error occurred.")
        self.choose_btn.configure(state="normal")
        if self.pipeline is not None:
            self.camera_btn.configure(state="normal")
        self._set_results(f"Error:\n{error_message}")
