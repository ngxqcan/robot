import os
import glob
import time
import math
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import messagebox

from config import config
from mouse import Mouse, connect_to_logitech, connect_to_makcu, test_move, is_connected as mouse_is_connected
import main
from main import (
    start_aimbot, stop_aimbot, is_aimbot_running,
    reload_model, get_model_classes, get_model_size, get_latest_preview_frame
)
from gui_constants import (
    NEON, NEON_GREEN, NEON_CYAN, NEON_ORANGE,
    BG_DARK, BG_CARD, BG_CARD_HOVER, BORDER_COLOR,
    TEXT_MUTED, TEXT_LIGHT, neon_button, cyber_button, green_button
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class CapkfaPlusGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CapkfaPlus - AI Vision & Assist Engine (1PC Logitech Driver)")
        
        # Responsive sizing
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        init_w = min(1200, int(screen_width * 0.88))
        init_h = min(920, int(screen_height * 0.90))
        x = (screen_width - init_w) // 2
        y = (screen_height - init_h) // 2
        
        self.geometry(f"{init_w}x{init_h}+{x}+{y}")
        self.configure(fg_color=BG_DARK)
        self.minsize(960, 720)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Configure Grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Internal State & Variables
        self.active_tab = "dashboard"
        self._updating_conf = False
        self._updating_fov = False
        self._updating_imgsz = False
        self.preview_image_tk = None
        self.preview_paused = False

        # Status Variables
        self.error_text = ctk.StringVar(value="")
        self.aimbot_status = ctk.StringVar(value="Stopped")
        self.connection_status = ctk.StringVar(value="Disconnected")
        self.connection_color = ctk.StringVar(value="#ff1744")
        self.model_name = ctk.StringVar(value=os.path.basename(config.model_path))
        self.model_size = ctk.StringVar(value="")
        self.fps_var = ctk.StringVar(value="0.0 FPS")
        self.capture_fps_var = ctk.StringVar(value="0.0 FPS")
        self.latency_var = ctk.StringVar(value="0.00 ms")
        self.resolution_var = ctk.StringVar(value=f"{config.region_size}x{config.region_size}")

        # Controls Variables
        self.aim_humanize_var = ctk.BooleanVar(value=bool(config.aim_humanization))
        self.debug_checkbox_var = ctk.BooleanVar(value=bool(config.show_debug_window))
        self.input_check_var = ctk.BooleanVar(value=False)
        self.button_mask_var = ctk.BooleanVar(value=bool(getattr(config, "button_mask", False)))
        self.always_on_var = ctk.BooleanVar(value=bool(getattr(config, "always_on_aim", False)))
        self.head_priority_var = ctk.BooleanVar(value=bool(getattr(config, "head_priority", True)))
        
        # Anti-Shake & RCS Variables
        self.rcs_enabled_var = ctk.BooleanVar(value=bool(getattr(config, "rcs_enabled", False)))
        
        # Triggerbot Variables
        self.trigger_enabled_var = ctk.BooleanVar(value=bool(getattr(config, "trigger_enabled", False)))
        self.trigger_always_on_var = ctk.BooleanVar(value=bool(getattr(config, "trigger_always_on", False)))
        self.trigger_btn_var = ctk.IntVar(value=int(getattr(config, "trigger_button", 1)))
        self.btn_var = ctk.IntVar(value=int(getattr(config, "selected_mouse_button", 3)))
        self.mode_var = ctk.StringVar(value=config.mode)
        self.capture_mode_var = ctk.StringVar(value=config.capturer_mode.upper())
        self.preview_fov_var = ctk.BooleanVar(value=bool(getattr(config, "preview_fov", True)))
        self.preview_boxes_var = ctk.BooleanVar(value=bool(getattr(config, "preview_boxes", True)))
        self.preview_vectors_var = ctk.BooleanVar(value=bool(getattr(config, "preview_vectors", True)))

        # Build Interface
        self.build_top_nav()
        self.build_views()
        
        # Initial Refresh & Connection
        self.refresh_all()
        self.on_connect()
        self.poll_telemetry()
        self.poll_preview()

    # =========================================================================
    # TOP NAVIGATION BAR (Dashboard / Preview Tabs)
    # =========================================================================
    def build_top_nav(self):
        nav_bar = ctk.CTkFrame(self, fg_color="#101115", height=60, corner_radius=0)
        nav_bar.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        nav_bar.grid_columnconfigure(1, weight=1)
        nav_bar.grid_propagate(False)

        # Brand / Title
        brand_frame = ctk.CTkFrame(nav_bar, fg_color="transparent")
        brand_frame.grid(row=0, column=0, padx=20, pady=12, sticky="w")
        
        ctk.CTkLabel(
            brand_frame, text="⚡ CapkfaPlus",
            font=("Segoe UI", 18, "bold"), text_color=NEON_CYAN
        ).pack(side="left")
        ctk.CTkLabel(
            brand_frame, text=" AI Engine",
            font=("Segoe UI", 14), text_color=TEXT_MUTED
        ).pack(side="left", padx=(4, 0))

        # Center Navigation Tabs (Pills)
        tabs_frame = ctk.CTkFrame(nav_bar, fg_color="#181920", corner_radius=8)
        tabs_frame.grid(row=0, column=1, pady=10)

        self.btn_tab_dashboard = ctk.CTkButton(
            tabs_frame, text="📊 Dashboard", width=140, height=36,
            fg_color="#242633", hover_color="#2f3242", text_color="#ffffff",
            font=("Segoe UI", 13, "bold"), corner_radius=6,
            command=lambda: self.switch_tab("dashboard")
        )
        self.btn_tab_dashboard.pack(side="left", padx=4, pady=4)

        self.btn_tab_preview = ctk.CTkButton(
            tabs_frame, text="🎯 Preview HUD", width=140, height=36,
            fg_color="transparent", hover_color="#222430", text_color=TEXT_MUTED,
            font=("Segoe UI", 13, "bold"), corner_radius=6,
            command=lambda: self.switch_tab("preview")
        )
        self.btn_tab_preview.pack(side="left", padx=4, pady=4)

        # Right Quick Telemetry & Status
        status_frame = ctk.CTkFrame(nav_bar, fg_color="transparent")
        status_frame.grid(row=0, column=2, padx=20, pady=12, sticky="e")

        self.nav_driver_pill = ctk.CTkLabel(
            status_frame, text="● Driver: Disconnected",
            text_color="#ff1744", font=("Segoe UI", 12, "bold")
        )
        self.nav_driver_pill.pack(side="left", padx=10)

        self.nav_aimbot_btn = neon_button(
            status_frame, text="START AIMBOT", width=130, height=34,
            command=self.toggle_aimbot
        )
        self.nav_aimbot_btn.pack(side="left", padx=(5, 0))

    def switch_tab(self, tab_name):
        self.active_tab = tab_name
        if tab_name == "dashboard":
            self.view_preview.grid_remove()
            self.view_dashboard.grid(row=1, column=0, sticky="nsew", padx=15, pady=(10, 15))
            self.btn_tab_dashboard.configure(fg_color="#242633", text_color="#ffffff")
            self.btn_tab_preview.configure(fg_color="transparent", text_color=TEXT_MUTED)
        else:
            self.view_dashboard.grid_remove()
            self.view_preview.grid(row=1, column=0, sticky="nsew", padx=15, pady=(10, 15))
            self.btn_tab_preview.configure(fg_color="#242633", text_color="#ffffff")
            self.btn_tab_dashboard.configure(fg_color="transparent", text_color=TEXT_MUTED)

    # =========================================================================
    # BUILD VIEWS (Dashboard & Preview)
    # =========================================================================
    def build_views(self):
        # 1. Dashboard View (Scrollable)
        self.view_dashboard = ctk.CTkScrollableFrame(
            self, fg_color=BG_DARK,
            scrollbar_button_color="#242633",
            scrollbar_button_hover_color=NEON_CYAN
        )
        self.view_dashboard.grid(row=1, column=0, sticky="nsew", padx=15, pady=(10, 15))
        self.view_dashboard.grid_columnconfigure(0, weight=1)
        self.view_dashboard.grid_columnconfigure(1, weight=1)

        # 2. Preview View (Centered Live Cyber Viewport)
        self.view_preview = ctk.CTkFrame(self, fg_color=BG_DARK)
        self.view_preview.grid_columnconfigure(0, weight=1)
        self.view_preview.grid_rowconfigure(1, weight=1)

        self.build_dashboard_content()
        self.build_preview_content()

    # =========================================================================
    # DASHBOARD VIEW CONTENT
    # =========================================================================
    def build_dashboard_content(self):
        # Status Header Banner
        self.build_status_banner(self.view_dashboard)

        # 2-Column Responsive Layout
        left_col = ctk.CTkFrame(self.view_dashboard, fg_color="transparent")
        left_col.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=0)
        left_col.grid_columnconfigure(0, weight=1)

        right_col = ctk.CTkFrame(self.view_dashboard, fg_color="transparent")
        right_col.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=0)
        right_col.grid_columnconfigure(0, weight=1)

        # Left Column Cards
        row = 0
        self.build_card_driver(left_col, row); row += 1
        self.build_card_capture(left_col, row); row += 1
        self.build_card_aim_settings(left_col, row); row += 1
        self.build_card_recoil_control(left_col, row); row += 1
        self.build_card_dynamic_mode(left_col, row); row += 1

        # Right Column Cards
        row = 0
        self.build_card_model_and_classes(right_col, row); row += 1
        self.build_card_triggerbot(right_col, row); row += 1
        self.build_card_profiles(right_col, row); row += 1

    def build_status_banner(self, parent):
        banner = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
        banner.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        banner.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Status 1: Driver
        f1 = ctk.CTkFrame(banner, fg_color="transparent")
        f1.grid(row=0, column=0, padx=15, pady=12, sticky="w")
        ctk.CTkLabel(f1, text="LOGITECH DRIVER", font=("Segoe UI", 10, "bold"), text_color=TEXT_MUTED).pack(anchor="w")
        self.lbl_banner_driver = ctk.CTkLabel(f1, textvariable=self.connection_status, font=("Segoe UI", 14, "bold"), text_color="#ff1744")
        self.lbl_banner_driver.pack(anchor="w")

        # Status 2: Aimbot
        f2 = ctk.CTkFrame(banner, fg_color="transparent")
        f2.grid(row=0, column=1, padx=15, pady=12, sticky="w")
        ctk.CTkLabel(f2, text="AIMBOT ENGINE", font=("Segoe UI", 10, "bold"), text_color=TEXT_MUTED).pack(anchor="w")
        self.lbl_banner_aimbot = ctk.CTkLabel(f2, textvariable=self.aimbot_status, font=("Segoe UI", 14, "bold"), text_color=TEXT_MUTED)
        self.lbl_banner_aimbot.pack(anchor="w")

        # Status 3: Model Loaded
        f3 = ctk.CTkFrame(banner, fg_color="transparent")
        f3.grid(row=0, column=2, padx=15, pady=12, sticky="w")
        ctk.CTkLabel(f3, text="ACTIVE AI MODEL", font=("Segoe UI", 10, "bold"), text_color=TEXT_MUTED).pack(anchor="w")
        self.lbl_banner_model = ctk.CTkLabel(f3, textvariable=self.model_name, font=("Segoe UI", 13, "bold"), text_color=NEON_CYAN)
        self.lbl_banner_model.pack(anchor="w")

        # Status 4: Performance Telemetry
        f4 = ctk.CTkFrame(banner, fg_color="transparent")
        f4.grid(row=0, column=3, padx=15, pady=12, sticky="e")
        ctk.CTkLabel(f4, text="PERFORMANCE", font=("Segoe UI", 10, "bold"), text_color=TEXT_MUTED).pack(anchor="e")
        self.lbl_banner_fps = ctk.CTkLabel(f4, textvariable=self.fps_var, font=("Segoe UI", 13, "bold"), text_color=NEON_GREEN)
        self.lbl_banner_fps.pack(anchor="e")

    # ---------------- Card: Driver Controls ----------------
    def build_card_driver(self, parent, row):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(card, text="🔌 Driver & System Controls", font=("Segoe UI", 14, "bold"), text_color=NEON_GREEN)\
            .grid(row=0, column=0, columnspan=3, padx=15, pady=(12, 10), sticky="w")

        self.btn_connect = cyber_button(card, text="Connect Driver", command=self.on_connect, height=34)
        self.btn_connect.grid(row=1, column=0, padx=(15, 6), pady=(0, 12), sticky="ew")

        ctk.CTkButton(card, text="Test Move", command=test_move, height=34, fg_color="#20222b", hover_color="#2b2e3b", text_color=TEXT_LIGHT)\
            .grid(row=1, column=1, padx=6, pady=(0, 12), sticky="ew")

        ctk.CTkSwitch(card, text="Button Mask", variable=self.button_mask_var, command=self.on_button_mask_toggle, text_color=TEXT_LIGHT)\
            .grid(row=1, column=2, padx=(6, 15), pady=(0, 12), sticky="w")

    # ---------------- Card: Screen Capture ----------------
    def build_card_capture(self, parent, row):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(card, text="📷 Screen Capture Engine", font=("Segoe UI", 14, "bold"), text_color=NEON_CYAN)\
            .grid(row=0, column=0, columnspan=2, padx=15, pady=(12, 10), sticky="w")

        ctk.CTkLabel(card, text="Capture Mode", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=1, column=0, padx=15, pady=(0, 12), sticky="w")

        self.capture_mode_menu = ctk.CTkOptionMenu(
            card, values=["DXGI", "MSS", "NDI"], variable=self.capture_mode_var,
            command=self.on_capture_mode_change, width=150, fg_color="#20222b", button_color="#282a36"
        )
        self.capture_mode_menu.grid(row=1, column=1, padx=15, pady=(0, 12), sticky="e")

        self.ndi_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.ndi_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.ndi_frame, text="NDI Source", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=0, column=0, padx=15, pady=(0, 10), sticky="w")
        self.ndi_source_var = ctk.StringVar(value=self._initial_ndi_source_value())
        self.ndi_source_menu = ctk.CTkOptionMenu(
            self.ndi_frame, values=self._ndi_menu_values(), variable=self.ndi_source_var,
            command=self.on_ndi_source_change, width=200, fg_color="#20222b", button_color="#282a36"
        )
        self.ndi_source_menu.grid(row=0, column=1, padx=15, pady=(0, 10), sticky="e")

        self._update_ndi_controls_state()

    # ---------------- Card: Aim Settings (Anti-Shake Included) ----------------
    def build_card_aim_settings(self, parent, row):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(card, text="🎮 Aiming & Anti-Shake Smooth", font=("Segoe UI", 14, "bold"), text_color=NEON_GREEN)\
            .grid(row=0, column=0, columnspan=3, padx=15, pady=(12, 10), sticky="w")

        # Aim Mode Dropdown
        ctk.CTkLabel(card, text="Aim Algorithm", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=1, column=0, padx=15, pady=6, sticky="w")
        self.mode_menu = ctk.CTkOptionMenu(
            card, values=["normal", "bezier", "silent", "smooth"], variable=self.mode_var,
            command=self.update_mode, width=150, fg_color="#20222b", button_color="#282a36"
        )
        self.mode_menu.grid(row=1, column=1, columnspan=2, padx=15, pady=6, sticky="e")

        # FOV Size Slider
        ctk.CTkLabel(card, text="FOV Region Size", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=2, column=0, padx=15, pady=6, sticky="w")
        self.fov_slider = ctk.CTkSlider(card, from_=50, to=400, number_of_steps=175, command=self.update_fov)
        self.fov_slider.grid(row=2, column=1, padx=(5, 10), pady=6, sticky="ew")
        self.fov_entry = ctk.CTkEntry(card, width=50, justify="center")
        self.fov_entry.grid(row=2, column=2, padx=(0, 15), pady=6)
        self.fov_entry.bind("<Return>", self.on_fov_entry_commit)
        self.fov_entry.bind("<FocusOut>", self.on_fov_entry_commit)

        # In-Game Sensitivity
        ctk.CTkLabel(card, text="In-Game Sensitivity", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=3, column=0, padx=15, pady=6, sticky="w")
        self.sens_slider = ctk.CTkSlider(card, from_=0.1, to=10.0, number_of_steps=99, command=self.update_in_game_sens)
        self.sens_slider.grid(row=3, column=1, padx=(5, 10), pady=6, sticky="ew")
        self.sens_entry = ctk.CTkEntry(card, width=50, justify="center")
        self.sens_entry.grid(row=3, column=2, padx=(0, 15), pady=6)
        self.sens_entry.bind("<Return>", self.on_sens_entry_commit)
        self.sens_entry.bind("<FocusOut>", self.on_sens_entry_commit)

        # Anti-Shake Deadzone (px)
        ctk.CTkLabel(card, text="Anti-Shake Deadzone (px)", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=4, column=0, padx=15, pady=6, sticky="w")
        self.deadzone_slider = ctk.CTkSlider(card, from_=0.0, to=8.0, number_of_steps=80, command=self.update_deadzone)
        self.deadzone_slider.set(getattr(config, "aim_deadzone", 2.0))
        self.deadzone_slider.grid(row=4, column=1, padx=(5, 10), pady=6, sticky="ew")
        self.deadzone_label = ctk.CTkLabel(card, text=f"{getattr(config, 'aim_deadzone', 2.0):.1f}", font=("Segoe UI", 12, "bold"), text_color=NEON_CYAN, width=50)
        self.deadzone_label.grid(row=4, column=2, padx=(0, 15), pady=6)

        # Anti-Shake EMA Smoothing
        ctk.CTkLabel(card, text="Target EMA Smooth Filter", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=5, column=0, padx=15, pady=6, sticky="w")
        self.smoothing_slider = ctk.CTkSlider(card, from_=0.0, to=0.95, number_of_steps=95, command=self.update_smoothing)
        self.smoothing_slider.set(getattr(config, "aim_smoothing_factor", 0.60))
        self.smoothing_slider.grid(row=5, column=1, padx=(5, 10), pady=6, sticky="ew")
        self.smoothing_label = ctk.CTkLabel(card, text=f"{getattr(config, 'aim_smoothing_factor', 0.60):.2f}", font=("Segoe UI", 12, "bold"), text_color=NEON_CYAN, width=50)
        self.smoothing_label.grid(row=5, column=2, padx=(0, 15), pady=6)

        # Player Y Offset
        ctk.CTkLabel(card, text="Player Y-Offset", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=6, column=0, padx=15, pady=6, sticky="w")
        self.offset_slider = ctk.CTkSlider(card, from_=-20, to=30, number_of_steps=50, command=self.update_offset)
        self.offset_slider.grid(row=6, column=1, padx=(5, 10), pady=6, sticky="ew")
        self.offset_label = ctk.CTkLabel(card, text=str(config.player_y_offset), font=("Segoe UI", 12, "bold"), text_color=NEON_CYAN, width=50)
        self.offset_label.grid(row=6, column=2, padx=(0, 15), pady=6)

        # Aim Activation Button
        ctk.CTkLabel(card, text="Activation Button", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=7, column=0, padx=15, pady=(6, 12), sticky="w")
        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.grid(row=7, column=1, columnspan=2, padx=15, pady=(6, 12), sticky="e")

        self.btn_menu = ctk.CTkOptionMenu(
            btn_frame, values=["Left (0)", "Right (1)", "Middle (2)", "Side 4 (3)", "Side 5 (4)"],
            command=self.update_mouse_btn, width=120, fg_color="#20222b", button_color="#282a36"
        )
        self.btn_menu.pack(side="left", padx=(0, 10))

        ctk.CTkSwitch(btn_frame, text="Always-On", variable=self.always_on_var, command=self.on_always_on_toggle, text_color=TEXT_LIGHT)\
            .pack(side="left")

    # ---------------- Card: Recoil Control System (RCS) ----------------
    def build_card_recoil_control(self, parent, row):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure(1, weight=1)

        # Header + Switch
        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.grid(row=0, column=0, columnspan=3, padx=15, pady=(12, 10), sticky="ew")
        ctk.CTkLabel(hdr, text="🔫 Recoil Control System (RCS)", font=("Segoe UI", 14, "bold"), text_color=NEON_ORANGE).pack(side="left")
        ctk.CTkSwitch(hdr, text="Enable RCS", variable=self.rcs_enabled_var, command=self.on_rcs_toggle, text_color=TEXT_LIGHT).pack(side="right")

        # Vertical Recoil (Pull Down)
        ctk.CTkLabel(card, text="Vertical Pull Strength", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=1, column=0, padx=15, pady=6, sticky="w")
        self.rcs_y_slider = ctk.CTkSlider(card, from_=0.0, to=15.0, number_of_steps=150, command=self.update_rcs_y)
        self.rcs_y_slider.set(getattr(config, "rcs_strength_y", 2.8))
        self.rcs_y_slider.grid(row=1, column=1, padx=(5, 10), pady=6, sticky="ew")
        self.rcs_y_lbl = ctk.CTkLabel(card, text=f"{getattr(config, 'rcs_strength_y', 2.8):.1f}", font=("Segoe UI", 12, "bold"), text_color=NEON_ORANGE, width=50)
        self.rcs_y_lbl.grid(row=1, column=2, padx=(0, 15), pady=6)

        # Horizontal Recoil (Compensate X)
        ctk.CTkLabel(card, text="Horizontal Compensation", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=2, column=0, padx=15, pady=6, sticky="w")
        self.rcs_x_slider = ctk.CTkSlider(card, from_=-5.0, to=5.0, number_of_steps=100, command=self.update_rcs_x)
        self.rcs_x_slider.set(getattr(config, "rcs_strength_x", 0.0))
        self.rcs_x_slider.grid(row=2, column=1, padx=(5, 10), pady=6, sticky="ew")
        self.rcs_x_lbl = ctk.CTkLabel(card, text=f"{getattr(config, 'rcs_strength_x', 0.0):.1f}", font=("Segoe UI", 12, "bold"), text_color=NEON_ORANGE, width=50)
        self.rcs_x_lbl.grid(row=2, column=2, padx=(0, 15), pady=6)

        # RCS Activation Delay (ms)
        ctk.CTkLabel(card, text="RCS Start Delay (ms)", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=3, column=0, padx=15, pady=(6, 12), sticky="w")
        self.rcs_delay_entry = ctk.CTkEntry(card, width=70, justify="center")
        self.rcs_delay_entry.insert(0, str(getattr(config, "rcs_delay_ms", 45)))
        self.rcs_delay_entry.grid(row=3, column=1, columnspan=2, padx=15, pady=(6, 12), sticky="e")
        self.rcs_delay_entry.bind("<FocusOut>", self.save_rcs_params)

    # ---------------- Card: Dynamic Mode Settings ----------------
    def build_card_dynamic_mode(self, parent, row):
        self.dynamic_frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
        self.dynamic_frame.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        self.dynamic_frame.grid_columnconfigure(1, weight=1)
        self.update_dynamic_frame()

    def update_dynamic_frame(self):
        for w in self.dynamic_frame.winfo_children():
            w.destroy()

        mode = config.mode.lower()
        ctk.CTkLabel(self.dynamic_frame, text=f"⚙️ {mode.capitalize()} Mode Parameters", font=("Segoe UI", 14, "bold"), text_color=NEON_CYAN)\
            .grid(row=0, column=0, columnspan=3, padx=15, pady=(12, 10), sticky="w")

        if mode == "normal":
            ctk.CTkLabel(self.dynamic_frame, text="Speed X", text_color=TEXT_LIGHT).grid(row=1, column=0, padx=15, pady=6, sticky="w")
            self.normal_x_slider = ctk.CTkSlider(self.dynamic_frame, from_=0.05, to=2.0, command=self.update_normal_x)
            self.normal_x_slider.set(config.normal_x_speed)
            self.normal_x_slider.grid(row=1, column=1, padx=(5, 10), pady=6, sticky="ew")
            self.normal_x_label = ctk.CTkLabel(self.dynamic_frame, text=f"{config.normal_x_speed:.2f}", text_color=NEON_CYAN, width=50)
            self.normal_x_label.grid(row=1, column=2, padx=(0, 15), pady=6)

            ctk.CTkLabel(self.dynamic_frame, text="Speed Y", text_color=TEXT_LIGHT).grid(row=2, column=0, padx=15, pady=(6, 12), sticky="w")
            self.normal_y_slider = ctk.CTkSlider(self.dynamic_frame, from_=0.05, to=2.0, command=self.update_normal_y)
            self.normal_y_slider.set(config.normal_y_speed)
            self.normal_y_slider.grid(row=2, column=1, padx=(5, 10), pady=(6, 12), sticky="ew")
            self.normal_y_label = ctk.CTkLabel(self.dynamic_frame, text=f"{config.normal_y_speed:.2f}", text_color=NEON_CYAN, width=50)
            self.normal_y_label.grid(row=2, column=2, padx=(0, 15), pady=(6, 12))

        elif mode in ("bezier", "silent"):
            is_silent = (mode == "silent")
            ctk.CTkLabel(self.dynamic_frame, text="Segments", text_color=TEXT_LIGHT).grid(row=1, column=0, padx=15, pady=6, sticky="w")
            seg_val = config.silent_segments if is_silent else config.bezier_segments
            self.bez_seg_slider = ctk.CTkSlider(self.dynamic_frame, from_=2, to=30, number_of_steps=28, command=lambda v: self.update_bezier_param('seg', v, is_silent))
            self.bez_seg_slider.set(seg_val)
            self.bez_seg_slider.grid(row=1, column=1, padx=(5, 10), pady=6, sticky="ew")
            self.bez_seg_lbl = ctk.CTkLabel(self.dynamic_frame, text=str(seg_val), text_color=NEON_CYAN, width=50)
            self.bez_seg_lbl.grid(row=1, column=2, padx=(0, 15), pady=6)

            ctk.CTkLabel(self.dynamic_frame, text="Curve Strength", text_color=TEXT_LIGHT).grid(row=2, column=0, padx=15, pady=(6, 12), sticky="w")
            ctrl_val = config.silent_ctrl_x if is_silent else config.bezier_ctrl_x
            self.bez_ctrl_slider = ctk.CTkSlider(self.dynamic_frame, from_=1, to=50, number_of_steps=49, command=lambda v: self.update_bezier_param('ctrl', v, is_silent))
            self.bez_ctrl_slider.set(ctrl_val)
            self.bez_ctrl_slider.grid(row=2, column=1, padx=(5, 10), pady=(6, 12), sticky="ew")
            self.bez_ctrl_lbl = ctk.CTkLabel(self.dynamic_frame, text=str(ctrl_val), text_color=NEON_CYAN, width=50)
            self.bez_ctrl_lbl.grid(row=2, column=2, padx=(0, 15), pady=(6, 12))

        elif mode == "smooth":
            ctk.CTkLabel(self.dynamic_frame, text="Gravity", text_color=TEXT_LIGHT).grid(row=1, column=0, padx=15, pady=6, sticky="w")
            self.smooth_grav_slider = ctk.CTkSlider(self.dynamic_frame, from_=1.0, to=20.0, command=self.update_smooth_gravity)
            self.smooth_grav_slider.set(config.smooth_gravity)
            self.smooth_grav_slider.grid(row=1, column=1, padx=(5, 10), pady=6, sticky="ew")
            self.smooth_grav_lbl = ctk.CTkLabel(self.dynamic_frame, text=f"{config.smooth_gravity:.1f}", text_color=NEON_CYAN, width=50)
            self.smooth_grav_lbl.grid(row=1, column=2, padx=(0, 15), pady=6)

            ctk.CTkLabel(self.dynamic_frame, text="Wind Randomness", text_color=TEXT_LIGHT).grid(row=2, column=0, padx=15, pady=(6, 12), sticky="w")
            self.smooth_wind_slider = ctk.CTkSlider(self.dynamic_frame, from_=0.0, to=15.0, command=self.update_smooth_wind)
            self.smooth_wind_slider.set(config.smooth_wind)
            self.smooth_wind_slider.grid(row=2, column=1, padx=(5, 10), pady=6, sticky="ew")
            self.smooth_wind_lbl = ctk.CTkLabel(self.dynamic_frame, text=f"{config.smooth_wind:.1f}", text_color=NEON_CYAN, width=50)
            self.smooth_wind_lbl.grid(row=2, column=2, padx=(0, 15), pady=(6, 12))

    # ---------------- Card: AI Model & Detection ----------------
    def build_card_model_and_classes(self, parent, row):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(card, text="🎯 AI Model & Detection", font=("Segoe UI", 14, "bold"), text_color=NEON_ORANGE)\
            .grid(row=0, column=0, columnspan=3, padx=15, pady=(12, 10), sticky="w")

        # Model Selector
        ctk.CTkLabel(card, text="Select Model", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=1, column=0, padx=15, pady=6, sticky="w")
        
        m_frame = ctk.CTkFrame(card, fg_color="transparent")
        m_frame.grid(row=1, column=1, columnspan=2, padx=15, pady=6, sticky="ew")
        m_frame.grid_columnconfigure(0, weight=1)

        self.model_menu = ctk.CTkOptionMenu(
            m_frame, values=self.get_model_list(), command=self.select_model,
            fg_color="#20222b", button_color="#282a36"
        )
        self.model_menu.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        cyber_button(m_frame, text="🔄", width=34, height=28, command=self.reload_current_model)\
            .grid(row=0, column=1)

        # Player Class & Head Class
        ctk.CTkLabel(card, text="Player Class", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=2, column=0, padx=15, pady=6, sticky="w")
        self.player_class_menu = ctk.CTkOptionMenu(
            card, values=["Loading..."], command=self.select_player_class,
            fg_color="#20222b", button_color="#282a36"
        )
        self.player_class_menu.grid(row=2, column=1, columnspan=2, padx=15, pady=6, sticky="ew")

        ctk.CTkLabel(card, text="Head Class", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=3, column=0, padx=15, pady=6, sticky="w")
        self.head_class_menu = ctk.CTkOptionMenu(
            card, values=["Loading..."], command=self.select_head_class,
            fg_color="#20222b", button_color="#282a36"
        )
        self.head_class_menu.grid(row=3, column=1, columnspan=2, padx=15, pady=6, sticky="ew")

        # Head Priority Switch
        ctk.CTkSwitch(card, text="Headshot Priority (Prioritize Head over Body)", variable=self.head_priority_var, command=self.on_head_priority_toggle, text_color=TEXT_LIGHT)\
            .grid(row=4, column=0, columnspan=3, padx=15, pady=6, sticky="w")

        # Confidence Threshold
        ctk.CTkLabel(card, text="Confidence Threshold", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=5, column=0, padx=15, pady=6, sticky="w")
        self.conf_slider = ctk.CTkSlider(card, from_=0.05, to=0.95, number_of_steps=90, command=self.update_conf)
        self.conf_slider.grid(row=5, column=1, padx=(5, 10), pady=6, sticky="ew")
        self.conf_entry = ctk.CTkEntry(card, width=50, justify="center")
        self.conf_entry.grid(row=5, column=2, padx=(0, 15), pady=6)
        self.conf_entry.bind("<Return>", self.on_conf_entry_commit)
        self.conf_entry.bind("<FocusOut>", self.on_conf_entry_commit)

        # Resolution (imgsz) & Max Detections
        ctk.CTkLabel(card, text="Detection Resolution", font=("Segoe UI", 12), text_color=TEXT_LIGHT)\
            .grid(row=6, column=0, padx=15, pady=(6, 12), sticky="w")
        self.imgsz_slider = ctk.CTkSlider(card, from_=256, to=1024, number_of_steps=12, command=self.update_imgsz)
        self.imgsz_slider.grid(row=6, column=1, padx=(5, 10), pady=(6, 12), sticky="ew")
        self.imgsz_entry = ctk.CTkEntry(card, width=50, justify="center")
        self.imgsz_entry.grid(row=6, column=2, padx=(0, 15), pady=(6, 12))
        self.imgsz_entry.bind("<Return>", self.on_imgsz_entry_commit)
        self.imgsz_entry.bind("<FocusOut>", self.on_imgsz_entry_commit)

    # ---------------- Card: Triggerbot ----------------
    def build_card_triggerbot(self, parent, row):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure((1, 3), weight=1)

        # Header + Switch
        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.grid(row=0, column=0, columnspan=4, padx=15, pady=(12, 10), sticky="ew")
        ctk.CTkLabel(hdr, text="⚡ Fast Triggerbot", font=("Segoe UI", 14, "bold"), text_color=NEON_GREEN).pack(side="left")
        ctk.CTkSwitch(hdr, text="Enable", variable=self.trigger_enabled_var, command=self.on_trigger_enabled_toggle, text_color=TEXT_LIGHT).pack(side="right")

        # Mode: Always on vs Dedicated Key
        ctk.CTkLabel(card, text="Trigger Key", font=("Segoe UI", 12), text_color=TEXT_LIGHT).grid(row=1, column=0, padx=15, pady=6, sticky="w")
        self.tb_btn_menu = ctk.CTkOptionMenu(
            card, values=["Left (0)", "Right (1)", "Middle (2)", "Side 4 (3)", "Side 5 (4)"],
            command=self.update_trigger_button, width=110, fg_color="#20222b", button_color="#282a36"
        )
        self.tb_btn_menu.grid(row=1, column=1, padx=5, pady=6, sticky="w")

        ctk.CTkSwitch(card, text="Always-On Fire", variable=self.trigger_always_on_var, command=self.on_trigger_always_on_toggle, text_color=TEXT_LIGHT)\
            .grid(row=1, column=2, columnspan=2, padx=15, pady=6, sticky="e")

        # Parameters
        ctk.CTkLabel(card, text="Radius (px)", font=("Segoe UI", 12), text_color=TEXT_LIGHT).grid(row=2, column=0, padx=15, pady=6, sticky="w")
        self.tb_radius_entry = ctk.CTkEntry(card, width=70, justify="center")
        self.tb_radius_entry.grid(row=2, column=1, padx=5, pady=6, sticky="w")
        self.tb_radius_entry.bind("<FocusOut>", self.save_trigger_params)

        ctk.CTkLabel(card, text="Min Conf", font=("Segoe UI", 12), text_color=TEXT_LIGHT).grid(row=2, column=2, padx=10, pady=6, sticky="w")
        self.tb_conf_entry = ctk.CTkEntry(card, width=70, justify="center")
        self.tb_conf_entry.grid(row=2, column=3, padx=(5, 15), pady=6, sticky="w")
        self.tb_conf_entry.bind("<FocusOut>", self.save_trigger_params)

        ctk.CTkLabel(card, text="Delay (ms)", font=("Segoe UI", 12), text_color=TEXT_LIGHT).grid(row=3, column=0, padx=15, pady=(6, 12), sticky="w")
        self.tb_delay_entry = ctk.CTkEntry(card, width=70, justify="center")
        self.tb_delay_entry.grid(row=3, column=1, padx=5, pady=(6, 12), sticky="w")
        self.tb_delay_entry.bind("<FocusOut>", self.save_trigger_params)

        ctk.CTkLabel(card, text="Cooldown (ms)", font=("Segoe UI", 12), text_color=TEXT_LIGHT).grid(row=3, column=2, padx=10, pady=(6, 12), sticky="w")
        self.tb_cd_entry = ctk.CTkEntry(card, width=70, justify="center")
        self.tb_cd_entry.grid(row=3, column=3, padx=(5, 15), pady=(6, 12), sticky="w")
        self.tb_cd_entry.bind("<FocusOut>", self.save_trigger_params)

    # ---------------- Card: Profiles & Management ----------------
    def build_card_profiles(self, parent, row):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(card, text="💾 Profiles & Presets", font=("Segoe UI", 14, "bold"), text_color=TEXT_LIGHT)\
            .grid(row=0, column=0, columnspan=3, padx=15, pady=(12, 10), sticky="w")

        cyber_button(card, text="Save Profile", command=self.save_profile, height=32)\
            .grid(row=1, column=0, padx=(15, 5), pady=(0, 12), sticky="ew")

        cyber_button(card, text="Load Profile", command=self.load_profile, height=32)\
            .grid(row=1, column=1, padx=5, pady=(0, 12), sticky="ew")

        ctk.CTkButton(card, text="Reset Defaults", command=self.reset_profile, height=32, fg_color="#20222b", hover_color="#303240", text_color="#ff5e69")\
            .grid(row=1, column=2, padx=(5, 15), pady=(0, 12), sticky="ew")

    # =========================================================================
    # PREVIEW VIEW CONTENT (Live Cyber HUD - Exact Match to Screenshot!)
    # =========================================================================
    def build_preview_content(self):
        # 1. Top HUD Controls Bar
        ctrl_bar = ctk.CTkFrame(self.view_preview, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
        ctrl_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(0, 10))
        ctrl_bar.grid_columnconfigure(4, weight=1)

        self.btn_preview_pause = cyber_button(
            ctrl_bar, text="⏸ Pause Stream", width=120, height=32,
            command=self.toggle_preview_pause
        )
        self.btn_preview_pause.grid(row=0, column=0, padx=10, pady=8)

        ctk.CTkCheckBox(ctrl_bar, text="FOV Circle", variable=self.preview_fov_var, command=self.update_preview_toggles, text_color=TEXT_LIGHT)\
            .grid(row=0, column=1, padx=10, pady=8)

        ctk.CTkCheckBox(ctrl_bar, text="Target Boxes & Pills", variable=self.preview_boxes_var, command=self.update_preview_toggles, text_color=TEXT_LIGHT)\
            .grid(row=0, column=2, padx=10, pady=8)

        ctk.CTkCheckBox(ctrl_bar, text="Lock Vector", variable=self.preview_vectors_var, command=self.update_preview_toggles, text_color=TEXT_LIGHT)\
            .grid(row=0, column=3, padx=10, pady=8)

        ctk.CTkCheckBox(ctrl_bar, text="OpenCV Debug Window", variable=self.debug_checkbox_var, command=self.on_debug_toggle, text_color=TEXT_LIGHT)\
            .grid(row=0, column=5, padx=15, pady=8, sticky="e")

        # 2. Centered Live HUD Viewport Container
        viewport_card = ctk.CTkFrame(self.view_preview, fg_color="#090a0c", corner_radius=12, border_width=1, border_color="#1d2029")
        viewport_card.grid(row=1, column=0, sticky="nsew", padx=10, pady=0)
        viewport_card.grid_rowconfigure(0, weight=1)
        viewport_card.grid_columnconfigure(0, weight=1)

        # Image Display Label
        self.preview_canvas_label = ctk.CTkLabel(
            viewport_card, text="[ Standby / Start Aimbot to Stream ]",
            font=("Segoe UI", 14), text_color=TEXT_MUTED
        )
        self.preview_canvas_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # 3. Bottom Telemetry Status Bar (Matching Screenshot: Resolution | Capture FPS | Detection Latency)
        telemetry_bar = ctk.CTkFrame(self.view_preview, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
        telemetry_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(10, 0))
        telemetry_bar.grid_columnconfigure((0, 1, 2), weight=1)

        # Telemetry Item 1: Resolution
        t1 = ctk.CTkFrame(telemetry_bar, fg_color="transparent")
        t1.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        ctk.CTkLabel(t1, text="Resolution:", font=("Segoe UI", 13), text_color=TEXT_LIGHT).pack(side="left", padx=(0, 8))
        self.lbl_hud_res = ctk.CTkLabel(t1, textvariable=self.resolution_var, font=("Segoe UI", 13, "bold"), text_color=TEXT_LIGHT)
        self.lbl_hud_res.pack(side="left")

        # Telemetry Item 2: Capture FPS
        t2 = ctk.CTkFrame(telemetry_bar, fg_color="transparent")
        t2.grid(row=0, column=1, padx=20, pady=10)
        ctk.CTkLabel(t2, text="Capture FPS:", font=("Segoe UI", 13), text_color=TEXT_LIGHT).pack(side="left", padx=(0, 8))
        self.lbl_hud_fps = ctk.CTkLabel(t2, textvariable=self.capture_fps_var, font=("Segoe UI", 13, "bold"), text_color=NEON_GREEN)
        self.lbl_hud_fps.pack(side="left")

        # Telemetry Item 3: Detection Latency
        t3 = ctk.CTkFrame(telemetry_bar, fg_color="transparent")
        t3.grid(row=0, column=2, padx=20, pady=10, sticky="e")
        ctk.CTkLabel(t3, text="Detection Latency:", font=("Segoe UI", 13), text_color=TEXT_LIGHT).pack(side="left", padx=(0, 8))
        self.lbl_hud_latency = ctk.CTkLabel(t3, textvariable=self.latency_var, font=("Segoe UI", 13, "bold"), text_color=NEON_GREEN)
        self.lbl_hud_latency.pack(side="left")

    def toggle_preview_pause(self):
        self.preview_paused = not self.preview_paused
        self.btn_preview_pause.configure(text="▶ Resume Stream" if self.preview_paused else "⏸ Pause Stream")

    def update_preview_toggles(self):
        config.preview_fov = bool(self.preview_fov_var.get())
        config.preview_boxes = bool(self.preview_boxes_var.get())
        config.preview_vectors = bool(self.preview_vectors_var.get())

    # =========================================================================
    # REAL-TIME POLLING & TELEMETRY
    # =========================================================================
    def poll_telemetry(self):
        # Aimbot Running State
        is_running = is_aimbot_running()
        self.aimbot_status.set("RUNNING" if is_running else "STOPPED")
        self.lbl_banner_aimbot.configure(text_color=NEON_GREEN if is_running else TEXT_MUTED)
        self.nav_aimbot_btn.configure(
            text="STOP AIMBOT" if is_running else "START AIMBOT",
            fg_color="#d50000" if is_running else NEON
        )

        # Logitech Driver Connection
        driver_ok = mouse_is_connected or getattr(config, "logitech_connected", False)
        self.connection_status.set("Connected" if driver_ok else "Disconnected")
        self.connection_color.set(NEON_GREEN if driver_ok else "#ff1744")
        self.lbl_banner_driver.configure(text_color=self.connection_color.get())
        self.nav_driver_pill.configure(
            text="● Driver: Connected" if driver_ok else "● Driver: Disconnected",
            text_color=NEON_GREEN if driver_ok else "#ff1744"
        )

        # Performance Stats
        c_fps = getattr(config, "capture_fps", 0.0)
        d_lat = getattr(config, "detection_latency", 0.0)
        
        self.fps_var.set(f"{main.fps:.1f} FPS (Aimbot)")
        self.capture_fps_var.set(f"{c_fps:.1f} (GOOD)")
        self.latency_var.set(f"{d_lat:.2f} ms (GOOD)")
        self.resolution_var.set(f"{config.region_size}x{config.region_size}")

        self.after(250, self.poll_telemetry)

    def poll_preview(self):
        """Ultra-smooth in-app live preview update loop."""
        if self.active_tab == "preview" and not self.preview_paused:
            frame = get_latest_preview_frame()
            if frame is not None:
                try:
                    target_w = max(256, self.preview_canvas_label.winfo_width() - 20)
                    target_h = max(256, self.preview_canvas_label.winfo_height() - 20)
                    side = min(target_w, target_h, 600)
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    pil_img = pil_img.resize((side, side), Image.Resampling.BILINEAR)
                    
                    self.preview_image_tk = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(side, side))
                    self.preview_canvas_label.configure(image=self.preview_image_tk, text="")
                except Exception:
                    pass
            elif not is_aimbot_running():
                self.preview_canvas_label.configure(image=None, text="[ Aimbot Stopped — Click START AIMBOT to Launch Stream ]")

        self.after(33, self.poll_preview)

    # =========================================================================
    # CALLBACKS & ACTIONS
    # =========================================================================
    def refresh_all(self):
        self.fov_slider.set(config.region_size)
        self._set_entry_text(self.fov_entry, str(config.region_size))
        
        self.sens_slider.set(config.in_game_sens)
        self._set_entry_text(self.sens_entry, f"{config.in_game_sens:.2f}")

        self.deadzone_slider.set(getattr(config, "aim_deadzone", 2.0))
        self.deadzone_label.configure(text=f"{getattr(config, 'aim_deadzone', 2.0):.1f}")

        self.smoothing_slider.set(getattr(config, "aim_smoothing_factor", 0.60))
        self.smoothing_label.configure(text=f"{getattr(config, 'aim_smoothing_factor', 0.60):.2f}")

        self.offset_slider.set(config.player_y_offset)
        self.offset_label.configure(text=str(config.player_y_offset))

        self.rcs_enabled_var.set(bool(getattr(config, "rcs_enabled", False)))
        self.rcs_y_slider.set(getattr(config, "rcs_strength_y", 2.8))
        self.rcs_y_lbl.configure(text=f"{getattr(config, 'rcs_strength_y', 2.8):.1f}")
        self.rcs_x_slider.set(getattr(config, "rcs_strength_x", 0.0))
        self.rcs_x_lbl.configure(text=f"{getattr(config, 'rcs_strength_x', 0.0):.1f}")
        self._set_entry_text(self.rcs_delay_entry, str(getattr(config, "rcs_delay_ms", 45)))

        self.conf_slider.set(config.conf)
        self._set_entry_text(self.conf_entry, f"{config.conf:.2f}")

        self.imgsz_slider.set(config.imgsz)
        self._set_entry_text(self.imgsz_entry, str(config.imgsz))

        self.btn_var.set(config.selected_mouse_button)
        self.btn_menu.set(self._mouse_btn_idx_to_str(config.selected_mouse_button))

        self.mode_var.set(config.mode)
        self.mode_menu.set(config.mode)

        self.model_name.set(os.path.basename(config.model_path))
        self.model_menu.set(os.path.basename(config.model_path))
        self.model_size.set(get_model_size(config.model_path))

        self.always_on_var.set(bool(getattr(config, "always_on_aim", False)))
        self.head_priority_var.set(bool(getattr(config, "head_priority", True)))
        self.button_mask_var.set(bool(getattr(config, "button_mask", False)))
        self.debug_checkbox_var.set(bool(config.show_debug_window))

        self.trigger_enabled_var.set(bool(getattr(config, "trigger_enabled", False)))
        self.trigger_always_on_var.set(bool(getattr(config, "trigger_always_on", False)))
        self.trigger_btn_var.set(int(getattr(config, "trigger_button", 1)))
        self.tb_btn_menu.set(self._mouse_btn_idx_to_str(config.trigger_button))

        self._set_entry_text(self.tb_radius_entry, str(getattr(config, "trigger_radius_px", 10)))
        self._set_entry_text(self.tb_conf_entry, f"{getattr(config, 'trigger_min_conf', 0.35):.2f}")
        self._set_entry_text(self.tb_delay_entry, str(getattr(config, "trigger_delay_ms", 25)))
        self._set_entry_text(self.tb_cd_entry, str(getattr(config, "trigger_cooldown_ms", 120)))

        self.load_class_list()
        self.update_dynamic_frame()

    def _mouse_btn_idx_to_str(self, idx):
        mapping = {0: "Left (0)", 1: "Right (1)", 2: "Middle (2)", 3: "Side 4 (3)", 4: "Side 5 (4)"}
        return mapping.get(idx, "Side 4 (3)")

    def _mouse_str_to_idx(self, val_str):
        if "0" in val_str: return 0
        if "1" in val_str: return 1
        if "2" in val_str: return 2
        if "3" in val_str: return 3
        if "4" in val_str: return 4
        return 3

    def _set_entry_text(self, entry, text):
        try:
            entry.delete(0, "end")
            entry.insert(0, str(text))
        except Exception:
            pass

    def on_connect(self):
        if connect_to_logitech():
            config.logitech_connected = True
            config.logitech_status_msg = "Connected"
            self.error_text.set("Logitech driver connected!")
        else:
            config.logitech_connected = False
            config.logitech_status_msg = "Disconnected"
            self.error_text.set("Failed to load logitech.driver.dll!")

    def toggle_aimbot(self):
        if is_aimbot_running():
            stop_aimbot()
        else:
            start_aimbot()

    def on_capture_mode_change(self, value):
        m = {"MSS": "mss", "NDI": "ndi", "DXGI": "dxgi"}
        val = m.get((value or "").upper(), "dxgi")
        config.capturer_mode = val
        self._update_ndi_controls_state()
        if is_aimbot_running():
            stop_aimbot()
            start_aimbot()
        config.save()

    def _update_ndi_controls_state(self):
        if self.capture_mode_var.get().upper() == "NDI":
            self.ndi_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        else:
            self.ndi_frame.grid_remove()

    def _initial_ndi_source_value(self):
        return str(config.ndi_selected_source) if config.ndi_selected_source else "Searching NDI..."

    def _ndi_menu_values(self):
        return config.ndi_sources if config.ndi_sources else ["No NDI Sources"]

    def on_ndi_source_change(self, val):
        if val and not val.startswith("No"):
            config.ndi_selected_source = val
            config.save()

    def update_fov(self, val):
        if getattr(self, "_updating_fov", False): return
        self._updating_fov = True
        try:
            val = int(round(val))
            config.region_size = val
            self._set_entry_text(self.fov_entry, str(val))
            self.resolution_var.set(f"{val}x{val}")
        finally:
            self._updating_fov = False

    def on_fov_entry_commit(self, event=None):
        try:
            val = int(self.fov_entry.get().strip())
            val = max(32, min(800, val))
            config.region_size = val
            self.fov_slider.set(val)
            self.resolution_var.set(f"{val}x{val}")
        except Exception:
            self._set_entry_text(self.fov_entry, str(config.region_size))

    def update_in_game_sens(self, val):
        config.in_game_sens = round(float(val), 2)
        self._set_entry_text(self.sens_entry, f"{config.in_game_sens:.2f}")

    def on_sens_entry_commit(self, event=None):
        try:
            val = float(self.sens_entry.get().strip())
            config.in_game_sens = max(0.05, min(20.0, val))
            self.sens_slider.set(config.in_game_sens)
        except Exception:
            self._set_entry_text(self.sens_entry, f"{config.in_game_sens:.2f}")

    def update_deadzone(self, val):
        config.aim_deadzone = round(float(val), 1)
        self.deadzone_label.configure(text=f"{config.aim_deadzone:.1f}")

    def update_smoothing(self, val):
        config.aim_smoothing_factor = round(float(val), 2)
        self.smoothing_label.configure(text=f"{config.aim_smoothing_factor:.2f}")

    def on_rcs_toggle(self):
        config.rcs_enabled = bool(self.rcs_enabled_var.get())

    def update_rcs_y(self, val):
        config.rcs_strength_y = round(float(val), 1)
        self.rcs_y_lbl.configure(text=f"{config.rcs_strength_y:.1f}")

    def update_rcs_x(self, val):
        config.rcs_strength_x = round(float(val), 1)
        self.rcs_x_lbl.configure(text=f"{config.rcs_strength_x:.1f}")

    def save_rcs_params(self, event=None):
        try:
            config.rcs_delay_ms = int(self.rcs_delay_entry.get().strip())
        except Exception:
            pass

    def update_offset(self, val):
        val = int(round(val))
        config.player_y_offset = val
        self.offset_label.configure(text=str(val))

    def update_mouse_btn(self, val):
        config.selected_mouse_button = self._mouse_str_to_idx(val)

    def on_always_on_toggle(self):
        config.always_on_aim = bool(self.always_on_var.get())

    def on_head_priority_toggle(self):
        config.head_priority = bool(self.head_priority_var.get())

    def on_button_mask_toggle(self):
        config.button_mask = bool(self.button_mask_var.get())

    def on_debug_toggle(self):
        config.show_debug_window = bool(self.debug_checkbox_var.get())
        if not config.show_debug_window:
            try: cv2.destroyWindow("CapkfaPlus Live Debug")
            except Exception: pass

    def update_mode(self, val):
        config.mode = str(val).lower()
        self.update_dynamic_frame()

    def update_normal_x(self, val):
        config.normal_x_speed = round(float(val), 2)
        self.normal_x_label.configure(text=f"{config.normal_x_speed:.2f}")

    def update_normal_y(self, val):
        config.normal_y_speed = round(float(val), 2)
        self.normal_y_label.configure(text=f"{config.normal_y_speed:.2f}")

    def update_bezier_param(self, param, val, is_silent=False):
        val = int(round(float(val)))
        if param == 'seg':
            if is_silent: config.silent_segments = val
            else: config.bezier_segments = val
            self.bez_seg_lbl.configure(text=str(val))
        else:
            if is_silent: config.silent_ctrl_x = config.silent_ctrl_y = val
            else: config.bezier_ctrl_x = config.bezier_ctrl_y = val
            self.bez_ctrl_lbl.configure(text=str(val))

    def update_smooth_gravity(self, val):
        config.smooth_gravity = round(float(val), 1)
        self.smooth_grav_lbl.configure(text=f"{config.smooth_gravity:.1f}")

    def update_smooth_wind(self, val):
        config.smooth_wind = round(float(val), 1)
        self.smooth_wind_lbl.configure(text=f"{config.smooth_wind:.1f}")

    def update_conf(self, val):
        if getattr(self, "_updating_conf", False): return
        self._updating_conf = True
        try:
            config.conf = round(float(val), 2)
            self._set_entry_text(self.conf_entry, f"{config.conf:.2f}")
        finally:
            self._updating_conf = False

    def on_conf_entry_commit(self, event=None):
        try:
            val = float(self.conf_entry.get().strip())
            config.conf = max(0.01, min(0.99, val))
            self.conf_slider.set(config.conf)
        except Exception:
            self._set_entry_text(self.conf_entry, f"{config.conf:.2f}")

    def update_imgsz(self, val):
        if getattr(self, "_updating_imgsz", False): return
        self._updating_imgsz = True
        try:
            val = int(round(val))
            config.imgsz = val
            self._set_entry_text(self.imgsz_entry, str(val))
        finally:
            self._updating_imgsz = False

    def on_imgsz_entry_commit(self, event=None):
        try:
            val = int(self.imgsz_entry.get().strip())
            config.imgsz = max(128, min(1920, val))
            self.imgsz_slider.set(config.imgsz)
        except Exception:
            self._set_entry_text(self.imgsz_entry, str(config.imgsz))

    def get_model_list(self):
        models = config.list_models()
        return models if models else ["No Models in /models"]

    def select_model(self, val):
        path = os.path.join("models", val)
        if os.path.isfile(path):
            config.model_path = path
            self.model_name.set(os.path.basename(path))
            self.model_size.set(get_model_size(path))
            reload_model(path)
            self.load_class_list()
            config.save()

    def reload_current_model(self):
        if os.path.isfile(config.model_path):
            reload_model(config.model_path)
            self.load_class_list()
            self.model_size.set(get_model_size(config.model_path))

    def load_class_list(self):
        classes = get_model_classes()
        vals = list(classes.values()) if isinstance(classes, dict) else (classes or ["0", "1"])
        if not vals: vals = ["0", "1"]
        
        self.player_class_menu.configure(values=[str(v) for v in vals])
        self.head_class_menu.configure(values=[str(v) for v in vals])

        if str(config.custom_player_label) in [str(v) for v in vals]:
            self.player_class_menu.set(str(config.custom_player_label))
        elif vals:
            config.custom_player_label = vals[0]
            self.player_class_menu.set(str(vals[0]))

        if str(config.custom_head_label) in [str(v) for v in vals]:
            self.head_class_menu.set(str(config.custom_head_label))
        elif len(vals) > 1:
            config.custom_head_label = vals[1]
            self.head_class_menu.set(str(vals[1]))
        elif vals:
            config.custom_head_label = vals[0]
            self.head_class_menu.set(str(vals[0]))

    def select_player_class(self, val):
        config.custom_player_label = val

    def select_head_class(self, val):
        config.custom_head_label = val

    def on_trigger_enabled_toggle(self):
        config.trigger_enabled = bool(self.trigger_enabled_var.get())

    def on_trigger_always_on_toggle(self):
        config.trigger_always_on = bool(self.trigger_always_on_var.get())

    def update_trigger_button(self, val):
        config.trigger_button = self._mouse_str_to_idx(val)

    def save_trigger_params(self, event=None):
        try:
            config.trigger_radius_px = int(self.tb_radius_entry.get().strip())
            config.trigger_min_conf = float(self.tb_conf_entry.get().strip())
            config.trigger_delay_ms = int(self.tb_delay_entry.get().strip())
            config.trigger_cooldown_ms = int(self.tb_cd_entry.get().strip())
        except Exception:
            pass

    def save_profile(self):
        self.save_trigger_params()
        self.save_rcs_params()
        config.save()
        messagebox.showinfo("CapkfaPlus", "Profile configuration saved successfully!")

    def load_profile(self):
        config.load()
        self.refresh_all()
        messagebox.showinfo("CapkfaPlus", "Profile loaded successfully!")

    def reset_profile(self):
        if messagebox.askyesno("CapkfaPlus", "Are you sure you want to reset all settings to defaults?"):
            config.reset_to_defaults()
            self.refresh_all()

    def on_close(self):
        stop_aimbot()
        self.destroy()


if __name__ == "__main__":
    app = CapkfaPlusGUI()
    app.mainloop()