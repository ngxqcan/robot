import os
import json
import ctypes
from ctypes import wintypes

# Structures
class RECT(ctypes.Structure):
    _fields_ = [
        ("left",   ctypes.c_long),
        ("top",    ctypes.c_long),
        ("right",  ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]

class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize",   ctypes.c_ulong),
        ("rcMonitor", RECT),
        ("rcWork",    RECT),
        ("dwFlags",   ctypes.c_ulong),
    ]

def get_foreground_monitor_resolution():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    try:
        user32 = ctypes.windll.user32
        monitor = user32.MonitorFromWindow(user32.GetForegroundWindow(), 2)
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)

        if user32.GetMonitorInfoW(monitor, ctypes.byref(mi)):
            w = mi.rcMonitor.right - mi.rcMonitor.left
            h = mi.rcMonitor.bottom - mi.rcMonitor.top
            return w, h
        else:
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080

w, h = get_foreground_monitor_resolution()

class Config:
    def __init__(self):
        # --- General Settings (1PC Radiant Profile) ---
        self.region_size = 180            # Tight FOV for legit Radiant micro-flicks
        w, h = get_foreground_monitor_resolution()
        self.screen_width = w
        self.screen_height = h
        self.player_y_offset = 3          # Direct Head/Upper Neck alignment
        self.capturer_mode = "DXGI"       # DXGI Hardware ROI (Ultra-fast)
        self.target_fps = 240
        self.always_on_aim = False
        self.head_priority = True         # 100% Headshot Lock Priority
        self.target_lock_hysteresis = True
        self.main_pc_width = w
        self.main_pc_height = h

        # --- Anti-Shaking & Deadzone Settings ---
        self.aim_deadzone = 1.8           # 1.8px deadzone for crisp lock with 0 shaking
        self.aim_smoothing_factor = 0.70  # EMA Smooth Filter (Smooth Radiant curve)

        # --- Recoil Control System (RCS) ---
        self.rcs_enabled = True           # Enable Recoil Control
        self.rcs_strength_y = 2.2         # Subtle vertical pull (First bullets one-tap, then pull)
        self.rcs_strength_x = 0.0         # Horizontal compensation
        self.rcs_delay_ms = 60            # 60ms first-bullet accuracy delay

        # --- Model and Detection ---
        self.models_dir = "models"
        self.model_path = os.path.join(self.models_dir, "Click here to Load a model")
        self.custom_player_label = "0"    # Class 0: Body
        self.custom_head_label = "1"      # Class 1: Head
        self.model_file_size = 0
        self.model_load_error = ""
        self.conf = 0.35                  # 0.35 clean confidence threshold
        self.imgsz = 640
        self.max_detect = 30
        
        # --- Mouse / Logitech Driver ---
        self.selected_mouse_button = 3   # Side 4 Mouse Button
        self.logitech_connected = False
        self.logitech_status_msg = "Disconnected"
        self.makcu_connected = False
        self.makcu_status_msg = "Disconnected"
        self.aim_humanization = 1
        self.in_game_sens = 1.15         # Sensitivity multiplier
        self.button_mask = False

        # --- Trigger Settings (Radiant Instant One-Tap) ---
        self.trigger_enabled         = True
        self.trigger_always_on       = False
        self.trigger_button          = 1        # Right Click or Side Key
        self.trigger_radius_px       = 7        # Tight 7px headshot zone
        self.trigger_delay_ms        = 18       # Sub-human 18ms assist delay
        self.trigger_cooldown_ms     = 160      # Cooldown for tap-fire rifles (Vandal/Phantom)
        self.trigger_min_conf        = 0.42     # High confidence for trigger

        # --- Aimbot Mode (Radiant Smooth Curves) ---
        self.mode = "bezier"             # Bezier curves for natural pro aim
        self.aimbot_running = False
        self.aimbot_status_msg = "Stopped"

        # --- Normal Aim ---
        self.normal_x_speed = 0.65
        self.normal_y_speed = 0.65

        # --- Bezier Aim ---
        self.bezier_segments = 10
        self.bezier_ctrl_x = 8.0
        self.bezier_ctrl_y = 8.0

        # --- Silent Aim ---
        self.silent_segments = 7
        self.silent_ctrl_x = 14.0
        self.silent_ctrl_y = 14.0
        self.silent_speed = 3
        self.silent_cooldown = 0.18

        # --- Smooth Aim (WindMouse Radiant Params) ---
        self.smooth_gravity = 12.0
        self.smooth_wind = 1.5
        self.smooth_min_delay = 0.0
        self.smooth_max_delay = 0.001
        self.smooth_max_step = 45.0
        self.smooth_min_step = 1.5
        self.smooth_max_step_ratio = 0.22
        self.smooth_target_area_ratio = 0.04
        self.smooth_reaction_min = 0.03
        self.smooth_reaction_max = 0.09
        self.smooth_close_range = 30
        self.smooth_far_range = 220
        self.smooth_close_speed = 0.85
        self.smooth_far_speed = 1.05
        self.smooth_acceleration = 1.25
        self.smooth_deceleration = 1.15
        self.smooth_fatigue_effect = 0.2
        self.smooth_micro_corrections = 0

        # --- Preview & HUD Overlays ---
        self.show_preview = True
        self.preview_fov = True
        self.preview_boxes = True
        self.preview_vectors = True
        self.show_debug_window = False

        # --- Real-Time Telemetry ---
        self.capture_fps = 0.0
        self.detection_latency = 0.0

        # --- Last error/status for GUI display ---
        self.last_error = ""
        self.last_info = ""

        # --- NDI Settings (Legacy/Optional) ---
        self.ndi_width = 0
        self.ndi_height = 0
        self.ndi_sources = []
        self.ndi_selected_source = None

    # -- Profile functions --
    def save(self, path="config_profile.json"):
        data = self.__dict__.copy()
        filtered = {k: v for k, v in data.items() if isinstance(v, (int, float, str, bool, list, dict))}
        with open(path, "w") as f:
            json.dump(filtered, f, indent=2)

    def load(self, path="config_profile.json"):
        if os.path.exists(path):
            with open(path, "r") as f:
                self.__dict__.update(json.load(f))

    def reset_to_defaults(self):
        self.__init__()

    # --- Utility ---
    def list_models(self):
        if not os.path.exists(self.models_dir):
            return []
        return [f for f in os.listdir(self.models_dir)
                if f.endswith(".engine") or f.endswith(".pt") or f.endswith(".onnx")]

config = Config()