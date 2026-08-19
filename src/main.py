import numpy as np
import time
import threading
import os
import math
import cv2
import queue
import random

from mouse import Mouse, is_button_pressed, test_move
from capture import get_camera
from detection import load_model, perform_detection, get_model_size, get_class_names
from config import config
from windmouse_smooth import smooth_aimer

# --- Global state for aimbot control ---
_aimbot_running = False
_aimbot_thread = None
_capture_thread = None
_smooth_thread = None
fps = 0.0

frame_queue = queue.Queue(maxsize=1)
smooth_move_queue = queue.Queue(maxsize=10)
driver = None  # Mouse driver instance
makcu = None   # Backward compatibility alias

_last_trigger_time_ms = 0.0
_in_zone_since_ms = 0.0
_last_locked_target_pos = None

# --- Thread-Safe Preview Buffer ---
_preview_frame_lock = threading.Lock()
_latest_preview_frame = None


def get_latest_preview_frame():
    """Retrieve the latest annotated cyber HUD preview frame for the GUI."""
    global _latest_preview_frame
    with _preview_frame_lock:
        if _latest_preview_frame is not None:
            return _latest_preview_frame.copy()
        return None


def set_latest_preview_frame(frame):
    """Update the latest annotated preview frame."""
    global _latest_preview_frame
    with _preview_frame_lock:
        _latest_preview_frame = frame


def smooth_movement_loop():
    """
    Dedicated thread for executing smooth mouse movements with micro-second precision.
    """
    global _aimbot_running, driver, makcu
    active_driver = driver or makcu
    while _aimbot_running:
        try:
            move_data = smooth_move_queue.get(timeout=0.05)
            dx, dy, delay = move_data
            if active_driver is not None:
                active_driver.move(dx, dy)
            if delay > 0:
                time.sleep(delay)
        except queue.Empty:
            continue
        except Exception:
            time.sleep(0.005)


def _now_ms():
    return time.perf_counter() * 1000.0


def capture_loop():
    """PRODUCER: Ultra-fast frame capture loop on dedicated thread."""
    global _aimbot_running
    camera, _ = get_camera()
    last_selected = None

    while _aimbot_running:
        try:
            if hasattr(camera, "list_sources"):
                try:
                    config.ndi_sources = camera.list_sources(refresh=False)
                except Exception:
                    config.ndi_sources = []

            if config.capturer_mode.lower() == "ndi":
                desired = config.ndi_selected_source
                if isinstance(desired, str) and desired in config.ndi_sources:
                    if (desired != last_selected) or not camera.connected:
                        camera.select_source(desired)
                        last_selected = desired

            image = camera.get_latest_frame()
            if image is not None:
                try:
                    frame_queue.put(image, block=False)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put(image, block=False)
                    except queue.Full:
                        pass
            else:
                time.sleep(0.001)

        except Exception as e:
            time.sleep(0.01)

    try:
        camera.stop()
    except Exception:
        pass


def draw_corner_rect(img, pt1, pt2, color, thickness=2, corner_len=12):
    """Draw stylish corner brackets around bounding box."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    cl = min(corner_len, w // 2, h // 2)

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + cl, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + cl), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - cl, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + cl), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + cl, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - cl), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - cl, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - cl), color, thickness)


def draw_cyber_pill(img, text, center_pt, bg_color=(20, 20, 24), border_color=(34, 96, 255), text_color=(240, 240, 240)):
    """Draw a modern pill tag with border above target."""
    cx, cy = center_pt
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

    pad_x = 8
    pad_y = 4
    x1 = cx - (tw // 2) - pad_x
    y1 = cy - th - pad_y
    x2 = cx + (tw // 2) + pad_x
    y2 = cy + pad_y

    # Clamp
    h, w = img.shape[:2]
    if y1 < 2:
        y1 = 2
        y2 = y1 + th + 2 * pad_y

    # Draw rounded-like pill background
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)
    cv2.putText(img, text, (x1 + pad_x, y2 - pad_y - baseline // 2 + 2), font, scale, text_color, thick, cv2.LINE_AA)


def detection_and_aim_loop():
    """CONSUMER: High-performance GPU inference, target tracking, and Cyber HUD renderer."""
    global _aimbot_running, fps, driver, makcu, _last_locked_target_pos
    model, class_names = load_model(config.model_path)
    active_driver = driver or makcu

    frame_count = 0
    start_time = time.perf_counter()
    debug_window_moved = False

    while _aimbot_running:
        try:
            image = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Region coordinates calculation
        if config.capturer_mode.lower() in ("mss", "dxgi"):
            region_left = (config.screen_width - config.region_size) // 2
            region_top  = (config.screen_height - config.region_size) // 2
            crosshair_x = config.screen_width // 2
            crosshair_y = config.screen_height // 2
            crop_center_x = config.region_size / 2.0
            crop_center_y = config.region_size / 2.0
            fov_radius = config.region_size / 2.0
        else:
            region_left = (config.main_pc_width - config.ndi_width) // 2
            region_top  = (config.main_pc_height - config.ndi_height) // 2
            crosshair_x = config.main_pc_width // 2
            crosshair_y = config.main_pc_height // 2
            crop_center_x = config.ndi_width / 2.0
            crop_center_y = config.ndi_height / 2.0
            fov_radius = min(config.ndi_width, config.ndi_height) / 2.0

        if config.button_mask:
            Mouse.mask_manager_tick(selected_idx=config.selected_mouse_button, aimbot_running=is_aimbot_running())
            Mouse.mask_manager_tick(selected_idx=config.trigger_button, aimbot_running=is_aimbot_running())
        else:
            Mouse.mask_manager_tick(selected_idx=config.selected_mouse_button, aimbot_running=False)
            Mouse.mask_manager_tick(selected_idx=config.trigger_button, aimbot_running=False)

        # Run AI detection
        results = perform_detection(model, image)

        all_targets = []
        head_targets = []
        body_targets = []
        detected_boxes_info = []

        if results:
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    coords = [val.item() for val in box.xyxy[0]]
                    if any(math.isnan(c) for c in coords):
                        continue

                    x1, y1, x2, y2 = [int(c) for c in coords]
                    conf = float(box.conf[0].item())
                    cls = int(box.cls[0].item())
                    class_name = class_names.get(cls, f"class_{cls}")

                    is_player = False
                    is_head = False

                    player_label = str(config.custom_player_label).lower() if config.custom_player_label else ""
                    head_label = str(config.custom_head_label).lower() if config.custom_head_label else ""
                    class_name_str = str(class_name).lower()
                    cls_str = str(cls)

                    # Check head match
                    if head_label and (head_label == class_name_str or head_label == cls_str or (len(head_label) > 1 and head_label in class_name_str)):
                        is_head = True
                    # Check player match
                    elif player_label and (player_label == class_name_str or player_label == cls_str or (len(player_label) > 1 and player_label in class_name_str)):
                        is_player = True

                    target_type = "head" if is_head else ("player" if is_player else "other")
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    if is_player:
                        # Player offset
                        center_y = y1 + config.player_y_offset

                    dist = math.hypot(center_x - crop_center_x, center_y - crop_center_y)

                    target_entry = {
                        'dist': dist,
                        'center_x': center_x,
                        'center_y': center_y,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'type': target_type,
                        'class': class_name,
                        'cls': cls,
                        'conf': conf
                    }

                    detected_boxes_info.append(target_entry)

                    if is_head or is_player:
                        # Only target if within FOV circle
                        if dist <= fov_radius:
                            if is_head:
                                head_targets.append(target_entry)
                            else:
                                body_targets.append(target_entry)
                            all_targets.append(target_entry)

        # Smart Head Priority Target Selection
        best_target = None
        if getattr(config, "head_priority", True) and head_targets:
            best_target = min(head_targets, key=lambda t: t['dist'])
        elif all_targets:
            # Target hysteresis: if we were tracking a target nearby, prefer it
            if config.target_lock_hysteresis and _last_locked_target_pos is not None:
                lx, ly = _last_locked_target_pos
                close_candidates = [t for t in all_targets if math.hypot(t['center_x'] - lx, t['center_y'] - ly) < 30]
                if close_candidates:
                    best_target = min(close_candidates, key=lambda t: t['dist'])
                else:
                    best_target = min(all_targets, key=lambda t: t['dist'])
            else:
                best_target = min(all_targets, key=lambda t: t['dist'])

        if best_target is not None:
            _last_locked_target_pos = (best_target['center_x'], best_target['center_y'])
        else:
            _last_locked_target_pos = None

        # --- Mouse Aim Execution ---
        button_held = is_button_pressed(config.selected_mouse_button)
        should_aim = (button_held or config.always_on_aim) and (best_target is not None) and (active_driver is not None)

        if should_aim and best_target:
            target_screen_x = region_left + best_target['center_x']
            target_screen_y = region_top + best_target['center_y']

            dx = target_screen_x - crosshair_x
            dy = target_screen_y - crosshair_y

            # Sensitivity multiplier
            sens = config.in_game_sens
            distance = 1.07437623 * math.pow(sens, -0.9936827126)
            dx *= distance
            dy *= distance

            if config.mode == "normal":
                dx *= config.normal_x_speed
                dy *= config.normal_y_speed
                active_driver.move(dx, dy)
            elif config.mode == "bezier":
                active_driver.move_bezier(dx, dy, config.bezier_segments, config.bezier_ctrl_x, config.bezier_ctrl_y)
            elif config.mode == "silent":
                active_driver.move_bezier(dx, dy, config.silent_segments, config.silent_ctrl_x, config.silent_ctrl_y)
            elif config.mode == "smooth":
                path = smooth_aimer.calculate_smooth_path(dx, dy, config)
                for move_dx, move_dy, delay in path:
                    if not smooth_move_queue.full():
                        smooth_move_queue.put((move_dx, move_dy, delay))
                    else:
                        try:
                            while not smooth_move_queue.empty():
                                smooth_move_queue.get_nowait()
                        except queue.Empty:
                            pass
                        smooth_move_queue.put((move_dx, move_dy, delay))
                        break
                if len(path) == 0:
                    active_driver.move(dx, dy)
        else:
            smooth_aimer.reset_fatigue()

        # --- Fast Triggerbot Logic ---
        try:
            if getattr(config, "trigger_enabled", False) and active_driver is not None:
                trigger_active = bool(getattr(config, "trigger_always_on", False))
                if not trigger_active:
                    trigger_btn_idx = int(getattr(config, "trigger_button", 0))
                    trigger_active = is_button_pressed(trigger_btn_idx)

                if trigger_active and all_targets:
                    min_conf = float(getattr(config, "trigger_min_conf", 0.35))
                    radius_px = int(getattr(config, "trigger_radius_px", 10))
                    delay_ms = int(getattr(config, "trigger_delay_ms", 25) * random.uniform(0.9, 1.1))
                    cooldown_ms = int(getattr(config, "trigger_cooldown_ms", 120) * random.uniform(0.9, 1.1))

                    candidates = [t for t in all_targets if (t['conf'] >= min_conf and t['dist'] <= radius_px)]

                    now = _now_ms()
                    global _in_zone_since_ms, _last_trigger_time_ms

                    if candidates:
                        if _in_zone_since_ms == 0.0:
                            _in_zone_since_ms = now

                        linger_ok = (now - _in_zone_since_ms) >= delay_ms
                        cooldown_ok = (now - _last_trigger_time_ms) >= cooldown_ms

                        if linger_ok and cooldown_ok:
                            try:
                                active_driver.click()
                            except Exception:
                                pass
                            _last_trigger_time_ms = now
                            _in_zone_since_ms = 0.0
                    else:
                        _in_zone_since_ms = 0.0
                else:
                    _in_zone_since_ms = 0.0
        except Exception:
            pass

        # --- Cyber Live HUD Renderer (In-App Preview & Debug Window) ---
        if config.show_preview or config.show_debug_window:
            hud_img = image.copy()
            ih, iw = hud_img.shape[:2]
            cx, cy = int(crop_center_x), int(crop_center_y)

            # 1. Draw Cyan FOV circle
            if getattr(config, "preview_fov", True):
                fov_color = (255, 229, 0)  # BGR Cyan (#00e5ff)
                cv2.circle(hud_img, (cx, cy), int(fov_radius), fov_color, 1, cv2.LINE_AA)

            # 2. Draw Detections with Cyber Styling
            if getattr(config, "preview_boxes", True):
                for box in detected_boxes_info:
                    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                    conf_pct = int(box['conf'] * 100)
                    cls_idx = box['cls']
                    btype = box['type']

                    if btype == "head":
                        # Green Head Box (#00e676 -> BGR: 118, 230, 0)
                        cv2.rectangle(hud_img, (x1, y1), (x2, y2), (118, 230, 0), 2, cv2.LINE_AA)
                    elif btype == "player":
                        # Orange Corner Brackets (#ff6022 -> BGR: 34, 96, 255)
                        draw_corner_rect(hud_img, (x1, y1), (x2, y2), (34, 96, 255), thickness=2, corner_len=14)
                    else:
                        # Neutral Box
                        cv2.rectangle(hud_img, (x1, y1), (x2, y2), (160, 160, 160), 1)

                    # Pill Badge: `[0] 92% | [1] 90%`
                    pill_text = f"[{cls_idx}] {conf_pct}%"
                    mid_x = (x1 + x2) // 2
                    draw_cyber_pill(hud_img, pill_text, (mid_x, y1 - 4), border_color=(34, 96, 255) if btype != "head" else (118, 230, 0))

            # 3. Draw Target Lock Point & Vector Line
            if getattr(config, "preview_vectors", True) and best_target is not None:
                tx = int(best_target['center_x'])
                ty = int(best_target['center_y'])
                # Pink / Magenta Target Lock (#ff007f -> BGR: 127, 0, 255)
                cv2.circle(hud_img, (tx, ty), 6, (127, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(hud_img, (tx, ty), 9, (255, 255, 255), 1, cv2.LINE_AA)
                # Cyan connection vector line
                cv2.line(hud_img, (cx, cy), (tx, ty), (255, 229, 0), 2, cv2.LINE_AA)

            # 4. Center Crosshair Dot
            cv2.circle(hud_img, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

            # Store for in-app Preview tab
            set_latest_preview_frame(hud_img)

            # Optional Separate OpenCV Debug Window
            if config.show_debug_window:
                win_name = "CapkfaPlus Live Debug"
                cv2.imshow(win_name, hud_img)
                if not debug_window_moved:
                    cv2.moveWindow(win_name, (config.screen_width - iw) // 2, (config.screen_height - ih) // 2)
                    debug_window_moved = True
                cv2.waitKey(1)

        # FPS Calculation
        frame_count += 1
        elapsed = time.perf_counter() - start_time
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            start_time = time.perf_counter()
            frame_count = 0


def start_aimbot():
    """Start all aimbot threads."""
    global _aimbot_running, _aimbot_thread, _capture_thread, _smooth_thread
    if _aimbot_running:
        return
    _aimbot_running = True
    config.aimbot_running = True
    config.aimbot_status_msg = "Running"

    _capture_thread = threading.Thread(target=capture_loop, daemon=True, name="CaptureThread")
    _capture_thread.start()

    _aimbot_thread = threading.Thread(target=detection_and_aim_loop, daemon=True, name="DetectionThread")
    _aimbot_thread.start()

    _smooth_thread = threading.Thread(target=smooth_movement_loop, daemon=True, name="SmoothThread")
    _smooth_thread.start()


def stop_aimbot():
    """Stop all aimbot threads."""
    global _aimbot_running
    _aimbot_running = False
    config.aimbot_running = False
    config.aimbot_status_msg = "Stopped"
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def is_aimbot_running():
    return _aimbot_running


def get_model_classes():
    return get_class_names()
