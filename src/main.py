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
_last_locked_entity = None
_smoothed_aim_point = None
_subpixel_carry_x = 0.0
_subpixel_carry_y = 0.0
_firing_start_ms = 0.0

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

        except Exception:
            time.sleep(0.01)

    try:
        camera.stop()
    except Exception:
        pass


def draw_corner_rect(img, pt1, pt2, color, thickness=2, corner_len=14):
    """Draw stylish corner brackets around bounding box."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    cl = min(corner_len, max(4, w // 3), max(4, h // 3))

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + cl, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + cl), color, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - cl, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - cl, y1), color, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + cl, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - cl), color, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - cl, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - cl), color, thickness, cv2.LINE_AA)


def draw_cyber_pill(img, text, center_pt, bg_color=(10, 10, 14), border_color=(34, 96, 255), text_color=(255, 255, 255), is_active=False):
    """Draw a modern pill tag with border above target (matching screenshot)."""
    cx, cy = center_pt
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thick = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

    pad_x = 6
    pad_y = 3
    x1 = max(2, cx - (tw // 2) - pad_x)
    x2 = min(img.shape[1] - 2, cx + (tw // 2) + pad_x)
    y2 = cy
    y1 = max(2, y2 - th - (pad_y * 2))

    if is_active:
        # Highlighted Active Target Pill
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (34, 96, 255), 2, cv2.LINE_AA)
    else:
        # Standard Target Pill
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

    cv2.putText(img, text, (x1 + pad_x, y2 - pad_y), font, scale, text_color, thick, cv2.LINE_AA)


def group_detections_into_entities(raw_detections):
    """
    Groups raw YOLO detections into coherent Bot Entities (Head + Body pairs).
    """
    heads = [d for d in raw_detections if d['type'] == "head"]
    bodies = [d for d in raw_detections if d['type'] == "player"]
    others = [d for d in raw_detections if d['type'] == "other"]

    entities = []
    used_heads = set()
    used_bodies = set()

    for bi, body in enumerate(bodies):
        bx1, by1, bx2, by2 = body['x1'], body['y1'], body['x2'], body['y2']
        bw = bx2 - bx1
        bh = by2 - by1

        matched_head = None
        min_dist_to_body_top = float('inf')

        for hi, head in enumerate(heads):
            if hi in used_heads:
                continue
            hx1, hy1, hx2, hy2 = head['x1'], head['y1'], head['x2'], head['y2']
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0

            if (bx1 - bw * 0.4) <= hcx <= (bx2 + bw * 0.4) and (by1 - bh * 0.5) <= hcy <= (by1 + bh * 0.6):
                d = abs(hcx - (bx1 + bx2) / 2.0) + abs(hcy - by1)
                if d < min_dist_to_body_top:
                    min_dist_to_body_top = d
                    matched_head = (hi, head)

        if matched_head is not None:
            hi, head = matched_head
            used_heads.add(hi)
            used_bodies.add(bi)
            entities.append({
                'has_head': True,
                'has_body': True,
                'head': head,
                'body': body,
                'head_center': ((head['x1'] + head['x2']) / 2.0, (head['y1'] + head['y2']) / 2.0),
                'body_center': ((body['x1'] + body['x2']) / 2.0, (body['y1'] + body['y2']) / 2.0),
                'top_y': min(head['y1'], body['y1']),
                'center_x': (head['x1'] + head['x2']) / 2.0,
                'conf_body': body['conf'],
                'conf_head': head['conf'],
                'cls_body': body['cls'],
                'cls_head': head['cls']
            })
        else:
            used_bodies.add(bi)
            entities.append({
                'has_head': False,
                'has_body': True,
                'head': None,
                'body': body,
                'head_center': None,
                'body_center': ((body['x1'] + body['x2']) / 2.0, (body['y1'] + body['y2']) / 2.0),
                'top_y': body['y1'],
                'center_x': (body['x1'] + body['x2']) / 2.0,
                'conf_body': body['conf'],
                'conf_head': None,
                'cls_body': body['cls'],
                'cls_head': None
            })

    for hi, head in enumerate(heads):
        if hi not in used_heads:
            entities.append({
                'has_head': True,
                'has_body': False,
                'head': head,
                'body': None,
                'head_center': ((head['x1'] + head['x2']) / 2.0, (head['y1'] + head['y2']) / 2.0),
                'body_center': None,
                'top_y': head['y1'],
                'center_x': (head['x1'] + head['x2']) / 2.0,
                'conf_body': None,
                'conf_head': head['conf'],
                'cls_body': None,
                'cls_head': head['cls']
            })

    for other in others:
        entities.append({
            'has_head': False,
            'has_body': False,
            'head': None,
            'body': other,
            'head_center': None,
            'body_center': ((other['x1'] + other['x2']) / 2.0, (other['y1'] + other['y2']) / 2.0),
            'top_y': other['y1'],
            'center_x': (other['x1'] + other['x2']) / 2.0,
            'conf_body': other['conf'],
            'conf_head': None,
            'cls_body': other['cls'],
            'cls_head': None
        })

    return entities


def detection_and_aim_loop():
    """CONSUMER: High-performance GPU inference, anti-shaking targeting, and recoil control."""
    global _aimbot_running, fps, driver, makcu
    global _last_locked_target_pos, _last_locked_entity, _smoothed_aim_point
    global _subpixel_carry_x, _subpixel_carry_y, _firing_start_ms

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

        raw_detections = []

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

                    if head_label and (head_label == class_name_str or head_label == cls_str or (len(head_label) > 1 and head_label in class_name_str)):
                        is_head = True
                    elif player_label and (player_label == class_name_str or player_label == cls_str or (len(player_label) > 1 and player_label in class_name_str)):
                        is_player = True
                    else:
                        if cls == 1 or "head" in class_name_str:
                            is_head = True
                        elif cls == 0 or "player" in class_name_str or "enemy" in class_name_str or "bot" in class_name_str:
                            is_player = True

                    target_type = "head" if is_head else ("player" if is_player else "other")

                    raw_detections.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'type': target_type,
                        'class': class_name,
                        'cls': cls,
                        'conf': conf
                    })

        # Group raw detections into Bot Entities
        entities = group_detections_into_entities(raw_detections)

        # Calculate Aim Point and Distance from Crosshair
        valid_aim_targets = []
        for ent in entities:
            aim_x = None
            aim_y = None

            if ent['has_head'] and getattr(config, "head_priority", True):
                aim_x, aim_y = ent['head_center']
                target_conf = ent['conf_head']
            elif ent['has_body']:
                bx1, by1, bx2, by2 = ent['body']['x1'], ent['body']['y1'], ent['body']['x2'], ent['body']['y2']
                aim_x = (bx1 + bx2) / 2.0
                aim_y = by1 + config.player_y_offset
                target_conf = ent['conf_body']
            elif ent['has_head']:
                aim_x, aim_y = ent['head_center']
                target_conf = ent['conf_head']

            if aim_x is not None and aim_y is not None:
                dist = math.hypot(aim_x - crop_center_x, aim_y - crop_center_y)
                ent['aim_x'] = aim_x
                ent['aim_y'] = aim_y
                ent['dist'] = dist
                ent['target_conf'] = target_conf

                if dist <= fov_radius:
                    valid_aim_targets.append(ent)
            else:
                ent['dist'] = float('inf')

        # Target Selection: Closest to Crosshair with Hysteresis
        best_target = None
        if valid_aim_targets:
            if config.target_lock_hysteresis and _last_locked_target_pos is not None:
                lx, ly = _last_locked_target_pos
                close_candidates = [t for t in valid_aim_targets if math.hypot(t['aim_x'] - lx, t['aim_y'] - ly) < 35]
                if close_candidates:
                    best_target = min(close_candidates, key=lambda t: t['dist'])
                else:
                    best_target = min(valid_aim_targets, key=lambda t: t['dist'])
            else:
                best_target = min(valid_aim_targets, key=lambda t: t['dist'])

        if best_target is not None:
            _last_locked_target_pos = (best_target['aim_x'], best_target['aim_y'])
        else:
            _last_locked_target_pos = None

        # --- Anti-Shaking EMA Filter ---
        if best_target is not None:
            raw_tx, raw_ty = best_target['aim_x'], best_target['aim_y']
            if _smoothed_aim_point is None or _last_locked_entity is not best_target:
                _smoothed_aim_point = (raw_tx, raw_ty)
            else:
                # Exponential Moving Average (EMA) smoothing to eliminate bounding box flicker
                smooth_factor = getattr(config, "aim_smoothing_factor", 0.60)
                alpha = max(0.15, min(1.0, 1.0 - (smooth_factor * 0.65)))
                sx = _smoothed_aim_point[0] * (1.0 - alpha) + raw_tx * alpha
                sy = _smoothed_aim_point[1] * (1.0 - alpha) + raw_ty * alpha
                _smoothed_aim_point = (sx, sy)
            _last_locked_entity = best_target
        else:
            _smoothed_aim_point = None
            _last_locked_entity = None

        # --- Recoil Control Calculation (RCS) ---
        now_ms = _now_ms()
        is_firing = is_button_pressed(0)  # Left Mouse Button / Shooting Key
        rcs_offset_x = 0.0
        rcs_offset_y = 0.0

        if getattr(config, "rcs_enabled", False):
            if is_firing:
                if _firing_start_ms == 0.0:
                    _firing_start_ms = now_ms
                
                # Check if firing duration exceeded RCS activation delay
                rcs_delay = getattr(config, "rcs_delay_ms", 45)
                if (now_ms - _firing_start_ms) >= rcs_delay:
                    rcs_offset_y = float(getattr(config, "rcs_strength_y", 2.8))
                    rcs_offset_x = float(getattr(config, "rcs_strength_x", 0.0))
            else:
                _firing_start_ms = 0.0
        else:
            _firing_start_ms = 0.0

        # --- Mouse Aim Execution ---
        button_held = is_button_pressed(config.selected_mouse_button)
        should_aim = (button_held or config.always_on_aim) and (_smoothed_aim_point is not None) and (active_driver is not None)

        if should_aim and _smoothed_aim_point is not None:
            aim_px, aim_py = _smoothed_aim_point
            target_screen_x = region_left + aim_px
            target_screen_y = region_top + aim_py

            raw_dx = target_screen_x - crosshair_x
            raw_dy = target_screen_y - crosshair_y

            # --- Anti-Shaking Deadzone & Progressive Damping ---
            dist_to_crosshair = math.hypot(raw_dx, raw_dy)
            deadzone = float(getattr(config, "aim_deadzone", 2.0))

            if dist_to_crosshair <= deadzone:
                # Inside deadzone: no aim movement to guarantee zero shaking!
                dx = 0.0
                dy = 0.0
            else:
                # Progressive Damping for close ranges (smooth glide into deadzone without oscillation)
                damping = 1.0
                if dist_to_crosshair < 14.0:
                    damping = (dist_to_crosshair - deadzone) / max(0.5, 14.0 - deadzone)
                
                dx = raw_dx * damping
                dy = raw_dy * damping

                # Sensitivity scaling
                sens = config.in_game_sens
                sens_multiplier = 1.07437623 * math.pow(sens, -0.9936827126)
                dx *= sens_multiplier
                dy *= sens_multiplier

            # Add RCS recoil compensation to Y axis
            dy += rcs_offset_y
            dx += rcs_offset_x

            # Subpixel accumulation for ultra smooth stepping
            dx += _subpixel_carry_x
            dy += _subpixel_carry_y

            if config.mode == "normal":
                dx *= config.normal_x_speed
                dy *= config.normal_y_speed
                
                step_x = int(round(dx))
                step_y = int(round(dy))
                _subpixel_carry_x = dx - step_x
                _subpixel_carry_y = dy - step_y

                if step_x != 0 or step_y != 0:
                    active_driver.move(step_x, step_y)

            elif config.mode == "bezier":
                _subpixel_carry_x = 0.0
                _subpixel_carry_y = 0.0
                active_driver.move_bezier(dx, dy, config.bezier_segments, config.bezier_ctrl_x, config.bezier_ctrl_y)

            elif config.mode == "silent":
                _subpixel_carry_x = 0.0
                _subpixel_carry_y = 0.0
                active_driver.move_bezier(dx, dy, config.silent_segments, config.silent_ctrl_x, config.silent_ctrl_y)

            elif config.mode == "smooth":
                _subpixel_carry_x = 0.0
                _subpixel_carry_y = 0.0
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
                if len(path) == 0 and (dx != 0 or dy != 0):
                    active_driver.move(int(round(dx)), int(round(dy)))

        elif getattr(config, "rcs_enabled", False) and is_firing and active_driver is not None and (rcs_offset_y != 0 or rcs_offset_x != 0):
            # Standalone Recoil Control when spraying without target lock
            active_driver.move(int(round(rcs_offset_x)), int(round(rcs_offset_y)))
            smooth_aimer.reset_fatigue()
            _subpixel_carry_x = 0.0
            _subpixel_carry_y = 0.0
        else:
            smooth_aimer.reset_fatigue()
            _subpixel_carry_x = 0.0
            _subpixel_carry_y = 0.0

        # --- Fast Triggerbot Logic ---
        try:
            if getattr(config, "trigger_enabled", False) and active_driver is not None:
                trigger_active = bool(getattr(config, "trigger_always_on", False))
                if not trigger_active:
                    trigger_btn_idx = int(getattr(config, "trigger_button", 0))
                    trigger_active = is_button_pressed(trigger_btn_idx)

                if trigger_active and valid_aim_targets:
                    min_conf = float(getattr(config, "trigger_min_conf", 0.35))
                    radius_px = int(getattr(config, "trigger_radius_px", 10))
                    delay_ms = int(getattr(config, "trigger_delay_ms", 25) * random.uniform(0.9, 1.1))
                    cooldown_ms = int(getattr(config, "trigger_cooldown_ms", 120) * random.uniform(0.9, 1.1))

                    candidates = [t for t in valid_aim_targets if (t['target_conf'] >= min_conf and t['dist'] <= radius_px)]

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

        # --- Cyber Live HUD Renderer (Matching Exact Screenshot) ---
        if config.show_preview or config.show_debug_window:
            hud_img = image.copy()
            ih, iw = hud_img.shape[:2]
            cx, cy = int(crop_center_x), int(crop_center_y)

            # 1. Draw Detections & Entities
            if getattr(config, "preview_boxes", True):
                for ent in entities:
                    is_active_target = (ent is best_target)

                    # Draw Head Box (Neon Green)
                    if ent['has_head']:
                        hx1, hy1, hx2, hy2 = ent['head']['x1'], ent['head']['y1'], ent['head']['x2'], ent['head']['y2']
                        cv2.rectangle(hud_img, (hx1, hy1), (hx2, hy2), (118, 230, 0), 2, cv2.LINE_AA)

                    # Draw Body Box (Orange Corner Brackets)
                    if ent['has_body']:
                        bx1, by1, bx2, by2 = ent['body']['x1'], ent['body']['y1'], ent['body']['x2'], ent['body']['y2']
                        draw_corner_rect(hud_img, (bx1, by1), (bx2, by2), (34, 96, 255), thickness=2, corner_len=14)

                    # Draw Pill Header: `[0] 82% | [1] 81%`
                    pill_parts = []
                    if ent['has_body']:
                        pill_parts.append(f"[{ent['cls_body']}] {int(ent['conf_body'] * 100)}%")
                    if ent['has_head']:
                        pill_parts.append(f"[{ent['cls_head']}] {int(ent['conf_head'] * 100)}%")
                    
                    if pill_parts:
                        pill_text = " | ".join(pill_parts)
                        pill_cx = int(ent['center_x'])
                        pill_cy = int(ent['top_y'] - 6)
                        draw_cyber_pill(
                            hud_img, pill_text, (pill_cx, pill_cy),
                            border_color=(34, 96, 255) if is_active_target else (100, 100, 120),
                            is_active=is_active_target
                        )

            # 2. Draw Center Crosshair: Cyan Circle with Center Dot
            if getattr(config, "preview_fov", True):
                cv2.circle(hud_img, (cx, cy), 12, (255, 229, 0), 2, cv2.LINE_AA)
                cv2.circle(hud_img, (cx, cy), 2, (255, 229, 0), -1, cv2.LINE_AA)

            # 3. Draw Aim Vector Line & Target Lock Marker (Matching Screenshot)
            if getattr(config, "preview_vectors", True) and best_target is not None:
                tx = int(best_target['aim_x'])
                ty = int(best_target['aim_y'])

                # Cyan Aim Vector connecting center crosshair circle to target head
                cv2.line(hud_img, (cx, cy), (tx, ty), (255, 229, 0), 2, cv2.LINE_AA)

                # Pink / Magenta Target Lock Marker (#ff007f -> BGR: 127, 0, 255)
                cv2.circle(hud_img, (tx, ty), 6, (127, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(hud_img, (tx, ty), 8, (255, 255, 255), 1, cv2.LINE_AA)
                # Inner crosshair in pink target dot
                cv2.line(hud_img, (tx - 4, ty), (tx + 4, ty), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(hud_img, (tx, ty - 4), (tx, ty + 4), (255, 255, 255), 1, cv2.LINE_AA)

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
