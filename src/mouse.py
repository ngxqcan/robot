import os
import sys
import time
import ctypes
import threading

# Global state
driver = None
driver_lock = threading.Lock()
is_connected = False
button_states = {i: False for i in range(5)}
button_states_lock = threading.Lock()
_listener_running = False
_listener_thread = None

# Backward compatibility alias
makcu = None
makcu_lock = driver_lock

# Virtual Key Mapping for 1PC mouse button detection
# 0: Left, 1: Right, 2: Middle, 3: Side 4 (XBUTTON1), 4: Side 5 (XBUTTON2)
VK_MOUSE_MAP = {
    0: 0x01,  # VK_LBUTTON
    1: 0x02,  # VK_RBUTTON
    2: 0x04,  # VK_MBUTTON
    3: 0x05,  # VK_XBUTTON1
    4: 0x06,  # VK_XBUTTON2
}

# Function pointers from logitech.driver.dll
_device_open_fn = None
_moveR_fn = None
_mouse_down_fn = None
_mouse_up_fn = None
_click_fn = None
_device_close_fn = None


def find_driver_dll():
    """Locate logitech.driver.dll across common directories."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "logitech.driver.dll"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logitech.driver.dll"),
        os.path.join(os.getcwd(), "logitech.driver.dll"),
        os.path.join(os.getcwd(), "src", "logitech.driver.dll"),
        os.path.join(os.getcwd(), "driver", "logitech.driver.dll"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "driver", "logitech.driver.dll"),
        "logitech.driver.dll",
    ]
    for p in candidates:
        norm_path = os.path.abspath(p)
        if os.path.isfile(norm_path):
            return norm_path
    return "logitech.driver.dll"


def connect_to_logitech():
    """
    Load logitech.driver.dll and initialize Logitech driver connection for 1PC.
    """
    global driver, is_connected, makcu
    global _device_open_fn, _moveR_fn, _mouse_down_fn, _mouse_up_fn, _click_fn, _device_close_fn
    global _listener_running, _listener_thread

    dll_path = find_driver_dll()
    print(f"[INFO] Looking for Logitech driver DLL at: {dll_path}")

    try:
        if os.name == "nt":
            loaded_dll = ctypes.CDLL(dll_path)
        else:
            # Fallback for testing on non-Windows
            try:
                loaded_dll = ctypes.CDLL(dll_path)
            except Exception:
                print(f"[WARN] Non-Windows OS detected or DLL missing: {dll_path}")
                loaded_dll = None
    except Exception as e:
        print(f"[ERROR] Could not load Logitech driver DLL '{dll_path}': {e}")
        is_connected = False
        driver = None
        makcu = None
        return False

    if loaded_dll is None:
        is_connected = False
        driver = None
        makcu = None
        return False

    driver = loaded_dll
    makcu = driver

    # 1. Resolve device_open / init
    _device_open_fn = None
    for name in ["device_open", "device_initialize", "initialize", "init", "Init", "Open", "device_open_lghub"]:
        if hasattr(driver, name):
            _device_open_fn = getattr(driver, name)
            try:
                _device_open_fn.restype = ctypes.c_int
            except Exception:
                pass
            break

    # 2. Resolve moveR / move
    _moveR_fn = None
    for name in ["moveR", "move", "mouse_move", "move_relative", "MoveR", "Move"]:
        if hasattr(driver, name):
            _moveR_fn = getattr(driver, name)
            try:
                _moveR_fn.argtypes = [ctypes.c_int, ctypes.c_int]
                _moveR_fn.restype = ctypes.c_int
            except Exception:
                pass
            break

    # 3. Resolve mouse_down
    _mouse_down_fn = None
    for name in ["mouse_down", "MouseDown", "mouse_press"]:
        if hasattr(driver, name):
            _mouse_down_fn = getattr(driver, name)
            try:
                _mouse_down_fn.argtypes = [ctypes.c_int]
            except Exception:
                pass
            break

    # 4. Resolve mouse_up
    _mouse_up_fn = None
    for name in ["mouse_up", "MouseUp", "mouse_release"]:
        if hasattr(driver, name):
            _mouse_up_fn = getattr(driver, name)
            try:
                _mouse_up_fn.argtypes = [ctypes.c_int]
            except Exception:
                pass
            break

    # 5. Resolve click
    _click_fn = None
    for name in ["click", "Click", "mouse_click"]:
        if hasattr(driver, name):
            _click_fn = getattr(driver, name)
            break

    # 6. Resolve device_close
    _device_close_fn = None
    for name in ["device_close", "close", "Close", "device_cleanup"]:
        if hasattr(driver, name):
            _device_close_fn = getattr(driver, name)
            break

    # Initialize device if open function exists
    if _device_open_fn is not None:
        try:
            res = _device_open_fn()
            print(f"[INFO] Logitech device_open returned: {res}")
            # Usually 1 or >0 means success, or 0 on success in some implementations
            # If function executes without throwing, driver is loaded
        except Exception as e:
            print(f"[WARN] Error calling device_open: {e}")

    is_connected = True
    print("[INFO] Logitech driver connected successfully (1PC Mode).")

    # Start button listener thread for 1PC UI input monitoring
    if not _listener_running:
        _listener_running = True
        _listener_thread = threading.Thread(target=_listen_mouse_buttons, daemon=True)
        _listener_thread.start()

    return True


# Alias for backward compatibility
connect_to_makcu = connect_to_logitech


def _listen_mouse_buttons():
    """Background listener to update button_states dictionary on 1PC."""
    global _listener_running
    while _listener_running:
        try:
            if os.name == "nt":
                user32 = ctypes.windll.user32
                with button_states_lock:
                    for idx, vk in VK_MOUSE_MAP.items():
                        state = bool(user32.GetAsyncKeyState(vk) & 0x8000)
                        button_states[idx] = state
            time.sleep(0.01)  # 100Hz polling for GUI status
        except Exception:
            time.sleep(0.05)


def is_button_pressed(idx: int) -> bool:
    """
    Check if a mouse button is pressed on 1PC.
    0: Left, 1: Right, 2: Middle, 3: Side 4, 4: Side 5
    """
    if os.name == "nt":
        vk = VK_MOUSE_MAP.get(idx)
        if vk is not None:
            try:
                return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
            except Exception:
                pass
    with button_states_lock:
        return button_states.get(idx, False)


def test_move():
    """Perform a test movement with Logitech driver."""
    if not is_connected:
        print("[WARN] Logitech driver is not connected.")
        return
    try:
        with driver_lock:
            if _moveR_fn:
                _moveR_fn(50, 50)
                time.sleep(0.05)
                _moveR_fn(-50, -50)
                print("[INFO] Test move executed successfully.")
    except Exception as e:
        print(f"[ERROR] Test move failed: {e}")


def mask_manager_tick(selected_idx: int, aimbot_running: bool):
    """
    Compatibility wrapper for button masking.
    In 1PC with Logitech driver, raw driver injection does not require COM port masking.
    """
    pass


class Mouse:
    _instance = None
    _listener = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_inited"):
            if not connect_to_logitech():
                print("[ERROR] Mouse init: Failed to connect to logitech.driver.dll")
            self._inited = True

    def move(self, x: float, y: float):
        """Move mouse relative by dx, dy using Logitech driver."""
        if not is_connected or _moveR_fn is None:
            return
        dx, dy = int(round(x)), int(round(y))
        if dx == 0 and dy == 0:
            return
        with driver_lock:
            try:
                _moveR_fn(dx, dy)
            except Exception as e:
                print(f"[WARN] Logitech move error: {e}")

    def move_bezier(self, x: float, y: float, segments: int, ctrl_x: float, ctrl_y: float):
        """
        Bezier curve movement using Logitech driver relative steps.
        """
        if not is_connected or _moveR_fn is None:
            return

        segments = max(1, int(segments))
        target_x = float(x)
        target_y = float(y)
        cx = float(ctrl_x)
        cy = float(ctrl_y)

        last_px = 0.0
        last_py = 0.0

        for i in range(1, segments + 1):
            t = i / float(segments)
            # Quadratic Bezier: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2 (where P0 is 0,0)
            cur_px = (2.0 * (1.0 - t) * t * cx) + (t * t * target_x)
            cur_py = (2.0 * (1.0 - t) * t * cy) + (t * t * target_y)

            step_x = int(round(cur_px - last_px))
            step_y = int(round(cur_py - last_py))

            if step_x != 0 or step_y != 0:
                with driver_lock:
                    try:
                        _moveR_fn(step_x, step_y)
                    except Exception:
                        pass
                last_px += step_x
                last_py += step_y

            time.sleep(0.001)

    def click(self):
        """Send a left mouse click using Logitech driver."""
        if not is_connected:
            return
        with driver_lock:
            try:
                if _click_fn is not None:
                    try:
                        _click_fn()
                    except TypeError:
                        try:
                            _click_fn(1)
                        except Exception:
                            pass
                elif _mouse_down_fn is not None and _mouse_up_fn is not None:
                    _mouse_down_fn(1)  # 1 = Left button
                    time.sleep(0.015)
                    _mouse_up_fn(1)
            except Exception as e:
                print(f"[WARN] Logitech click error: {e}")

    def press(self, button_code: int = 1):
        """Press a mouse button (1=Left, 2=Right, 3=Middle)."""
        if not is_connected or _mouse_down_fn is None:
            return
        with driver_lock:
            try:
                _mouse_down_fn(int(button_code))
            except Exception as e:
                print(f"[WARN] Logitech press error: {e}")

    def release(self, button_code: int = 1):
        """Release a mouse button (1=Left, 2=Right, 3=Middle)."""
        if not is_connected or _mouse_up_fn is None:
            return
        with driver_lock:
            try:
                _mouse_up_fn(int(button_code))
            except Exception as e:
                print(f"[WARN] Logitech release error: {e}")

    @staticmethod
    def mask_manager_tick(selected_idx: int, aimbot_running: bool):
        mask_manager_tick(selected_idx, aimbot_running)

    @staticmethod
    def cleanup():
        global is_connected, driver, makcu, _listener_running
        _listener_running = False
        if _device_close_fn is not None and is_connected:
            try:
                _device_close_fn()
            except Exception:
                pass
        is_connected = False
        driver = None
        makcu = None
        Mouse._instance = None
        print("[INFO] Logitech driver cleaned up.")
