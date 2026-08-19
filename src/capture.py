import time
import numpy as np
import mss
import cv2
import dxcam
from config import config

# NDI imports
try:
    from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
    from cyndilib.finder import Finder
    from cyndilib.receiver import Receiver
    from cyndilib.video_frame import VideoFrameSync
    from cyndilib.audio_frame import AudioFrameSync
    NDI_AVAILABLE = True
except Exception:
    NDI_AVAILABLE = False


def get_region():
    """Center capture region for 1PC mode (MSS / DXGI)."""
    left = (config.screen_width - config.region_size) // 2
    top = (config.screen_height - config.region_size) // 2
    right = left + config.region_size
    bottom = top + config.region_size
    return (int(left), int(top), int(right), int(bottom))


class MSSCamera:
    def __init__(self, region):
        self.region = region
        self.sct = mss.mss()
        self.monitor = {
            "top": region[1],
            "left": region[0],
            "width": region[2] - region[0],
            "height": region[3] - region[1],
        }
        self.running = True
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()

    def get_latest_frame(self):
        try:
            img = np.array(self.sct.grab(self.monitor))
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Update capture FPS
            self._frame_count += 1
            now = time.perf_counter()
            elapsed = now - self._last_fps_time
            if elapsed >= 0.5:
                config.capture_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

            return img
        except Exception:
            return None

    def stop(self):
        self.running = False
        try:
            self.sct.close()
        except Exception:
            pass


class DXGICamera:
    def __init__(self, region=None, target_fps=None):
        self.region = region
        # dxcam handles direct BGR color output at hardware level
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        fps = int(getattr(config, "target_fps", 240) if target_fps is None else target_fps)
        
        # Start DXCAM high-speed hardware capture thread with ROI
        try:
            if self.region:
                self.camera.start(target_fps=fps, region=self.region)
            else:
                self.camera.start(target_fps=fps)
        except Exception:
            self.camera.start(target_fps=fps)

        self.running = True
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()

    def get_latest_frame(self):
        frame = self.camera.get_latest_frame()
        if frame is None:
            return None
        
        # If DXCAM captured full screen fallback, crop to region
        if frame.shape[0] != config.region_size or frame.shape[1] != config.region_size:
            if self.region:
                x1, y1, x2, y2 = self.region
                frame = frame[y1:y2, x1:x2]

        # Update capture FPS
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 0.5:
            config.capture_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

        return frame

    def stop(self):
        self.running = False
        try:
            self.camera.stop()
        except Exception:
            pass


class NDICamera:
    def __init__(self):
        if not NDI_AVAILABLE:
            raise RuntimeError("NDI library is not available.")
        self.finder = Finder()
        self.finder.set_change_callback(self.on_finder_change)
        self.finder.open()

        self.receiver = Receiver(
            color_format=RecvColorFormat.RGBX_RGBA,
            bandwidth=RecvBandwidth.highest,
        )
        self.video_frame = VideoFrameSync()
        self.audio_frame = AudioFrameSync()
        self.receiver.frame_sync.set_video_frame(self.video_frame)
        self.receiver.frame_sync.set_audio_frame(self.audio_frame)

        self.available_sources = []     
        self.desired_source_name = None
        self._pending_index = None
        self._pending_connect = False
        self._last_connect_try = 0.0
        self._retry_interval = 0.5

        self.connected = False
        self._source_name = None
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()

        try:
            self.available_sources = self.finder.get_source_names() or []
        except Exception:
            self.available_sources = []

    def select_source(self, name_or_index):
        if self.available_sources is None:
            self.available_sources = []

        self._pending_connect = True
        if isinstance(name_or_index, int):
            self._pending_index = name_or_index
            if 0 <= name_or_index < len(self.available_sources):
                self.desired_source_name = self.available_sources[name_or_index]
            else:
                return
        else:
            self.desired_source_name = str(name_or_index)

        if self.desired_source_name in self.available_sources:
            self._try_connect_throttled()

    def on_finder_change(self):
        self.available_sources = self.finder.get_source_names() or []
        if self._pending_index is not None and 0 <= self._pending_index < len(self.available_sources):
            self.desired_source_name = self.available_sources[self._pending_index]
        if self._pending_connect and not self.connected and self.desired_source_name in self.available_sources:
            self._try_connect_throttled()

    def _try_connect_throttled(self):
        now = time.time()
        if now - self._last_connect_try < self._retry_interval:
            return
        self._last_connect_try = now
        if self.desired_source_name:
            self.connect_to_source(self.desired_source_name)

    def connect_to_source(self, source_name):
        source = self.finder.get_source(source_name)
        if not source:
            return
        self.receiver.set_source(source)
        self._source_name = source.name
        for _ in range(100):
            if self.receiver.is_connected():
                self.connected = True
                self._pending_connect = False
                break
            time.sleep(0.01)
        else:
            self.connected = False

    def list_sources(self, refresh=True):
        if refresh:
            try:
                self.available_sources = self.finder.get_source_names() or []
            except Exception:
                self.available_sources = self.available_sources or []
        return list(self.available_sources)

    def get_latest_frame(self):
        if not self.receiver.is_connected():
            time.sleep(0.002)
            return None

        self.receiver.frame_sync.capture_video()
        if min(self.video_frame.xres, self.video_frame.yres) == 0:
            time.sleep(0.002)
            return None
        config.ndi_width, config.ndi_height = self.video_frame.xres, self.video_frame.yres

        frame = np.frombuffer(self.video_frame, dtype=np.uint8).copy()
        frame = frame.reshape((self.video_frame.yres, self.video_frame.xres, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 0.5:
            config.capture_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

        return frame

    def stop(self):
        try:
            try: self.receiver.set_source(None)
            except Exception: pass
            self.finder.close()
        except Exception:
            pass


def get_camera():
    """Factory function to return the right camera based on config."""
    mode = config.capturer_mode.lower()
    if mode == "mss":
        region = get_region()
        cam = MSSCamera(region)
        return cam, region
    elif mode == "ndi":
        cam = NDICamera()
        return cam, None
    elif mode == "dxgi":
        region = get_region()
        cam = DXGICamera(region)
        return cam, region
    else:
        # Fallback to DXGI
        region = get_region()
        cam = DXGICamera(region)
        return cam, region