import os
import time
import numpy as np
from config import config

try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE = 0
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        DEVICE = "cpu"
except Exception:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

_model = None
_class_names = {}
_is_half = (DEVICE == 0)


def load_model(model_path=None):
    global _model, _class_names, _is_half
    if model_path is None:
        model_path = config.model_path

    if not os.path.exists(model_path):
        config.model_load_error = f"Model file not found: {model_path}"
        _model = None
        _class_names = {}
        return None, {}

    try:
        _model = YOLO(model_path, task="detect")
        
        # Extract class names
        if hasattr(_model, "names") and _model.names:
            _class_names = _model.names
        elif hasattr(_model, "model") and hasattr(_model.model, "names") and _model.model.names:
            _class_names = _model.model.names
        else:
            _class_names = {}

        config.model_classes = list(_class_names.values())
        config.model_file_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        config.model_load_error = ""

        # Model GPU Warm-up pass to eliminate first-frame latency stutter
        try:
            dummy_img = np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8)
            _is_half = bool(DEVICE == 0 and not str(model_path).endswith(".onnx"))
            if TORCH_AVAILABLE:
                with torch.inference_mode():
                    _model.predict(
                        source=dummy_img,
                        imgsz=config.imgsz,
                        device=DEVICE,
                        half=_is_half,
                        verbose=False,
                        conf=0.25
                    )
            else:
                _model.predict(
                    source=dummy_img,
                    imgsz=config.imgsz,
                    device=DEVICE,
                    verbose=False,
                    conf=0.25
                )
        except Exception:
            pass

        return _model, _class_names

    except Exception as e:
        config.model_load_error = f"Failed to load model: {e}"
        _model = None
        _class_names = {}
        return None, {}


def reload_model(model_path):
    return load_model(model_path)


def perform_detection(model, image):
    """
    High-performance batched inference with precision timing.
    """
    if model is None or image is None:
        return None

    t0 = time.perf_counter()
    try:
        if TORCH_AVAILABLE:
            with torch.inference_mode():
                results = model.predict(
                    source=image,
                    imgsz=config.imgsz,
                    stream=True,
                    conf=config.conf,
                    iou=0.45,
                    device=DEVICE,
                    half=_is_half,
                    max_det=config.max_detect,
                    agnostic_nms=False,
                    augment=False,
                    vid_stride=False,
                    visualize=False,
                    verbose=False,
                    show_boxes=False,
                    show_labels=False,
                    show_conf=False,
                    save=False,
                    show=False
                )
        else:
            results = model.predict(
                source=image,
                imgsz=config.imgsz,
                stream=True,
                conf=config.conf,
                iou=0.45,
                device=DEVICE,
                max_det=config.max_detect,
                agnostic_nms=False,
                augment=False,
                vid_stride=False,
                visualize=False,
                verbose=False,
                show_boxes=False,
                show_labels=False,
                show_conf=False,
                save=False,
                show=False
            )
        
        t1 = time.perf_counter()
        config.detection_latency = (t1 - t0) * 1000.0
        return results
    except Exception:
        return None


def get_class_names():
    return _class_names


def get_model_size(model_path=None):
    if not model_path:
        model_path = config.model_path
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        if size_bytes > 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        return f"{size_bytes / 1024:.1f} KB"
    return "0 KB"
