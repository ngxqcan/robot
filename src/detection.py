import os
import re
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


def get_model_native_imgsz(model, model_path=""):
    """Extract native / expected image size from loaded model or filename."""
    if model is None and not model_path:
        return None

    # 1. Try AutoBackend session inputs for ONNX / Engine
    try:
        if hasattr(model, "model") and hasattr(model.model, "session") and model.model.session is not None:
            inputs = model.model.session.get_inputs()
            if inputs and len(inputs) > 0:
                shape = inputs[0].shape
                if len(shape) >= 4 and isinstance(shape[2], int) and shape[2] > 0:
                    return int(shape[2])
    except Exception:
        pass

    # 2. Try AutoBackend imgsz attribute
    try:
        if hasattr(model, "model") and hasattr(model.model, "imgsz") and model.model.imgsz:
            ims = model.model.imgsz
            if isinstance(ims, (list, tuple)) and len(ims) > 0 and isinstance(ims[0], int) and ims[0] > 0:
                return int(ims[0])
            elif isinstance(ims, int) and ims > 0:
                return int(ims)
    except Exception:
        pass

    # 3. Try overrides
    try:
        if hasattr(model, "overrides") and model.overrides.get("imgsz"):
            ims = model.overrides["imgsz"]
            if isinstance(ims, (list, tuple)) and len(ims) > 0 and isinstance(ims[0], int) and ims[0] > 0:
                return int(ims[0])
            elif isinstance(ims, (int, float)) and ims > 0:
                return int(ims)
    except Exception:
        pass

    # 4. Check filename pattern like *_256.onnx, *_320.onnx, etc.
    p = model_path or (getattr(model, "model_path", "") if hasattr(model, "model_path") else "")
    if p:
        base = os.path.basename(str(p))
        match = re.search(r'[_x\-](128|160|192|224|256|288|320|384|416|480|512|576|640|736|800|960|1024|1280)\b', base)
        if match:
            return int(match.group(1))

    return None


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

        # Auto-detect and sync imgsz for fixed-dimension models (e.g. 256x256 ONNX)
        detected_size = get_model_native_imgsz(_model, model_path)
        if detected_size is not None and detected_size > 0:
            config.imgsz = detected_size
            print(f"[INFO] Auto-detected model input resolution: {config.imgsz}x{config.imgsz}")

        # Model GPU Warm-up pass to eliminate first-frame latency stutter
        try:
            target_size = detected_size or config.imgsz
            dummy_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            _is_half = bool(DEVICE == 0 and not str(model_path).endswith(".onnx"))
            if TORCH_AVAILABLE:
                with torch.inference_mode():
                    _model.predict(
                        source=dummy_img,
                        imgsz=target_size,
                        device=DEVICE,
                        half=_is_half,
                        verbose=False,
                        conf=0.25
                    )
            else:
                _model.predict(
                    source=dummy_img,
                    imgsz=target_size,
                    device=DEVICE,
                    verbose=False,
                    conf=0.25
                )
        except Exception as warmup_err:
            print(f"[WARN] Model warm-up skipped: {warmup_err}")

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

    target_imgsz = getattr(config, "imgsz", 640)
    native_size = get_model_native_imgsz(model)
    if native_size is not None and native_size > 0:
        target_imgsz = native_size

    t0 = time.perf_counter()
    try:
        if TORCH_AVAILABLE:
            with torch.inference_mode():
                results = model.predict(
                    source=image,
                    imgsz=target_imgsz,
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
                imgsz=target_imgsz,
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
    except Exception as e:
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
