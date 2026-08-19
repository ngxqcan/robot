# EVENTURI-AI (1PC Logitech Driver)

The ultimate AI aimbot and detection GUI for Windows (1PC setup), supporting YOLOv8–v12 models and Logitech Driver (`logitech.driver.dll`).
Features custom class selection for multiple games and super-smooth, modern UI.

---
## Disclaimer
This program is intended for 1PC setup using Logitech driver injection.
I am not responsible for any account bans, penalties, or any other consequences that may result from using this program.
Use it at your own risk and be aware of the potential implications.

---
## Features

- **1PC Architecture**: High speed, ultra-low latency capture (DXGI / MSS) directly on your main machine.
- **Logitech Driver Support**: Direct mouse injection via `logitech.driver.dll` (G HUB / Logitech Gaming Software).
- **YOLOv8–v12 Support**: PyTorch (`.pt`), ONNX (`.onnx`), TensorRT (`.engine`).
- **Aim Modes**: Normal, Bezier, Silent, and WindMouse Smooth aim.
- **Triggerbot**: Integrated triggerbot with customizable radius, delay, cooldown, and confidence threshold.
- **Profile System**: Save, load, and reset your custom configurations.
- **Modern Dark GUI**: Built with CustomTkinter for a responsive and clean layout.
- **Dual Acceleration**: DirectML (AMD/Intel/NVIDIA) and CUDA 12.6 support (NVIDIA).

---

## Installation & Requirements

### 1. Requirements
- Windows 10/11
- Logitech G HUB (or Logitech Gaming Software) installed and running.
- Place `logitech.driver.dll` in the project folder (or `src/`).

### 2. Setup

- **NVIDIA GPU (CUDA 12.6)**:
  Run `install_setup_cuda.bat`
- **AMD / Intel / Any GPU (DirectML)**:
  Run `install_setup_directml.bat`

---

## Usage

1. Put your `logitech.driver.dll` in the root folder or `src/` directory.
2. Launch the app by running `run_eventuri_ai.bat`.
3. In the GUI, click **Connect Driver** to initialize `logitech.driver.dll`.
4. Click **Test Move** to verify cursor movement.
5. Select your AI model (`models/` folder) and configure your target class and sensitivity.
6. Click **START AIMBOT**, hold your designated mouse button, and enjoy!

---

## Troubleshooting

- **"Failed to load logitech.driver.dll"**:
  Make sure `logitech.driver.dll` is placed in the project directory or `src/`, and ensure Logitech G HUB is running in the background.
- **"CUDA not found"**:
  Make sure CUDA 12.6 is installed, or switch to DirectML.
