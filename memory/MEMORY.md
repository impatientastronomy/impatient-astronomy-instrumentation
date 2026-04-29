# Memory Index

- [Code style and quality guidance](feedback_code_style.md) — Use MATLAB as reference only; write clean idiomatic Python; user wants to learn good practices
- [Camera stack design decisions](project_camera_stack.md) — naming convention, catalog columns, FrameGrabber API, DPC mask rationale, zwoasi API corrections, opencv/macOS quirks
- [SDL2 symlink fix for macOS](feedback_sdl2_symlink.md) — opencv-python and pygame both bundle libSDL2; fix with symlink after pip install
- [Augmented eyepiece display decisions](project_augmented_eyepiece.md) — guest display must be horizontally flipped via Wayland (wlr-randr) so cursor movement is also mirrored
