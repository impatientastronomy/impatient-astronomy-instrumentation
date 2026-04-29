---
name: SDL2 symlink fix for macOS
description: opencv-python and pygame both bundle libSDL2 on macOS, causing duplicate class warnings. Fix with a symlink.
type: project
---

Both `opencv-python` and `pygame` bundle their own `libSDL2-2.0.0.dylib` on macOS. This causes harmless but noisy "Class X is implemented in both..." warnings. Deleting cv2's copy breaks cv2 (libavdevice depends on it). The fix is to replace cv2's copy with a symlink to pygame's copy:

```
ln -sf .venv/lib/python3.10/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib \
       .venv/lib/python3.10/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib
```

**Why:** cv2's libavdevice links against libSDL2 at load time, so the file must exist. Symlinking to pygame's copy means only one binary is loaded, eliminating the duplicate class warnings.

**How to apply:** Include this in the macOS installation script after `pip install -r requirements.txt`. Not needed on Linux.
