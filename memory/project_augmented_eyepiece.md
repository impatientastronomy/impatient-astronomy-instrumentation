---
name: Augmented eyepiece display decisions
description: Key decisions for the augmented eyepiece project, including display flip approach
type: project
---

## Guest display horizontal flip

The augmented eyepiece guest display must be horizontally flipped because the projected image is reflected by a half-silvered mirror on one axis. This affects text, mouse cursor movement, and the image.

**Decision: use Wayland display-level flip on the Raspberry Pi.**

```
wlr-randr --output HDMI-2 --transform flipped
```

**Why Wayland over application-level flip:** cursor movements must also be mirrored, which requires an OS-level flip. Application-level `pygame.transform.flip()` would only flip the rendered frame, leaving cursor movement backwards.

**How to apply:** add `wlr-randr` command to the augmented eyepiece startup script. Pi OS Bookworm uses Wayland by default so wlr-randr should be available. Verify with `wlr-randr --help` on the target Pi.

**Note:** also check whether the camera input image needs to be flipped — the half-silvered mirror may mirror the camera image too, which would require updating the Bayer pattern and/or a flip in the camera config.
