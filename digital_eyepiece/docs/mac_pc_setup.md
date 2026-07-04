# Mac / Windows Setup — Digital Eyepiece

This guide walks through installing and running the digital eyepiece software on a Mac
or Windows PC.  This is the recommended path for development, testing with a virtual
camera, and verifying hardware before deploying to a Raspberry Pi.

---

## What you need

- Mac (Apple Silicon or Intel) or Windows 10/11 PC
- ZWO ASI camera connected via USB (optional — see [Running without a camera](#8-running-without-a-camera-vcam))
- Internet connection during setup

---

## 1. Install uv (Python package manager)

The project uses [uv](https://docs.astral.sh/uv/) to manage its Python environment.

**Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Open a new terminal so that the `uv` command is on your PATH.

**Windows (PowerShell):**
```powershell
winget install --id=astral-sh.uv
```
Or:
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Verify on any platform:
```bash
uv --version
```

---

## 2. Clone the repository

**Mac / Linux:**
```bash
cd ~
git clone https://github.com/impatientastronomy/impatient-astronomy-instrumentation.git
cd impatient-astronomy-instrumentation
```

**Windows (PowerShell):**
```powershell
cd $HOME
git clone https://github.com/impatientastronomy/impatient-astronomy-instrumentation.git
cd impatient-astronomy-instrumentation
```

> Replace the URL above with the actual repository URL when it is published on GitHub.

---

## 3. Run the install script

A single script installs Python dependencies and sets up your data directory.

**Mac:**
```bash
uv run python utilities/install.py
```

**Windows (PowerShell):**
```powershell
uv run python utilities\install.py
```

> **Note:** Always use `uv run python` rather than bare `python` or `python3`.  If you
> have pyenv, conda, or another version manager installed, bare `python` will go through
> that tool's shim and may fail if the pinned version isn't installed.  `uv run` bypasses
> all of that and uses uv's own managed Python.

The script:
- Installs all Python packages via `uv sync`
- Creates `~/digital_eyepiece/{cals,sessions,images}/`
- Copies `digital_eyepiece/config/configuration-example.yaml` to
  `digital_eyepiece/config/configuration.yaml` (in the repo, already gitignored)
  if no configuration file exists there yet
- **Mac only:** fixes a libSDL2 conflict between opencv and pygame

It is safe to run more than once — each step checks whether it has already been applied.

---

## 4. Find your camera ID

Connect the camera via USB and run:

**Mac:**
```bash
uv run python -c "from astrocore.camera.zwo_asi import list_cameras; print(list_cameras())"
```

**Windows (PowerShell):**
```powershell
uv run python -c "from astrocore.camera.zwo_asi import list_cameras; print(list_cameras())"
```

This prints each connected camera's model, USB index, and camera ID (stored in the
camera's EEPROM).  If the camera ID shows `0`, you need to assign one — see
`utilities/set_camera_id.py`.

---

## 5. Edit the configuration file

The configuration file lives inside the repository at `digital_eyepiece/config/configuration.yaml`
(gitignored — safe to edit without affecting version control).

Open it in a text editor:

**Mac:**
```bash
open -e digital_eyepiece/config/configuration.yaml
```
Or use any editor (`nano`, VS Code, etc.):
```bash
code digital_eyepiece/config/configuration.yaml
```

**Windows:**
```powershell
notepad "digital_eyepiece\config\configuration.yaml"
```

Key fields to set:

| Field | What to change |
|---|---|
| `latitude` / `longitude` | Your observing location in decimal degrees |
| `mount_driver` | Driver module for your mount (e.g. `zwo_am5`), or leave blank if no mount |
| `cameras."primary".id` | The camera ID from step 4 |
| `cameras."primary".focal_length_mm` | Focal length of your telescope in mm |
| `cameras."primary".cam_size` | `[width, height]` of the ROI, or `[-1, -1]` for full sensor |
| `cameras."primary".pattern` | Bayer pattern for your sensor (e.g. `GBRG`, `RGGB`) |
| `cameras."primary".telescope_description` | A short label used in saved file metadata |

**Paths** (`cal_path`, `record_path`, `image_path`) default to subdirectories of
`~/digital_eyepiece/` — no changes needed unless you want data elsewhere.

See `digital_eyepiece/config/configuration-example.yaml` for a fully commented reference.

---

## 6. Connect to the mount (optional)

The ZWO AM5 (and similar mounts) broadcasts a WiFi hotspot.  The eyepiece software
connects to it at a fixed IP address (`192.168.4.1`) — no configuration needed beyond
having your computer on that network.

Click the WiFi icon in the menu bar (Mac) or taskbar (Windows) and select the mount's
hotspot (e.g. `ZWO-AM5-XXXXXX`).  Your internet connection will drop while connected
to the mount — this is expected and fine for testing.

To test mount-related features without a physical mount, use `--vmount` when launching
the app (see [Running without a camera](#8-running-without-a-camera-vcam)).

> On the Raspberry Pi, a dedicated USB WiFi dongle handles the mount connection
> automatically so the Pi can stay on the home network at the same time.  That
> dongle-based setup is not needed on a Mac or PC used for development.

---

## 7. Run the app

From the repository root:

**Mac:**
```bash
uv run python -m digital_eyepiece.main
```

**Windows (PowerShell):**
```powershell
uv run python -m digital_eyepiece.main
```

The application opens fullscreen by default.  To run in a window (recommended during
development):

```bash
uv run python -m digital_eyepiece.main --windowed
```

The ZWO camera connects automatically.  If no camera is found, an alert is shown.

**To exit:** press `Q` or `Escape`.

---

## 8. Running without a camera (vcam)

To test without hardware, replay a previously recorded session:

```bash
uv run python -m digital_eyepiece.main --vcam <SessionFolderName>
```

The session folder must be inside `~/digital_eyepiece/sessions/` (or the `record_path`
set in your configuration file).  Use `--vmount` alongside `--vcam` to simulate a
connected mount:

```bash
uv run python -m digital_eyepiece.main --vcam <SessionFolderName> --vmount
```

---

## 9. View saved images (gallery server)

When you save an image (Action Menu → Save), the software:

1. Writes a JPEG to `~/digital_eyepiece/images/` (or your configured `image_path`)
2. Automatically opens the gallery in your default browser at `http://localhost:8080`
3. Shows a QR code on screen — scan it with a phone on the same WiFi network to view
   images there too

The gallery page lists all saved images in reverse chronological order and
auto-refreshes every 15 seconds.

> On the Raspberry Pi the QR code is a WiFi join code for the dedicated guest hotspot.
> On Mac and Windows it encodes the gallery URL directly so the phone can open it
> without switching networks.

---

## Troubleshooting

**Camera not found**
On Mac, ZWO cameras are recognised automatically via the bundled SDK library — no driver
installation is needed.  On Windows, install the ZWO ASI Studio software once to get the
USB driver, then the eyepiece software will be able to open the camera.

**`libASICamera2` not found (Mac)**
The macOS SDK library (`libASICamera2.dylib`) is included in the repository under
`astrocore/camera/asi_sdk/lib/mac_arm64/` (Apple Silicon) or `mac/` (Intel).  If the
error appears, confirm you are running the correct architecture: `uname -m` should print
`arm64` (Apple Silicon) or `x86_64` (Intel).

**pygame `No available video device` (Mac)**
If you launch the app over SSH without a display attached, pygame cannot open a window.
Run the app directly in a terminal session on the Mac itself, not over SSH.

**SDL2 warnings on Mac**
Both opencv and pygame bundle `libSDL2`.  The install script symlinks one to the other
to silence the warning.  If you see `Class X is implemented in both...` messages, re-run
`python utilities/install.py`.

**App opens but image is black**
Check that the camera ID in `configuration.yaml` matches the value printed by
`list_cameras()`.  A mismatch means the wrong config is loaded and settings like gain
and pattern may be wrong.
