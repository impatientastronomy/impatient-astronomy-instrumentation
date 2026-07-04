# Raspberry Pi Setup — Digital Eyepiece

This guide walks through installing and configuring the digital eyepiece software on a
Raspberry Pi 5.  It covers the operating system, software dependencies, camera driver,
and configuration file.  Physical assembly is covered in a separate document.

---

## What you need

- Raspberry Pi 5 (4 GB or 8 GB RAM recommended)
- MicroSD card (32 GB or larger, Class 10 or faster)
- Waveshare 4inch HDMI LCD (C), 720×720 (the display used in this project)
- A second screen or SSH access for the initial setup steps
- ZWO ASI camera connected via USB
- USB power supply (official Pi 5 27 W supply recommended)
- Internet connection during setup (wired or WiFi)

---

## 1. Flash the operating system

Download and install **Raspberry Pi Imager** on your computer:
https://www.raspberrypi.com/software/

Install the SD card on your computer.

In the imager:

1. **Device** — Raspberry Pi 5
2. **OS** — Raspberry Pi OS (64-bit) — the Desktop version is recommended so the
   eyepiece window displays without extra configuration.
3. **Storage** — select your microSD card
4. Choose your hostname, choose a username, and password. This documentation assumes:
hostname=astro-eye, username=pi, password=raspberry
5. Toggle to "Enable SSH"
6. Set your WiFi SSID and password so the Pi connects on first boot


Write the card, insert it into the Pi, and power it on.  After about 60 seconds the Pi
should be reachable from your laptop:

```bash
ssh pi@astro-eye.local
```
Be sure to replace "pi" and "astro-eye" with your chosen username and hostname.

> All remaining setup steps run over SSH.  You do not need a keyboard, mouse, or display
> connected to the Pi until you are ready to run the eyepiece software for the first time.

---

## 2. Connect and update

SSH into the Pi or open a terminal on the desktop:

```bash
ssh pi@astro-eye.local
```

Update the system packages:

```bash
sudo apt update && sudo apt full-upgrade -y
```

Install Git and a few required system libraries:

```bash
sudo apt install -y git libgl1 libglib2.0-0
```

> **Why libgl1 and libglib2.0-0?**  OpenCV requires these at runtime on a minimal OS
> install.  They are usually present on the Desktop image but are listed here for
> completeness.

---

## 3. Install uv (Python package manager)

The project uses [uv](https://docs.astral.sh/uv/) to manage its Python environment.

**Mac / Linux / Raspberry Pi:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Open a new terminal (or run `source ~/.bashrc`) so that the `uv` command is on your PATH.

**Windows:**
```powershell
winget install --id=astral-sh.uv
```
Or in PowerShell:
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Verify on any platform:

```bash
uv --version
```

---

## 4. Clone the repository

```bash
cd ~
git clone https://github.com/impatientastronomy/impatient-astronomy-instrumentation.git
cd impatient-astronomy-instrumentation
```

> Replace the URL above with the actual repository URL when it is published on GitHub.

---

## 5. Install dependencies and configure hardware

A single script handles Python dependencies, data directory setup, the ZWO ASI camera
udev rule, and the display timing settings.  The script works on Mac, Windows, and
Raspberry Pi without modification.

```bash
uv run python utilities/install.py
```

On Mac and Raspberry Pi you can also use the bash shortcut:

```bash
bash utilities/install.sh
```

The script:
- Installs all Python packages via `uv sync`
- Creates `~/digital_eyepiece/{cals,sessions,images}/`
- Copies `digital_eyepiece/config/configuration-example.yaml` to
  `digital_eyepiece/config/configuration.yaml` (in the repo, already gitignored)
  if no configuration file exists there yet
- On macOS: fixes a libSDL2 conflict between opencv and pygame
- On Raspberry Pi: copies the ZWO ASI udev rule to `/etc/udev/rules.d/` (requires sudo)
- On Raspberry Pi: appends the Waveshare display timings and USB current setting to
  `/boot/firmware/config.txt` (requires sudo)
- On Raspberry Pi: pins the two USB WiFi adapters to stable names (`wlan-mount` and
  `wlan-guest`) via a udev rule — both Edimax adapters must be plugged in when this step runs

It is safe to run more than once — each step checks whether it has already been applied.

**Note:** plug in both Edimax WiFi adapters before running the script so the WiFi naming
step can detect their MAC addresses.

If hardware settings were added, reboot before continuing:

```bash
sudo reboot
```

The ZWO ASI camera library for arm64 (`libASICamera2.so.1.41`, with a `libASICamera2.so`
symlink) is included in the repository and is loaded automatically — no additional
environment variables are required.

---

## 6. Configure mount WiFi

This step runs after the reboot from step 5, so the `wlan-mount` interface name is active.

Open `utilities/setup_mount_wifi.sh` in a text editor and set the SSID and password to
match your mount's WiFi hotspot.  For the ZWO AM5, find these in the ZWO app under
**Network → Hotspot settings**.

```bash
nano utilities/setup_mount_wifi.sh
```

Then run it:

```bash
sudo bash utilities/setup_mount_wifi.sh
```

`wlan-mount` will now connect to the mount automatically whenever the mount is powered
on.  No action is needed in the eyepiece software — selecting **Mount → Connect** from
the menu is sufficient.  If the mount is off or out of range when you
attempt to connect, the software will show an alert and continue running.

To connect immediately (if the mount is already on):

```bash
nmcli connection up mount-wifi
```

---

## 7. Edit the configuration file

The install script created your personal configuration file inside the repository at:

```
digital_eyepiece/config/configuration.yaml
```

Open it in a text editor:

```bash
nano ~/impatient-astronomy-instrumentation/digital_eyepiece/config/configuration.yaml
```

Key fields to set:

| Field | What to change |
|---|---|
| `latitude` / `longitude` | Your observing location in decimal degrees |
| `cameras.primary.id` | The camera ID stored in your ASI camera's EEPROM |
| `cameras.primary.focal_length_mm` | Focal length of your telescope in mm |
| `cameras.primary.cam_size` | `[width, height]` of the ROI to read from the sensor, or `[-1, -1]` for full sensor |
| `cameras.primary.pattern` | Bayer pattern for your sensor (e.g. `GBRG`, `RGGB`) — check your camera's datasheet |
| `cameras.primary.telescope_description` | A short label used in saved file metadata |
| `mount_driver` | Driver module for your mount (e.g. `zwo_am5`, `lx200`) |

**Paths** (`cal_path`, `record_path`, `image_path`) default to subdirectories of
`~/digital_eyepiece/` — no changes needed unless you want data elsewhere (e.g. an
external drive).  Uncomment and edit the relevant lines to override.

**Finding your camera ID:**  Connect the camera via USB and run:

```bash
uv run python -c "from astrocore.camera.zwo_asi import list_cameras; print(list_cameras())"
```

This prints each connected camera's model and ID.

**Calibration data** is optional.  If `~/digital_eyepiece/cals/` is empty the software
starts without dark or flat correction and prints a warning.  You can add calibration
data later.

**Transferring files from your Mac/PC:**  If you already have a working setup on
another computer, run these commands from that computer to copy files to the Pi:

```bash
# Copy your configuration file (run from the repo root on your Mac/PC)
scp digital_eyepiece/config/configuration.yaml pi@astro-eye.local:~/impatient-astronomy-instrumentation/digital_eyepiece/config/

# Copy your calibration files
scp -r ~/digital_eyepiece/cals/ pi@astro-eye.local:~/digital_eyepiece/
```

Replace `pi@astro-eye.local` with your Pi's username and hostname if different.

---

## 8. Connect the display

After rebooting in step 5, the Waveshare display should come up at 720×720.  If it remains blank, check that:
- The micro-HDMI cable is plugged into the **HDMI 0** port on the Pi 5 (the one closest
  to the USB-C power connector).
Also plug in the USB-USBC cable from the Pi to the display to provide power to the display.

### Touch input (optional)

Connect a USB-C cable from the display to any USB port on the Pi for driver-free
capacitive touch.  Touch is not required — the eyepiece software uses a wireless mouse.

---

## 9. Pair the Bluetooth mouse

You will need a wired USB mouse plugged into the Pi for this step.

1. Click the **Bluetooth icon** in the upper-right corner of the Pi desktop taskbar.
2. Select **Add Device**.
3. On the Bluetooth mouse, long-press the button on the right side for about 1 second
   until the blue light starts flashing — this puts the mouse into pairing mode.
4. **D13** should appear in the list of available devices.  Select it.
5. Press the side button on the mouse once more to confirm pairing.
6. You should see **"Connection complete"**.

The Pi remembers the mouse and reconnects automatically on every subsequent boot.  You
can now unplug the wired USB mouse.

---

## 10. Run the eyepiece software

From the repository root:

```bash
uv run python -m digital_eyepiece.main
```

The application opens fullscreen by default.  To run in a window instead (useful during
development):

```bash
uv run python -m digital_eyepiece.main --windowed
```

The ZWO camera connects automatically.  If no camera is found, an alert is shown and
the software waits.

**To exit:** press `Q` or `Escape`.

### Running without a camera (virtual camera)

For testing without hardware, replay a previously recorded session:

```bash
uv run python -m digital_eyepiece.main --vcam <SessionFolderName>
```

The session folder must be inside `~/digital_eyepiece/sessions/` (or the `record_path`
override set in your configuration file).

---

## 11. Optional: guest image sharing hotspot

The eyepiece can broadcast a WiFi hotspot on `wlan-guest` so observers nearby can view
saved images on their phones.  `wlan0` remains free for SSH and setup; `wlan-mount`
connects to the mount's WiFi network.

**Hardware required:** a USB WiFi adapter that supports AP mode (most common adapters do).

Run the one-time setup script:

```bash
sudo bash utilities/setup_hotspot.sh
```

This installs `hostapd` and `dnsmasq`, configures the `wlan-guest` interface with a
static IP, and enables the captive portal redirect.  See the script header for
customisation options.

Edit the `hotspot:` section in `digital_eyepiece/config/configuration.yaml` if you
changed the SSID, password, or IP address in the script.

After setup and a reboot, the hotspot starts automatically.  To print the QR code for
guests to scan:

```bash
uv run python utilities/print_qr.py
```

---

## 12. Optional: auto-start on boot

To launch the eyepiece automatically when the Pi boots into the desktop, create an
autostart entry:

```bash
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/digital_eyepiece.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=Digital Eyepiece
Exec=/home/pi/impatient-astronomy-instrumentation/.venv/bin/python -m digital_eyepiece.main
WorkingDirectory=/home/pi/impatient-astronomy-instrumentation
EOF
```

Adjust the paths above if your username or repo location is different.

> This uses the `.venv` Python directly rather than `uv run` to avoid the overhead of
> uv's environment resolution at every boot.

---

## Troubleshooting

**Low-power warning on screen**
If you are powering the Pi from a non-USB-C-PD supply (such as a 12V battery with a buck
converter), the Pi may display a low-power warning even when the supply is capable of
delivering 5A.  This is because the Pi expects USB-C Power Delivery negotiation to confirm
the supply rating.  The install script adds `usb_max_current_enable=1` to
`/boot/firmware/config.txt` to unlock the full 1.6A USB current budget regardless — the
warning can safely be ignored once this setting is in place.

**Camera not found / permission denied**
Make sure the udev rule was installed (step 6) and that you reconnected the camera after
running `udevadm trigger`.  Confirm with: `ls -l /dev/bus/usb/...` — the camera device
should have world-readable permissions (`rw-rw-rw-`).

**`libASICamera2.so` not found**
The repository includes both `libASICamera2.so` (a symlink) and the real versioned library
`libASICamera2.so.1.41`.  If the error appears, check that both files are present:
```bash
ls -lh astrocore/camera/asi_sdk/lib/armv8/
```
You should see `libASICamera2.so.1.41` (~3.9 MB) and `libASICamera2.so` pointing to it.
If either is missing, run `git pull`.  Also confirm you are running a 64-bit OS:
`uname -m` should print `aarch64`.

**Blank or flickering display**
Confirm the `hdmi_timings` line in `/boot/firmware/config.txt` is present and exactly as
shown in step 8 (no line breaks).  Confirm the cable is in the **HDMI 0** port (closest
to the USB-C power connector on the Pi 5).  Try a different micro-HDMI cable — the
Waveshare display is powered by HDMI and a marginal cable can cause instability.

**pygame `No available video device`**
The desktop environment must be running before the eyepiece starts.  If you are connecting
over SSH, the application cannot open a window.  Connect a keyboard/mouse to the Pi or
enable VNC, then launch the software from the Pi's own desktop session.

**Calibration data warnings at startup**
These are non-fatal.  The software runs without calibration; dark and flat correction is
simply skipped.  Add calibration data to `~/digital_eyepiece/cals/` when you have it;
see `digital_eyepiece/config/configuration-example.yaml` for the expected folder layout.
