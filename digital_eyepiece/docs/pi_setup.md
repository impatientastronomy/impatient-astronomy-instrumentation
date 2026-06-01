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

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal (or run `source ~/.bashrc`) so that the `uv` command is on your PATH.

Verify:

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

A single script handles Python dependencies, the ZWO ASI camera udev rule, and the
display timing settings:

```bash
bash utilities/install.sh
```

On a Raspberry Pi the script automatically:
- Installs all Python packages via `uv sync`
- Copies the ZWO ASI udev rule to `/etc/udev/rules.d/` (requires sudo, will prompt)
- Appends the Waveshare display timings to `/boot/firmware/config.txt` (requires sudo)

It is safe to run more than once — each step checks whether it has already been applied.

If the display settings were added, reboot before continuing:

```bash
sudo reboot
```

The ZWO ASI camera library for arm64 (`libASICamera2.so.1.41`, with a `libASICamera2.so`
symlink) is included in the repository and is loaded automatically — no additional
environment variables are required.

---

## 6. Create the configuration file

Copy the example configuration to the repo root and edit it for your setup:

```bash
cp configuration.yaml configuration.yaml.bak   # keep the original as reference
nano configuration.yaml
```

Key fields to set:

| Field | What to change |
|---|---|
| `latitude` / `longitude` | Your observing location in decimal degrees |
| `cal_path` | Absolute path to your calibration data folder (darks, flats, DPC masks) |
| `record_path` | Absolute path where captured sessions are saved |
| `image_path` | Path where saved eyepiece JPEG images are written |
| `cameras.primary.id` | The camera ID stored in your ASI camera's EEPROM |
| `cameras.primary.focal_length_mm` | Focal length of your telescope in mm |
| `cameras.primary.cam_size` | `[width, height]` of the ROI to read from the sensor, or `[-1, -1]` for full sensor |
| `cameras.primary.pattern` | Bayer pattern for your sensor (e.g. `GBRG`, `RGGB`) — check your camera's datasheet |
| `cameras.primary.telescope_description` | A short label used in saved file metadata |
| `cameras.primary.mount_driver` | Driver module for your mount (e.g. `zwo_am5`, `lx200`) |

**Finding your camera ID:**  Connect the camera via USB and run:

```bash
uv run python -c "from astrocore.camera.zwo_asi import list_cameras; print(list_cameras())"
```

This prints each connected camera's model and ID.

**Calibration data** is optional.  If `cal_path` doesn't exist yet, the software starts
without dark or flat correction and prints a warning.  You can add calibration data later.

---

## 7. Connect the display

After rebooting in step 5, the Waveshare display should come up at 720×720.  If it
remains blank:
- Confirm the micro-HDMI cable is in the **HDMI 0** port (closest to the USB-C power
  connector on the Pi 5).
- The display is HDMI-powered; a marginal cable can cause instability — try a different
  one if the image flickers or doesn't appear.

### Touch input (optional)

Connect a USB-C cable from the display to any USB port on the Pi for driver-free
capacitive touch.  Touch is not required — the eyepiece software uses a wireless mouse.

---

## 8. Run the eyepiece software

From the repository root:

```bash
uv run python -m digital_eyepiece.main
```

Or in fullscreen mode (recommended when the Pi is mounted in the eyepiece housing):

```bash
uv run python -m digital_eyepiece.main --fullscreen
```

The application opens a pygame window.  The ZWO camera connects automatically.  If no
camera is found, an alert is shown and the software waits.

**To exit:** press `Q` or `Escape`.

### Running without a camera (virtual camera)

For testing without hardware, replay a previously recorded session:

```bash
uv run python -m digital_eyepiece.main --vcam <SessionFolderName>
```

The session folder must be inside the `record_path` directory set in
`configuration.yaml`.

---

## 9. Optional: guest image sharing hotspot

The eyepiece can broadcast a WiFi hotspot on a second wireless interface (e.g. a USB
WiFi dongle) so observers nearby can view saved images on their phones.  The main
`wlan0` interface remains free to connect to your mount's WiFi network.

**Hardware required:** a USB WiFi adapter that supports AP mode (most common adapters do).

Run the one-time setup script:

```bash
sudo bash utilities/setup_hotspot.sh
```

This installs `hostapd` and `dnsmasq`, configures the `wlan1` interface with a static IP,
and enables the captive portal redirect.  See the script header for customisation options.

Edit the `hotspot:` section in `configuration.yaml` if you changed the SSID, password,
or IP address in the script.

After setup and a reboot, the hotspot starts automatically.  To print the QR code for
guests to scan:

```bash
uv run python utilities/print_qr.py
```

---

## 10. Optional: auto-start on boot

To launch the eyepiece automatically when the Pi boots into the desktop, create an
autostart entry:

```bash
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/digital_eyepiece.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=Digital Eyepiece
Exec=/home/pi/impatient-astronomy-instrumentation/.venv/bin/python -m digital_eyepiece.main --fullscreen
WorkingDirectory=/home/pi/impatient-astronomy-instrumentation
EOF
```

Adjust the paths above if your username or repo location is different.

> This uses the `.venv` Python directly rather than `uv run` to avoid the overhead of
> uv's environment resolution at every boot.

---

## Troubleshooting

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
simply skipped.  See `configuration.yaml` comments for how to organise calibration data
once you have recorded it.
