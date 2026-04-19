# CLAUDE.md — Astro Instruments Project Context

This file provides persistent context for Claude (claude.ai Projects and Claude CLI) across all
conversations about this repository. Update it as key decisions are made.

---

## About the author

- Electrical engineer with a background in camera design
- Hobbyist astrophotographer and electronically assisted astronomy (EAA) enthusiast
- Experienced in MATLAB for camera control and image processing; transitioning to Python
- Primary development machine: Mac; target deployment platform: Raspberry Pi 5 (or similar)
- Goal: build open source astronomy instrumentation projects, share on GitHub, and enable
  community contributions

---

## Repository overview

This is a **monorepo** containing three astronomy instrument projects plus a shared Python
library (`astrocore`) that all three projects depend on. The monorepo approach was chosen
because the projects share significant code (camera drivers, mount control, image pipeline)
and are being developed by a single author at this stage.

**Repo name:** `impatient-astronomy-instrumentation`
**License:** TBD (MIT or GPL — to be decided)
**Language:** Python (migrating from MATLAB)
**Target platform:** Develop on Mac w/ Apple Silicon. Port over to Raspberry Pi 5
**Python packaging:** `pyproject.toml` at repo root defining `astrocore` as a local package

---

## Shared library: `astrocore/`

The heart of the repo. All three projects import from here. Designed to be modular so
contributors can swap in their own hardware drivers without touching higher-level logic.

### Key modules

| Module | Purpose |
|---|---|
| `astrocore/camera/` | Abstract base class + concrete drivers (e.g. ZWO ASI). New cameras implement the base class. |
| `astrocore/mount/` | Abstract base class + INDI/ASCOM drivers for telescope mount control |
| `astrocore/pipeline/` | Live image stacking, dark/flat/bias calibration, histogram stretching |
| `astrocore/display/` | Rendering frames to HDMI display or microdisplay; overlay compositing |
| `astrocore/config/` | YAML-based configuration loading and validation |
| `astrocore/scheduler/` | Observation scheduling and target sequencing |
| `astrocore/logging/` | Structured logging and telemetry |

### Design principles

- Camera, mount, and display classes use abstract base classes (Python `abc` module) so
  hardware can be swapped without changing project-level code
- All modules should be independently testable with `pytest`

---

## Project 1: Digital eyepiece (`digital_eyepiece/`)

### Concept
A camera, mini HDMI display, and optical eyepiece packaged together. The user looks into
the eyepiece and sees a live stacked image from the camera — like looking through a traditional
eyepiece, but with the benefit of long-exposure stacking revealing dim detail in real time.

### Key features
- Live image stacking with progressive display (image builds up over time)
- Multi-camera zoom: four cameras at different focal lengths are mounted on the telescope;
  the user scrolls a wireless mouse wheel to zoom in/out, and the software selects the
  appropriate camera automatically. The user is unaware of the camera switching.
- Single interface for image stacking, target selection, and mount control
- Similar in concept to the Pegasus Astro SmartEye

### Hardware
- Camera(s) with USB connection to Raspberry Pi
- Raspberry Pi drives HDMI display inside the eyepiece housing
- Optical eyepiece element over the display
- Wireless mouse for user input (scroll wheel = zoom)
- Physical enclosure: CAD files in `digital_eyepiece/hardware/cad/`

### Key source files
- `digital_eyepiece/main.py` — application entry point
- `digital_eyepiece/zoom_controller.py` — multi-camera selection logic

---

## Project 2: Augmented eyepiece (`augmented_eyepiece/`)

### Concept
A half-silvered teleprompter mirror arrangement that lets the user see real starlight through
the eyepiece *and* an overlaid processed image from a camera simultaneously. A second mirror
injects light from a microdisplay into the optical path. The result: natural starlight with
color-enhanced long-exposure detail and digital annotations (star names, object labels)
overlaid — like augmented reality for astronomy.

### Key features
- Real starlight preserved through the eyepiece (unlike a purely electronic system)
- Camera captures a parallel light path via the first teleprompter mirror
- Microdisplay injects stacked/processed imagery and annotations via the second mirror
- AR overlay compositor: aligns camera image with eyepiece field of view
- Potentially novel as a hobbyist-buildable device

### Hardware
- Two teleprompter (half-silvered) mirrors in the optical path
- Camera on the reflected path of mirror 1
- Microdisplay on mirror 2 injection path
- Optical eyepiece at the exit
- Physical enclosure: `augmented_eyepiece/hardware/cad/`

### Key source files
- `augmented_eyepiece/main.py` — application entry point
- `augmented_eyepiece/compositor.py` — AR overlay alignment and compositing logic

---

## Project 3: Sky survey (`sky_survey_automation/`)

### Concept
An automated all-sky survey collecting imagery and data not available in existing public
surveys. Intended to support a future fourth project (details TBD). Fully automated
acquisition: the mount slews through a target list while the camera acquires and stores data.

### Key features
- Automated observation scheduling and mount control
- Multi-target sequencing across a night
- Data storage and indexing for later processing
- Relies heavily on `astrocore` scheduler, mount, and camera modules

### Key source files
- `sky_survey/main.py` — application entry point
- `sky_survey/scheduler.py` — survey target sequencing
- `sky_survey/data_store.py` — image and metadata storage

---

## Repository folder structure

```
impatient-astronomy-instrumentation/
├── CLAUDE.md                  ← this file
├── README.md                  ← project overview for GitHub visitors
├── LICENSE
├── CONTRIBUTING.md
├── pyproject.toml             ← workspace / packaging config
│
├── astrocore/                 ← shared Python library
│   ├── camera/
│   │   ├── base.py            ← abstract base class
│   │   └── zwo_asi.py         ← example ZWO ASI driver
│   ├── mount/
│   │   ├── base.py
│   │   └── skywatcher.py
│   ├── pipeline/
│   │   ├── stacker.py
│   │   ├── calibration.py
│   │   └── stretch.py
│   ├── display/
│   ├── config/
│   ├── scheduler/
│   ├── logging/
│   └── tests/
│
├── digital_eyepiece/
│   ├── main.py
│   ├── zoom_controller.py
│   ├── hardware/
│   │   ├── cad/               ← .step, .stl, and source CAD files
│   │   └── BOM.csv
│   └── docs/                  ← assembly and wiring guides
│
├── augmented_eyepiece/
│   ├── main.py
│   ├── compositor.py
│   ├── hardware/
│   │   ├── cad/
│   │   └── BOM.csv
│   └── docs/
│
├── sky_survey_automation/
│   ├── main.py
│   ├── scheduler.py
│   ├── data_store.py
│   ├── hardware/
│   │   ├── cad/
│   │   └── BOM.csv
│   └── docs/
│
├── docs/                      ← repo-level documentation
└── .github/                   ← CI workflows and issue templates
```

---

## Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2025-04 | Python over MATLAB | Free, runs on Raspberry Pi, large astronomy ecosystem |
| 2025-04 | Monorepo | Single author, heavy code sharing, easy to split later if needed |
| 2025-04 | Raspberry Pi as target | Compact, inexpensive, GPU for image processing, mounts to telescope |
| 2025-04 | INDI for mount control | Open standard, cross-platform, broad mount support |
| 2025-04 | Abstract base classes for hardware | Allows contributors to swap camera/mount without rewriting projects |

---

## Open questions / TODO

- [ ] Choose license (MIT vs GPL)
- [ ] Evaluate ZWO ASI SDK Python bindings vs `asi-python` community library
- [ ] Decide on GUI framework for eyepiece UI (pygame? Qt? custom OpenGL?)
- [ ] Detail the fourth project that `sky_survey` is intended to support
- [ ] Set up GitHub Actions CI (lint, test on push)
- [ ] Choose CAD package (Fusion 360, FreeCAD, or other)

---

*Last updated: April 2026*