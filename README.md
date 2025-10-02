---

# 🛰️ Satellite Simulator Advanced

A terminal-based satellite simulation written in **Python**, featuring orbital mechanics, subsystem management, telemetry, ground station interaction, and AI-assisted autonomous control.
The simulation uses **asyncio** for concurrency and **rich** for a live, interactive console dashboard.

---

## ✨ Features

* **Satellite Subsystems**

  * Communications (COMMS)
  * Radar
  * Camera
  * Thermal Control
  * Attitude Control
  * Propulsion

* **Space Environment Simulation**

  * Orbit tracking (latitude/longitude/altitude)
  * Day/night cycle with solar incidence factor
  * Variable solar radiation & solar flares
  * Micrometeorite impacts

* **Energy System**

  * Battery with health degradation, cycle tracking, and voltage modeling
  * Solar panels with deploy/stow, damage, and efficiency degradation

* **Ground Station Link**

  * Telemetry downlink (with packet loss possibility)
  * Uplink commands (e.g., `PING`, `REQUEST_STATUS`, `DEPLOY_SOLAR`, `SET_POWER_MODE conserve`)
  * Automated ground station agent issuing recovery actions

* **AI Controller**

  * Automatic mode switching (`NOMINAL` ↔ `CONSERVE`)
  * Responds to critical conditions like low battery or high radiation
  * Attempts subsystem reboots

* **Interactive Rich UI**

  * Live updating dashboard with telemetry, power, subsystems, and logs
  * Event log with color-coded severity
  * Summary footer with system mode, battery, solar, and radiation

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/satellite-simulator.git
cd satellite-simulator
pip install -r requirements.txt
```

**Requirements:**

* Python 3.9+
* [`rich`](https://github.com/Textualize/rich)

---

## 🖥️ Usage

Run the simulation with default parameters:

```bash
python simulator.py
```

Available options:

```bash
python simulator.py --duration 180 --update 0.5
```

* `--duration` → total simulation time in seconds (default `120`)
* `--update` → dashboard update interval in seconds (default `1.0`)

Stop anytime with **Ctrl+C**.

---

## 📡 Commands

The ground station agent (or manual uplink) can send commands such as:

* `PING` → Satellite responds with `PONG`
* `REQUEST_STATUS` → Sends back latest telemetry
* `SET_POWER_MODE conserve|nominal` → Adjusts power consumption strategy
* `DEPLOY_SOLAR` → Deploy or re-enable solar panels
* `STOW_SOLAR` → Stow solar panels
* `REBOOT_COMMS` → Attempt to reboot communications
* `ADJUST_ORBIT delta_alt=X` → Adjust orbit altitude

---

## 📂 Logs

* Live logs are displayed in the **Event Log panel**.
* All logs are also written asynchronously to `simulation.log`.

---

## 🧠 Architecture Overview

```
[ SpaceEnvironment ] 
       │
       ▼
[ Satellite ]─── manages ───[ Battery / Solar Panel / Subsystems ]
       │
       ├── AIController (autonomous decisions)
       ├── CriticalConditionManager (random critical events)
       └── GroundLink ⇄ GroundStationAgent (uplink/downlink)
```

---

## 📝 License

MIT License – feel free to modify and use for research, teaching, or fun simulations.

---
## 📬 Contact

If you have new ideas, suggestions, or want to get in touch:

* **Email**: [lloydlewizzz@gmail.com](mailto:lloydlewizzz@gmail.com)
* **Discord**: `lloydlewizzz` 

---
