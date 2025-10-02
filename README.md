---

# 🛰️ Satellite Simulator Advanced v1.1

A next-generation, terminal-based **satellite operations simulator** written in Python.
It models real-time orbital mechanics, power systems, subsystem behavior, ground station communications, and AI-driven autonomous control — now with improved physics, stability, and realism.

The simulation uses **`asyncio`** for concurrency and **`rich`** for a live, interactive dashboard.

---

## ✨ What's New in v1.1

* ✅ **Per-tick stability:** Radiation and solar calculations are now sampled once per simulation tick.
* 🌍 **Physics overhaul:** Orbital dynamics now evolve based on simulation time (`dt`) — no more jumpy motion.
* 🧠 **Smarter AI:** Hysteresis-based mode switching, gradual subsystem recovery, and improved reboot logic.
* ⚙️ **More realistic environment:**

  * Ground-station visibility windows (no more 24/7 link)
  * Radiation spikes and solar effects influence subsystems and energy systems consistently
* ⚡ **Better power modeling:** Battery degradation, panel efficiency loss, and thermal dynamics scale with time.
* 📊 **Telemetry & reporting:** Optional CSV export and an end-of-run performance summary.
* 🔄 **New CLI options:** `--seed`, `--timescale`, `--csv`, `--scenario`, `--deterministic`
* 📶 **Comms improvements:** Backpressure on telemetry link and prioritized uplink queue prevent overload.
* 🪵 **Logging system:** Rewritten with a background worker and queue for more reliable file logging.

---

## 🛰️ Core Features

### 🧩 Satellite Subsystems

* **COMMS:** Communication and telemetry
* **Radar / Camera:** Payload instruments
* **Thermal Control:** Body temperature management
* **Attitude Control:** Orientation handling
* **Propulsion:** Orbit adjustment and maneuvers

### ☀️ Space Environment

* Real-time orbit tracking (lat/lon/alt)
* Day/night cycle with solar incidence
* Variable solar radiation, flares, and micrometeorite impacts
* **Ground-station visibility windows** for realistic link sessions

### 🔋 Power System

* Detailed battery modeling (health, voltage, cycle life, degradation)
* Solar panels with deploy/stow, damage, and radiation-linked efficiency loss

### 📡 Ground Link & Commands

* Telemetry downlink with realistic packet loss
* Command uplink with priority scheduling and capacity limits
* Built-in ground-station agent that analyzes telemetry and issues corrective actions automatically

### 🧠 AI Controller

* Autonomous switching between `NOMINAL` and `CONSERVE` power modes
* Smart recovery logic for failed subsystems
* Radiation-aware load shedding
* Gradual subsystem reactivation

### 🖥️ Interactive Dashboard

* Real-time **Rich** interface with:

  * Telemetry view
  * Subsystem status panel
  * Power and thermal graphs
  * Event log (color-coded severity)
* Live summary of mode, power, solar generation, and environment

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

Run the simulator with defaults:

```bash
python simulator.py
```

Or customize:

```bash
python simulator.py --duration 300 --update 0.5 --seed 42 --csv run.csv
```

**Main options:**

| Option            | Description                                        |
| ----------------- | -------------------------------------------------- |
| `--duration`      | Total simulation time in seconds (default: 120)    |
| `--update`        | UI update interval in seconds (default: 1.0)       |
| `--seed`          | Set random seed for reproducibility                |
| `--timescale`     | Accelerate or slow down the simulation             |
| `--csv`           | Export telemetry to a CSV file                     |
| `--scenario`      | Load external scenario configuration (JSON)        |
| `--deterministic` | Run with deterministic behavior (no random jitter) |

---

## 📡 Commands

Ground station agent or manual uplink can send:

* `PING` – Satellite replies `PONG`
* `REQUEST_STATUS` – Send latest telemetry
* `SET_POWER_MODE conserve|nominal` – Adjust power strategy
* `DEPLOY_SOLAR` / `STOW_SOLAR` – Control solar panels
* `REBOOT_COMMS` – Attempt communications reboot
* `ADJUST_ORBIT delta_alt=X` – Change orbital altitude

---

## 📂 Logs & Data

* Live logs shown in the **Event Log panel**.
* All logs stored asynchronously in `simulation.log`.
* Optional CSV telemetry export for long-term analysis.

---

## 📊 Simulation Architecture

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

## 📈 End-of-Run Summary

At the end of a run, the simulator prints a mission summary including:

* Battery min/avg/max
* Total solar energy produced
* Total load consumed
* Link uptime
* Number of subsystem failures and recoveries

---

## 📝 License

MIT License – use, modify, and build upon this simulator for research, education, or hobby projects.

---

## 📬 Contact

* **Email:** [lloydlewizzz@gmail.com](mailto:lloydlewizzz@gmail.com)
* **Discord:** `lloydlewizzz`

---
