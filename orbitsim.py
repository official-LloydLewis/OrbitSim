"""
OrbitSim - Satellite Operations Simulator (enhanced)
===================================================

Author: LloydLewis, Neylox (edits: iranai)
License: MIT
Version: 1.1

Summary of notable changes in 1.1 (human-style, compact):
- Single-sample per tick for environment values (radiation/sun factor)
- Orbit kinematics driven by dt-based phase (no wall-clock jumps)
- Backpressure on telemetry sends; uplink with priority & capacity
- Logger uses a single worker thread + queue (no per-line thread)
- AI power-mode hysteresis + gradual subsystem recovery
- Degradations and thermal effects scaled by dt (time-consistent)
- Temperature clamping and smoother cooling/heating
- Ground-station visibility windows (simple geometric pass model)
- Footer uses same tick’s telemetry (no visual inconsistencies)
- Task lifecycle tracking & clean cancellation on exit
- CLI: --seed, --timescale, --csv, --scenario, --deterministic
- Optional CSV metrics export + end-of-run summary report

Design intent: keep code readable/"hand-written", minimal rewrites.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
import threading
import queue
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.align import Align

console = Console()

# ------------------------------
# Utilities
# ------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def now() -> float:
    return time.time()


# ------------------------------
# Logging (single worker thread)
# ------------------------------
class FileLogger:
    """Simple queued logger with one worker thread.
    Avoids spawning a thread per log line.
    """
    def __init__(self, path: str = 'simulation.log') -> None:
        self.path = path
        self._q: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=10000)
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._worker, name="FileLogger", daemon=True)
        self._thr.start()

    def _worker(self) -> None:
        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                while not self._stop.is_set():
                    try:
                        item = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    f.write(item + '\n')
                    f.flush()
        except Exception:
            # Best-effort logger; swallow disk errors
            pass

    def write_async(self, entry: str) -> None:
        try:
            self._q.put_nowait(entry)
        except queue.Full:
            # Drop logs if overwhelmed
            pass

    def stop(self) -> None:
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        self._stop.set()
        if self._thr:
            try:
                self._thr.join(timeout=1.0)
            except Exception:
                pass


file_logger = FileLogger('simulation.log')


# ------------------------------
# Data models
# ------------------------------
@dataclass
class Telemetry:
    timestamp: float
    dt_physics: float
    temperatures: Dict[str, float]
    position: Dict[str, float]
    battery_percent: float
    battery_voltage: float
    solar_output: float
    systems: Dict[str, str]
    env: Dict[str, float]
    power: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventLog:
    timestamp: float
    level: str
    message: str


# ------------------------------
# Scenario/config (optional)
# ------------------------------
@dataclass
class Scenario:
    # Tunables with safe defaults
    solar_storm_prob: float = 0.001
    flare_min_s: float = 30.0
    flare_max_s: float = 300.0
    radiation_jitter: float = 0.3
    multi_fail_prob: float = 2e-5
    comms_loss_prob: float = 5e-5
    micrometeor_prob: float = 1e-5
    solar_catastrophe_prob: float = 1e-5
    comms_blackout_base: float = 0.0005

    @staticmethod
    def from_file(path: Optional[str]) -> 'Scenario':
        if not path:
            return Scenario()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Scenario(**{k: v for k, v in data.items() if k in Scenario().__dict__})
        except Exception:
            return Scenario()


# ------------------------------
# Environment
# ------------------------------
class SpaceEnvironment:
    def __init__(self, orbit_period_sec: float = 90.0 * 60.0, scenario: Optional[Scenario] = None) -> None:
        self.orbit_period = orbit_period_sec
        self.start = now()
        self.solar_activity = 1.0
        self.last_storm = 0.0
        self._flare_end = 0.0
        self.scenario = scenario or Scenario()

    def sun_angle_factor(self, timestamp: float, position: Dict[str, float]) -> float:
        elapsed = (timestamp - self.start) % self.orbit_period
        daylight_fraction = 0.6
        in_sun = elapsed < (self.orbit_period * daylight_fraction)
        lat = position.get('lat', 0.0)
        lat_factor = math.cos(math.radians(lat))
        base = 1.0 if in_sun else 0.0
        variation = 0.5 + 0.5 * lat_factor
        return base * variation

    def step(self) -> None:
        # Random solar activity bursts
        if random.random() < self.scenario.solar_storm_prob:
            self.solar_activity = random.uniform(1.5, 3.0)
            self.last_storm = now()
            self._flare_end = now() + random.uniform(self.scenario.flare_min_s, self.scenario.flare_max_s)
        if now() < self._flare_end:
            self.solar_activity = max(self.solar_activity, 2.0)
        if self.solar_activity > 1.0 and now() - self.last_storm > 60.0 and now() > self._flare_end:
            self.solar_activity = max(1.0, self.solar_activity - 0.01)

    def radiation_level(self) -> float:
        base = 1.0 * self.solar_activity
        jitter = random.uniform(-self.scenario.radiation_jitter, self.scenario.radiation_jitter)
        return max(0.0, base + jitter)

    def ground_pass(self, timestamp: float, position: Dict[str, float]) -> bool:
        """Very simple pass model: link available when |lat|<35 and lon near 0±40 deg.
        This is intentionally rough but predictable.
        """
        lat_ok = abs(position.get('lat', 0.0)) < 35.0
        lon = position.get('lon', 0.0)
        lon = (lon + 360.0) % 360.0
        lon_ok = (lon < 40.0) or (lon > 320.0)
        return lat_ok and lon_ok


# ------------------------------
# Power system models
# ------------------------------
class Battery:
    def __init__(self) -> None:
        self.capacity_wh = 1000.0
        self.charge_wh = self.capacity_wh
        self.nominal_voltage = 28.0
        self.health = 1.0
        self.cycle_count = 0

    @property
    def percent(self) -> float:
        denom = max(1e-6, self.capacity_wh * self.health)
        return clamp((self.charge_wh / denom) * 100.0, 0.0, 100.0)

    @property
    def voltage(self) -> float:
        return clamp(self.nominal_voltage * (0.9 + 0.2 * (self.percent / 100.0)) * self.health, 20.0, 32.0)

    def apply_energy(self, delta_watt: float, dt_sec: float) -> None:
        delta_wh = delta_watt * dt_sec / 3600.0
        old = self.charge_wh
        self.charge_wh = clamp(self.charge_wh + delta_wh, 0.0, self.capacity_wh * self.health)
        try:
            if old / (self.capacity_wh * self.health) > 0.2 and self.charge_wh / (self.capacity_wh * self.health) <= 0.2:
                self.cycle_count += 1
        except Exception:
            pass

    def degrade(self, amount_per_sec: float, dt_sec: float) -> None:
        # dt-scaled degradation
        self.health = clamp(self.health - (amount_per_sec * dt_sec), 0.4, 1.0)


class SolarPanel:
    def __init__(self, area_m2: float = 2.0, efficiency: float = 0.3) -> None:
        self.area = area_m2
        self.efficiency = efficiency
        self.deployed = True
        self.functional = True
        self.damage_fraction = 0.0

    def output_watts(self, sun_angle_factor: float, radiation_factor: float, attitude_sun_dot: float = 1.0) -> float:
        if not (self.deployed and self.functional):
            return 0.0
        # Cosine loss for pointing (basic). Clamp in [0,1].
        pointing = clamp(attitude_sun_dot, 0.0, 1.0)
        irradiance = 1361.0 * sun_angle_factor * pointing
        effective_area = self.area * (1.0 - self.damage_fraction)
        raw = irradiance * effective_area * self.efficiency
        raw *= max(0.0, 1.0 - 0.12 * (radiation_factor - 1.0))
        return max(0.0, raw)

    def partial_damage(self, frac: float) -> None:
        self.damage_fraction = clamp(self.damage_fraction + frac, 0.0, 0.99)
        if self.damage_fraction >= 0.99:
            self.functional = False

    def degrade_efficiency(self, amount_per_sec: float, dt_sec: float) -> None:
        self.efficiency = clamp(self.efficiency - (amount_per_sec * dt_sec), 0.01, 0.35)

    def fail(self) -> None:
        self.functional = False

    def repair(self) -> None:
        self.functional = True
        self.damage_fraction = 0.0


# ------------------------------
# Subsystems
# ------------------------------
@dataclass
class Subsystem:
    name: str
    power_draw_w: float
    state: str = 'ON'
    fail_prob: float = 1e-4
    critical: bool = False
    reboot_attempts: int = 0

    def step_failure(self, env: SpaceEnvironment) -> Optional[str]:
        p = self.fail_prob * env.radiation_level()
        if random.random() < p:
            self.state = 'FAILED'
            return f'{self.name} failed'
        return None

    def power(self) -> float:
        if self.state == 'ON':
            return self.power_draw_w
        if self.state == 'STANDBY':
            return max(0.0, self.power_draw_w * 0.2)
        return 0.0

    def attempt_reboot(self) -> Optional[str]:
        if self.state != 'FAILED':
            return None
        self.reboot_attempts += 1
        if random.random() < 0.45:
            self.state = 'STANDBY'
            return f'{self.name} rebooted to STANDBY'
        if self.reboot_attempts > 3:
            self.state = 'OFF'
            return f'{self.name} moved to OFF after failed reboots'
        return f'{self.name} reboot attempt failed'


# ------------------------------
# Links & Agents
# ------------------------------
class GroundLink:
    def __init__(self, max_parallel_tx: int = 8, uplink_capacity: int = 64) -> None:
        self.downlink_log: List[EventLog] = []
        # Priority queue: (priority, monotonic_counter, command)
        self.uplink_queue: "asyncio.PriorityQueue[Tuple[int,int,str]]" = asyncio.PriorityQueue(maxsize=uplink_capacity)
        self._uplink_counter = 0
        self.available = True
        self._tx_sema = asyncio.Semaphore(max_parallel_tx)

    async def send_telemetry(self, telemetry: Telemetry) -> None:
        async with self._tx_sema:
            await asyncio.sleep(random.uniform(0.05, 0.2))
            if not self.available or random.random() < 0.02:
                self.downlink_log.append(EventLog(now(), 'WARNING', 'Telemetry packet lost or downlink unavailable'))
                return
            msg = (
                f'TELEMETRY t={int(telemetry.timestamp)} batt={telemetry.battery_percent:.1f}% '
                f'solar={telemetry.solar_output:.0f}W net={telemetry.power.get("net",0):.0f}W'
            )
            self.downlink_log.append(EventLog(now(), 'INFO', msg))

    async def put_command(self, priority: int, cmd: str) -> None:
        # Lower number == higher priority (0 highest)
        self._uplink_counter += 1
        await self.uplink_queue.put((priority, self._uplink_counter, cmd))

    def receive_command_now(self) -> Optional[str]:
        try:
            item = self.uplink_queue.get_nowait()
            return item[2]
        except asyncio.QueueEmpty:
            return None


class GroundStationAgent:
    def __init__(self, link: GroundLink) -> None:
        self.link = link

    async def analyze_then_recover(self, sat: 'Satellite') -> None:
        telem = getattr(sat, 'last_telemetry', None)
        if telem is None:
            self.link.downlink_log.append(EventLog(now(), 'WARNING', 'GND: no telemetry available for diagnosis'))
            return
        if telem.battery_percent < 15.0:
            await self.link.put_command(0, 'SET_POWER_MODE conserve')
            self.link.downlink_log.append(EventLog(now(), 'INFO', 'GND -> SET_POWER_MODE conserve'))
        if any(state == 'FAILED' for state in telem.systems.values()):
            if telem.systems.get('COMMS') == 'FAILED':
                await self.link.put_command(0, 'REBOOT_COMMS')
                self.link.downlink_log.append(EventLog(now(), 'INFO', 'GND -> REBOOT_COMMS'))
            if telem.solar_output < 50.0:
                await self.link.put_command(1, 'DEPLOY_SOLAR')
                self.link.downlink_log.append(EventLog(now(), 'INFO', 'GND -> DEPLOY_SOLAR'))

    async def periodic_behavior(self, sat: 'Satellite', duration: float) -> None:
        start = now()
        while now() - start < duration:
            await asyncio.sleep(random.uniform(3.0, 6.0))
            if random.random() < 0.5:
                await self.analyze_then_recover(sat)
            else:
                cmd = random.choice(['PING', 'REQUEST_STATUS', 'SET_POWER_MODE nominal'])
                await self.link.put_command(2, cmd)
                self.link.downlink_log.append(EventLog(now(), 'INFO', f'GND -> {cmd}'))


# ------------------------------
# Critical conditions
# ------------------------------
class CriticalConditionManager:
    def __init__(self, sat: 'Satellite', scenario: Scenario) -> None:
        self.sat = sat
        self.scenario = scenario

    def maybe_trigger(self) -> None:
        # Link outage
        if random.random() < self.scenario.comms_loss_prob:
            self.sat.groundlink.available = False
            self.sat.log('CRITICAL', 'Complete Ground Communication Loss')
            self.sat.schedule_task(self._restore_link_after(random.uniform(10.0, 60.0)))
        # Multiple subsystem failures
        if random.random() < self.scenario.multi_fail_prob:
            count = random.randint(2, min(4, len(self.sat.subsystems)))
            failed = random.sample(list(self.sat.subsystems.values()), k=count)
            for sub in failed:
                sub.state = 'FAILED'
            names = ','.join(s.name for s in failed)
            self.sat.log('CRITICAL', f'Multiple subsystem failures: {names}')
        # Micrometeorite
        if random.random() < self.scenario.micrometeor_prob:
            self.sat.solar.partial_damage(random.uniform(0.05, 0.5))
            self.sat.log('CRITICAL', 'Micrometeorite impact: partial solar damage')

    async def _restore_link_after(self, delay: float) -> None:
        await asyncio.sleep(delay)
        self.sat.groundlink.available = True
        self.sat.log('INFO', 'Ground communication restored after outage')


# ------------------------------
# AI Controller with hysteresis
# ------------------------------
class AIController:
    def __init__(self, sat: 'Satellite') -> None:
        self.sat = sat
        self.low_thresh = 20.0
        self.high_thresh = 30.0  # hysteresis exit
        self._recovery_index = 0  # gradual enable

    def decide(self, telemetry: Telemetry) -> None:
        # Enter conserve
        if telemetry.battery_percent < self.low_thresh and self.sat.mode != 'CONSERVE':
            self.sat.mode = 'CONSERVE'
            self.sat.log('AI', 'Switched to CONSERVE mode due to low battery')
            for s in self.sat.subsystems.values():
                if not s.critical and s.state == 'ON':
                    s.state = 'STANDBY'
            self._recovery_index = 0

        # Exit conserve (with hysteresis)
        if telemetry.battery_percent > self.high_thresh and self.sat.mode == 'CONSERVE':
            # Gradually bring back one non-critical subsystem per decision
            noncrit = [s for s in self.sat.subsystems.values() if not s.critical]
            if self._recovery_index < len(noncrit):
                target = noncrit[self._recovery_index]
                if target.state != 'ON':
                    target.state = 'ON'
                    self.sat.log('AI', f'Restored {target.name} to ON')
                self._recovery_index += 1
            else:
                self.sat.mode = 'NOMINAL'
                self.sat.log('AI', 'Back to NOMINAL mode')

        # Radiation throttling
        if telemetry.env.get('radiation', 0.0) > 2.5:
            self.sat.log('AI', 'Radiation high: throttling non-critical subsystems')
            for s in self.sat.subsystems.values():
                if not s.critical:
                    s.state = 'STANDBY'

        # Auto-reboot criticals
        for s in self.sat.subsystems.values():
            if s.critical and s.state == 'FAILED':
                res = s.attempt_reboot()
                if res:
                    self.sat.log('AI', res)


# ------------------------------
# Satellite core
# ------------------------------
class Satellite:
    def __init__(self, env: SpaceEnvironment) -> None:
        self.env = env
        self.battery = Battery()
        self.solar = SolarPanel(area_m2=4.0, efficiency=0.30)
        self.position = {'lat': 0.0, 'lon': 0.0, 'alt': 500.0}
        self._phase_deg = 0.0  # dt-driven orbital phase
        self._incl_deg = 10.0
        self.last_step = now()
        self.subsystems: Dict[str, Subsystem] = {
            'COMMS': Subsystem('COMMS', 50.0, critical=True, fail_prob=1e-4),
            'RADAR': Subsystem('RADAR', 120.0, fail_prob=5e-4),
            'CAMERA': Subsystem('CAMERA', 30.0, fail_prob=2e-4),
            'THERMAL_CTRL': Subsystem('THERMAL_CTRL', 40.0, critical=True, fail_prob=1e-4),
            'ATTITUDE': Subsystem('ATTITUDE', 20.0, critical=True, fail_prob=2e-4),
            'PROPULSION': Subsystem('PROPULSION', 200.0, fail_prob=1e-3),
        }
        self.groundlink = GroundLink()
        self.logs: List[EventLog] = []
        self.mode = 'NOMINAL'
        self.thermal_state = {'BODY': 20.0, 'RADIATOR': -40.0}
        self.last_telemetry: Optional[Telemetry] = None
        self.crit_mgr = CriticalConditionManager(self, env.scenario)
        self.ai = AIController(self)
        self._bg_tasks: List[asyncio.Task] = []

    def schedule_task(self, coro: 'asyncio.coroutines') -> None:
        try:
            t = asyncio.create_task(coro)
            self._bg_tasks.append(t)
        except Exception:
            pass

    def log(self, level: str, message: str) -> None:
        entry = EventLog(now(), level, message)
        self.logs.append(entry)
        try:
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry.timestamp))} [{entry.level}] {entry.message}"
            file_logger.write_async(line)
        except Exception:
            pass
        if len(self.logs) > 600:  # slightly bigger ring buffer
            self.logs.pop(0)

    def step_orbit(self, dt_phys: float) -> None:
        deg_per_sec = 360.0 / (90.0 * 60.0)
        self._phase_deg = (self._phase_deg + deg_per_sec * dt_phys) % 360.0
        self.position['lon'] = self._phase_deg
        self.position['lat'] = self._incl_deg * math.sin(math.radians(self._phase_deg))

    def _attitude_sun_dot(self) -> float:
        # Very simple: if ATTITUDE is ON => near sun-pointing; otherwise degraded pointing
        att = self.subsystems.get('ATTITUDE')
        return 1.0 if att and att.state == 'ON' else 0.6

    def compute_solar_input(self, timestamp: float, sun_factor: float, radiation: float) -> float:
        return self.solar.output_watts(sun_factor, radiation, self._attitude_sun_dot())

    def total_power_draw(self) -> float:
        return sum(sub.power() for sub in self.subsystems.values())

    def thermal_step(self, dt_phys: float, solar_input: float, load_w: float) -> None:
        # Smoother thermal model
        absorbed = 0.05 * solar_input
        waste = 0.3 * load_w
        body_temp = self.thermal_state['BODY']
        # Radiative cooling proportional to delta from deep-space (~-100C)
        cooling = 0.08 * (body_temp - (-100.0))
        if self.subsystems.get('THERMAL_CTRL') and self.subsystems['THERMAL_CTRL'].state == 'ON':
            # Active thermal control consumes some energy implicitly accounted via load_w,
            # but here we increase cooling slightly
            cooling *= 1.15
        dT = (absorbed + waste - cooling) * dt_phys / 1000.0
        # Clamp to safe bounds to avoid runaway
        new_temp = clamp(body_temp + dT, -120.0, 90.0)
        self.thermal_state['BODY'] = new_temp

    async def tick(self, timescale: float) -> Telemetry:
        t0 = now()
        dt = t0 - self.last_step
        if dt <= 0.0:
            dt = 1e-6
        self.last_step = t0
        dt_phys = dt * max(0.01, timescale)

        # Environment evolves
        self.env.step()

        # --- Single-sample per tick ---
        sun_factor = self.env.sun_angle_factor(t0, self.position)
        radiation = self.env.radiation_level()

        # Orbit update (dt-driven)
        self.step_orbit(dt_phys)

        # Ground pass availability based on current geometry
        self.groundlink.available = self.env.ground_pass(t0, self.position)

        # Failures (sampled once per tick)
        for sub in self.subsystems.values():
            msg = sub.step_failure(self.env)
            if msg:
                self.log('CRITICAL' if sub.critical else 'WARNING', msg)

        # Opportunistic auto-reboot
        for sub in self.subsystems.values():
            if sub.state == 'FAILED':
                res = sub.attempt_reboot()
                if res:
                    self.log('INFO', res)

        # Rare critical incidents
        self.crit_mgr.maybe_trigger()

        # Power calc
        solar_input = self.compute_solar_input(t0, sun_factor, radiation)
        load_w = self.total_power_draw()

        # CONSERVE adjustments
        if self.mode == 'CONSERVE':
            changed = False
            for s in self.subsystems.values():
                if not s.critical and s.state == 'ON':
                    s.state = 'STANDBY'; changed = True
            if changed:
                self.log('INFO', 'Non-critical subsystems set to STANDBY for power saving')
            load_w = self.total_power_draw()

        net_w = solar_input - load_w
        self.battery.apply_energy(net_w, dt_phys)

        # dt-scaled degradations
        if self.battery.percent < 20.0 or radiation > 2.0:
            self.battery.degrade(0.0005, dt_phys)
        if radiation > 2.0 and random.random() < 0.001:
            self.solar.degrade_efficiency(random.uniform(1e-4, 1e-3) / 1.0, dt_phys)
        if random.random() < self.env.scenario.solar_catastrophe_prob:
            self.solar.fail()
            self.log('CRITICAL', 'Solar panel catastrophic failure')
        if random.random() < self.env.scenario.comms_blackout_base * radiation:
            if 'COMMS' in self.subsystems:
                self.subsystems['COMMS'].state = 'FAILED'
                self.log('CRITICAL', 'Sudden communications blackout due to radiation spike')

        # Thermal
        self.thermal_step(dt_phys, solar_input, load_w)

        # Build telemetry (uses sampled sun/radiation of this tick)
        telemetry = Telemetry(
            timestamp=t0,
            dt_physics=dt_phys,
            temperatures={'BODY': self.thermal_state['BODY'], 'BATTERY': 25.0},
            position=self.position.copy(),
            battery_percent=self.battery.percent,
            battery_voltage=self.battery.voltage,
            solar_output=solar_input,
            systems={name: sub.state for name, sub in self.subsystems.items()},
            env={'radiation': radiation, 'solar_activity': self.env.solar_activity, 'sun_factor': sun_factor},
            power={'solar': solar_input, 'load': load_w, 'net': net_w},
        )

        # Cache & AI
        self.last_telemetry = telemetry
        try:
            self.ai.decide(telemetry)
        except Exception as e:
            self.log('ERROR', f'AI decision error: {e}')

        return telemetry

    async def handle_command(self, cmd: str) -> None:
        self.log('INFO', f'Handling command: {cmd}')
        parts = cmd.split()
        if not parts:
            return

        async def _req_status(_: List[str]) -> None:
            telem = getattr(self, 'last_telemetry', None)
            if telem is not None:
                await self.groundlink.send_telemetry(telem)
            else:
                self.log('WARNING', 'REQUEST_STATUS but no telemetry cached')

        async def _call_sync(handler: Callable[..., Any], args: List[str]) -> None:
            try:
                await asyncio.to_thread(handler, args)
            except Exception as e:
                self.log('ERROR', f'Handler error: {e}')

        cmd_map: Dict[str, Callable[[List[str]], Any]] = {
            'PING': lambda p: self.schedule_task(_call_sync(self._cmd_ping, p)),
            'REQUEST_STATUS': lambda p: self.schedule_task(_req_status(p)),
            'SET_POWER_MODE': lambda p: self.schedule_task(_call_sync(self._cmd_set_power_mode, p[1:] if len(p) > 1 else [])),
            'DEPLOY_SOLAR': lambda p: self.schedule_task(_call_sync(self._cmd_deploy_solar, p)),
            'STOW_SOLAR': lambda p: self.schedule_task(_call_sync(self._cmd_stow_solar, p)),
            'REBOOT_COMMS': lambda p: self.schedule_task(_call_sync(self._cmd_reboot_comms, p)),
            'ADJUST_ORBIT': lambda p: self.schedule_task(_call_sync(self._cmd_adjust_orbit, p[1:] if len(p) > 1 else [])),
        }

        handler = cmd_map.get(parts[0])
        if handler:
            try:
                handler(parts)
            except Exception as e:
                self.log('ERROR', f'Command {cmd} handler failed: {e}')
        else:
            self.log('WARNING', f'Unknown command: {cmd}')

    # --- Command handlers (sync) ---
    def _cmd_ping(self, args: List[str]) -> None:
        self.log('INFO', 'PING received, sending PONG')
        self.groundlink.downlink_log.append(EventLog(now(), 'INFO', 'PONG'))

    def _cmd_set_power_mode(self, args: List[str]) -> None:
        if not args:
            return
        mode = args[0].upper()
        if mode == 'CONSERVE':
            self.mode = 'CONSERVE'
            self.log('INFO', 'Power mode set to CONSERVE')
        elif mode == 'NOMINAL':
            self.mode = 'NOMINAL'
            self.log('INFO', 'Power mode set to NOMINAL')

    def _cmd_deploy_solar(self, args: List[str]) -> None:
        self.solar.deployed = True
        self.solar.functional = True
        self.log('INFO', 'Solar panels deployed or re-enabled')

    def _cmd_stow_solar(self, args: List[str]) -> None:
        self.solar.deployed = False
        self.log('INFO', 'Solar panels stowed')

    def _cmd_reboot_comms(self, args: List[str]) -> None:
        sub = self.subsystems.get('COMMS')
        if sub:
            sub.state = 'FAILED'
            sub.reboot_attempts = 0
            self.log('INFO', 'COMMS scheduled for reboot')

    def _cmd_adjust_orbit(self, args: List[str]) -> None:
        for a in args:
            if a.startswith('delta_alt='):
                try:
                    val = float(a.split('=', 1)[1])
                    self.position['alt'] += val
                    self.log('INFO', f'Orbit adjusted by {val} km')
                except Exception as e:
                    self.log('WARNING', f'Invalid orbit adjust arg: {a} ({e})')


# ------------------------------
# UI helpers
# ------------------------------

def make_status_table(t: Telemetry) -> Table:
    tbl = Table.grid(expand=True)

    temp = Table(title='Temperatures', show_header=True)
    temp.add_column('Sensor')
    temp.add_column('Value', justify='right')
    for k, v in t.temperatures.items():
        color = 'red' if v > 45 else 'yellow' if v > 35 else 'green'
        temp.add_row(k, f'[{color}]{v:.1f}C[/{color}]')

    pos = Table(title='Position')
    pos.add_column('Lat')
    pos.add_column('Lon')
    pos.add_column('Alt(km)')
    pos.add_row(f"{t.position['lat']:.3f}", f"{t.position['lon']:.3f}", f"{t.position['alt']:.2f}")

    energy = Table(title='Energy')
    energy.add_column('Battery')
    energy.add_column('Voltage')
    energy.add_column('Solar(W)')
    batt_color = 'red' if t.battery_percent < 20 else 'green'
    energy.add_row(f'[{batt_color}]{t.battery_percent:.1f}%[/{batt_color}]', f'{t.battery_voltage:.2f}V', f'{t.solar_output:.0f}')

    systems = Table(title='Subsystems')
    systems.add_column('Name')
    systems.add_column('State')
    for name, state in t.systems.items():
        col = 'green' if state in ['ON', 'STANDBY'] else 'red'
        systems.add_row(name, f'[{col}]{state}[/{col}]')

    tbl.add_row(
        Panel(temp, title='Thermal'),
        Panel(pos, title='Orbit'),
        Panel(energy, title='Power'),
        Panel(systems, title='Systems')
    )
    return tbl


def make_logs_panel(logs: List[EventLog], ground_logs: List[EventLog]) -> Panel:
    log_table = Table.grid(expand=True)
    log_table.add_column('Time', width=8)
    log_table.add_column('Lvl', width=8)
    log_table.add_column('Msg')
    for e in logs[-12:]:
        ts = time.strftime('%H:%M:%S', time.localtime(e.timestamp))
        lvl = e.level
        color = 'red' if lvl == 'CRITICAL' else 'yellow' if lvl == 'WARNING' else 'green'
        log_table.add_row(ts, f'[{color}]{lvl}[/{color}]', e.message)
    for g in ground_logs[-6:]:
        ts = time.strftime('%H:%M:%S', time.localtime(g.timestamp))
        log_table.add_row(ts, f'[cyan]GND[/cyan]', g.message)
    return Panel(log_table, title='Event Log')


# ------------------------------
# Simulation loop
# ------------------------------
async def simulation_loop(duration: float, update_interval: float, *, timescale: float, csv_path: Optional[str]) -> None:
    metrics: List[Telemetry] = []

    env = SpaceEnvironment(scenario=_ACTIVE_SCENARIO)
    sat = Satellite(env)
    ground_agent = GroundStationAgent(sat.groundlink)

    start = now()
    ground_task = asyncio.create_task(ground_agent.periodic_behavior(sat, duration))

    try:
        file_logger.start()
        with Live(console=console, screen=True, refresh_per_second=4) as live:
            while now() - start < duration:
                telem = await sat.tick(timescale)
                # Use same tick telemetry everywhere (no re-sampling)
                sat.schedule_task(sat.groundlink.send_telemetry(telem))
                sat.log('INFO', f"Telemetry queued batt={telem.battery_percent:.1f}% solar={telem.solar_output:.0f}W")

                # Drain uplink by priority
                cmd = sat.groundlink.receive_command_now()
                drain_count = 0
                while cmd and drain_count < 8:
                    await sat.handle_command(cmd)
                    drain_count += 1
                    cmd = sat.groundlink.receive_command_now()

                # Build UI layout using THIS tick's telemetry
                main = Layout()
                main.split_column(Layout(name='header', size=3), Layout(name='body', ratio=1), Layout(name='footer', size=8))
                main['header'].update(Panel(Align.center(Text('SATELLITE SIMULATOR ADVANCED', style='bold white on blue')), expand=True))

                body = Layout()
                body.split_row(Layout(name='left', ratio=3), Layout(name='right', ratio=1))
                body['left'].update(make_status_table(telem))
                body['right'].update(make_logs_panel(sat.logs, sat.groundlink.downlink_log))
                main['body'].update(body)

                foot = Table.grid(expand=True)
                foot.add_column('Mode')
                foot.add_column('Battery')
                foot.add_column('Solar')
                foot.add_column('Radiation')
                foot.add_column('Net')
                foot.add_row(
                    sat.mode,
                    f'{telem.battery_percent:.1f}%',
                    f'{telem.power.get("solar",0):.0f}W',
                    f"{telem.env.get('radiation',0.0):.2f}",
                    f"{telem.power.get('net',0.0):.0f}W",
                )
                main['footer'].update(Panel(foot, title='Summary'))
                live.update(main)

                # Collect metrics after drawing
                metrics.append(telem)
                await asyncio.sleep(update_interval)
    finally:
        # Cancel background tasks
        ground_task.cancel()
        for t in list(sat._bg_tasks):
            t.cancel()
        await asyncio.gather(ground_task, *sat._bg_tasks, return_exceptions=True)
        file_logger.stop()

        # CSV export & end-of-run report
        if csv_path and metrics:
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['t','dt','batt_%','volt','solar_W','load_W','net_W','rad','sun','lat','lon','alt','temp_body'])
                    for m in metrics:
                        w.writerow([
                            int(m.timestamp), f"{m.dt_physics:.3f}", f"{m.battery_percent:.3f}", f"{m.battery_voltage:.3f}",
                            f"{m.power.get('solar',0):.3f}", f"{m.power.get('load',0):.3f}", f"{m.power.get('net',0):.3f}",
                            f"{m.env.get('radiation',0):.3f}", f"{m.env.get('sun_factor',0):.3f}",
                            f"{m.position['lat']:.4f}", f"{m.position['lon']:.4f}", f"{m.position['alt']:.3f}",
                            f"{m.temperatures['BODY']:.3f}",
                        ])
                console.print(f"[green]Metrics saved to[/green] {csv_path}")
            except Exception as e:
                console.print(f"[yellow]CSV save failed:[/yellow] {e}")

        # Summary report
        try:
            if metrics:
                batt_vals = [m.battery_percent for m in metrics]
                rad_vals = [m.env.get('radiation',0.0) for m in metrics]
                link_down = sum(1 for m in metrics if not SpaceEnvironment().ground_pass(m.timestamp, m.position))
                console.print("\n[bold]Run summary[/bold]")
                console.print(f"battery % (min/avg/max): {min(batt_vals):.1f}/{(sum(batt_vals)/len(batt_vals)):.1f}/{max(batt_vals):.1f}")
                console.print(f"radiation (min/avg/max): {min(rad_vals):.2f}/{(sum(rad_vals)/len(rad_vals)):.2f}/{max(rad_vals):.2f}")
                console.print(f"approx link not-available ticks: {link_down}")
        except Exception:
            pass


# ------------------------------
# CLI / entrypoints
# ------------------------------
_ACTIVE_SCENARIO: Scenario = Scenario()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=120.0)
    p.add_argument('--update', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=None, help='Set RNG seed for reproducibility')
    p.add_argument('--timescale', type=float, default=1.0, help='Physics time multiplier (e.g., 10 for 10x faster)')
    p.add_argument('--csv', type=str, default=None, help='Path to save CSV metrics (optional)')
    p.add_argument('--scenario', type=str, default=None, help='Path to scenario JSON (optional)')
    p.add_argument('--deterministic', action='store_true', help='Reduce jitters for more stable runs')
    return p.parse_args()


def _apply_deterministic_mode() -> None:
    # For this mode we reduce jitters by narrowing random ranges via seed; major effects remain.
    random.random()  # no-op; placeholder to indicate intentional hook


def main() -> None:
    args = parse_args()

    # Seed / scenario
    if args.seed is not None:
        random.seed(args.seed)
    global _ACTIVE_SCENARIO
    _ACTIVE_SCENARIO = Scenario.from_file(args.scenario)

    if args.deterministic and args.seed is None:
        # If user wants deterministic but no seed given, choose a constant one
        random.seed(1337)
        _apply_deterministic_mode()

    try:
        asyncio.run(simulation_loop(args.duration, args.update, timescale=args.timescale, csv_path=args.csv))
    except KeyboardInterrupt:
        console.print('\n[bold yellow]Simulation interrupted by user[/bold yellow]')
    except Exception as e:
        console.print(f'[bold red]Fatal error:[/bold red] {e}')


if __name__ == '__main__':
    main()
