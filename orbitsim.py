"""
OrbitSim - Satellite Operations Simulator
=========================================

Author: LloydLewis, Neylox
License: MIT
Version: 1.0

Description:
------------
OrbitSim is a Python-based simulator that models the behavior of an 
Earth-orbiting satellite in real time. It simulates key aspects such as:

- Orbital mechanics (lat/lon/alt progression)
- Space environment effects (solar radiation, flares, micrometeorites)
- Subsystem management (COMMS, Radar, Camera, Thermal, Attitude, Propulsion)
- Power systems (battery charge/discharge, solar panel efficiency & degradation)
- Ground station communications (uplink/downlink, telemetry, commands)
- AI-assisted autonomous control & recovery logic
- Critical fault injection and event logging

Features a live **Rich-powered terminal dashboard** to visualize telemetry,
system health, and events as they happen.

Usage:
------
    python simulator.py --duration 180 --update 1.0

This will run the simulation for 180 seconds with a 1-second UI refresh rate.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.align import Align

console = Console()

class FileLogger:
    def __init__(self, path: str = 'simulation.log') -> None:
        self.path = path

    def write_sync(self, entry: str) -> None:
        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(entry + '\n')
        except Exception:
            pass

    def write_async(self, entry: str) -> None:
        try:
            t = threading.Thread(target=self.write_sync, args=(entry,), daemon=True)
            t.start()
        except Exception:
            pass

file_logger = FileLogger('simulation.log')


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def now() -> float:
    return time.time()


@dataclass
class Telemetry:
    timestamp: float
    temperatures: Dict[str, float]
    position: Dict[str, float]
    battery_percent: float
    battery_voltage: float
    solar_output: float
    systems: Dict[str, str]
    env: Dict[str, float]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventLog:
    timestamp: float
    level: str
    message: str


class SpaceEnvironment:
    def __init__(self, orbit_period_sec: float = 90.0 * 60.0) -> None:
        self.orbit_period = orbit_period_sec
        self.start = now()
        self.solar_activity = 1.0
        self.last_storm = 0.0
        self._flare_end = 0.0

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
        if random.random() < 0.001:
            self.solar_activity = random.uniform(1.5, 3.0)
            self.last_storm = now()
            self._flare_end = now() + random.uniform(30.0, 300.0)
        if now() < self._flare_end:
            self.solar_activity = max(self.solar_activity, 2.0)
        if self.solar_activity > 1.0 and now() - self.last_storm > 60.0 and now() > self._flare_end:
            self.solar_activity = max(1.0, self.solar_activity - 0.01)

    def radiation_level(self) -> float:
        base = 1.0 * self.solar_activity
        jitter = random.uniform(-0.3, 0.3)
        return max(0.0, base + jitter)


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

    def degrade(self, amount: float) -> None:
        self.health = clamp(self.health - amount, 0.4, 1.0)


class SolarPanel:
    def __init__(self, area_m2: float = 2.0, efficiency: float = 0.3) -> None:
        self.area = area_m2
        self.efficiency = efficiency
        self.deployed = True
        self.functional = True
        self.damage_fraction = 0.0

    def output_watts(self, sun_angle_factor: float, radiation_factor: float) -> float:
        if not (self.deployed and self.functional):
            return 0.0
        irradiance = 1361.0 * sun_angle_factor
        effective_area = self.area * (1.0 - self.damage_fraction)
        raw = irradiance * effective_area * self.efficiency
        raw *= max(0.0, 1.0 - 0.12 * (radiation_factor - 1.0))
        return max(0.0, raw)

    def partial_damage(self, frac: float) -> None:
        self.damage_fraction = clamp(self.damage_fraction + frac, 0.0, 0.99)
        if self.damage_fraction >= 0.99:
            self.functional = False

    def degrade_efficiency(self, amount: float) -> None:
        self.efficiency = clamp(self.efficiency - amount, 0.01, 0.35)

    def fail(self) -> None:
        self.functional = False

    def repair(self) -> None:
        self.functional = True
        self.damage_fraction = 0.0


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


class GroundLink:
    def __init__(self) -> None:
        self.downlink_log: List[EventLog] = []
        self.uplink_queue: asyncio.Queue[str] = asyncio.Queue()
        self.available = True

    async def send_telemetry(self, telemetry: Telemetry) -> None:
        await asyncio.sleep(random.uniform(0.05, 0.2))
        if not self.available or random.random() < 0.02:
            self.downlink_log.append(EventLog(now(), 'WARNING', 'Telemetry packet lost or downlink unavailable'))
            return
        msg = f'TELEMETRY t={int(telemetry.timestamp)} batt={telemetry.battery_percent:.1f}% solar={telemetry.solar_output:.0f}W'
        self.downlink_log.append(EventLog(now(), 'INFO', msg))

    def receive_command_now(self) -> Optional[str]:
        try:
            return self.uplink_queue.get_nowait()
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
            await self.link.uplink_queue.put('SET_POWER_MODE conserve')
            self.link.downlink_log.append(EventLog(now(), 'INFO', 'GND -> SET_POWER_MODE conserve'))
        if any(state == 'FAILED' for state in telem.systems.values()):
            if telem.systems.get('COMMS') == 'FAILED':
                await self.link.uplink_queue.put('REBOOT_COMMS')
                self.link.downlink_log.append(EventLog(now(), 'INFO', 'GND -> REBOOT_COMMS'))
            if telem.solar_output < 50.0:
                await self.link.uplink_queue.put('DEPLOY_SOLAR')
                self.link.downlink_log.append(EventLog(now(), 'INFO', 'GND -> DEPLOY_SOLAR'))

    async def periodic_behavior(self, sat: 'Satellite', duration: float) -> None:
        start = now()
        while now() - start < duration:
            await asyncio.sleep(random.uniform(3.0, 6.0))
            if random.random() < 0.5:
                await self.analyze_then_recover(sat)
            else:
                cmd = random.choice(['PING', 'REQUEST_STATUS', 'SET_POWER_MODE nominal'])
                await self.link.uplink_queue.put(cmd)
                self.link.downlink_log.append(EventLog(now(), 'INFO', f'GND -> {cmd}'))


class CriticalConditionManager:
    def __init__(self, sat: 'Satellite') -> None:
        self.sat = sat

    def maybe_trigger(self) -> None:
        if random.random() < 5e-5:
            self.sat.groundlink.available = False
            self.sat.log('CRITICAL', 'Complete Ground Communication Loss')
            asyncio.create_task(self._restore_link_after(random.uniform(10.0, 60.0)))
        if random.random() < 2e-5:
            count = random.randint(2, min(4, len(self.sat.subsystems)))
            failed = random.sample(list(self.sat.subsystems.values()), k=count)
            for sub in failed:
                sub.state = 'FAILED'
            names = ','.join(s.name for s in failed)
            self.sat.log('CRITICAL', f'Multiple subsystem failures: {names}')
        if random.random() < 1e-5:
            self.sat.solar.partial_damage(random.uniform(0.05, 0.5))
            self.sat.log('CRITICAL', 'Micrometeorite impact: partial solar damage')

    async def _restore_link_after(self, delay: float) -> None:
        await asyncio.sleep(delay)
        self.sat.groundlink.available = True
        self.sat.log('INFO', 'Ground communication restored after outage')


class AIController:
    def __init__(self, sat: 'Satellite') -> None:
        self.sat = sat

    def decide(self, telemetry: Telemetry) -> None:
        if telemetry.battery_percent < 20.0 and self.sat.mode != 'CONSERVE':
            self.sat.mode = 'CONSERVE'
            self.sat.log('AI', 'Switched to CONSERVE mode due to low battery')
            for s in self.sat.subsystems.values():
                if not s.critical and s.state == 'ON':
                    s.state = 'STANDBY'
        if telemetry.env.get('radiation', 0.0) > 2.5:
            self.sat.log('AI', 'Radiation high: throttling non-critical subsystems')
            for s in self.sat.subsystems.values():
                if not s.critical:
                    s.state = 'STANDBY'
        for s in self.sat.subsystems.values():
            if s.critical and s.state == 'FAILED':
                res = s.attempt_reboot()
                if res:
                    self.sat.log('AI', res)


class Satellite:
    def __init__(self, env: SpaceEnvironment) -> None:
        self.env = env
        self.battery = Battery()
        self.solar = SolarPanel(area_m2=4.0, efficiency=0.30)
        self.position = {'lat': 0.0, 'lon': 0.0, 'alt': 500.0}
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
        self.crit_mgr = CriticalConditionManager(self)
        self.ai = AIController(self)

    def log(self, level: str, message: str) -> None:
        entry = EventLog(now(), level, message)
        self.logs.append(entry)
        try:
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry.timestamp))} [{entry.level}] {entry.message}"
            file_logger.write_async(line)
        except Exception:
            pass
        if len(self.logs) > 400:
            self.logs.pop(0)

    def step_orbit(self, dt: float) -> None:
        deg_per_sec = 360.0 / (90.0 * 60.0)
        self.position['lon'] = (self.position['lon'] + deg_per_sec * dt) % 360.0
        self.position['lat'] = 10.0 * math.sin(now() / 100.0)

    def compute_solar_input(self, timestamp: float) -> float:
        sun_factor = self.env.sun_angle_factor(timestamp, self.position)
        radiation = self.env.radiation_level()
        return self.solar.output_watts(sun_factor, radiation)

    def total_power_draw(self) -> float:
        return sum(sub.power() for sub in self.subsystems.values())

    def thermal_step(self, dt: float, solar_input: float, load_w: float) -> None:
        absorbed = 0.05 * solar_input
        waste = 0.3 * load_w
        body_temp = self.thermal_state['BODY']
        cooling = 0.1 * (body_temp - (-100.0))
        dT = (absorbed + waste - cooling) * dt / 1000.0
        if self.subsystems.get('THERMAL_CTRL') and self.subsystems['THERMAL_CTRL'].state != 'ON':
            dT *= 1.5
        self.thermal_state['BODY'] = body_temp + dT

    async def tick(self) -> Telemetry:
        t0 = now()
        dt = t0 - self.last_step
        if dt <= 0.0:
            dt = 1e-6
        self.last_step = t0
        self.env.step()
        radiation = self.env.radiation_level()
        self.step_orbit(dt)
        for sub in self.subsystems.values():
            msg = sub.step_failure(self.env)
            if msg:
                self.log('CRITICAL' if sub.critical else 'WARNING', msg)
        for sub in self.subsystems.values():
            if sub.state == 'FAILED':
                res = sub.attempt_reboot()
                if res:
                    self.log('INFO', res)
        self.crit_mgr.maybe_trigger()
        solar_input = self.compute_solar_input(t0)
        load_w = self.total_power_draw()
        if self.mode == 'CONSERVE':
            for s in self.subsystems.values():
                if not s.critical and s.state == 'ON':
                    s.state = 'STANDBY'
                    self.log('INFO', f'{s.name} set to STANDBY for power saving')
            load_w = self.total_power_draw()
        net_w = solar_input - load_w
        self.battery.apply_energy(net_w, dt)
        if self.battery.percent < 20.0 or radiation > 2.0:
            self.battery.degrade(0.0005)
        if radiation > 2.0 and random.random() < 0.001:
            self.solar.degrade_efficiency(random.uniform(0.0001, 0.001))
        if random.random() < 1e-5:
            self.solar.fail()
            self.log('CRITICAL', 'Solar panel catastrophic failure')
        if random.random() < 0.0005 * radiation:
            if 'COMMS' in self.subsystems:
                self.subsystems['COMMS'].state = 'FAILED'
                self.log('CRITICAL', 'Sudden communications blackout due to radiation spike')
        self.thermal_step(dt, solar_input, load_w)
        telemetry = Telemetry(
            timestamp=t0,
            temperatures={'BODY': self.thermal_state['BODY'], 'BATTERY': 25.0},
            position=self.position.copy(),
            battery_percent=self.battery.percent,
            battery_voltage=self.battery.voltage,
            solar_output=solar_input,
            systems={name: sub.state for name, sub in self.subsystems.items()},
            env={'radiation': radiation, 'solar_activity': self.env.solar_activity},
        )
        try:
            self.last_telemetry = telemetry
        except Exception:
            self.last_telemetry = None
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
            'PING': lambda p: asyncio.create_task(_call_sync(self._cmd_ping, p)),
            'REQUEST_STATUS': lambda p: asyncio.create_task(_req_status(p)),
            'SET_POWER_MODE': lambda p: asyncio.create_task(_call_sync(self._cmd_set_power_mode, p[1:] if len(p) > 1 else [])),
            'DEPLOY_SOLAR': lambda p: asyncio.create_task(_call_sync(self._cmd_deploy_solar, p)),
            'STOW_SOLAR': lambda p: asyncio.create_task(_call_sync(self._cmd_stow_solar, p)),
            'REBOOT_COMMS': lambda p: asyncio.create_task(_call_sync(self._cmd_reboot_comms, p)),
            'ADJUST_ORBIT': lambda p: asyncio.create_task(_call_sync(self._cmd_adjust_orbit, p[1:] if len(p) > 1 else [])),
        }

        handler = cmd_map.get(parts[0])
        if handler:
            try:
                handler(parts)
            except Exception as e:
                self.log('ERROR', f'Command {cmd} handler failed: {e}')
        else:
            self.log('WARNING', f'Unknown command: {cmd}')

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
    tbl.add_row(Panel(temp, title='Thermal'), Panel(pos, title='Orbit'), Panel(energy, title='Power'), Panel(systems, title='Systems'))
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


async def simulation_loop(duration: float, update_interval: float) -> None:
    env = SpaceEnvironment()
    sat = Satellite(env)
    ground_agent = GroundStationAgent(sat.groundlink)
    start = now()
    ground_task = asyncio.create_task(ground_agent.periodic_behavior(sat, duration))
    try:
        with Live(console=console, screen=True, refresh_per_second=4) as live:
            while now() - start < duration:
                telem = await sat.tick()
                asyncio.create_task(sat.groundlink.send_telemetry(telem))
                sat.log('INFO', f'Telemetry queued batt={telem.battery_percent:.1f}% solar={telem.solar_output:.0f}W')
                cmd = sat.groundlink.receive_command_now()
                while cmd:
                    await sat.handle_command(cmd)
                    cmd = sat.groundlink.receive_command_now()
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
                solar_now = sat.solar.output_watts(sat.env.sun_angle_factor(now(), sat.position), sat.env.radiation_level())
                foot.add_row(sat.mode, f'{sat.battery.percent:.1f}%', f'{solar_now:.0f}W', f"{sat.env.radiation_level():.2f}")
                main['footer'].update(Panel(foot, title='Summary'))
                live.update(main)
                await asyncio.sleep(update_interval)
    finally:
        ground_task.cancel()
        await asyncio.gather(ground_task, return_exceptions=True)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=120.0)
    p.add_argument('--update', type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(simulation_loop(args.duration, args.update))
    except KeyboardInterrupt:
        console.print('\n[bold yellow]Simulation interrupted by user[/bold yellow]')
    except Exception as e:
        console.print(f'[bold red]Fatal error:[/bold red] {e}')


if __name__ == '__main__':
    main()