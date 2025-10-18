#!/usr/bin/env python3
"""
🐧 S25+ Thermal Intelligence System
===================================
Copyright (c) 2025 PNGN-Tec LLC
Author: Jesse Vogeler-Wunsch (@DaSettingsPNGN)

Predictive Thermal Management for Android
=========================================
Real-time thermal monitoring and prediction system using physics-based
modeling for Samsung Galaxy S25+ and compatible Android devices.

Core Features:
- Multi-zone thermal monitoring (CPU, GPU, battery, modem, etc.)
- Physics-based temperature prediction using Newton's law of cooling
- Pattern recognition and thermal signature learning
- Statistical anomaly detection with z-score analysis
- Network-aware thermal impact tracking (WiFi vs 5G)
- Persistent pattern storage with async file operations

Technical Implementation:
- Thermal mass: 50 J/°C (device heat capacity)
- Thermal resistance: 5°C/W (cooling efficiency)
- Ambient coupling: 0.3 (heat transfer coefficient)
- Prediction horizon: 60 seconds with confidence scoring
- Sampling interval: 10 seconds with adaptive rate
- History size: 1000 samples (~3 hours of data)

System Architecture:
- ThermalTelemetryCollector: Reads from sysfs and Termux API
- ThermalPhysicsEngine: Calculates velocity, acceleration, predictions
- ThermalPatternEngine: Learns command signatures and correlations
- ThermalStatisticalAnalyzer: Computes percentiles and detects anomalies

Module Interface:
- create_thermal_intelligence(): Factory function for system creation
- ThermalIntelligenceSystem: Main monitoring coordinator
- ThermalSample/Statistics/Intelligence: Data structures for telemetry

Usage:
    thermal = create_thermal_intelligence()
    await thermal.start()
    
    intel = thermal.get_current_intelligence()
    print(f"Temperature: {intel.temperature:.1f}°C")
    print(f"State: {intel.state.name}")
    
    await thermal.stop()

Requirements:
- Python 3.11+
- numpy for physics calculations
- Android device with accessible thermal zones (optional)
- Termux environment for full functionality (optional)

Version: 1.0.0
Optimized for: Samsung Galaxy S25+ (Snapdragon 8 Elite)
"""

import os
import json
import time
import asyncio
import subprocess
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict, OrderedDict
from enum import Enum, auto
from datetime import datetime, timedelta
import math
import statistics
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger('PNGN.S25Thermal')

# ============================================================================
# CONFIGURATION IMPORTS
# ============================================================================

from config import (
    # Thermal zones
    THERMAL_ZONES,
    THERMAL_ZONE_NAMES,
    THERMAL_ZONE_WEIGHTS,
    
    # Sampling
    THERMAL_SAMPLE_INTERVAL_MS,
    THERMAL_HISTORY_SIZE,
    THERMAL_PREDICTION_HORIZON,
    
    # Network impact
    NETWORK_THERMAL_IMPACT,
    
    # Statistical thresholds
    THERMAL_PERCENTILE_WINDOWS,
    THERMAL_ANOMALY_THRESHOLD,
    
    # Pattern recognition
    THERMAL_SIGNATURE_WINDOW,
    THERMAL_CORRELATION_THRESHOLD,
    
    # Termux API
    TERMUX_BATTERY_STATUS_CMD,
    TERMUX_WIFI_INFO_CMD,
    TERMUX_TELEPHONY_INFO_CMD,
    TERMUX_SENSORS_CMD,
    
    # Features
    THERMAL_PREDICTION_ENABLED,
    THERMAL_PATTERN_RECOGNITION_ENABLED,
    THERMAL_NETWORK_AWARENESS_ENABLED,
    
    # S25+ thermal characteristics
    S25_THERMAL_MASS,
    S25_THERMAL_RESISTANCE,
    S25_AMBIENT_COUPLING,
    S25_MAX_TDP,
    
    # Temperature thresholds
    THERMAL_TEMP_COLD,
    THERMAL_TEMP_OPTIMAL_MIN,
    THERMAL_TEMP_OPTIMAL_MAX,
    THERMAL_TEMP_WARM,
    THERMAL_TEMP_HOT,
    THERMAL_TEMP_CRITICAL,
    
    # Hysteresis values
    THERMAL_HYSTERESIS_UP,
    THERMAL_HYSTERESIS_DOWN,
    
    # Timeouts
    THERMAL_SUBPROCESS_TIMEOUT,
    THERMAL_TELEMETRY_TIMEOUT,
    THERMAL_SHUTDOWN_TIMEOUT,
    
    # Processing
    THERMAL_TELEMETRY_BATCH_SIZE,
    THERMAL_TELEMETRY_PROCESSING_INTERVAL,
    THERMAL_SIGNATURE_MAX_COUNT,
    THERMAL_LEARNING_RATE,
    THERMAL_SIGNATURE_MIN_DELTA,
    THERMAL_COMMAND_HASH_LENGTH,
    
    # Persistence
    THERMAL_PERSISTENCE_INTERVAL,
    THERMAL_PERSISTENCE_KEY,
    THERMAL_PERSISTENCE_FILE,
    
    # Velocity thresholds
    THERMAL_VELOCITY_RAPID_COOLING,
    THERMAL_VELOCITY_COOLING,
    THERMAL_VELOCITY_WARMING,
    THERMAL_VELOCITY_RAPID_WARMING,
    
    # Confidence calculations
    THERMAL_MIN_SAMPLES_CONFIDENCE,
    THERMAL_PREDICTION_CONFIDENCE_DECAY,
    THERMAL_SENSOR_TEMP_MIN,
    THERMAL_SENSOR_TEMP_MAX,
    THERMAL_SENSOR_CONFIDENCE_REDUCED,
    
    # Network detection
    THERMAL_WIFI_5G_FREQ_MIN,
    THERMAL_NETWORK_TIMEOUT,
    
    # Recommendations
    THERMAL_BUDGET_WARNING_SECONDS,
    THERMAL_NETWORK_IMPACT_WARNING,
    THERMAL_CHARGING_IMPACT_WARNING,
    THERMAL_COMMAND_IMPACT_WARNING,
)

# ============================================================================
# SHARED TYPES
# ============================================================================

try:
    from shared_types import ThermalState, ThermalZone, ThermalTrend, NetworkType, MemoryPressureLevel
except ImportError:
    logger.warning("Shared types not available - using fallback definitions")
    class ThermalState(Enum):
        OPTIMAL = auto()
        WARM = auto()
        HOT = auto()
        CRITICAL = auto()
        UNKNOWN = auto()
        COLD = auto()
    
    class ThermalZone(Enum):
        CPU_BIG = auto()
        CPU_LITTLE = auto()
        GPU = auto()
        BATTERY = auto()
        MODEM = auto()
        AMBIENT = auto()
    
    class ThermalTrend(Enum):
        RAPID_COOLING = auto()
        COOLING = auto()
        STABLE = auto()
        WARMING = auto()
        RAPID_WARMING = auto()
    
    class NetworkType(Enum):
        UNKNOWN = auto()
        OFFLINE = auto()
        WIFI_2G = auto()
        WIFI_5G = auto()
        MOBILE_3G = auto()
        MOBILE_4G = auto()
        MOBILE_5G = auto()

    class MemoryPressureLevel(IntEnum):
        NORMAL = 0
        MODERATE = 1
        HIGH = 2
        CRITICAL = 3

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ThermalSample:
    """Temperature measurement at single point in time"""
    timestamp: float
    zones: Dict[ThermalZone, float]
    confidence: Dict[ThermalZone, float]
    ambient: Optional[float] = None
    network: NetworkType = NetworkType.UNKNOWN
    charging: bool = False
    screen_on: bool = True
    workload_hash: Optional[str] = None

@dataclass
class ThermalVelocity:
    """Temperature rate of change"""
    zones: Dict[ThermalZone, float]  # °C/second
    overall: float
    trend: ThermalTrend
    acceleration: float  # °C/second²

@dataclass
class ThermalPrediction:
    """Future temperature prediction"""
    timestamp: float
    horizon: float
    predicted_temps: Dict[ThermalZone, float]
    confidence: float
    thermal_budget: float
    recommended_delay: float

@dataclass
class ThermalSignature:
    """Thermal impact of command"""
    command: str
    avg_delta_temp: float
    peak_delta_temp: float
    duration: float
    zones_affected: List[ThermalZone]
    sample_count: int
    confidence: float

@dataclass
class ThermalStatistics:
    """Statistical thermal analysis"""
    current: ThermalSample
    velocity: ThermalVelocity
    mean: Dict[ThermalZone, float]
    median: Dict[ThermalZone, float]
    std_dev: Dict[ThermalZone, float]
    percentiles: Dict[int, Dict[ThermalZone, float]]
    min_1m: Dict[ThermalZone, float]
    max_1m: Dict[ThermalZone, float]
    mean_1m: Dict[ThermalZone, float]
    thermal_cycles: int
    time_above_warm: float
    last_critical: Optional[float] = None
    workload_correlation: float = 0.5
    network_impact: float = 0.0
    charging_impact: float = 0.0

@dataclass
class ThermalIntelligence:
    """Thermal telemetry package"""
    stats: ThermalStatistics
    prediction: Optional[ThermalPrediction]
    signatures: Dict[str, ThermalSignature]
    anomalies: List[Tuple[float, str]]
    recommendations: List[str]
    state: ThermalState
    confidence: float

@dataclass
class ThermalEvent:
    """Thermal event log entry"""
    timestamp: float
    type: str
    description: str
    state: ThermalState
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# SUBPROCESS UTILITIES
# ============================================================================

async def _safe_subprocess_call(cmd: List[str], timeout: float = THERMAL_SUBPROCESS_TIMEOUT) -> Optional[str]:
    """Execute subprocess with timeout and error handling"""
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, _ = await asyncio.wait_for(
            proc.communicate(), 
            timeout=timeout
        )
        
        if proc.returncode == 0:
            return stdout.decode()
        return None
        
    except asyncio.TimeoutError:
        logger.debug(f"Subprocess {cmd[0]} timed out")
        if proc:
            try:
                proc.kill()
                await proc.wait()
            except:
                pass
        return None
    except Exception as e:
        logger.debug(f"Subprocess {cmd[0]} failed: {e}")
        return None

# ============================================================================
# TELEMETRY COLLECTOR
# ============================================================================

class ThermalTelemetryCollector:
    """
    Collects temperature data from system sensors.
    Reads from /sys/class/thermal zones and Termux API.
    """
    
    def __init__(self):
        self.zone_readers = {}
        self.last_readings = {}
        self.read_failures = defaultdict(int)
        self._shutting_down = False
        
        # Initialize readers with error handling
        try:
            self.zone_readers = self._initialize_readers()
        except Exception as e:
            logger.warning(f"Failed to initialize some thermal readers: {e}")
        
    def shutdown(self):
        """Mark collector as shutting down"""
        self._shutting_down = True
        
    def _initialize_readers(self) -> Dict[ThermalZone, Callable]:
        """Map zones to reader functions"""
        readers = {}
        
        # Only add readers for zones that actually exist
        if os.path.exists('/sys/class/thermal'):
            thermal_zones = os.listdir('/sys/class/thermal')
            
            if 'thermal_zone0' in thermal_zones:
                readers[ThermalZone.CPU_BIG] = lambda: self._read_thermal_zone(0)
                
            if 'thermal_zone1' in thermal_zones:
                readers[ThermalZone.CPU_LITTLE] = lambda: self._read_thermal_zone(1)
                
            if 'thermal_zone2' in thermal_zones:
                readers[ThermalZone.GPU] = lambda: self._read_thermal_zone(2)
        else:
            logger.warning("/sys/class/thermal not found - limited thermal monitoring")
        
        # These readers use alternative methods
        readers[ThermalZone.BATTERY] = self._read_battery_temp
        readers[ThermalZone.AMBIENT] = self._read_ambient_temp
        
        logger.info(f"Initialized {len(readers)} thermal zone readers")
        return readers
    
    async def collect(self) -> ThermalSample:
        """Collect temperature from all zones"""
        if self._shutting_down:
            return ThermalSample(
                timestamp=time.time(),
                zones={},
                confidence={},
                network=NetworkType.UNKNOWN,
                charging=False
            )
            
        zones = {}
        confidence = {}
        
        # Parallel collection
        tasks = []
        for zone, reader in self.zone_readers.items():
            if self._shutting_down:
                break
            tasks.append(self._read_zone_async(zone, reader))
        
        if not tasks:
            return ThermalSample(
                timestamp=time.time(),
                zones={},
                confidence={},
                network=NetworkType.UNKNOWN,
                charging=False
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for zone, result in zip(self.zone_readers.keys(), results):
            if isinstance(result, Exception):
                # Use cached value with reduced confidence
                if zone in self.last_readings:
                    zones[zone] = self.last_readings[zone]
                    confidence[zone] = 0.5
                else:
                    zones[zone] = 25.0  # Room temperature
                    confidence[zone] = 0.1
                self.read_failures[zone] += 1
            else:
                temp, conf = result
                zones[zone] = temp
                confidence[zone] = conf
                self.last_readings[zone] = temp
                self.read_failures[zone] = 0
        
        # Auxiliary data
        network = await self._detect_network_type() if not self._shutting_down else NetworkType.UNKNOWN
        charging = await self._is_charging() if not self._shutting_down else False
        
        return ThermalSample(
            timestamp=time.time(),
            zones=zones,
            confidence=confidence,
            network=network,
            charging=charging,
            ambient=zones.get(ThermalZone.AMBIENT)
        )
    
    async def _read_zone_async(self, zone: ThermalZone, reader: Callable) -> Tuple[float, float]:
        """Read zone temperature asynchronously"""
        try:
            if asyncio.iscoroutinefunction(reader):
                temp = await reader()
            else:
                temp = await asyncio.get_event_loop().run_in_executor(None, reader)
            
            # Validate temperature range
            if THERMAL_SENSOR_TEMP_MIN < temp < THERMAL_SENSOR_TEMP_MAX:
                return temp, 1.0
            else:
                logger.warning(f"Temperature {temp}°C outside expected range for {zone}")
                return temp, THERMAL_SENSOR_CONFIDENCE_REDUCED
                
        except Exception as e:
            logger.debug(f"Failed to read {zone}: {e}")
            raise
    
    def _read_thermal_zone(self, zone_idx: int) -> float:
        """Read from /sys/class/thermal/thermal_zoneX/temp"""
        if self._shutting_down:
            raise RuntimeError("Shutting down")
            
        try:
            path = f'/sys/class/thermal/thermal_zone{zone_idx}/temp'
            with open(path, 'r') as f:
                # Value in millidegrees
                return int(f.read().strip()) / 1000.0
        except:
            raise
    
    async def _read_battery_temp(self) -> float:
        """Read battery temperature"""
        if self._shutting_down:
            raise RuntimeError("Shutting down")
            
        # Try Termux API
        result = await _safe_subprocess_call(TERMUX_BATTERY_STATUS_CMD)
        if result:
            try:
                data = json.loads(result)
                return data.get('temperature', 25.0)
            except:
                pass
        
        # Fallback to hwmon
        try:
            for hwmon in os.listdir('/sys/class/hwmon/'):
                name_path = f'/sys/class/hwmon/{hwmon}/name'
                if os.path.exists(name_path):
                    with open(name_path, 'r') as f:
                        if 'battery' in f.read().lower():
                            temp_path = f'/sys/class/hwmon/{hwmon}/temp1_input'
                            if os.path.exists(temp_path):
                                with open(temp_path, 'r') as f:
                                    return int(f.read().strip()) / 1000.0
        except:
            pass
        
        raise RuntimeError("Cannot read battery temperature")
    
    async def _read_ambient_temp(self) -> float:
        """Read ambient temperature from sensors"""
        if self._shutting_down:
            raise RuntimeError("Shutting down")
            
        # Try Termux sensors
        result = await _safe_subprocess_call(TERMUX_SENSORS_CMD)
        if result:
            try:
                data = json.loads(result)
                for sensor in data.get('sensors', []):
                    if 'ambient' in sensor.get('name', '').lower():
                        values = sensor.get('values', [])
                        if values:
                            return values[0]
            except:
                pass
        
        # Estimate from battery
        if ThermalZone.BATTERY in self.last_readings:
            # Battery typically 10°C warmer than ambient
            return self.last_readings[ThermalZone.BATTERY] - 10.0
        
        return 25.0  # Room temperature
    
    async def _detect_network_type(self) -> NetworkType:
        """Detect network connection type"""
        if not THERMAL_NETWORK_AWARENESS_ENABLED or self._shutting_down:
            return NetworkType.UNKNOWN
        
        # Check WiFi
        result = await _safe_subprocess_call(TERMUX_WIFI_INFO_CMD, THERMAL_NETWORK_TIMEOUT)
        if result:
            try:
                data = json.loads(result)
                if data.get('connection_info', {}).get('ssid') != '<unknown ssid>':
                    freq = data.get('connection_info', {}).get('frequency', 0)
                    if freq > THERMAL_WIFI_5G_FREQ_MIN:
                        return NetworkType.WIFI_5G
                    else:
                        return NetworkType.WIFI_2G
            except:
                pass
        
        # Check mobile
        result = await _safe_subprocess_call(TERMUX_TELEPHONY_INFO_CMD, THERMAL_NETWORK_TIMEOUT)
        if result:
            try:
                data = json.loads(result)
                network_type = data.get('data_network_type', '').lower()
                
                if 'nr' in network_type or '5g' in network_type:
                    return NetworkType.MOBILE_5G
                elif 'lte' in network_type:
                    return NetworkType.MOBILE_4G
                elif network_type:
                    return NetworkType.MOBILE_3G
            except:
                pass
        
        return NetworkType.OFFLINE
    
    async def _is_charging(self) -> bool:
        """Check charging status"""
        if self._shutting_down:
            return False
            
        result = await _safe_subprocess_call(TERMUX_BATTERY_STATUS_CMD)
        if result:
            try:
                data = json.loads(result)
                return data.get('plugged', 'UNPLUGGED') != 'UNPLUGGED'
            except:
                pass
        
        return False

# ============================================================================
# PHYSICS ENGINE
# ============================================================================

class ThermalPhysicsEngine:
    """
    Temperature physics modeling.
    Calculates velocity, acceleration, and predictions.
    """
    
    def __init__(self):
        self.thermal_mass = S25_THERMAL_MASS
        self.thermal_resistance = S25_THERMAL_RESISTANCE
        self.ambient_coupling = S25_AMBIENT_COUPLING
        self.max_tdp = S25_MAX_TDP
        
    def calculate_velocity(self, samples: List[ThermalSample], 
                          window: float = 60.0) -> ThermalVelocity:
        """Calculate temperature rate of change"""
        if len(samples) < 2:
            return ThermalVelocity(
                zones={},
                overall=0.0,
                trend=ThermalTrend.STABLE,
                acceleration=0.0
            )
        
        # Get samples within window
        cutoff = time.time() - window
        recent = [s for s in samples if s.timestamp > cutoff]
        
        if len(recent) < 2:
            recent = samples[-2:]
        
        # Per-zone velocity
        zone_velocities = {}
        
        for zone in ThermalZone:
            temps = [(s.timestamp, s.zones.get(zone, 0)) for s in recent if zone in s.zones]
            if len(temps) >= 2:
                times = np.array([t[0] for t in temps])
                values = np.array([t[1] for t in temps])
                
                # Normalize time
                times = times - times[0]
                
                if len(times) > 2:
                    # Polynomial fit
                    coeffs = np.polyfit(times, values, 2)
                    velocity = coeffs[1] + 2 * coeffs[0] * times[-1]
                    acceleration = 2 * coeffs[0]
                else:
                    # Linear velocity
                    velocity = (values[-1] - values[0]) / (times[-1] - times[0]) if times[-1] != times[0] else 0
                    acceleration = 0
                
                zone_velocities[zone] = velocity
        
        # Weighted average
        weights = THERMAL_ZONE_WEIGHTS
        
        overall = sum(
            zone_velocities.get(zone, 0) * weights.get(zone, 0)
            for zone in weights
        )
        
        # Classify trend
        if overall < THERMAL_VELOCITY_RAPID_COOLING:
            trend = ThermalTrend.RAPID_COOLING
        elif overall < THERMAL_VELOCITY_COOLING:
            trend = ThermalTrend.COOLING
        elif overall < THERMAL_VELOCITY_WARMING:
            trend = ThermalTrend.STABLE
        elif overall < THERMAL_VELOCITY_RAPID_WARMING:
            trend = ThermalTrend.WARMING
        else:
            trend = ThermalTrend.RAPID_WARMING
        
        return ThermalVelocity(
            zones=zone_velocities,
            overall=overall,
            trend=trend,
            acceleration=acceleration
        )
    
    def predict_temperature(self, current: ThermalSample, 
                          velocity: ThermalVelocity,
                          horizon: float) -> ThermalPrediction:
        """Predict future temperature"""
        predicted = {}
        
        ambient = current.ambient or 25.0
        
        for zone, current_temp in current.zones.items():
            if zone in velocity.zones:
                zone_velocity = velocity.zones[zone]
                
                # Temperature delta from ambient
                delta_from_ambient = current_temp - ambient
                
                # Newton's law of cooling
                cooling_rate = -self.ambient_coupling * delta_from_ambient / self.thermal_resistance
                
                # Net rate
                net_rate = zone_velocity + cooling_rate
                
                # Include acceleration
                if velocity.acceleration != 0:
                    predicted_temp = current_temp + net_rate * horizon + 0.5 * velocity.acceleration * horizon**2
                else:
                    predicted_temp = current_temp + net_rate * horizon
                
                # Physical limits
                predicted[zone] = max(ambient, min(85.0, predicted_temp))
        
        # Calculate thermal budget
        thermal_budget = float('inf')
        hottest_zone = max(current.zones.items(), key=lambda x: x[1])
        
        if hottest_zone[0] in velocity.zones:
            zone_velocity = velocity.zones[hottest_zone[0]]
            if zone_velocity > 0:
                time_to_throttle = (60.0 - hottest_zone[1]) / zone_velocity
                thermal_budget = max(0, time_to_throttle)
        
        # Confidence decreases with horizon
        confidence = 1.0 / (1.0 + horizon / THERMAL_PREDICTION_CONFIDENCE_DECAY)
        confidence *= min(current.confidence.values()) if current.confidence else 0.5
        
        # Recommended delay
        if velocity.trend == ThermalTrend.RAPID_WARMING:
            recommended_delay = 2.0
        elif velocity.trend == ThermalTrend.WARMING:
            recommended_delay = 1.0
        elif velocity.trend == ThermalTrend.STABLE:
            recommended_delay = 0.5
        else:
            recommended_delay = 0.0
        
        return ThermalPrediction(
            timestamp=time.time(),
            horizon=horizon,
            predicted_temps=predicted,
            confidence=confidence,
            thermal_budget=thermal_budget,
            recommended_delay=recommended_delay
        )
    
    def calculate_power_draw(self, sample: ThermalSample, 
                           previous: Optional[ThermalSample]) -> float:
        """Calculate power draw from temperature change"""
        if not previous:
            return 0.0
        
        dt = sample.timestamp - previous.timestamp
        if dt <= 0:
            return 0.0
        
        # Energy = mass * specific_heat * delta_T
        total_energy = 0.0
        
        for zone in [ThermalZone.CPU_BIG, ThermalZone.CPU_LITTLE, ThermalZone.GPU]:
            if zone in sample.zones and zone in previous.zones:
                delta_t = sample.zones[zone] - previous.zones[zone]
                # Each zone has 1/3 of thermal mass
                zone_energy = (self.thermal_mass / 3) * delta_t
                total_energy += zone_energy
        
        # Power = Energy / Time
        power = total_energy / dt
        
        # Account for cooling
        avg_temp = np.mean(list(sample.zones.values()))
        ambient = sample.ambient or 25.0
        cooling_power = (avg_temp - ambient) / self.thermal_resistance
        
        total_power = power + cooling_power
        
        return max(0, min(self.max_tdp, total_power))

# ============================================================================
# PATTERN ENGINE
# ============================================================================

class ThermalPatternEngine:
    """
    Thermal pattern recognition and learning.
    Tracks command signatures and correlates with temperature.
    """
    
    def __init__(self):
        self.command_signatures = OrderedDict()
        self.max_signatures = THERMAL_SIGNATURE_MAX_COUNT
        self.learning_rate = THERMAL_LEARNING_RATE
        self.telemetry_signatures = OrderedDict()
        
    def learn_signature(self, command: str, 
                       before: ThermalSample, 
                       after: ThermalSample,
                       duration: float):
        """Learn thermal signature of command"""
        # Calculate deltas
        deltas = {}
        affected_zones = []
        
        for zone in ThermalZone:
            if zone in before.zones and zone in after.zones:
                delta = after.zones[zone] - before.zones[zone]
                if abs(delta) > THERMAL_SIGNATURE_MIN_DELTA:
                    deltas[zone] = delta
                    affected_zones.append(zone)
        
        if not deltas:
            return
        
        # Update or create signature
        if command in self.command_signatures:
            sig = self.command_signatures[command]
            # Exponential moving average
            alpha = self.learning_rate
            
            sig.avg_delta_temp = (1 - alpha) * sig.avg_delta_temp + alpha * np.mean(list(deltas.values()))
            sig.peak_delta_temp = max(sig.peak_delta_temp, max(deltas.values()))
            sig.duration = (1 - alpha) * sig.duration + alpha * duration
            sig.sample_count += 1
            sig.confidence = min(1.0, sig.sample_count / 10.0)
            
            for zone in affected_zones:
                if zone not in sig.zones_affected:
                    sig.zones_affected.append(zone)
        else:
            sig = ThermalSignature(
                command=command,
                avg_delta_temp=np.mean(list(deltas.values())),
                peak_delta_temp=max(deltas.values()),
                duration=duration,
                zones_affected=affected_zones,
                sample_count=1,
                confidence=0.1
            )
            
            self.command_signatures[command] = sig
            
            # LRU eviction
            if len(self.command_signatures) > self.max_signatures:
                self.command_signatures.popitem(last=False)
    
    def learn_from_telemetry(self, telemetry: Dict[str, Any]):
        """Learn from render telemetry"""
        command = telemetry.get('command', 'unknown')
        thermal_cost = telemetry.get('thermal_cost_mw', 0) / 1000.0  # Convert to watts
        duration = telemetry.get('render_duration', 0)
        
        # Estimate temperature impact
        # 1W approximates 0.5°C rise after cooling
        estimated_delta = thermal_cost * 0.5
        
        # Update telemetry signature
        if command in self.telemetry_signatures:
            sig = self.telemetry_signatures[command]
            alpha = self.learning_rate
            
            sig['avg_power'] = (1 - alpha) * sig['avg_power'] + alpha * thermal_cost
            sig['avg_duration'] = (1 - alpha) * sig['avg_duration'] + alpha * duration
            sig['avg_delta'] = (1 - alpha) * sig['avg_delta'] + alpha * estimated_delta
            sig['sample_count'] += 1
        else:
            self.telemetry_signatures[command] = {
                'avg_power': thermal_cost,
                'avg_duration': duration,
                'avg_delta': estimated_delta,
                'sample_count': 1
            }
            
            # LRU eviction
            if len(self.telemetry_signatures) > self.max_signatures:
                self.telemetry_signatures.popitem(last=False)
    
    def get_thermal_impact(self, command: str) -> Optional[ThermalSignature]:
        """Get thermal impact of command"""
        return self.command_signatures.get(command)
    
    def predict_impact(self, commands: List[str]) -> float:
        """Predict cumulative thermal impact"""
        total_impact = 0.0
        
        for cmd in commands:
            sig = self.get_thermal_impact(cmd)
            if sig and sig.confidence > 0.5:
                total_impact += sig.avg_delta_temp
            elif cmd in self.telemetry_signatures:
                telem_sig = self.telemetry_signatures[cmd]
                if telem_sig['sample_count'] > 5:
                    total_impact += telem_sig['avg_delta']
        
        return total_impact
    
    def find_anomalies(self, sample: ThermalSample, 
                       history: List[ThermalSample]) -> List[str]:
        """Detect thermal anomalies"""
        anomalies = []
        
        if len(history) < 10:
            return anomalies
        
        # Calculate z-scores
        for zone in sample.zones:
            historical = [s.zones.get(zone, 0) for s in history[-100:] if zone in s.zones]
            if len(historical) < 10:
                continue
            
            mean = statistics.mean(historical)
            stdev = statistics.stdev(historical)
            
            current = sample.zones[zone]
            z_score = abs(current - mean) / stdev if stdev > 0 else 0
            
            if z_score > THERMAL_ANOMALY_THRESHOLD:
                anomalies.append(
                    f"{zone.name} anomaly: {current:.1f}°C "
                    f"(expected {mean:.1f}±{stdev:.1f}°C)"
                )
        
        # Check unusual zone relationships
        if ThermalZone.CPU_BIG in sample.zones and ThermalZone.GPU in sample.zones:
            cpu_temp = sample.zones[ThermalZone.CPU_BIG]
            gpu_temp = sample.zones[ThermalZone.GPU]
            
            # GPU typically cooler than CPU
            if gpu_temp > cpu_temp + 10:
                anomalies.append(
                    f"GPU temperature unusual: {gpu_temp:.1f}°C "
                    f"(CPU: {cpu_temp:.1f}°C)"
                )
        
        return anomalies
    
    def export_signatures(self) -> Dict[str, Any]:
        """Export signatures for persistence"""
        return {
            'command_signatures': {
                cmd: {
                    'avg_delta_temp': sig.avg_delta_temp,
                    'peak_delta_temp': sig.peak_delta_temp,
                    'duration': sig.duration,
                    'zones_affected': [z.name for z in sig.zones_affected],
                    'sample_count': sig.sample_count,
                    'confidence': sig.confidence
                }
                for cmd, sig in self.command_signatures.items()
            },
            'telemetry_signatures': dict(self.telemetry_signatures),
            'metadata': {
                'version': '1.0',
                'timestamp': time.time(),
                'total_patterns': len(self.command_signatures) + len(self.telemetry_signatures)
            }
        }
    
    def import_signatures(self, data: Dict[str, Any]) -> None:
        """Import signatures from persistence - FIXED VERSION"""
        try:
            # Validate input data
            if not isinstance(data, dict):
                logger.error(f"Invalid signature data type: {type(data)}")
                return
                
            # Import command signatures
            command_sigs = data.get('command_signatures', {})
            if isinstance(command_sigs, dict):
                for cmd, sig_data in command_sigs.items():
                    try:
                        self.command_signatures[cmd] = ThermalSignature(
                            command=cmd,
                            avg_delta_temp=sig_data.get('avg_delta_temp', 0.0),
                            peak_delta_temp=sig_data.get('peak_delta_temp', 0.0),
                            duration=sig_data.get('duration', 0.0),
                            zones_affected=[
                                ThermalZone[z] for z in sig_data.get('zones_affected', [])
                                if z in [e.name for e in ThermalZone]
                            ],
                            sample_count=sig_data.get('sample_count', 0),
                            confidence=sig_data.get('confidence', 0.0)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to import signature for {cmd}: {e}")
            
            # Import telemetry signatures
            telemetry_sigs = data.get('telemetry_signatures', {})
            if isinstance(telemetry_sigs, dict):
                self.telemetry_signatures.update(telemetry_sigs)
            
            logger.info(
                f"Imported {len(self.command_signatures)} command signatures, "
                f"{len(self.telemetry_signatures)} telemetry signatures"
            )
        except Exception as e:
            logger.error(f"Failed to import signatures: {e}")

# ============================================================================
# STATISTICAL ANALYZER
# ============================================================================

class ThermalStatisticalAnalyzer:
    """Statistical analysis of thermal data"""
    
    def __init__(self):
        self.percentile_calculator = lambda data, p: np.percentile(data, p) if len(data) > 0 else 0
        
    def analyze(self, samples: List[ThermalSample], 
                velocity: ThermalVelocity) -> ThermalStatistics:
        """Perform statistical analysis"""
        if not samples:
            raise ValueError("No samples to analyze")
        
        current = samples[-1]
        
        # Zone data collection
        zone_data = defaultdict(list)
        for sample in samples:
            for zone, temp in sample.zones.items():
                zone_data[zone].append(temp)
        
        # Calculate statistics
        mean = {}
        median = {}
        std_dev = {}
        percentiles = {5: {}, 25: {}, 75: {}, 95: {}}
        
        for zone, temps in zone_data.items():
            if temps:
                mean[zone] = statistics.mean(temps)
                median[zone] = statistics.median(temps)
                std_dev[zone] = statistics.stdev(temps) if len(temps) > 1 else 0
                
                for p in [5, 25, 75, 95]:
                    percentiles[p][zone] = self.percentile_calculator(temps, p)
        
        # One minute statistics
        one_minute_ago = time.time() - 60
        recent_samples = [s for s in samples if s.timestamp > one_minute_ago]
        
        min_1m = {}
        max_1m = {}
        mean_1m = {}
        
        for zone in ThermalZone:
            recent_temps = [s.zones.get(zone, 0) for s in recent_samples if zone in s.zones]
            if recent_temps:
                min_1m[zone] = min(recent_temps)
                max_1m[zone] = max(recent_temps)
                mean_1m[zone] = statistics.mean(recent_temps)
        
        # Pattern analysis
        thermal_cycles = self._count_thermal_cycles(samples)
        time_above_warm = self._calculate_time_above_threshold(samples, THERMAL_TEMP_WARM)
        
        # Last critical event
        last_critical = None
        for sample in reversed(samples):
            max_temp = max(sample.zones.values()) if sample.zones else 0
            if max_temp > THERMAL_TEMP_CRITICAL:
                last_critical = sample.timestamp
                break
        
        # Correlations
        workload_correlation = self._calculate_workload_correlation(samples)
        network_impact = self._calculate_network_impact(samples)
        charging_impact = self._calculate_charging_impact(samples)
        
        return ThermalStatistics(
            current=current,
            velocity=velocity,
            mean=mean,
            median=median,
            std_dev=std_dev,
            percentiles=percentiles,
            min_1m=min_1m,
            max_1m=max_1m,
            mean_1m=mean_1m,
            thermal_cycles=thermal_cycles,
            time_above_warm=time_above_warm,
            last_critical=last_critical,
            workload_correlation=workload_correlation,
            network_impact=network_impact,
            charging_impact=charging_impact
        )
    
    def _count_thermal_cycles(self, samples: List[ThermalSample]) -> int:
        """Count heat/cool cycles"""
        if len(samples) < 3:
            return 0
        
        temps = [s.zones.get(ThermalZone.CPU_BIG, 0) for s in samples 
                if ThermalZone.CPU_BIG in s.zones]
        
        if len(temps) < 3:
            return 0
        
        # Find direction changes
        cycles = 0
        increasing = temps[1] > temps[0]
        
        for i in range(2, len(temps)):
            if increasing and temps[i] < temps[i-1]:
                cycles += 1
                increasing = False
            elif not increasing and temps[i] > temps[i-1]:
                increasing = True
        
        return cycles
    
    def _calculate_time_above_threshold(self, samples: List[ThermalSample], 
                                       threshold: float) -> float:
        """Calculate time above temperature threshold"""
        if len(samples) < 2:
            return 0.0
        
        time_above = 0.0
        
        for i in range(1, len(samples)):
            prev = samples[i-1]
            curr = samples[i]
            
            prev_above = any(t > threshold for t in prev.zones.values())
            curr_above = any(t > threshold for t in curr.zones.values())
            
            if prev_above and curr_above:
                time_above += curr.timestamp - prev.timestamp
        
        return time_above
    
    def _calculate_workload_correlation(self, samples: List[ThermalSample]) -> float:
        """Calculate workload correlation coefficient"""
        # Simplified - would correlate with actual workload metrics
        return 0.5
    
    def _calculate_network_impact(self, samples: List[ThermalSample]) -> float:
        """Calculate network temperature impact"""
        network_samples = defaultdict(list)
        
        for sample in samples:
            if sample.network != NetworkType.UNKNOWN:
                max_temp = max(sample.zones.values()) if sample.zones else 0
                network_samples[sample.network].append(max_temp)
        
        if not network_samples:
            return 0.0
        
        # Average by network type
        network_avgs = {}
        for network, temps in network_samples.items():
            if temps:
                network_avgs[network] = statistics.mean(temps)
        
        if not network_avgs:
            return 0.0
        
        # Compare to baseline
        baseline = network_avgs.get(NetworkType.WIFI_2G, min(network_avgs.values()))
        
        max_impact = 0.0
        for network, avg_temp in network_avgs.items():
            impact = avg_temp - baseline
            max_impact = max(max_impact, impact)
        
        return max_impact
    
    def _calculate_charging_impact(self, samples: List[ThermalSample]) -> float:
        """Calculate charging temperature impact"""
        charging_temps = []
        not_charging_temps = []
        
        for sample in samples:
            max_temp = max(sample.zones.values()) if sample.zones else 0
            if sample.charging:
                charging_temps.append(max_temp)
            else:
                not_charging_temps.append(max_temp)
        
        if not charging_temps or not not_charging_temps:
            return 0.0
        
        return statistics.mean(charging_temps) - statistics.mean(not_charging_temps)

# ============================================================================
# THERMAL INTELLIGENCE SYSTEM - FIXED WITH ASYNC PERSISTENCE
# ============================================================================

class ThermalIntelligenceSystem:
    """
    Main thermal telemetry coordinator.
    Collects data, learns patterns, provides predictions.
    FIXED: Async persistence loading to avoid 'bool' object errors
    """
    
    def __init__(self):
        # Components
        self.telemetry = ThermalTelemetryCollector()
        self.physics = ThermalPhysicsEngine()
        self.patterns = ThermalPatternEngine()
        self.analyzer = ThermalStatisticalAnalyzer()
        
        # Data storage
        self.samples = deque(maxlen=THERMAL_HISTORY_SIZE)
        self.predictions = deque(maxlen=100)
        self.events = deque(maxlen=1000)
        
        # State
        self.current_state = ThermalState.UNKNOWN
        self.last_update = 0
        self.update_interval = THERMAL_SAMPLE_INTERVAL_MS / 1000.0
        
        # Command tracking
        self.command_history = deque(maxlen=100)
        self.command_timestamps = {}
        
        # Telemetry queue
        self.telemetry_queue = deque(maxlen=THERMAL_TELEMETRY_BATCH_SIZE * 2)
        self.telemetry_batch = []
        self.last_telemetry_process = 0
        
        # Persistence
        self.persistence_enabled = True
        self.persistence_key = THERMAL_PERSISTENCE_KEY
        self.auto_save_interval = THERMAL_PERSISTENCE_INTERVAL
        self.last_save_time = 0
        
        # Monitoring
        self.monitor_task = None
        self.running = False
        
        # Event callbacks
        self.event_callbacks = []
        
        # Task tracking
        self._pending_tasks = set()
        
        # NOTE: Removed synchronous _load_signatures() call
        # Signatures will be loaded asynchronously in start()
        
        logger.info("Thermal Intelligence System initialized")
    
    async def _load_signatures_async(self) -> None:
        """Load persisted signatures from file"""
        if not self.persistence_enabled:
            return
        
        # Use file-based persistence
        self._load_signatures_from_file()
    
    def _load_signatures_from_file(self) -> None:
        """Load signatures from local file"""
        try:
            signatures_file = Path(THERMAL_PERSISTENCE_FILE)
            if signatures_file.exists():
                with open(signatures_file, 'r') as f:
                    data = json.load(f)
                    
                    # Validate loaded data
                    if isinstance(data, dict):
                        self.patterns.import_signatures(data)
                        logger.info(f"Loaded signatures from {signatures_file}")
                    else:
                        logger.warning(f"Invalid data format in {signatures_file}")
            else:
                logger.info(f"No existing signatures file found")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in signatures file: {e}")
        except Exception as e:
            logger.warning(f"Could not load signatures: {e}")
    
    async def save_signatures(self) -> bool:
        """Save signatures to file"""
        if not self.persistence_enabled:
            return False
        
        try:
            data = self.patterns.export_signatures()
            return await self._save_signatures_to_file(data)
        except Exception as e:
            logger.error(f"Failed to save signatures: {e}")
            return False
    
    async def _save_signatures_to_file(self, data: Dict[str, Any]) -> bool:
        """Save signatures to local file"""
        try:
            signatures_file = Path(THERMAL_PERSISTENCE_FILE)
            
            # Create directory if needed
            signatures_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(signatures_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.last_save_time = time.time()
            logger.info(f"Saved {data['metadata']['total_patterns']} patterns")
            return True
        except Exception as e:
            logger.error(f"Failed to save signatures: {e}")
            return False
    
    def get_persistence_stats(self) -> Dict[str, Any]:
        """Get persistence statistics"""
        total_command_sigs = len(self.patterns.command_signatures)
        total_telemetry_sigs = len(self.patterns.telemetry_signatures)
        
        high_confidence = sum(
            1 for sig in self.patterns.command_signatures.values()
            if sig.confidence > 0.8
        )
        
        avg_samples = 0
        if total_command_sigs > 0:
            avg_samples = sum(
                sig.sample_count 
                for sig in self.patterns.command_signatures.values()
            ) / total_command_sigs
        
        return {
            'command_signatures': total_command_sigs,
            'telemetry_signatures': total_telemetry_sigs,
            'high_confidence_patterns': high_confidence,
            'average_samples_per_pattern': avg_samples,
            'last_save': self.last_save_time,
            'persistence_enabled': self.persistence_enabled,
            'auto_save_interval': self.auto_save_interval
        }
    
    async def start(self):
        """Start thermal monitoring"""
        if self.running:
            return
        
        # Load signatures asynchronously at startup
        try:
            await self._load_signatures_async()
        except Exception as e:
            logger.warning(f"Failed to load signatures: {e}")
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Thermal monitoring started")
    
    async def stop(self):
        """Stop thermal monitoring"""
        self.running = False
        
        # Save signatures
        if self.persistence_enabled:
            try:
                await asyncio.wait_for(
                    self.save_signatures(), 
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Signature save timed out")
        
        # Shutdown collector
        self.telemetry.shutdown()
        
        # Cancel monitor task
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await asyncio.wait_for(self.monitor_task, timeout=THERMAL_SHUTDOWN_TIMEOUT)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            finally:
                self.monitor_task = None
        
        # Cancel pending tasks
        if self._pending_tasks:
            for task in list(self._pending_tasks):
                if not task.done():
                    task.cancel()
            await asyncio.sleep(0.1)
            self._pending_tasks.clear()
        
        logger.info("Thermal monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                if not self.running:
                    break
                    
                # Adaptive sampling
                if self.current_state in [ThermalState.HOT, ThermalState.CRITICAL]:
                    interval = self.update_interval / 2
                else:
                    interval = self.update_interval
                
                # Collect telemetry
                if not self.running:
                    break
                    
                try:
                    sample = await asyncio.wait_for(
                        self.telemetry.collect(), 
                        timeout=THERMAL_TELEMETRY_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.warning("Telemetry collection timed out")
                    await asyncio.sleep(interval)
                    continue
                    
                if not self.running:
                    break
                
                self.samples.append(sample)
                
                # Process telemetry
                self._process_telemetry_batch()
                
                # Calculate velocity
                velocity = self.physics.calculate_velocity(list(self.samples))
                
                # Generate prediction
                prediction = None
                if THERMAL_PREDICTION_ENABLED and len(self.samples) > 10:
                    prediction = self.physics.predict_temperature(
                        sample, velocity, THERMAL_PREDICTION_HORIZON
                    )
                    self.predictions.append(prediction)
                
                if not self.running:
                    break
                
                # Statistical analysis
                stats = self.analyzer.analyze(list(self.samples), velocity)
                
                # Pattern recognition
                anomalies = []
                if THERMAL_PATTERN_RECOGNITION_ENABLED:
                    anomalies = self.patterns.find_anomalies(sample, list(self.samples))
                
                # State assessment
                self._assess_thermal_state(stats)
                
                # Recommendations
                recommendations = self._generate_recommendations(stats, prediction)
                
                # Build intelligence
                intelligence = ThermalIntelligence(
                    stats=stats,
                    prediction=prediction,
                    signatures=dict(self.patterns.command_signatures),
                    anomalies=[(time.time(), a) for a in anomalies],
                    recommendations=recommendations,
                    state=self.current_state,
                    confidence=self._calculate_confidence()
                )
                
                # Fire callbacks
                if self.running:
                    await self._fire_callbacks(intelligence)
                
                # Log events
                self._log_thermal_events(intelligence)
                
                # Auto-save
                if (self.persistence_enabled and 
                    time.time() - self.last_save_time > self.auto_save_interval):
                    await self.save_signatures()
                
                self.last_update = time.time()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}", exc_info=True)
                if self.running:
                    await asyncio.sleep(self.update_interval)
    
    def track_command(self, command: str, command_hash: str):
        """Track command execution"""
        if not self.running:
            return
            
        self.command_history.append((time.time(), command, command_hash))
        self.command_timestamps[command_hash] = time.time()
        
        # Set workload hash
        for sample in list(self.samples)[-5:]:
            sample.workload_hash = command_hash
    
    def complete_command(self, command: str, command_hash: str):
        """Complete command tracking"""
        if command_hash not in self.command_timestamps:
            return
        
        start_time = self.command_timestamps[command_hash]
        duration = time.time() - start_time
        
        # Find before/after samples
        before = None
        after = None
        
        for sample in self.samples:
            if sample.timestamp < start_time:
                before = sample
            elif sample.timestamp > start_time + duration:
                after = sample
                break
        
        if before and after:
            self.patterns.learn_signature(command, before, after, duration)
        
        del self.command_timestamps[command_hash]
    
    def track_render(self, command: str, thermal_cost_mw: float, duration: float):
        """Track render operation"""
        telemetry = {
            'command': command,
            'thermal_cost_mw': thermal_cost_mw,
            'render_duration': duration,
            'timestamp': time.time()
        }
        self.telemetry_queue.append(telemetry)
        
        if len(self.telemetry_queue) >= THERMAL_TELEMETRY_BATCH_SIZE:
            self._process_telemetry_batch()
    
    def process_telemetry_batch(self, telemetry_batch: List[Dict[str, Any]]):
        """Process telemetry batch"""
        for telemetry in telemetry_batch:
            self.telemetry_queue.append(telemetry)
        
        self._process_telemetry_batch()
        
        self.events.append(ThermalEvent(
            timestamp=time.time(),
            type='telemetry',
            description=f"Processed {len(telemetry_batch)} entries",
            state=self.current_state,
            metadata={'batch_size': len(telemetry_batch)}
        ))
    
    def _process_telemetry_batch(self):
        """Process pending telemetry"""
        now = time.time()
        if now - self.last_telemetry_process < THERMAL_TELEMETRY_PROCESSING_INTERVAL:
            return
        
        if not self.telemetry_queue:
            return
        
        batch = list(self.telemetry_queue)
        self.telemetry_queue.clear()
        self.last_telemetry_process = now
        
        for telemetry in batch:
            self.patterns.learn_from_telemetry(telemetry)
        
        logger.debug(f"Processed {len(batch)} telemetry entries")
    
    def _assess_thermal_state(self, stats: ThermalStatistics):
        """Assess thermal state with hysteresis"""
        max_temp = max(stats.current.zones.values()) if stats.current.zones else 0
        
        warming_fast = stats.velocity.trend in [ThermalTrend.WARMING, ThermalTrend.RAPID_WARMING]
        
        # State machine with hysteresis
        if self.current_state == ThermalState.UNKNOWN:
            if max_temp < THERMAL_TEMP_COLD:
                self.current_state = ThermalState.COLD
            elif max_temp < THERMAL_TEMP_OPTIMAL_MAX:
                self.current_state = ThermalState.OPTIMAL
            elif max_temp < THERMAL_TEMP_WARM:
                self.current_state = ThermalState.WARM
            elif max_temp < THERMAL_TEMP_HOT:
                self.current_state = ThermalState.HOT
            else:
                self.current_state = ThermalState.CRITICAL
        else:
            # Apply hysteresis
            if self.current_state == ThermalState.COLD and max_temp > THERMAL_TEMP_COLD + THERMAL_HYSTERESIS_UP:
                self.current_state = ThermalState.OPTIMAL
            elif self.current_state == ThermalState.OPTIMAL:
                if max_temp < THERMAL_TEMP_COLD - THERMAL_HYSTERESIS_DOWN:
                    self.current_state = ThermalState.COLD
                elif max_temp > THERMAL_TEMP_OPTIMAL_MAX + THERMAL_HYSTERESIS_UP or (max_temp > THERMAL_TEMP_OPTIMAL_MAX - 2 and warming_fast):
                    self.current_state = ThermalState.WARM
            elif self.current_state == ThermalState.WARM:
                if max_temp < THERMAL_TEMP_OPTIMAL_MAX - THERMAL_HYSTERESIS_DOWN:
                    self.current_state = ThermalState.OPTIMAL
                elif max_temp > THERMAL_TEMP_WARM + THERMAL_HYSTERESIS_UP or (max_temp > THERMAL_TEMP_WARM - 2 and warming_fast):
                    self.current_state = ThermalState.HOT
            elif self.current_state == ThermalState.HOT:
                if max_temp < THERMAL_TEMP_WARM - THERMAL_HYSTERESIS_DOWN:
                    self.current_state = ThermalState.WARM
                elif max_temp > THERMAL_TEMP_HOT + THERMAL_HYSTERESIS_UP or (max_temp > THERMAL_TEMP_HOT - 2 and warming_fast):
                    self.current_state = ThermalState.CRITICAL
            elif self.current_state == ThermalState.CRITICAL and max_temp < THERMAL_TEMP_HOT - THERMAL_HYSTERESIS_DOWN:
                self.current_state = ThermalState.HOT
    
    def _generate_recommendations(self, stats: ThermalStatistics,
                                prediction: Optional[ThermalPrediction]) -> List[str]:
        """Generate thermal recommendations"""
        recommendations = []
        
        # State recommendations
        if self.current_state == ThermalState.HOT:
            recommendations.append("Reduce workload intensity")
            if stats.velocity.trend == ThermalTrend.RAPID_WARMING:
                recommendations.append("Temperature rising rapidly")
        
        elif self.current_state == ThermalState.CRITICAL:
            recommendations.append("Critical temperature - minimize operations")
            recommendations.append("Check device ventilation")
        
        # Prediction recommendations
        if prediction and prediction.thermal_budget < THERMAL_BUDGET_WARNING_SECONDS:
            recommendations.append(
                f"Thermal budget: {prediction.thermal_budget:.0f}s"
            )
            if prediction.recommended_delay > 0:
                recommendations.append(
                    f"Delay: {prediction.recommended_delay:.1f}s"
                )
        
        # Network recommendations
        if stats.current.network == NetworkType.MOBILE_5G and stats.network_impact > THERMAL_NETWORK_IMPACT_WARNING:
            recommendations.append(f"5G: +{stats.network_impact:.1f}°C")
        
        # Charging recommendations
        if stats.charging_impact > THERMAL_CHARGING_IMPACT_WARNING and stats.current.charging:
            recommendations.append(f"Charging: +{stats.charging_impact:.1f}°C")
        
        # Pattern recommendations
        recent_commands = [cmd for t, cmd, _ in self.command_history if time.time() - t < 300]
        if recent_commands:
            predicted_impact = self.patterns.predict_impact(recent_commands[-5:])
            if predicted_impact > THERMAL_COMMAND_IMPACT_WARNING:
                recommendations.append(
                    f"Command impact: +{predicted_impact:.1f}°C"
                )
        
        return recommendations
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence score"""
        if not self.samples:
            return 0.0
        
        factors = []
        
        # Sample count
        sample_confidence = min(1.0, len(self.samples) / THERMAL_MIN_SAMPLES_CONFIDENCE)
        factors.append(sample_confidence)
        
        # Sensor confidence
        current = self.samples[-1]
        sensor_confidence = statistics.mean(current.confidence.values()) if current.confidence else 0.5
        factors.append(sensor_confidence)
        
        # Pattern confidence
        if self.patterns.command_signatures:
            pattern_confidence = statistics.mean(
                sig.confidence for sig in self.patterns.command_signatures.values()
            )
            factors.append(pattern_confidence)
        
        # Prediction confidence
        if len(self.predictions) > 10:
            factors.append(0.7)
        
        return statistics.mean(factors)
    
    def _log_thermal_events(self, intelligence: ThermalIntelligence):
        """Log thermal events"""
        # State changes
        if len(self.events) > 0:
            last_event = self.events[-1]
            if hasattr(last_event, 'state') and last_event.state != self.current_state:
                self.events.append(ThermalEvent(
                    timestamp=time.time(),
                    type='state_change',
                    description=f"{last_event.state} → {self.current_state}",
                    state=self.current_state
                ))
        
        # Anomalies
        for timestamp, anomaly in intelligence.anomalies:
            self.events.append(ThermalEvent(
                timestamp=timestamp,
                type='anomaly',
                description=anomaly,
                state=self.current_state
            ))
        
        # Critical events
        if self.current_state == ThermalState.CRITICAL:
            max_temp = max(intelligence.stats.current.zones.values())
            self.events.append(ThermalEvent(
                timestamp=time.time(),
                type='critical',
                description=f"Critical: {max_temp:.1f}°C",
                state=self.current_state
            ))
    
    async def _fire_callbacks(self, intelligence: ThermalIntelligence):
        """Fire event callbacks"""
        for callback in self.event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(intelligence)
                else:
                    callback(intelligence)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def register_callback(self, callback: Callable):
        """Register event callback"""
        self.event_callbacks.append(callback)
    
    def get_current_intelligence(self) -> Optional[ThermalIntelligence]:
        """Get current thermal intelligence"""
        if not self.samples:
            return None
        
        try:
            velocity = self.physics.calculate_velocity(list(self.samples))
            stats = self.analyzer.analyze(list(self.samples), velocity)
            
            prediction = None
            if self.predictions:
                prediction = self.predictions[-1]
            
            return ThermalIntelligence(
                stats=stats,
                prediction=prediction,
                signatures=dict(self.patterns.command_signatures),
                anomalies=[],
                recommendations=self._generate_recommendations(stats, prediction),
                state=self.current_state,
                confidence=self._calculate_confidence()
            )
        except Exception as e:
            logger.error(f"Failed to get intelligence: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        persistence_stats = self.get_persistence_stats()
        
        return {
            'samples_collected': len(self.samples),
            'predictions_made': len(self.predictions),
            'patterns_learned': len(self.patterns.command_signatures),
            'telemetry_patterns': len(self.patterns.telemetry_signatures),
            'thermal_events': len(self.events),
            'current_state': self.current_state.name,
            'confidence': self._calculate_confidence(),
            'update_interval': self.update_interval,
            'telemetry_failures': dict(self.telemetry.read_failures),
            'telemetry_queue': len(self.telemetry_queue),
            'persistence': persistence_stats
        }

# ============================================================================
# INTEGRATION
# ============================================================================

def create_thermal_intelligence() -> ThermalIntelligenceSystem:
    """Create thermal intelligence system"""
    thermal = ThermalIntelligenceSystem()
    
    # Integrate with persistence if available
    try:
        from persistence_system import get_global_persistence
        persistence = get_global_persistence()
        
        async def save_thermal_data():
            await thermal.save_signatures()
        
        if hasattr(persistence, 'engine'):
            original_persist = persistence.engine.persist_to_disk
            
            async def persist_with_thermal():
                await save_thermal_data()
                return await original_persist()
            
            persistence.engine.persist_to_disk = persist_with_thermal
            
        logger.info("Integrated with persistence")
    except ImportError:
        logger.info("Using local storage")
    
    return thermal

async def integrate_with_performance_system(thermal: ThermalIntelligenceSystem,
                                          performance_system):
    """
    Integrate with performance system.
    Thermal provides telemetry, performance makes decisions.
    """
    
    async def thermal_callback(intelligence: ThermalIntelligence):
        """Feed thermal data to performance system"""
        if hasattr(performance_system, 'update_thermal_intelligence'):
            performance_system.update_thermal_intelligence(intelligence)
        
        if intelligence.state in [ThermalState.HOT, ThermalState.CRITICAL]:
            logger.warning(
                f"State: {intelligence.state.name}, "
                f"Max: {max(intelligence.stats.current.zones.values()):.1f}°C, "
                f"Trend: {intelligence.stats.velocity.trend.name}"
            )
    
    thermal.register_callback(thermal_callback)
    
    # Command tracking integration
    if hasattr(performance_system, 'track_command'):
        original_track = performance_system.track_command
        
        def wrapped_track(command: str, *args, **kwargs):
            import hashlib
            command_hash = hashlib.md5(f"{command}{time.time()}".encode()).hexdigest()[:THERMAL_COMMAND_HASH_LENGTH]
            
            thermal.track_command(command, command_hash)
            
            result = original_track(command, *args, **kwargs)
            
            async def delayed_complete():
                await asyncio.sleep(5.0)
                thermal.complete_command(command, command_hash)
            
            task = asyncio.create_task(delayed_complete())
            thermal._pending_tasks.add(task)
            
            def cleanup_task(t):
                thermal._pending_tasks.discard(t)
            
            task.add_done_callback(cleanup_task)
            
            return result
        
        performance_system.track_command = wrapped_track
    
    logger.info("Integrated with performance system")

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

logger.info("S25+ Thermal Telemetry Module loaded - Fixed version with async persistence")
