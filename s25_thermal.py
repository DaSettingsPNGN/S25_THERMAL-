#!/usr/bin/env python3
"""
🔥 S25+ Thermal Intelligence System v2.24
==========================================
Battery-centric thermal management with physics-based prediction.

CORE APPROACH:
- Newton's law of cooling with measured thermal constants
- Battery backdating only (10s from 30s moving average)
- Die sensors report current temp - no backdating needed
- Battery τ=540s >> prediction horizon → pure power integration
- Tank model: simple battery-focused throttle decisions
- Dual-confidence learning: per-prediction × sample-size weighting

NEW IN v2.24:
- Removed die sensor backdating (test showed 0s optimal vs 7.9s)
- MAE restored to ~1.5°C range (was 10x worse with die backdating)
- Battery keeps 10s backdate (30s moving average confirmed)
- Measurement test validated: die sensors report current, not lagged

HARDWARE (Samsung Galaxy S25+ / Snapdragon 8 Elite):
- Zone 20 (cpuss-1-0): CPU_BIG - 2× Oryon Prime (τ_meas = 6.6s)
- Zone 13 (cpuss-0-0): CPU_LITTLE - 6× Oryon efficiency (τ_meas = 6.9s)
- Zone 23 (gpuss-0): GPU - Adreno 830 (τ_meas = 9.1s)
- Zone 31 (mdmss-0): Modem - 5G/WiFi (τ_meas = 9.0s)
- Zone 60 (battery): Battery thermistor (τ = 540s)
- Zone 52 (sys-therm-5): Chassis reference

THROTTLE POINTS:
- Samsung throttles at 40°C battery
- We throttle at 38.5°C (1.5°C safety margin = our MAE)
- Tank provides simple bool decision: can accept work or not

PHYSICS:
T(t) = T_amb + (T₀ - T_amb)·exp(-t/τ) + (P·R/k)·(1 - exp(-t/τ))

Battery simplification (τ >> horizon):
ΔT = (P/C) × Δt

Measured from step response testing, validated in production.
"""

import sys
import os
import json
import time
import asyncio
import subprocess
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any, Deque, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict, OrderedDict
from enum import Enum, auto, IntEnum
from datetime import datetime, timedelta
import math
import statistics
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger('PNGN.S25Thermal')
# ============================================================================
# TUNING CONSTANTS - ALL CONFIGURABLE PARAMETERS IN ONE PLACE
# ============================================================================

# ============================================================================
# CORE PREDICTION PARAMETERS
# ============================================================================
CHASSIS_DAMPING_FACTOR = 0.90          # α for chassis thermal inertia (0.9 = production optimal)
THERMAL_PREDICTION_HORIZON = 30.0      # seconds ahead to predict (70s total: battery t=-10 to t=60)
THERMAL_SAMPLE_INTERVAL_MS = 10000     # 10s uniform sampling
THERMAL_HISTORY_SIZE = 300             # samples to keep (300 × 10s = 50 minutes)
MIN_SAMPLES_FOR_PREDICTIONS = 12        # minimum samples before making predictions (configurable)
CHASSIS_CALIBRATION_OFFSET = 0.0       # °C (manual bias correction, default 0)

# Adaptive damping history sizes (multiples of MIN_SAMPLES_FOR_PREDICTIONS)
DAMPING_HISTORY_SLOW_ZONES = 10        # Battery (τ=540s): 10x = 120 samples = 20 min
DAMPING_HISTORY_FAST_ZONES = 1         # CPU/GPU (τ<10s): 1x = 12 samples = 2 min
DAMPING_HISTORY_MEDIUM_ZONES = 2       # Chassis (medium τ): 2x = 24 samples = 4 min

# Subprocess and timeouts
THERMAL_SUBPROCESS_TIMEOUT = 2.0       # seconds (general subprocess timeout)
THERMAL_NETWORK_TIMEOUT = 3.0          # seconds (network API calls)

# Network awareness
THERMAL_NETWORK_AWARENESS_ENABLED = True
THERMAL_WIFI_5G_FREQ_MIN = 5000        # MHz (5GHz detection threshold)

# Confidence and safety
CONFIDENCE_SAFETY_SCALE = 0.5          # prediction safety scaling
THERMAL_SENSOR_CONFIDENCE_REDUCED = 0.3  # fallback confidence for bad sensors

# Termux API commands
TERMUX_BATTERY_STATUS_CMD = ['termux-battery-status']
TERMUX_TELEPHONY_INFO_CMD = ['termux-telephony-deviceinfo']
TERMUX_WIFI_INFO_CMD = ['termux-wifi-connectioninfo']

# ============================================================================
# HARDWARE CONSTANTS - Samsung S25+ (Snapdragon 8 Elite for Galaxy)
# ============================================================================

# SoC specifications
SD8_ELITE_TDP = 8.0              # W (typical sustained)
SD8_ELITE_PEAK = 15.0            # W (burst)
SD8_ELITE_PROCESS_NODE = 3       # nm (TSMC N3)

# Battery
S25_PLUS_BATTERY_CAPACITY = 4900              # mAh
S25_PLUS_BATTERY_INTERNAL_RESISTANCE = 0.150  # Ohms
S25_PLUS_VAPOR_CHAMBER_EFFICIENCY = 0.85      # heat transfer efficiency
S25_PLUS_SCREEN_SIZE = 6.7                    # inches

# Per-zone thermal characteristics
ZONE_THERMAL_CONSTANTS = {
    'CPU_BIG': {
        'thermal_mass': 0.025,         # J/K
        'thermal_resistance': 2.8,     # °C/W
        'ambient_coupling': 0.80,
        'peak_power': 6.0,             # W
        'idle_power': 0.1,
        'time_constant': 0.07,         # s (die response)
        'measurement_tau': 6.6,        # s (sensor lag)
    },
    'CPU_LITTLE': {
        'thermal_mass': 0.050,
        'thermal_resistance': 3.2,
        'ambient_coupling': 0.75,
        'peak_power': 4.0,
        'idle_power': 0.05,
        'time_constant': 0.16,
        'measurement_tau': 6.9,
    },
    'GPU': {
        'thermal_mass': 0.035,
        'thermal_resistance': 2.5,
        'ambient_coupling': 0.85,
        'peak_power': 5.0,
        'idle_power': 0.2,
        'time_constant': 0.09,
        'measurement_tau': 9.1,
    },
    'BATTERY': {
        'thermal_mass': 45.0,          # J/K - huge thermal mass
        'thermal_resistance': 12.0,    # °C/W - poor coupling
        'ambient_coupling': 0.30,
        'peak_power': 3.0,
        'idle_power': 0.5,
        'time_constant': 540.0,        # s (9 minutes)
        'measurement_tau': 0,          # No lag for battery
    },
    'MODEM': {
        'thermal_mass': 0.020,
        'thermal_resistance': 4.0,
        'ambient_coupling': 0.60,
        'peak_power': 3.0,
        'idle_power': 0.2,
        'time_constant': 0.08,
        'measurement_tau': 9.0,
    },
}

# ============================================================================
# POWER INJECTION MODELS
# ============================================================================

# Display power
DISPLAY_POWER_MIN = 0.5              # W (minimum backlight)
DISPLAY_POWER_MAX = 4.0              # W (full brightness 1440p 120Hz)
DISPLAY_POWER_OFF = 0.1              # W (AOD mode)

# Baseline system power
SOC_FABRIC_POWER = 0.3               # W (interconnect, caches)
SENSOR_HUB_POWER = 0.1               # W (sensors)
BACKGROUND_SERVICES_POWER = 0.2      # W (daemons)
BASELINE_SYSTEM_POWER = 0.6          # W (sum of above)

# Network power by type
NETWORK_POWER = {
    'UNKNOWN': 0.2,
    'OFFLINE': 0.0,
    'WIFI_2G': 0.5,
    'WIFI_5G': 1.0,
    'MOBILE_3G': 1.5,
    'MOBILE_4G': 2.0,
    'MOBILE_5G': 3.0,
}

# Charging power
CHARGING_POWER_SLOW = 1.5            # W (5V 1A)
CHARGING_POWER_NORMAL = 2.5          # W (9V 2A)
CHARGING_POWER_FAST = 3.5            # W (15V 3A, 45W charger)

# Display thermal distribution (fraction of display power per zone)
DISPLAY_THERMAL_DISTRIBUTION = {
    'CPU_BIG': 0.15,
    'CPU_LITTLE': 0.10,
    'GPU': 0.40,                     # composition - largest impact
    'BATTERY': 0.25,                 # behind display
    'MODEM': 0.10,
}

# ============================================================================
# THERMAL STATE THRESHOLDS
# ============================================================================
THERMAL_TEMP_COLD = 20.0             # °C
THERMAL_TEMP_OPTIMAL_MIN = 25.0
THERMAL_TEMP_OPTIMAL_MAX = 38.0
THERMAL_TEMP_WARM = 40.0
THERMAL_TEMP_HOT = 42.0
THERMAL_TEMP_CRITICAL = 45.0

# Hysteresis to prevent oscillation
THERMAL_HYSTERESIS_UP = 1.0          # °C (transition up threshold)
THERMAL_HYSTERESIS_DOWN = 2.0        # °C (transition down threshold)

# Hardware throttling points
THROTTLE_START_TEMP = 42.0           # °C (begins throttling)
THROTTLE_AGGRESSIVE_TEMP = 45.0      # °C (aggressive throttling)
SHUTDOWN_TEMP = 48.0                 # °C (emergency shutdown)

# Thermal tank thresholds (battery-centric)
SAMSUNG_THROTTLE_TEMP = 42.0         # °C (Samsung's actual hardware throttle point)
TANK_THROTTLE_TEMP = 40.0            # °C (our throttle, 2°C safety buffer)
TANK_WARNING_TEMP = 38.0             # °C (early warning, 4°C buffer)

# Thermal velocity thresholds (°C/s)
# Battery τ=540s → ~0.002°C/s steady, CPU faster at ~0.1-0.3°C/s during load
THERMAL_VELOCITY_RAPID_COOLING = -0.10    # °C/s (rapid cooldown)
THERMAL_VELOCITY_COOLING = -0.02          # °C/s (slow cooling)
THERMAL_VELOCITY_WARMING = 0.05           # °C/s (heating begins)
THERMAL_VELOCITY_RAPID_WARMING = 0.15     # °C/s (rapid heating)

# ============================================================================
# SENSOR & VALIDATION
# ============================================================================
THERMAL_SENSOR_TEMP_MIN = 15.0       # °C (sanity check lower bound)
THERMAL_SENSOR_TEMP_MAX = 75.0       # °C (sanity check upper bound)

# Thermal zone mapping (None = auto-discover from /sys/class/thermal)
THERMAL_ZONES = None

# Battery prediction horizon
BATTERY_PREDICTION_HORIZON = THERMAL_PREDICTION_HORIZON    # seconds (matches main horizon for consistency)

# ============================================================================
# FEATURE FLAGS
# ============================================================================
THERMAL_PREDICTION_ENABLED = True

# ============================================================================
# SHARED TYPES
# ============================================================================

class ThermalState(Enum):
    COLD = auto()
    OPTIMAL = auto()
    WARM = auto()
    HOT = auto()
    CRITICAL = auto()
    UNKNOWN = auto()

class ThermalZone(Enum):
    CPU_BIG = 20       # cpuss-1-0: 2× Oryon Prime aggregate
    CPU_LITTLE = 13    # cpuss-0-0: 6× Oryon efficiency aggregate  
    GPU = 23           # gpuss-0: Adreno 830
    BATTERY = 60       # battery thermistor
    MODEM = 31         # mdmss-0: 5G/WiFi modem
    CHASSIS = 52       # sys-therm-5: chassis thermal reference

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
    chassis: Optional[float] = None
    network: NetworkType = NetworkType.UNKNOWN
    charging: bool = False
    screen_on: bool = True
    workload_hash: Optional[str] = None
    cache_hit_rate: float = 0.85  # NEW: typical cache hit rate
    display_brightness: Optional[int] = None  # V2.3: 0-255 brightness level
    battery_current: Optional[float] = None  # V2.3: charging/discharge current in mA

@dataclass
class ThermalVelocity:
    """Temperature rate of change per zone"""
    zones: Dict[ThermalZone, float]  # °C/second per zone
    overall: float  # weighted average °C/second
    trend: ThermalTrend
    acceleration: float  # °C/second²

@dataclass
class ThermalPrediction:
    """Future temperature prediction per zone"""
    timestamp: float
    horizon: float  # seconds into future
    predicted_temps: Dict[ThermalZone, float]
    confidence: float  # Overall confidence
    confidence_by_zone: Dict[ThermalZone, float] = field(default_factory=dict)  # Per-zone confidence
    thermal_budget: float = 0.0  # seconds until throttling
    recommended_delay: float = 0.0  # seconds to wait before heavy work

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
    """Complete thermal telemetry package"""
    stats: ThermalStatistics
    prediction: Optional[ThermalPrediction]
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

@dataclass
class PredictionError:
    """Single prediction error measurement"""
    timestamp: float
    horizon: float
    zone: ThermalZone
    predicted: float
    actual: float
    error: float  # actual - predicted
    ambient_estimate: float

@dataclass
class FastCPUSample:
    """Lightweight CPU-only sample for high-frequency monitoring (10Hz)"""
    timestamp: float
    cpu_big: Optional[float] = None
    cpu_little: Optional[float] = None
    gpu: Optional[float] = None

# ============================================================================
# TELEMETRY COLLECTOR
# ============================================================================

class ThermalTelemetryCollector:
    """
    Collects thermal data from system interfaces.
    Reads /sys/class/thermal zones and Termux API.
    """
    
    def __init__(self):
        self.zone_paths = self._discover_thermal_zones()
        self.read_failures = defaultdict(int)
        
        # TWO-TIER CACHING SYSTEM
        # Prediction cache (long ages - these are context, not physics inputs)
        self.pred_battery = None
        self.pred_battery_time = 0
        self.pred_network = NetworkType.UNKNOWN
        self.pred_network_time = 0
        self.pred_brightness = None
        self.pred_brightness_time = 0
        
        # Display cache (shorter ages for UI responsiveness)
        self.ui_battery = None
        self.ui_battery_time = 0
        self.ui_network = NetworkType.UNKNOWN
        self.ui_network_time = 0
        self.ui_brightness = None
        self.ui_brightness_time = 0
        
        # Legacy compatibility (uses UI cache by default)
        self.battery_cache = {}
        self.last_battery_read = 0
        self.network_cache = NetworkType.UNKNOWN
        self.last_network_read = 0
        
        logger.info(f"Discovered {len(self.zone_paths)} thermal zones")
        logger.info("Two-tier caching: battery 30s, network 5min, brightness 1min (predictions)")
    
    def _discover_thermal_zones(self) -> Dict[ThermalZone, str]:
        """Map ThermalZone enums to filesystem paths - ONLY valid enum zones"""
        zone_map = {}
        
        # Valid zones from enum
        valid_zones = {zone.value: zone for zone in ThermalZone}
        
        # Use configured zones if available
        if THERMAL_ZONES:
            # Handle both list and dict formats
            if isinstance(THERMAL_ZONES, dict):
                # THERMAL_ZONES is a dict: zone_id -> path
                for zone_id, path in THERMAL_ZONES.items():
                    try:
                        # Only map if zone_id matches a valid enum value
                        if isinstance(zone_id, int) and zone_id in valid_zones:
                            zone_enum = valid_zones[zone_id]
                            if Path(path).exists():
                                zone_map[zone_enum] = path
                    except (IndexError, OSError):
                        pass
            else:
                # THERMAL_ZONES is a list - match by enum VALUE, not index
                for zone_enum in ThermalZone:
                    zone_id = zone_enum.value
                    if zone_id < len(THERMAL_ZONES):
                        try:
                            path = THERMAL_ZONES[zone_id]
                            if Path(path).exists():
                                zone_map[zone_enum] = path
                        except (IndexError, OSError):
                            pass
        
        # Fallback discovery - only for defined enum zones
        if not zone_map:
            thermal_base = Path('/sys/class/thermal')
            if thermal_base.exists():
                for zone_enum in ThermalZone:
                    zone_path = thermal_base / f'thermal_zone{zone_enum.value}' / 'temp'
                    if zone_path.exists():
                        zone_map[zone_enum] = str(zone_path)
        
        return zone_map
    
    async def collect_sample(self, for_prediction: bool = False) -> ThermalSample:
        """
        Collect complete thermal sample with optimized caching.
        
        Args:
            for_prediction: If True, uses 1s cache (accurate for predictions)
                          If False, uses 5s cache (efficient for display/UI)
        """
        timestamp = time.time()
        
        # Determine which cache to use and max age per API type
        if for_prediction:
            battery_cache_time = self.pred_battery_time
            battery_cache = self.pred_battery
            network_cache_time = self.pred_network_time
            network_cache = self.pred_network
            brightness_cache_time = self.pred_brightness_time
            brightness_cache = self.pred_brightness
            battery_max_age = 30.0      # 30s - expensive, ground truth for power
            network_max_age = 300.0     # 5min - rare regime change, not physics input
            brightness_max_age = 60.0   # 1min - slow changes, context only
        else:
            battery_cache_time = self.ui_battery_time
            battery_cache = self.ui_battery
            network_cache_time = self.ui_network_time
            network_cache = self.ui_network
            brightness_cache_time = self.ui_brightness_time
            brightness_cache = self.ui_brightness
            battery_max_age = 5.0       # 5s for UI
            network_max_age = 60.0      # 1min for UI
            brightness_max_age = 30.0   # 30s for UI
        
        # Read thermal zones (fast, local filesystem)
        zones, confidence = await self._read_thermal_zones_batch()
        
        # PARALLEL API CALLS (50% faster than sequential)
        # Only call APIs if cache is expired
        api_tasks = []
        need_battery = (timestamp - battery_cache_time) > battery_max_age
        need_network = (timestamp - network_cache_time) > network_max_age
        need_brightness = (timestamp - brightness_cache_time) > brightness_max_age
        
        if need_battery:
            api_tasks.append(self._read_battery_status())
        if need_network:
            api_tasks.append(self._detect_network_type())
        if need_brightness:
            api_tasks.append(self._get_display_brightness())
        
        # Run all API calls in parallel
        if api_tasks:
            try:
                results = await asyncio.gather(*api_tasks, return_exceptions=True)
                
                # Process results
                result_idx = 0
                if need_battery:
                    battery_result = results[result_idx]
                    result_idx += 1
                    if not isinstance(battery_result, Exception) and battery_result:
                        battery_cache = battery_result
                        battery_cache_time = timestamp
                        # Update the appropriate cache
                        if for_prediction:
                            self.pred_battery = battery_cache
                            self.pred_battery_time = timestamp
                        else:
                            self.ui_battery = battery_cache
                            self.ui_battery_time = timestamp
                
                if need_network:
                    network_result = results[result_idx]
                    result_idx += 1
                    if not isinstance(network_result, Exception) and network_result != NetworkType.UNKNOWN:
                        network_cache = network_result
                        network_cache_time = timestamp
                        # Update the appropriate cache
                        if for_prediction:
                            self.pred_network = network_cache
                            self.pred_network_time = timestamp
                        else:
                            self.ui_network = network_cache
                            self.ui_network_time = timestamp
                
                if need_brightness:
                    brightness_result = results[result_idx]
                    if not isinstance(brightness_result, Exception) and brightness_result is not None:
                        brightness_cache = brightness_result
                        brightness_cache_time = timestamp
                        # Update the appropriate cache
                        if for_prediction:
                            self.pred_brightness = brightness_cache
                            self.pred_brightness_time = timestamp
                        else:
                            self.ui_brightness = brightness_cache
                            self.ui_brightness_time = timestamp
            
            except Exception as e:
                logger.debug(f"Parallel API call error: {e}")
        
        # Update legacy cache for backward compatibility
        if battery_cache:
            self.battery_cache = battery_cache
            self.last_battery_read = battery_cache_time
        if network_cache != NetworkType.UNKNOWN:
            self.network_cache = network_cache
            self.last_network_read = network_cache_time
        
        # Ambient temperature (use AMBIENT zone or estimate)
        chassis_temp = zones.get(ThermalZone.CHASSIS)
        if chassis_temp is None:
            # Estimate ambient from battery (slowest changing zone)
            chassis_temp = zones.get(ThermalZone.BATTERY, 25.0) - 5.0
        
        # Extract data from cache
        charging = battery_cache.get('plugged', False) if battery_cache else False
        screen_on = True  # TODO: detect from API
        
        # V2.3: Use cached display brightness (fetched in parallel if needed)
        display_brightness = brightness_cache
        
        # V2.3: Extract battery current from battery status
        battery_current = None
        if battery_cache:
            # Current in mA (positive = charging, negative = discharging)
            battery_current = battery_cache.get('current', None)
        
        # Cache hit rate (TODO: read from perf counters, for now estimate)
        cache_hit_rate = 0.85  # typical for normal workloads
        
        return ThermalSample(
            timestamp=timestamp,
            zones=zones,
            confidence=confidence,
            chassis=chassis_temp,
            network=network_cache,
            charging=charging,
            screen_on=screen_on,
            workload_hash=None,
            cache_hit_rate=cache_hit_rate,
            display_brightness=display_brightness,
            battery_current=battery_current
        )
    
    async def _read_thermal_zones_batch(self) -> Tuple[Dict[ThermalZone, float], Dict[ThermalZone, float]]:
        """
        Read all thermal zones efficiently in batch.
        Uses single shell command instead of multiple file opens.
        """
        zones = {}
        confidence = {}
        
        # Whitelist: Only allow real hardware thermal sensors
        REAL_HARDWARE_ZONES = {'BATTERY', 'CPU_BIG', 'CPU_LITTLE', 'GPU', 'MODEM', 'CHASSIS'}
        
        # Fallback: read zones individually (compatible with all systems)
        for zone_enum, path in self.zone_paths.items():
            # Only read whitelisted zones
            zone_name = str(zone_enum).split('.')[-1]
            if zone_name not in REAL_HARDWARE_ZONES:
                continue
                
            try:
                with open(path, 'r') as f:
                    # Temperature in millidegrees
                    temp = float(f.read().strip()) / 1000.0
                    
                    # Validate temperature
                    if THERMAL_SENSOR_TEMP_MIN <= temp <= THERMAL_SENSOR_TEMP_MAX:
                        zones[zone_enum] = temp
                        confidence[zone_enum] = 1.0
                    else:
                        # Out of range - reduced confidence
                        zones[zone_enum] = temp
                        confidence[zone_enum] = THERMAL_SENSOR_CONFIDENCE_REDUCED
                    
                self.read_failures[zone_enum] = 0
            except Exception as e:
                self.read_failures[zone_enum] += 1
                if self.read_failures[zone_enum] < 5:
                    logger.debug(f"Failed to read {zone_enum.name}: {e}")
        
        return zones, confidence
    
    async def collect_sample_for_display(self) -> ThermalSample:
        """
        Convenience method for UI/display data.
        Uses 5s cache (efficient, can tolerate slight staleness).
        """
        return await self.collect_sample(for_prediction=False)
    
    async def _read_battery_status(self) -> Optional[Dict]:
        """Read battery status from Termux API"""
        try:
            proc = await asyncio.create_subprocess_exec(
                *TERMUX_BATTERY_STATUS_CMD,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), 
                timeout=THERMAL_SUBPROCESS_TIMEOUT
            )
            
            if proc.returncode == 0:
                data = json.loads(stdout.decode())
                return {
                    'plugged': data.get('plugged', 'UNPLUGGED') != 'UNPLUGGED',
                    'percentage': data.get('percentage', 0),
                    'temperature': data.get('temperature', 25.0),
                    'current': data.get('current', 0)  # mA (+ charging, - discharging)
                }
        except (asyncio.TimeoutError, json.JSONDecodeError, Exception) as e:
            logger.debug(f"Battery status read failed: {e}")
        
        return None
    
    async def _detect_network_type(self) -> NetworkType:
        """Detect current network type"""
        if not THERMAL_NETWORK_AWARENESS_ENABLED:
            return NetworkType.UNKNOWN
        
        try:
            # Try WiFi first
            proc = await asyncio.create_subprocess_exec(
                *TERMUX_WIFI_INFO_CMD,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=THERMAL_NETWORK_TIMEOUT
            )
            
            if proc.returncode == 0:
                data = json.loads(stdout.decode())
                frequency = data.get('frequency', 0)
                
                if frequency >= THERMAL_WIFI_5G_FREQ_MIN:
                    return NetworkType.WIFI_5G
                elif frequency > 0:
                    return NetworkType.WIFI_2G
            
            # Try mobile network
            proc = await asyncio.create_subprocess_exec(
                *TERMUX_TELEPHONY_INFO_CMD,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=THERMAL_NETWORK_TIMEOUT
            )
            
            if proc.returncode == 0:
                data = json.loads(stdout.decode())
                network_type = data.get('network_type', '').upper()
                
                if '5G' in network_type:
                    return NetworkType.MOBILE_5G
                elif '4G' in network_type or 'LTE' in network_type:
                    return NetworkType.MOBILE_4G
                elif '3G' in network_type:
                    return NetworkType.MOBILE_3G
        
        except Exception as e:
            logger.debug(f"Network detection failed: {e}")
        
        return NetworkType.UNKNOWN
    
    async def _get_display_brightness(self) -> Optional[int]:
        """
        Get current display brightness (0-255).
        Uses Termux API to read system brightness setting.
        """
        try:
            # Try to get brightness from settings
            proc = await asyncio.create_subprocess_exec(
                'termux-brightness',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=1.0
            )
            
            if proc.returncode == 0:
                # Parse brightness value (0-255)
                brightness_str = stdout.decode().strip()
                brightness = int(brightness_str)
                return max(0, min(brightness, 255))
        
        except Exception as e:
            logger.debug(f"Brightness detection failed: {e}")
        
        # Fallback: assume medium brightness if screen is on
        return 128  # 50% brightness
    
    def estimate_display_power(self, screen_on: bool, brightness: Optional[int] = None) -> float:
        """
        Estimate display power consumption based on state and brightness.
        
        Args:
            screen_on: Whether screen is on
            brightness: Brightness level 0-255 (optional)
        
        Returns:
            Power in Watts
        """
        if not screen_on:
            return DISPLAY_POWER_OFF
        
        if brightness is None:
            brightness = 128  # assume 50%
        
        # Linear interpolation between min and max
        brightness_fraction = brightness / 255.0
        power = DISPLAY_POWER_MIN + (DISPLAY_POWER_MAX - DISPLAY_POWER_MIN) * brightness_fraction
        
        return power

# ============================================================================
# ZONE-SPECIFIC PHYSICS ENGINE
# ============================================================================

class ZonePhysicsEngine:
    """
    Per-zone thermal physics using Newton's law of cooling.
    Each zone has unique thermal mass, resistance, and time constant.
    """
    
    def __init__(self):
        self.zone_constants = ZONE_THERMAL_CONSTANTS
        self.tuner = TransientResponseTuner()
        
        # Prediction error tracking for adaptive damping
        # History size scales with thermal time constant
        self.zone_history_sizes = {
            'BATTERY': MIN_SAMPLES_FOR_PREDICTIONS * DAMPING_HISTORY_SLOW_ZONES,
            'CPU_BIG': MIN_SAMPLES_FOR_PREDICTIONS * DAMPING_HISTORY_FAST_ZONES,
            'CPU_LITTLE': MIN_SAMPLES_FOR_PREDICTIONS * DAMPING_HISTORY_FAST_ZONES,
            'GPU': MIN_SAMPLES_FOR_PREDICTIONS * DAMPING_HISTORY_FAST_ZONES,
            'MODEM': MIN_SAMPLES_FOR_PREDICTIONS * DAMPING_HISTORY_FAST_ZONES,
            'CHASSIS': MIN_SAMPLES_FOR_PREDICTIONS * DAMPING_HISTORY_MEDIUM_ZONES,
        }
        
        self.prediction_errors: Dict[ThermalZone, Deque[Tuple[float, float, float]]] = {}
        # Each entry: (error, confidence, timestamp)
        
        logger.info(f"Physics engine initialized with {len(self.zone_constants)} zone models")
        logger.info(f"Transient tuner loaded: heating={self.tuner.momentum_heating:.3f}, "
                   f"cooling={self.tuner.momentum_cooling:.3f}, stable={self.tuner.momentum_stable:.3f}")
        logger.info(f"Adaptive damping: Battery={self.zone_history_sizes['BATTERY']} samples, "
                   f"Fast zones={self.zone_history_sizes['CPU_BIG']} samples")
    
    def _get_momentum_factor(self, velocity: float) -> float:
        """
        Select appropriate momentum factor based on velocity regime.
        
        Heating: velocity > +0.1°C/s
        Cooling: velocity < -0.1°C/s  
        Stable: |velocity| <= 0.1°C/s
        """
        if velocity > 0.1:
            return self.tuner.momentum_heating
        elif velocity < -0.1:
            return self.tuner.momentum_cooling
        else:
            return self.tuner.momentum_stable
    
    def _calculate_adaptive_damping(self, zone: ThermalZone, 
                                   current_temp: float,
                                   raw_predicted: float,
                                   current_confidence: float,
                                   sample_count: int) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate adaptive damping factor based on recent prediction errors.
        
        Args:
            zone: Which thermal zone
            current_temp: Current measured temperature
            raw_predicted: Raw physics-based prediction
            current_confidence: Confidence in current prediction
            sample_count: Total samples collected (for scaling damping strength)
        
        Returns:
            (damping_factor, debug_info)
            damping_factor: 0.0 to 0.5 (applied as: damped_delta = raw_delta * (1 - factor))
        """
        # No error history yet
        if zone not in self.prediction_errors or len(self.prediction_errors[zone]) < 3:
            return 0.0, {'reason': 'insufficient_history'}
        
        # Get zone-specific history size for ramp calculation
        zone_name = str(zone).split('.')[-1]
        zone_history_size = self.zone_history_sizes.get(zone_name, MIN_SAMPLES_FOR_PREDICTIONS)
        
        # Scale damping strength by sample count
        # MIN_SAMPLES_FOR_PREDICTIONS = 12: 0% strength
        # zone_history_size: 100% strength
        # Battery (120 samples): ramps 12→120
        # CPU (12 samples): ramps 12→12 (instant full strength)
        
        if sample_count < MIN_SAMPLES_FOR_PREDICTIONS:
            # Not enough data to make predictions at all
            return 0.0, {'reason': 'below_min_samples', 'count': sample_count}
        
        # Ramp from MIN_SAMPLES to zone_history_size
        ramp_range = zone_history_size - MIN_SAMPLES_FOR_PREDICTIONS
        if ramp_range > 0:
            strength_scale = min(1.0, (sample_count - MIN_SAMPLES_FOR_PREDICTIONS) / ramp_range)
        else:
            # Fast zones: instant full strength once MIN_SAMPLES reached
            strength_scale = 1.0
        
        # Get recent errors
        errors = list(self.prediction_errors[zone])
        
        # Confidence-weighted mean error (recent bias)
        weighted_errors = []
        total_conf = 0.0
        for error, conf, timestamp in errors:
            weighted_errors.append(error * conf)
            total_conf += conf
        
        if total_conf < 0.1:
            return 0.0, {'reason': 'low_confidence'}
        
        bias = sum(weighted_errors) / total_conf
        
        # Only damp if predicted delta is significant
        raw_delta = raw_predicted - current_temp
        if abs(raw_delta) < 0.5:
            return 0.0, {'reason': 'delta_too_small', 'delta': raw_delta}
        
        # Damping proportional to bias
        # Positive bias = overpredict → reduce delta
        # Negative bias = underpredict → increase delta (negative damping)
        
        # Get typical error magnitude
        recent_errors = [abs(e) for e, c, t in errors[-3:]]
        avg_error_mag = statistics.mean(recent_errors) if recent_errors else 1.0
        
        # Relative bias: bias normalized by typical error
        relative_bias = bias / max(avg_error_mag, 1.0)
        
        # Base damping strength (30% of bias)
        base_strength = 0.3
        
        # Scale by confidence and sample count
        confidence_scale = (current_confidence + 0.5) / 1.5  # 0.33 to 1.0
        
        damping = relative_bias * base_strength * confidence_scale * strength_scale
        
        # Clamp to reasonable range
        damping = max(-0.5, min(0.5, damping))
        
        debug = {
            'bias': bias,
            'relative_bias': relative_bias,
            'strength_scale': strength_scale,
            'confidence_scale': confidence_scale,
            'raw_delta': raw_delta,
            'damping': damping,
            'sample_count': sample_count,
            'zone_history_size': zone_history_size,
            'error_count': len(errors)
        }
        
        return damping, debug

    
    def validate_and_tune(self, prediction: 'ThermalPrediction', actual: ThermalSample, 
                          velocity: 'ThermalVelocity'):
        """
        Compare prediction against actual measurement and tune momentum factors.
        
        Called after collecting a sample that corresponds to a previous prediction.
        Classifies zones by state, accumulates errors with confidences, and triggers tuning.
        
        NEW: Also records per-zone errors for adaptive damping.
        """
        errors_by_state = {
            'heating': [],
            'cooling': [],
            'stable': []
        }
        
        confidences_by_state = {
            'heating': [],
            'cooling': [],
            'stable': []
        }
        
        # Collect errors and confidences by zone, classify by state
        for zone, predicted_temp in prediction.predicted_temps.items():
            if zone not in actual.zones:
                continue
            
            actual_temp = actual.zones[zone]
            error = predicted_temp - actual_temp  # positive = overprediction
            
            # Get confidence for this zone (default 0.7 if missing)
            conf = prediction.confidence_by_zone.get(zone, 0.7)
            
            # Get velocity for this zone to determine state
            vel = velocity.zones.get(zone, 0)
            
            # Classify state and record error + confidence
            if vel > 0.1:
                errors_by_state['heating'].append(error)
                confidences_by_state['heating'].append(conf)
            elif vel < -0.1:
                errors_by_state['cooling'].append(error)
                confidences_by_state['cooling'].append(conf)
            else:
                errors_by_state['stable'].append(error)
                confidences_by_state['stable'].append(conf)
            
            # NEW: Record per-zone error for adaptive damping
            if zone not in self.prediction_errors:
                # Get zone-specific history size
                zone_name = str(zone).split('.')[-1]
                history_size = self.zone_history_sizes.get(zone_name, MIN_SAMPLES_FOR_PREDICTIONS)
                self.prediction_errors[zone] = deque(maxlen=history_size)
            
            self.prediction_errors[zone].append((error, conf, time.time()))
        
        # Trigger tuning for each state that has sufficient data
        if errors_by_state['heating']:
            self.tuner.tune_heating(errors_by_state['heating'], confidences_by_state['heating'])
        
        if errors_by_state['cooling']:
            self.tuner.tune_cooling(errors_by_state['cooling'], confidences_by_state['cooling'])
        
        if errors_by_state['stable']:
            self.tuner.tune_stable(errors_by_state['stable'], confidences_by_state['stable'])
    
    def estimate_chassis(self, sample: ThermalSample) -> float:
        """
        Dynamically estimate ambient temperature from system state.
        
        IMPROVED v2.8: 
        - Removed 40°C clamp (was causing 10-27°C errors outdoors)
        - Dynamic offset based on battery-to-coolest delta
        - Sun exposure detection (battery cooler than active zones)
        - Wider valid range (10-70°C instead of 10-40°C)
        """
        # Try to find battery zone
        battery_temp = None
        for zone, temp in sample.zones.items():
            zone_name = str(zone).split('.')[-1]
            if 'BATTERY' in zone_name.upper():
                battery_temp = temp
                break
        
        # Get other zone temperatures (excluding battery and ambient)
        other_temps = [t for z, t in sample.zones.items() 
                      if z not in [ThermalZone.BATTERY, ThermalZone.CHASSIS]]
        
        if battery_temp is None or not other_temps:
            # Fallback: multi-zone average excluding outliers
            if sample.zones:
                temps = [t for z, t in sample.zones.items() if z != ThermalZone.CHASSIS]
                if len(temps) >= 3:
                    temps_sorted = sorted(temps)
                    n = max(1, len(temps_sorted) * 3 // 5)
                    avg_cool = sum(temps_sorted[:n]) / n
                    return max(10.0, min(avg_cool - 2.5, 70.0))
                elif temps:
                    return max(10.0, min(min(temps) - 2.5, 70.0))
            return sample.chassis if sample.chassis else 25.0
        
        coolest = min(other_temps)
        hottest = max(other_temps)
        
        # CASE 1: Battery hotter than all zones (normal operation)
        if battery_temp > coolest + 1.0:
            # Battery is thermally insulated and has massive thermal mass
            # It runs above ambient but below active zones
            
            # Get state-dependent base offset
            battery_current = getattr(sample, 'battery_current', None)
            
            if sample.charging:
                # Charging generates heat
                if battery_current and battery_current > 1500:  # Fast charge
                    base_offset = 8.0
                else:
                    base_offset = 6.0
            elif battery_current and battery_current < -1500:
                # Heavy discharge
                base_offset = 5.0
            else:
                # Light discharge/idle
                base_offset = 3.0
            
            # Adjust offset based on battery-to-coolest delta
            # Larger delta = more active system = larger offset
            battery_to_coolest = battery_temp - coolest
            if battery_to_coolest > 8.0:
                # Very hot battery relative to CPU - reduce offset
                # (ambient is probably also hot)
                offset = base_offset * 0.7
            elif battery_to_coolest < 3.0:
                # Battery barely above CPU - increase offset
                # (everything at near-ambient)
                offset = base_offset * 1.3
            else:
                offset = base_offset
            
            ambient_est = battery_temp - offset
        
        # CASE 2: Battery cooler than some zones (sun exposure / outdoor)
        else:
            # This happens when:
            # - Ambient sensor reading radiant heat (direct sun)
            # - Battery insulated, heats slower than air/sensors
            # - Active zones (CPU/GPU) running hot
            
            # Trust the coolest active zone as proxy
            # Active zones run ~2-3°C above true ambient
            ambient_est = coolest - 2.5
            
            # Cross-check: if battery is way cooler, ambient is probably even lower
            if battery_temp < coolest - 5.0:
                # Battery significantly cooler suggests very recent temp change
                # or strong external cooling (AC, wind)
                battery_ambient = battery_temp - 1.0  # Battery barely above ambient when cool
                ambient_est = min(ambient_est, battery_ambient)
        
        # Validate with hottest zone (sanity check)
        # Ambient can't be hotter than the hottest component
        if ambient_est > hottest:
            ambient_est = hottest - 1.0
        
        # Clamp to physically reasonable range
        # Removed 40°C cap - outdoor environments can be 50-60°C
        ambient_est = max(10.0, min(ambient_est, 70.0))
        
        return ambient_est
    
    def effective_thermal_resistance(self, zone_name: str, temp: float, 
                                     chassis: float) -> float:
        """
        Calculate effective thermal resistance with temperature-dependent corrections.
        R decreases at higher ΔT due to improved natural convection and radiation.
        """
        if zone_name not in self.zone_constants:
            return 5.0  # Default fallback
        
        base_R = self.zone_constants[zone_name]['thermal_resistance']
        dT = temp - chassis
        
        if dT < 1:
            return base_R  # No correction at small ΔT
        
        # Natural convection improvement: R decreases ~8% per 10°C
        # Based on Nusselt number scaling: Nu ∝ Ra^0.25, Ra ∝ ΔT
        convection_improvement = 1.0 - 0.008 * min(dT, 25.0)
        
        # Radiation starts to matter at high temps (T > ambient + 15°C)
        if dT > 15:
            # Stefan-Boltzmann: radiation ∝ T^4
            # Approximate as 4% improvement for every 10°C above 15°C delta
            radiation_improvement = 0.96 ** ((dT - 15) / 10.0)
        else:
            radiation_improvement = 1.0
        
        # Combined effect (multiplicative)
        effective_R = base_R * convection_improvement * radiation_improvement
        
        # Don't go below 60% of base R (physical limit)
        return max(0.6 * base_R, effective_R)
    
    def calculate_velocity(self, samples: List[ThermalSample]) -> ThermalVelocity:
        """Calculate dT/dt for each zone using linear regression"""
        if len(samples) < 2:
            return ThermalVelocity(
                zones={},
                overall=0.0,
                trend=ThermalTrend.STABLE,
                acceleration=0.0
            )
        
        # Use last 5 samples for velocity calculation (noise reduction)
        recent = samples[-5:] if len(samples) >= 5 else samples
        
        zone_velocities = {}
        
        for zone in ThermalZone:
            temps = []
            times = []
            
            for sample in recent:
                if zone in sample.zones:
                    temps.append(sample.zones[zone])
                    times.append(sample.timestamp)
            
            if len(temps) >= 2:
                # Linear regression: slope = dT/dt
                dt = times[-1] - times[0]
                dT = temps[-1] - temps[0]
                velocity = dT / dt if dt > 0 else 0.0
                zone_velocities[zone] = velocity
        
        # Overall velocity (weighted by zone importance)
        # CPU_BIG and GPU weighted higher as they throttle first
        if zone_velocities:
            weights = {
                ThermalZone.CPU_BIG: 2.0,
                ThermalZone.GPU: 1.5,
                ThermalZone.CPU_LITTLE: 1.0,
                ThermalZone.BATTERY: 0.5,
                ThermalZone.MODEM: 0.8,
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            for zone, vel in zone_velocities.items():
                w = weights.get(zone, 1.0)
                weighted_sum += vel * w
                total_weight += w
            
            overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            overall = 0.0
        
        # Determine trend from overall velocity
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
        
        # Calculate acceleration (change in velocity)
        accel = 0.0
        if len(samples) >= 3:
            # Use CPU_BIG as reference (fastest responding zone)
            old_vel = 0.0
            new_vel = zone_velocities.get(ThermalZone.CPU_BIG, 0)
            
            if len(samples) >= 3 and ThermalZone.CPU_BIG in samples[-2].zones and ThermalZone.CPU_BIG in samples[-3].zones:
                dt_old = samples[-2].timestamp - samples[-3].timestamp
                if dt_old > 0:
                    old_vel = (samples[-2].zones[ThermalZone.CPU_BIG] - 
                              samples[-3].zones[ThermalZone.CPU_BIG]) / dt_old
            
            dt = samples[-1].timestamp - samples[-2].timestamp
            if dt > 0:
                accel = (new_vel - old_vel) / dt
        
        return ThermalVelocity(
            zones=zone_velocities,
            overall=overall,
            trend=trend,
            acceleration=accel
        )
    
    def predict_temperature(self,
                          current: ThermalSample,
                          velocity: ThermalVelocity,
                          horizon: float,
                          samples: List[ThermalSample] = None) -> ThermalPrediction:
        """
        Predict future temperature using per-zone Newton's law:
        
        dT/dt = -k*(T - T_ambient)/R + P/C
        
        Solution:
        T(t) = T_ambient + (T0 - T_ambient)*e^(-kt/RC) + (PR/k)*(1 - e^(-kt/RC))
        
        Where:
          k = ambient coupling coefficient
          R = thermal resistance (°C/W)
          C = thermal mass (J/K)
          P = power dissipation (W)
          T0 = current temperature
          T_ambient = ambient temperature
        """
        
        predicted_temps = {}
        # Use dynamic ambient estimation instead of fixed value
        chassis = self.estimate_chassis(current)
        
        for zone, current_temp in current.zones.items():
            zone_name = str(zone).split('.')[-1]
            
            # Skip non-thermal zones (software metrics, not hardware)
            if zone_name in ['DISPLAY', 'CHARGER']:
                continue
            
            # Skip chassis - predict after components
            if zone == ThermalZone.CHASSIS:
                continue
            
            # Get zone-specific constants
            if zone_name not in self.zone_constants:
                # Fallback: simple linear extrapolation
                vel = velocity.zones.get(zone, 0)
                predicted_temps[zone] = current_temp + vel * horizon
                continue
            
            constants = self.zone_constants[zone_name]
            
            # ========================================================================
            # BATTERY SPECIAL CASE (v2.10): Integration of measured power
            # ========================================================================
            # Battery τ=540s >> 30s horizon, so Newton's law reduces to dT/dt ≈ P/C
            # Use measured current for ground truth power, skip ambient estimation
            if zone_name == 'BATTERY':
                C = constants['thermal_mass']  # 45 J/K
                
                # Measured battery current from sensor
                battery_current = getattr(current, 'battery_current', None)
                
                if battery_current is not None:
                    # CRITICAL: Sensor reports μA mislabeled as "mA"
                    # Validated: 1,207,704 "mA" → 1.208 A (bot running)
                    # Validated: 744,370 "mA" → 0.744 A (bot off)
                    I_amps = abs(battery_current) / 1_000_000.0  # μA → A
                    
                    # I²R losses
                    P_losses = I_amps ** 2 * S25_PLUS_BATTERY_INTERNAL_RESISTANCE
                    
                    # Charging inefficiency
                    if current.charging and battery_current > 0:
                        P_charge_heat = 0.15 * I_amps * 4.2  # 15% loss at 4.2V nominal
                        P_total = P_losses + P_charge_heat
                    else:
                        P_total = P_losses
                    
                    # Integration only (τ=540s >> 30s horizon)
                    delta_T = (P_total / C) * horizon
                    
                    # Prediction
                    predicted_temp = current_temp + delta_T
                else:
                    # No current data - assume stable (τ >> horizon)
                    predicted_temp = current_temp
                
                # Battery doesn't need measurement dynamics (measurement_tau=0)
                predicted_temps[zone] = max(10.0, min(predicted_temp, 60.0))
                continue  # Skip general Newton's law for battery
            
            # ========================================================================
            # GENERAL ZONES: Full Newton's law with ambient coupling
            # ========================================================================
            C = constants['thermal_mass']
            # Use temperature-dependent thermal resistance
            R = self.effective_thermal_resistance(zone_name, current_temp, chassis)
            k = constants['ambient_coupling']
            
            # ========================================================================
            # POWER INJECTION (V2.3 FIX) - Add all known power sources
            # ========================================================================
            
            # Start with velocity-based power estimate (observed heating rate)
            vel = velocity.zones.get(zone, 0)
            cooling_rate = -k * (current_temp - chassis) / R
            heating_rate = vel - cooling_rate
            P_observed = heating_rate * C
            
            # Initialize injected power for this zone
            P_injected = 0.0
            
            # 1. BASELINE SYSTEM POWER (screen-state aware)
            # Deep sleep when screen off, normal baseline when on
            screen_on = current.screen_on
            if screen_on:
                baseline_power = BASELINE_SYSTEM_POWER  # 0.6W normal operation
            else:
                baseline_power = 0.15  # Deep sleep - SoC + sensors + minimal background
            
            # Distribute baseline across CPU zones proportionally
            if zone_name in ['CPU_BIG', 'CPU_LITTLE']:
                baseline_fraction = 0.4 if zone_name == 'CPU_BIG' else 0.3
                P_injected += baseline_power * baseline_fraction
            elif zone_name == 'GPU':
                P_injected += baseline_power * 0.2
            elif zone_name == 'MODEM':
                P_injected += baseline_power * 0.1
            
            # 2. DISPLAY POWER (distributed according to thermal impact)
            if current.screen_on:
                # Try to get actual brightness (this would come from collect_sample)
                display_brightness = getattr(current, 'display_brightness', None)
                if display_brightness is not None:
                    # Scale based on actual brightness
                    display_power = DISPLAY_POWER_MIN + (DISPLAY_POWER_MAX - DISPLAY_POWER_MIN) * (display_brightness / 255.0)
                else:
                    # Assume 50% brightness if unknown
                    display_power = DISPLAY_POWER_MIN + (DISPLAY_POWER_MAX - DISPLAY_POWER_MIN) * 0.5
                
                # Distribute to zones
                zone_fraction = DISPLAY_THERMAL_DISTRIBUTION.get(zone_name, 0.0)
                P_injected += display_power * zone_fraction
            else:
                # Screen off - minimal AOD power
                zone_fraction = DISPLAY_THERMAL_DISTRIBUTION.get(zone_name, 0.0)
                P_injected += DISPLAY_POWER_OFF * zone_fraction
            
            # 3. NETWORK POWER (goes to modem zone)
            if zone_name == 'MODEM':
                network_type_name = current.network.name if hasattr(current.network, 'name') else 'UNKNOWN'
                P_injected += NETWORK_POWER.get(network_type_name, 0.2)
            
            # 4. BATTERY POWER (charging + discharge losses)
            if zone_name == 'BATTERY':
                battery_current = getattr(current, 'battery_current', None)
                battery_percent = None
                
                # Try to get battery percentage from cache
                if hasattr(self, 'telemetry') and self.telemetry.battery_cache:
                    battery_percent = self.telemetry.battery_cache.get('percentage', None)
                
                # Charging power with SoC-dependent efficiency
                if current.charging and battery_current is not None:
                    # Sensor reports μA labeled as "mA" - adjust thresholds
                    if battery_current > 1_000_000:  # > 1A (fast charging)
                        base_charging_power = CHARGING_POWER_FAST
                    elif battery_current > 500_000:  # > 0.5A (normal charging)
                        base_charging_power = CHARGING_POWER_NORMAL
                    else:
                        base_charging_power = CHARGING_POWER_NORMAL
                    
                    # Apply SoC-dependent efficiency curve
                    if battery_percent is not None:
                        if battery_percent < 15:
                            # Cold battery at low SoC, poor efficiency
                            charging_power = base_charging_power * 1.35
                        elif battery_percent > 85:
                            # CV phase at high SoC, more heat
                            charging_power = base_charging_power * 1.25
                        elif 50 <= battery_percent <= 80:
                            # Sweet spot - best efficiency
                            charging_power = base_charging_power * 1.05
                        else:
                            # Normal CC phase
                            charging_power = base_charging_power * 1.15
                    else:
                        # No SoC data, assume normal efficiency
                        charging_power = base_charging_power * 1.15
                    
                    P_injected += charging_power
                
                # Discharge power (I²R losses)
                elif battery_current is not None and battery_current < 0:
                    # Discharging - calculate resistive heating
                    current_amps = abs(battery_current) / 1_000_000.0  # μA → A
                    discharge_power = current_amps ** 2 * S25_PLUS_BATTERY_INTERNAL_RESISTANCE
                    P_injected += discharge_power
            
            # Combine observed and injected power
            # If observed power is much higher than injected (heavy workload), trust observed
            # If observed is low, use injected (we know these sources are active)
            if P_observed > P_injected * 1.5:
                # Heavy workload detected - trust velocity-based estimate
                P = P_observed
            elif P_observed < 0:
                # Cooling - use only injected power (positive sources)
                P = P_injected
            else:
                # Normal operation - use max of observed and injected
                # This ensures we don't miss either workload OR known sources
                P = max(P_observed, P_injected)
            
            # NEW: Trend-aware power adjustment
            # If accelerating, assume power will continue to rise (conservative)
            # If decelerating, assume power will drop (optimistic)
            accel = velocity.acceleration
            
            if abs(accel) < 0.05:
                # Steady state - use current power
                P_final = P
            elif accel > 0.1:
                # Accelerating - power is increasing (be conservative)
                P_final = P * 1.2
            elif accel < -0.1:
                # Decelerating - power is dropping (be optimistic)
                P_final = P * 0.8
            else:
                # Small acceleration - use current power
                P_final = P
            
            # Clamp power to realistic range
            P_final = max(constants['idle_power'], min(P_final, constants['peak_power']))
            
            # ========================================================================
            # TWO-STAGE PREDICTION: Component → Sensor (v2.9)
            # ========================================================================
            # Stage 1: Predict component temperature (fast response)
            tau = constants['time_constant']  # C * R
            exp_factor = math.exp(-k * horizon / (R * C))
            
            # Transient response: initial temperature difference decays
            temp_transient = (current_temp - chassis) * exp_factor
            
            # Steady-state response: power establishes new equilibrium
            temp_steady = (P_final * R / k) * (1 - exp_factor)
            
            component_temp = chassis + temp_transient + temp_steady
            
            # Stage 2: Measurement point thermal dynamics
            # Measured temp exponentially approaches component temp with time constant τ_meas
            # This accounts for: package thermal R, vapor chamber dynamics, sensor location
            tau_meas = constants.get('measurement_tau', 0)
            
            if tau_meas > 0 and horizon > 0:
                # First-order exponential approach: T_meas(t) = T_comp - (T_comp - T_meas_0)*exp(-t/τ)
                # For prediction: weight current measurement vs predicted component temp
                decay_factor = math.exp(-horizon / tau_meas)
                
                # Predicted measurement = old reading (decaying) + new component temp (growing)
                predicted_temp = current_temp * decay_factor + component_temp * (1 - decay_factor)
            else:
                # No measurement dynamics (τ >> horizon) - use component temp directly
                predicted_temp = component_temp
            
            # Apply ambient calibration offset (tune for systematic errors)
            predicted_temp += CHASSIS_CALIBRATION_OFFSET
            
            # Sanity check: don't predict impossible temperatures
            predicted_temp = max(chassis - 5, min(predicted_temp, 60.0))
            
            predicted_temps[zone] = predicted_temp
        
        # Predict chassis from component temperatures with damping (v2.19)
        # Chassis equilibrates based on vapor chamber coupling to all components
        # but has thermal inertia - doesn't instantly track component changes
        if ThermalZone.CHASSIS in current.zones:
            numerator = 0.0
            denominator = 0.0
            
            for zone, pred_temp in predicted_temps.items():
                zone_name = str(zone).split('.')[-1]
                if zone_name in self.zone_constants:
                    constants = self.zone_constants[zone_name]
                    k = constants['ambient_coupling']
                    R = constants['thermal_resistance']
                    conductance = k / R
                    
                    numerator += conductance * pred_temp
                    denominator += conductance
            
            if denominator > 0:
                # Calculate weighted average of predicted component temps
                weighted_avg = numerator / denominator
                
                # Damping: chassis moves partway toward weighted average
                # ΔT_chassis = α * (weighted_avg - chassis_current)
                weighted_delta = weighted_avg - chassis
                chassis_pred = chassis + CHASSIS_DAMPING_FACTOR * weighted_delta
                
                # Physical limits
                chassis_pred = max(10.0, min(chassis_pred, 80.0))
                predicted_temps[ThermalZone.CHASSIS] = chassis_pred
            else:
                # Fallback: use current estimate
                predicted_temps[ThermalZone.CHASSIS] = chassis
        
        # Calculate prediction confidence
        confidence = self._calculate_prediction_confidence(
            velocity, current, predicted_temps
        )
        
        # Thermal budget: time until hottest zone hits throttling
        thermal_budget = self._calculate_thermal_budget(
            current.zones, predicted_temps, velocity
        )
        
        # Recommended delay: wait time to avoid throttling
        recommended_delay = self._calculate_recommended_delay(
            predicted_temps, velocity
        )
        
        prediction = ThermalPrediction(
            timestamp=time.time() + horizon,
            horizon=horizon,
            predicted_temps=predicted_temps,
            confidence=confidence,
            thermal_budget=thermal_budget,
            recommended_delay=recommended_delay
        )
        
        return prediction
    def _calculate_prediction_confidence(self,
                                        velocity: ThermalVelocity,
                                        current: ThermalSample,
                                        predicted: Dict[ThermalZone, float]) -> float:
        """Calculate prediction confidence score"""
        factors = []
        
        # Factor 1: Velocity stability (low acceleration = high confidence)
        accel_confidence = 1.0 / (1.0 + abs(velocity.acceleration) * 10.0)
        factors.append(accel_confidence)
        
        # Factor 2: Temperature range (narrow range = high confidence)
        if current.zones:
            temp_range = max(current.zones.values()) - min(current.zones.values())
            range_confidence = 1.0 / (1.0 + temp_range / 10.0)
            factors.append(range_confidence)
        
        # Factor 3: Sensor confidence
        if current.confidence:
            sensor_confidence = statistics.mean(current.confidence.values())
            factors.append(sensor_confidence)
        
        # Factor 4: Prediction reasonableness
        if predicted and current.zones:
            max_delta = max(abs(predicted.get(z, 0) - t) 
                          for z, t in current.zones.items() if z in predicted)
            # Large deltas reduce confidence
            delta_confidence = 1.0 / (1.0 + max_delta / 5.0)
            factors.append(delta_confidence)
        
        return statistics.mean(factors) if factors else 0.5
    
    def _scale_temp_by_confidence(self,
                                  predicted_temp: float,
                                  confidence: float,
                                  ambient_temp: float) -> float:
        """
        Scale predicted temperature upward based on uncertainty.
        Low confidence → treat prediction as hotter → shorter thermal budget.
        
        Scales the temperature RISE above ambient, not absolute temp.
        """
        delta = predicted_temp - ambient_temp
        safety_factor = 1.0 + CONFIDENCE_SAFETY_SCALE * (1.0 - confidence)
        return ambient_temp + (delta * safety_factor)
    
    def _calculate_zone_confidence(self,
                                   zone_name: str,
                                   predicted_temp: float,
                                   current_temp: float,
                                   zone_velocity: float,
                                   sensor_confidence: float = 1.0) -> float:
        """
        Calculate prediction confidence for a specific zone.
        
        Factors:
        - Model quality (battery > CPU > GPU > modem)
        - Prediction magnitude (small changes = high confidence)
        - Velocity stability (steady = high confidence)
        - Sensor quality
        """
        factors = []
        
        # Model confidence by zone type
        model_conf = {
            'BATTERY': 0.95,      # Measured current = ground truth
            'CPU_BIG': 0.85,      # Fast response, well-understood
            'CPU_LITTLE': 0.85,
            'GPU': 0.75,          # Workload variability
            'MODEM': 0.70,        # Network unpredictable
            'CHASSIS': 0.80,      # Damped response
        }.get(zone_name, 0.60)
        factors.append(model_conf)
        
        # Delta confidence (small changes = high confidence)
        delta = abs(predicted_temp - current_temp)
        delta_conf = 1.0 / (1.0 + delta / 5.0)
        factors.append(delta_conf)
        
        # Velocity confidence (stable = high confidence)
        vel_conf = 1.0 / (1.0 + abs(zone_velocity) * 2.0)
        factors.append(vel_conf)
        
        # Sensor confidence
        factors.append(sensor_confidence)
        
        return statistics.mean(factors)
    
    def _calculate_thermal_budget(self,
                                 current: Dict[ThermalZone, float],
                                 predicted: Dict[ThermalZone, float],
                                 velocity: ThermalVelocity) -> float:
        """
        Calculate seconds until thermal throttling with confidence-scaled predictions.
        
        Low confidence → treat predictions as hotter → shorter (safer) thermal budget.
        """
        if not predicted:
            return 999.0
        
        # Get ambient/chassis temperature
        chassis = self.estimate_chassis(
            type('Sample', (), {'zones': current, 'charging': False, 'screen_on': True})()
        )
        
        # Calculate worst-case temperature across all zones
        worst_case_temp = 0.0
        
        for zone, pred_temp in predicted.items():
            # Get zone name and current state
            zone_name = str(zone).split('.')[-1]
            curr_temp = current.get(zone, pred_temp)
            zone_vel = velocity.zones.get(zone, 0)
            
            # Get sensor confidence if available
            sensor_conf = 1.0
            # TODO: Pull from current.confidence dict if you want to use it
            
            # Calculate zone-specific confidence
            confidence = self._calculate_zone_confidence(
                zone_name=zone_name,
                predicted_temp=pred_temp,
                current_temp=curr_temp,
                zone_velocity=zone_vel,
                sensor_confidence=sensor_conf
            )
            
            # Scale prediction by confidence
            conservative_temp = self._scale_temp_by_confidence(
                predicted_temp=pred_temp,
                confidence=confidence,
                ambient_temp=chassis
            )
            
            # Track worst case
            worst_case_temp = max(worst_case_temp, conservative_temp)
        
        # Already at or above throttle threshold
        if worst_case_temp >= THERMAL_TEMP_HOT:
            return 0.0
        
        # Cooling - unlimited budget
        if velocity.overall <= 0:
            return 999.0
        
        # Time until hit HOT threshold
        budget = (THERMAL_TEMP_HOT - worst_case_temp) / velocity.overall
        
        # Clamp to reasonable range
        return max(0, min(budget, 600.0))
    
    def _calculate_recommended_delay(self,
                                    predicted: Dict[ThermalZone, float],
                                    velocity: ThermalVelocity) -> float:
        """Calculate recommended delay before heavy operations"""
        if not predicted:
            return 0.0
        
        max_predicted = max(predicted.values())
        
        # Below warning threshold - no delay needed
        if max_predicted < THERMAL_TEMP_WARM:
            return 0.0
        
        # Approaching throttle - recommend delay
        if max_predicted >= THERMAL_TEMP_HOT - 2.0:
            # Delay until temp drops below warm
            if velocity.overall > 0:
                delay = (max_predicted - THERMAL_TEMP_WARM) / (abs(velocity.overall) + 0.01)
                return min(delay, 10.0)  # cap at 10s
        
        return 0.0
    
    def estimate_power(self, sample: ThermalSample, 
                      previous: Optional[ThermalSample]) -> float:
        """Estimate current power dissipation"""
        if not previous:
            return 0.0
        
        dt = sample.timestamp - previous.timestamp
        if dt <= 0:
            return 0.0
        
        # Calculate energy change per zone
        total_power = 0.0
        
        for zone in sample.zones:
            if zone not in previous.zones:
                continue
            
            zone_name = str(zone).split('.')[-1]
            if zone_name not in self.zone_constants:
                continue
            
            constants = self.zone_constants[zone_name]
            C = constants['thermal_mass']
            R = constants['thermal_resistance']
            k = constants['ambient_coupling']
            
            # Temperature change
            dT = sample.zones[zone] - previous.zones[zone]
            
            # Power = C * dT/dt + cooling_power
            heating_power = C * (dT / dt)
            
            chassis = sample.chassis or 25.0
            cooling_power = k * (sample.zones[zone] - chassis) / R
            
            zone_power = heating_power + cooling_power
            zone_power = max(0, min(zone_power, constants['peak_power']))
            
            total_power += zone_power
        
        return total_power

# ============================================================================
# ADAPTIVE LEARNING SYSTEM
# ============================================================================


# ============================================================================
# THERMAL TANK - Battery-Centric Throttle Control
# ============================================================================

@dataclass
class ThermalTankStatus:
    """Simple output from thermal tank"""
    battery_temp_current: float      # Current battery temp (°C)
    battery_temp_predicted: float    # Predicted battery temp at horizon (°C)
    should_throttle: bool            # True = reject new work
    headroom_seconds: float          # Seconds until throttle (0 if already hot)
    cooling_rate: float              # °C/s (negative = heating)

class ThermalTank:
    """
    Battery-centric thermal management using tank metaphor.
    
    Samsung throttles at 40°C battery. We stay below 38.5°C (1.5°C safety margin).
    Tank models battery as thermal capacity that fills (work) and drains (cooling).
    """
    
    def __init__(self, physics_engine: 'ZonePhysicsEngine'):
        self.physics = physics_engine
        
        # Thresholds
        self.throttle_temp = TANK_THROTTLE_TEMP  # 38.5°C
        self.warning_temp = TANK_WARNING_TEMP     # 37.5°C
        self.samsung_limit = SAMSUNG_THROTTLE_TEMP  # 40.0°C (actual hardware limit)
        
        logger.info(f"Thermal tank initialized: throttle at {self.throttle_temp}°C "
                   f"(Samsung limit: {self.samsung_limit}°C)")
    
    def get_status(self, 
                   current: ThermalSample,
                   velocity: ThermalVelocity,
                   prediction: ThermalPrediction) -> ThermalTankStatus:
        """
        Get current tank status with simple throttle decision.
        
        Uses existing physics predictions but exposes battery-focused interface.
        """
        # Extract battery data
        battery_zone = None
        for zone in current.zones:
            zone_name = str(zone).split('.')[-1]
            if 'BATTERY' in zone_name.upper():
                battery_zone = zone
                break
        
        if battery_zone is None:
            # No battery sensor - can't make decision
            return ThermalTankStatus(
                battery_temp_current=0.0,
                battery_temp_predicted=0.0,
                should_throttle=True,  # Conservative: throttle if no data
                headroom_seconds=0.0,
                cooling_rate=0.0
            )
        
        # Current state
        battery_current = current.zones[battery_zone]
        battery_velocity = velocity.zones.get(battery_zone, 0.0)
        
        # Predicted state (from existing physics)
        battery_predicted = prediction.predicted_temps.get(battery_zone, battery_current)
        
        # Throttle decision
        should_throttle = battery_predicted >= self.throttle_temp
        
        # Calculate headroom (seconds until throttle)
        if battery_velocity > 0.001:  # Heating
            headroom = (self.throttle_temp - battery_current) / battery_velocity
            headroom = max(0.0, headroom)
        elif battery_current >= self.throttle_temp:
            headroom = 0.0
        else:
            headroom = float('inf')  # Cooling or stable below limit
        
        return ThermalTankStatus(
            battery_temp_current=battery_current,
            battery_temp_predicted=battery_predicted,
            should_throttle=should_throttle,
            headroom_seconds=headroom,
            cooling_rate=battery_velocity
        )
    
    def can_accept_work(self, status: ThermalTankStatus, estimated_heating: float = 0.0) -> bool:
        """
        Check if we can accept new work without overflowing tank.
        
        Args:
            status: Current tank status
            estimated_heating: Expected temp increase from work (°C)
        
        Returns:
            True if work can be accepted safely
        """
        predicted_after_work = status.battery_temp_predicted + estimated_heating
        return predicted_after_work < self.throttle_temp


# ============================================================================
# CACHE-AWARE PATTERN RECOGNIZER
# ============================================================================

# ============================================================================
# TRANSIENT MOMENTUM TUNER WITH FULL PERSISTENT HISTORY
# ============================================================================

class TransientResponseTuner:
    """
    Self-tuning momentum factors for transient thermal predictions.
    
    IMPROVEMENTS v2.21:
    - 10K error history (up from 1K) with full persistence
    - Confidence-aware learning: adjustment scales with √(sample_count)
    - Adaptive learning rates based on data availability
    - Memory-mapped persistence via PNGN system
    
    Learning Algorithm:
    - Analyzes last N errors (N scales with total history size)
    - Computes mean error: positive = overprediction, negative = underprediction
    - Adjustment = -learning_rate × (mean_error / 2°C) × confidence_factor
    - Confidence factor = min(1.0, √(n_samples / 100))
    - Learning rate adaptive: 0.01 (sparse) → 0.03 (abundant)
    
    Momentum factors:
    - heating: velocity > +0.1°C/s
    - cooling: velocity < -0.1°C/s
    - stable: |velocity| <= 0.1°C/s
    """
    
    def __init__(self):
        # Import persistence system
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from persistence_system import save_data, load_data
            self.persistence_save = save_data
            self.persistence_load = load_data
            self.persistence_available = True
        except Exception as e:
            logger.warning(f"Persistence system unavailable: {e}")
            self.persistence_available = False
            self.persistence_save = lambda k, v: False
            self.persistence_load = lambda k: None
        
        # Persistence key
        self.persistence_key = "thermal.transient_tuner"
        
        # Momentum factors - start at 1.0 (perfect linear continuation)
        self.momentum_heating = 1.0
        self.momentum_cooling = 1.0
        self.momentum_stable = 1.0
        
        # Error history - 10K entries each, fully persisted
        self.errors_heating: Deque[float] = deque(maxlen=10000)
        self.errors_cooling: Deque[float] = deque(maxlen=10000)
        self.errors_stable: Deque[float] = deque(maxlen=10000)
        
        # Stats
        self.total_tuning_cycles = 0
        self.last_tune_time = 0
        
        # Load persisted state
        self._load()
        
        logger.info(f"TransientResponseTuner initialized with 10K history")
        logger.info(f"Momentum factors: heat={self.momentum_heating:.3f}, "
                   f"cool={self.momentum_cooling:.3f}, stable={self.momentum_stable:.3f}")
        logger.info(f"Error history: heat={len(self.errors_heating)}, "
                   f"cool={len(self.errors_cooling)}, stable={len(self.errors_stable)}")
    
    def tune_heating(self, errors: List[float], confidences: List[float]):
        """Tune heating momentum based on confidence-weighted errors"""
        if not errors or len(errors) != len(confidences):
            return
        
        # Add to history
        self.errors_heating.extend(errors)
        
        # Calculate analysis window size (adaptive: 2% of history, min 20, max 500)
        history_size = len(self.errors_heating)
        window_size = max(20, min(500, int(0.02 * history_size)))
        
        # Tune using recent errors with their confidences
        recent_errors = list(self.errors_heating)[-window_size:]
        recent_confidences = confidences if len(confidences) == len(errors) else [1.0] * len(errors)
        
        self.momentum_heating = self._tune_momentum(
            recent_errors,
            recent_confidences[-window_size:] if len(recent_confidences) >= window_size else recent_confidences,
            self.momentum_heating,
            "HEATING",
            history_size
        )
        
        self.total_tuning_cycles += 1
        self.last_tune_time = time.time()
        self._save()
    
    def tune_cooling(self, errors: List[float], confidences: List[float]):
        """Tune cooling momentum based on confidence-weighted errors"""
        if not errors or len(errors) != len(confidences):
            return
        
        self.errors_cooling.extend(errors)
        
        history_size = len(self.errors_cooling)
        window_size = max(20, min(500, int(0.02 * history_size)))
        
        recent_errors = list(self.errors_cooling)[-window_size:]
        recent_confidences = confidences if len(confidences) == len(errors) else [1.0] * len(errors)
        
        self.momentum_cooling = self._tune_momentum(
            recent_errors,
            recent_confidences[-window_size:] if len(recent_confidences) >= window_size else recent_confidences,
            self.momentum_cooling,
            "COOLING",
            history_size
        )
        
        self.total_tuning_cycles += 1
        self.last_tune_time = time.time()
        self._save()
    
    def tune_stable(self, errors: List[float], confidences: List[float]):
        """Tune stable momentum based on confidence-weighted errors"""
        if not errors or len(errors) != len(confidences):
            return
        
        self.errors_stable.extend(errors)
        
        history_size = len(self.errors_stable)
        window_size = max(20, min(500, int(0.02 * history_size)))
        
        recent_errors = list(self.errors_stable)[-window_size:]
        recent_confidences = confidences if len(confidences) == len(errors) else [1.0] * len(errors)
        
        self.momentum_stable = self._tune_momentum(
            recent_errors,
            recent_confidences[-window_size:] if len(recent_confidences) >= window_size else recent_confidences,
            self.momentum_stable,
            "STABLE",
            history_size
        )
        
        self.total_tuning_cycles += 1
        self.last_tune_time = time.time()
        self._save()
    
    def _tune_momentum(self, recent_errors: List[float], recent_confidences: List[float],
                       current_momentum: float, state_name: str, history_size: int) -> float:
        """
        Tune momentum factor with dual confidence weighting.
        
        PREDICTION CONFIDENCE (per-prediction quality):
        - High confidence predictions that fail → learn more from them
        - Low confidence predictions that fail → learn less from them
        - Weighted mean: sum(error × conf) / sum(conf)
        
        SAMPLE CONFIDENCE (data availability):
        - √(n_samples / 100) scaling factor
        - More data → more aggressive adjustments
        
        Args:
            recent_errors: Recent prediction errors (predicted - actual)
            recent_confidences: Confidence for each prediction
            current_momentum: Current momentum factor
            state_name: State being tuned (for logging)
            history_size: Total error history size (for sample confidence)
        
        Returns:
            Updated momentum factor
        """
        if len(recent_errors) < 10:
            return current_momentum  # Need minimum data
        
        # Ensure confidences match errors
        if len(recent_confidences) != len(recent_errors):
            recent_confidences = [1.0] * len(recent_errors)
        
        # Calculate confidence-weighted mean error
        # Weights errors by their prediction quality
        weighted_sum = sum(e * c for e, c in zip(recent_errors, recent_confidences))
        confidence_sum = sum(recent_confidences)
        
        if confidence_sum < 0.1:  # Avoid division by zero
            return current_momentum
        
        mean_error = weighted_sum / confidence_sum
        
        # Adaptive learning rate based on data availability
        if history_size < 100:
            learning_rate = 0.01  # Conservative - sparse data
        elif history_size < 1000:
            learning_rate = 0.02  # Standard - moderate data
        else:
            learning_rate = 0.03  # Aggressive - abundant data
        
        # Sample confidence scales with √(sample_count)
        # Reaches 1.0 at 100 samples, caps there
        sample_confidence = min(1.0, math.sqrt(len(recent_errors) / 100.0))
        
        # Calculate adjustment with both confidence factors
        # Positive error (overprediction) → reduce momentum
        # Negative error (underprediction) → increase momentum
        adjustment = -learning_rate * (mean_error / 2.0) * sample_confidence
        
        new_momentum = current_momentum + adjustment
        
        # Bounds: [0.5, 1.5]
        new_momentum = max(0.5, min(1.5, new_momentum))
        
        logger.debug(f"{state_name}: n={len(recent_errors)}, history={history_size}, "
                    f"weighted_error={mean_error:.3f}°C, lr={learning_rate:.3f}, "
                    f"sample_conf={sample_confidence:.3f}, momentum {current_momentum:.3f} → {new_momentum:.3f}")
        
        return new_momentum
    
    def _save(self):
        """Persist learned state to disk"""
        if not self.persistence_available:
            return
        
        data = {
            'version': '2.21',
            'momentum_heating': self.momentum_heating,
            'momentum_cooling': self.momentum_cooling,
            'momentum_stable': self.momentum_stable,
            'errors_heating': list(self.errors_heating),
            'errors_cooling': list(self.errors_cooling),
            'errors_stable': list(self.errors_stable),
            'total_tuning_cycles': self.total_tuning_cycles,
            'last_tune_time': self.last_tune_time,
            'timestamp': time.time()
        }
        
        try:
            self.persistence_save(self.persistence_key, data)
            logger.debug(f"Saved transient tuner state: {len(self.errors_heating)}/"
                        f"{len(self.errors_cooling)}/{len(self.errors_stable)} errors")
        except Exception as e:
            logger.warning(f"Could not save transient tuner state: {e}")
    
    def _load(self):
        """Load persisted state from disk"""
        if not self.persistence_available:
            return
        
        try:
            data = self.persistence_load(self.persistence_key)
            
            if data and isinstance(data, dict):
                self.momentum_heating = data.get('momentum_heating', 1.0)
                self.momentum_cooling = data.get('momentum_cooling', 1.0)
                self.momentum_stable = data.get('momentum_stable', 1.0)
                
                # Load full error history
                heating_errors = data.get('errors_heating', [])
                cooling_errors = data.get('errors_cooling', [])
                stable_errors = data.get('errors_stable', [])
                
                self.errors_heating.extend(heating_errors)
                self.errors_cooling.extend(cooling_errors)
                self.errors_stable.extend(stable_errors)
                
                self.total_tuning_cycles = data.get('total_tuning_cycles', 0)
                self.last_tune_time = data.get('last_tune_time', 0)
                
                logger.info(f"Loaded transient tuner: {len(heating_errors)}/{len(cooling_errors)}/"
                           f"{len(stable_errors)} errors, {self.total_tuning_cycles} cycles")
        except Exception as e:
            logger.warning(f"Could not load transient tuner state: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tuner statistics"""
        return {
            'momentum': {
                'heating': self.momentum_heating,
                'cooling': self.momentum_cooling,
                'stable': self.momentum_stable
            },
            'error_history': {
                'heating': len(self.errors_heating),
                'cooling': len(self.errors_cooling),
                'stable': len(self.errors_stable),
                'total': len(self.errors_heating) + len(self.errors_cooling) + len(self.errors_stable)
            },
            'tuning': {
                'total_cycles': self.total_tuning_cycles,
                'last_tune': self.last_tune_time,
                'time_since_tune': time.time() - self.last_tune_time if self.last_tune_time > 0 else None
            }
        }
# ============================================================================
# THERMAL INTELLIGENCE SYSTEM
# ============================================================================

class ThermalIntelligenceSystem:
    """
    Main thermal intelligence coordinator with hardware-accurate physics.
    Collects data, learns patterns, provides predictions.
    """
    
    def __init__(self):
        # Core components only
        self.telemetry = ThermalTelemetryCollector()
        self.physics = ZonePhysicsEngine()
        self.tank = ThermalTank(self.physics)  # Battery-centric throttle control
        
        # Data storage
        self.samples: Deque[ThermalSample] = deque(maxlen=THERMAL_HISTORY_SIZE)
        self.predictions: Deque[ThermalPrediction] = deque(maxlen=100)
        
        # State
        self.current_state = ThermalState.UNKNOWN
        self.last_update = 0
        self.update_interval = THERMAL_SAMPLE_INTERVAL_MS / 1000.0  # 10 seconds
        
        # Monitoring
        self.monitor_task = None
        self.running = False
        
        logger.info("Thermal Intelligence System v2.24 - Die backdate removed")
    
    async def start(self):
        """Start thermal monitoring and prediction"""
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Thermal monitoring started")
    
    async def stop(self):
        """Stop thermal monitoring"""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Thermal monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop - 10s uniform sampling"""
        while self.running:
            try:
                # Collect sample (uniform 10s rate)
                sample = await self.telemetry.collect_sample(for_prediction=True)
                self.samples.append(sample)
                
                # Validate against old predictions if we have any
                if self.predictions and len(self.samples) >= MIN_SAMPLES_FOR_PREDICTIONS:
                    # Check if current sample matches a previous prediction's target time
                    # Prediction horizon is 60s, so look for predictions ~60s ago
                    for prediction in list(self.predictions):
                        time_diff = abs(sample.timestamp - (prediction.timestamp + THERMAL_PREDICTION_HORIZON))
                        
                        # If within 15s tolerance, this sample validates that prediction
                        if time_diff < 15.0:
                            # Calculate velocity at prediction time for regime classification
                            velocity = self.physics.calculate_velocity(list(self.samples))
                            self.physics.validate_and_tune(prediction, sample, velocity)
                            break  # Only validate one prediction per sample
                
                # Need minimum samples for predictions
                if len(self.samples) >= MIN_SAMPLES_FOR_PREDICTIONS:
                    # Calculate velocity
                    velocity = self.physics.calculate_velocity(list(self.samples))
                    
                    # Generate prediction
                    if THERMAL_PREDICTION_ENABLED:
                        prediction = self.physics.predict_temperature(
                            sample, velocity, THERMAL_PREDICTION_HORIZON, list(self.samples)
                        )
                        self.predictions.append(prediction)
                
                # Update thermal state
                self._update_thermal_state(sample)
                
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(1)
    
    def _update_thermal_state(self, sample: ThermalSample):
        """Update thermal state with hysteresis"""
        if not sample.zones:
            return
        
        max_temp = max(sample.zones.values())
        
        # State machine with hysteresis
        if self.current_state == ThermalState.CRITICAL:
            if max_temp < THERMAL_TEMP_CRITICAL - THERMAL_HYSTERESIS_DOWN:
                self.current_state = ThermalState.HOT
        elif self.current_state == ThermalState.HOT:
            if max_temp >= THERMAL_TEMP_CRITICAL:
                self.current_state = ThermalState.CRITICAL
            elif max_temp < THERMAL_TEMP_HOT - THERMAL_HYSTERESIS_DOWN:
                self.current_state = ThermalState.WARM
        elif self.current_state == ThermalState.WARM:
            if max_temp >= THERMAL_TEMP_HOT + THERMAL_HYSTERESIS_UP:
                self.current_state = ThermalState.HOT
            elif max_temp < THERMAL_TEMP_WARM - THERMAL_HYSTERESIS_DOWN:
                self.current_state = ThermalState.OPTIMAL
        elif self.current_state == ThermalState.OPTIMAL:
            if max_temp >= THERMAL_TEMP_WARM + THERMAL_HYSTERESIS_UP:
                self.current_state = ThermalState.WARM
            elif max_temp < THERMAL_TEMP_OPTIMAL_MIN:
                self.current_state = ThermalState.COLD
        else:
            # Initial state determination
            if max_temp >= THERMAL_TEMP_CRITICAL:
                self.current_state = ThermalState.CRITICAL
            elif max_temp >= THERMAL_TEMP_HOT:
                self.current_state = ThermalState.HOT
            elif max_temp >= THERMAL_TEMP_WARM:
                self.current_state = ThermalState.WARM
            elif max_temp >= THERMAL_TEMP_OPTIMAL_MIN:
                self.current_state = ThermalState.OPTIMAL
            else:
                self.current_state = ThermalState.COLD
    
    
    def get_current(self) -> Optional[ThermalSample]:
        """Get most recent thermal sample"""
        return self.samples[-1] if self.samples else None
    
    def get_prediction(self) -> Optional[ThermalPrediction]:
        """Get most recent prediction"""
        return self.predictions[-1] if self.predictions else None
    
    def get_tank_status(self) -> Optional[ThermalTankStatus]:
        """
        Get current thermal tank status - battery temp + throttle decision.
        
        This is the clean API: just battery temp and a bool.
        """
        if not self.samples or len(self.samples) < MIN_SAMPLES_FOR_PREDICTIONS:
            return None
        
        if not self.predictions:
            return None
        
        current = self.samples[-1]
        velocity = self.physics.calculate_velocity(list(self.samples))
        prediction = self.predictions[-1]
        
        return self.tank.get_status(current, velocity, prediction)
    
    def get_display_status(self) -> Optional[Dict[str, Any]]:
        """
        Get rich thermal display data for visualization.
        
        Returns zone temperatures with color coding, trends, state, and metrics.
        Perfect for ASCII art thermal boxes in launchers.
        
        Color thresholds:
        - Green: < 35°C (cool)
        - Yellow: 35-40°C (warm)
        - Orange: 40-45°C (hot)
        - Red: > 45°C (critical)
        
        Trend indicators:
        - ↑↑ : rapid warming (> 0.15°C/s)
        - ↑  : warming (0.05-0.15°C/s)
        - →  : stable (-0.05 to 0.05°C/s)
        - ↓  : cooling (-0.15 to -0.05°C/s)
        - ↓↓ : rapid cooling (< -0.15°C/s)
        """
        if not self.samples or len(self.samples) < MIN_SAMPLES_FOR_PREDICTIONS:
            return None
        
        current = self.samples[-1]
        velocity = self.physics.calculate_velocity(list(self.samples))
        prediction = self.predictions[-1] if self.predictions else None
        
        # Zone data with colors and trends
        zones = {}
        peak_temp = 0.0
        
        for zone, temp in current.zones.items():
            zone_name = str(zone).split('.')[-1]
            
            # Skip non-hardware zones
            if zone_name in ['DISPLAY', 'CHARGER']:
                continue
            
            # Color based on temperature
            if temp < 35:
                color = 'green'
            elif temp < 40:
                color = 'yellow'
            elif temp < 45:
                color = 'orange'
            else:
                color = 'red'
            
            # Trend from velocity
            vel = velocity.zones.get(zone, 0.0)
            if vel > 0.15:
                trend = '↑↑'
            elif vel > 0.05:
                trend = '↑'
            elif vel < -0.15:
                trend = '↓↓'
            elif vel < -0.05:
                trend = '↓'
            else:
                trend = '→'
            
            # Confidence if available
            conf = prediction.confidence_by_zone.get(zone, 0.7) if prediction else 0.7
            
            zones[zone_name] = {
                'temp': temp,
                'color': color,
                'trend': trend,
                'velocity': vel,
                'confidence': conf
            }
            
            peak_temp = max(peak_temp, temp)
        
        # Overall metrics
        thermal_budget = prediction.thermal_budget if prediction else 999.0
        thermal_load = 0.0  # Could calculate actual load percentage
        budget_used_pct = 0.0 if thermal_budget > 900 else (1.0 - thermal_budget / 999.0) * 100
        
        return {
            'zones': zones,
            'peak_temp': peak_temp,
            'state': self.current_state.name,
            'trend': velocity.trend.name,
            'thermal_load': thermal_load,
            'thermal_budget': thermal_budget,
            'budget_used_pct': budget_used_pct,
            'overall_velocity': velocity.overall,
            'timestamp': current.timestamp
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic system statistics"""
        return {
            'samples_collected': len(self.samples),
            'predictions_made': len(self.predictions),
            'current_state': self.current_state.name,
            'update_interval': self.update_interval,
        }

# ============================================================================
# FACTORY
# ============================================================================

def create_thermal_intelligence() -> ThermalIntelligenceSystem:
    """Create thermal intelligence system"""
    return ThermalIntelligenceSystem()

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

logger.info("S25+ Thermal Intelligence System v2.24 loaded")
logger.info(f"Hardware constants: {len(ZONE_THERMAL_CONSTANTS)} zones configured")
logger.info("Features: Battery-only backdating | Dual-confidence | 10K history | Damped chassis")
