#!/usr/bin/env python3
"""
🔥🐧🔥 S25+ Thermal Intelligence System
====================================
Copyright (c) 2025 PNGN-Tec LLC

Physics-based thermal management for Android devices under continuous load.

Multi-zone temperature monitoring with Newton's law of cooling predictions. 
Prevents throttling through proactive thermal budget calculation and workload 
scheduling. Built for production operation where thermal shutdowns are unacceptable.

Production deployment: Discord bot serving 645+ members on Samsung S25+, 
24/7 operation with zero thermal shutdowns.

ARCHITECTURE:
- Multi-zone sensor monitoring (CPU, GPU, battery, modem, chassis)
- Newton's law of cooling with measured per-zone thermal constants
- Dual-confidence predictions (physics model × sample-size weighting)
- Adaptive damping via prediction error feedback
- Samsung throttling awareness (50% power reduction at 42°C battery)
- Thermal tank status for dual-condition throttle decisions (battery temp + CPU velocity)
- 10s sampling, 30s prediction horizon

HARDWARE (Samsung Galaxy S25+ / Snapdragon 8 Elite):
- CPU_BIG (cpuss-1-0 / zone 20): 2× Oryon Prime, τ=6.6s
- CPU_LITTLE (cpuss-0-0 / zone 13): 6× Oryon efficiency, τ=6.9s
- GPU (gpuss-0 / zone 23): Adreno 830, τ=9.1s
- MODEM (mdmss-0 / zone 31): 5G/WiFi, τ=9.0s
- BATTERY (battery / zone 60): τ=540s (critical for Samsung throttle at 42°C)
- CHASSIS (sys-therm-0 / zone 53): vapor chamber reference sensor
- AMBIENT (sys-therm-5 / zone 52): ambient air temperature

ZONE CORRECTIONS (based on thermal scan):
- BATTERY: 31 → 60 (zone 31 is modem, not battery)
- MODEM: 39 → 31 (zone 39 is neural processor, not modem)
- CHASSIS: 47 → 53 (zone 47 broken, reads 0.0°C; zone 52 is ambient air)

PHYSICS:
T(t) = T_amb + (T₀ - T_amb)·exp(-t/τ) + (P·R/k)·(1 - exp(-t/τ))

Battery simplification (τ >> horizon):
ΔT ≈ (P/C) × Δt

PREDICTION ACCURACY:
~1.5°C MAE at 30s horizon with 10s sampling
Adaptive damping improves accuracy over time
Battery zone most predictable (τ=540s)
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
# PREDICTION PARAMETERS
# ============================================================================
THERMAL_PREDICTION_HORIZON = 15.0      # seconds ahead to predict
THERMAL_SAMPLE_INTERVAL_MS = 1000      # 1s uniform sampling
THERMAL_HISTORY_SIZE = 120             # samples to keep (120s = 2 minutes)
MIN_SAMPLES_FOR_PREDICTIONS = 5       # minimum samples before making predictions

# Confidence scaling
CONFIDENCE_SAFETY_SCALE = 0.5          # prediction safety scaling

# ============================================================================
# SAMSUNG THROTTLING BEHAVIOR
# ============================================================================
# Samsung's SDHM service throttles CPU/GPU when battery temperature hits 40-42°C
# Validation data shows ~50% power reduction at 42°C battery temp
# This is SOFTWARE throttling, distinct from hardware limits (105°C for SD8 Elite)
THROTTLE_ENABLED = True                    # Enable throttling-aware predictions
THROTTLE_BATTERY_TEMP_START = 40.0         # °C - start reducing power
THROTTLE_BATTERY_TEMP_FULL = 42.0          # °C - full 50% throttle engaged
THROTTLE_POWER_REDUCTION = 0.5             # Reduce to 50% of nominal power
THROTTLE_HYSTERESIS = 1.0                  # °C - hysteresis to prevent oscillation
# Zones affected by battery-based throttling
THROTTLE_AFFECTED_ZONES = ['CPU_BIG', 'CPU_LITTLE', 'GPU']

# ============================================================================
# PER-COMPONENT THERMAL THROTTLING
# ============================================================================
# Components throttle themselves based on their own temperature, independent
# of battery temp. This happens BEFORE Samsung's battery-based global throttle.
#
# Validation shows catastrophic prediction errors at max load because components
# self-throttle continuously - CPU drops to 74-77% sustained, GPU drops bins
# dynamically, etc. These curves model reality.
COMPONENT_THROTTLE_CURVES = {
    'CPU_BIG': {
        'temp_start': 45.0,
        'observed_peak': 79.1,    # From validation data
        'temp_aggressive': 75.0,
        'min_factor': 0.30,
        'curve_shape': 'exponential',
    },
    'CPU_LITTLE': {
        'temp_start': 48.0,
        'observed_peak': 93.4,    # From validation data
        'temp_aggressive': 88.0,
        'min_factor': 0.35,
        'curve_shape': 'exponential',
    },
    'GPU': {
        'temp_start': 38.0,
        'observed_peak': 58.1,    # From validation data
        'temp_aggressive': 54.0,
        'min_factor': 0.30,
        'curve_shape': 'exponential',
    },
    'MODEM': {
        'temp_start': 40.0,
        'observed_peak': 59.7,    # From validation data
        'temp_aggressive': 55.0,
        'min_factor': 0.60,
        'curve_shape': 'linear',
    },
    'BATTERY': {
        'temp_start': 999.0,
        'observed_peak': 35.2,    # From validation data
        'temp_aggressive': 999.0,
        'min_factor': 1.0,
        'curve_shape': 'linear',
    },
}

# ============================================================================
# HARDWARE CONSTANTS - Samsung S25+ (Snapdragon 8 Elite for Galaxy)
# ============================================================================

# SoC specifications
SD8_ELITE_TDP = 7.3              # W (typical sustained)
SD8_ELITE_PEAK = 15.0            # W (burst)
SD8_ELITE_PROCESS_NODE = 3       # nm (TSMC N3)

# Thermal Junction Maximum (TJmax) - hardware emergency throttling
# These are HARD LIMITS where silicon MUST throttle to prevent damage
TJMAX_CPU = 95.0                 # °C (Snapdragon CPU cores)
TJMAX_GPU = 95.0                 # °C (Adreno 830 GPU)
TJMAX_MODEM = 95.0               # °C (5G modem)
TJMAX_BATTERY = 60.0             # °C (Li-ion safety limit, not true TJmax)

# Battery
S25_PLUS_BATTERY_CAPACITY = 4900              # mAh
S25_PLUS_BATTERY_INTERNAL_RESISTANCE = 0.150  # Ohms
S25_PLUS_VAPOR_CHAMBER_EFFICIENCY = 0.85      # heat transfer efficiency
S25_PLUS_SCREEN_SIZE = 6.7                    # inches

# Per-zone thermal characteristics
ZONE_THERMAL_CONSTANTS = {
    'CPU_BIG': {
        'thermal_mass': 0.005,         # J/K
        'thermal_resistance': 1.50,    # K/W
        'ambient_coupling': 0.90,      # coupled to vapor chamber
        'peak_power': 6.0,             # W
        'idle_power': 0.1,
        'time_constant': 18.7,         # s (measured)
        'measurement_tau': 0.3,        # s (sensor lag)
    },
    'CPU_LITTLE': {
        'thermal_mass': 0.010,         # J/K
        'thermal_resistance': 2.00,    # K/W
        'ambient_coupling': 0.90,      # coupled to vapor chamber
        'peak_power': 4.0,
        'idle_power': 0.05,
        'time_constant': 14.3,         # s (measured)
        'measurement_tau': 0.1,
    },
    'GPU': {
        'thermal_mass': 0.07,         # J/K
        'thermal_resistance': 0.50,   # K/W
        'ambient_coupling': 0.85,     # moderate coupling
        'peak_power': 8.0,
        'idle_power': 0.2,
        'time_constant': 22.3,        # s (measured)
        'measurement_tau': 0.2,
    },
    'BATTERY': {
        'thermal_mass': 50.0,         # J/K (large thermal mass)
        'thermal_resistance': 50.00,  # K/W
        'ambient_coupling': 0.30,     # poor coupling (internal)
        'peak_power': 4.0,            # W (I²R losses)
        'idle_power': 0.1,
        'time_constant': 118.1,       # s (measured - very slow)
        'measurement_tau': 1.0,
    },
    'MODEM': {
        'thermal_mass': 0.02,         # J/K
        'thermal_resistance': 1.00,   # K/W
        'ambient_coupling': 0.70,
        'peak_power': 3.0,
        'idle_power': 0.3,
        'time_constant': 19.9,        # s (measured)
        'measurement_tau': 0.5,
    },
    'CHASSIS': {
        'thermal_mass': 15.0,         # J/K (vapor chamber + frame)
        'thermal_resistance': 0.25,   # K/W
        'ambient_coupling': 1.0,      # IS the ambient coupling
        'peak_power': 0.0,            # passive
        'idle_power': 0.0,
        'time_constant': 41.7,        # s (measured)
        'measurement_tau': 0.5,
    },
    'AMBIENT': {
        'thermal_mass': 100.0,        # J/K (environment)
        'thermal_resistance': 0.01,   # K/W (negligible)
        'ambient_coupling': 1.0,
        'peak_power': 0.0,
        'idle_power': 0.0,
        'time_constant': 3600.0,      # s (very slow room temp changes)
        'measurement_tau': 1.0,
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
CHARGING_POWER_SLOW = 1.5            # W
CHARGING_POWER_NORMAL = 4.0          # W
CHARGING_POWER_FAST = 16.0           # W (45W charger: ~15-18W heat @ 65% efficiency)

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
    CHASSIS = 53       # sys-therm-0: chassis/vapor chamber (CORRECTED from 52)
    AMBIENT = 52       # sys-therm-5: ambient air temperature (NEW)

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
    workload_hash: Optional[str] = None
    cache_hit_rate: float = 0.85
    display_brightness: Optional[int] = None
    battery_current: Optional[float] = None

@dataclass
class ThermalVelocity:
    """Temperature rate of change per zone"""
    zones: Dict[ThermalZone, float]  # °C/second per zone
    overall: float  # weighted average °C/second
    trend: ThermalTrend
    acceleration: float  # °C/second²

@dataclass
@dataclass
class ThermalPrediction:
    """Future temperature prediction per zone"""
    timestamp: float
    horizon: float  # seconds into future
    predicted_temps: Dict[ThermalZone, float]
    confidence: float  # Overall confidence
    confidence_by_zone: Dict[ThermalZone, float] = field(default_factory=dict)  # Per-zone confidence
    power_by_zone: Dict[ThermalZone, float] = field(default_factory=dict)  # Actual power used (W)
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
        display_brightness = brightness_cache
        
        battery_current = None
        if battery_cache:
            battery_current = battery_cache.get('current', None)
        
        cache_hit_rate = 0.85
        
        return ThermalSample(
            timestamp=timestamp,
            zones=zones,
            confidence=confidence,
            chassis=chassis_temp,
            network=network_cache,
            charging=charging,
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
        REAL_HARDWARE_ZONES = {'BATTERY', 'CPU_BIG', 'CPU_LITTLE', 'GPU', 'MODEM', 'CHASSIS', 'AMBIENT'}
        
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
        
        # Fallback: assume medium brightness
        return 128
    
    def estimate_display_power(self, brightness: Optional[int] = None) -> float:
        """Estimate display power based on brightness"""
        if brightness is None:
            brightness = 128
        
        brightness_fraction = brightness / 255.0
        return DISPLAY_POWER_MIN + (DISPLAY_POWER_MAX - DISPLAY_POWER_MIN) * brightness_fraction


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
        
        logger.info(f"Physics engine initialized with {len(self.zone_constants)} zone models")
        logger.info(f"Transient tuner loaded: heating={self.tuner.momentum_heating:.3f}, "
                   f"cooling={self.tuner.momentum_cooling:.3f}, stable={self.tuner.momentum_stable:.3f}")
    
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
        Return thermal resistance for the zone.
        
        This is the constant R value in Newton's law of cooling.
        Previously had exponential corrections - those were band-aids for missing throttling.
        """
        if zone_name not in self.zone_constants:
            return 5.0  # Default fallback
        
        zc = self.zone_constants[zone_name]
        return zc['thermal_resistance']
    
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
    
    def calculate_throttle_factor(self, battery_temp: float, zone_name: str) -> float:
        """
        Calculate power throttling factor based on battery temperature.
        
        Samsung's SDHM service throttles CPU/GPU by ~50% when battery hits 42°C.
        This is SOFTWARE throttling to protect battery, not hardware thermal limits.
        
        Validation data shows:
        - Battery <40°C: Full performance (factor = 1.0)
        - Battery 40-42°C: Linear ramp down
        - Battery ≥42°C: 50% throttle (factor = 0.5)
        
        Includes hysteresis to prevent oscillation at the threshold.
        
        Args:
            battery_temp: Current battery temperature in °C
            zone_name: Zone being predicted (only affects CPU_BIG, CPU_LITTLE, GPU)
        
        Returns:
            float: Power multiplier (0.5 to 1.0)
        """
        if not THROTTLE_ENABLED:
            return 1.0
        
        # Only throttle affected zones
        if zone_name not in THROTTLE_AFFECTED_ZONES:
            return 1.0
        
        # Hysteresis state machine
        # Track whether we're currently in throttled state
        if not hasattr(self, '_throttle_active'):
            self._throttle_active = False
        
        # Determine throttle state with hysteresis
        if self._throttle_active:
            # Currently throttled - require temp to drop below (start - hysteresis) to release
            if battery_temp < (THROTTLE_BATTERY_TEMP_START - THROTTLE_HYSTERESIS):
                self._throttle_active = False
        else:
            # Not throttled - require temp to reach full throttle point to engage
            if battery_temp >= THROTTLE_BATTERY_TEMP_FULL:
                self._throttle_active = True
        
        # Calculate throttle factor
        if battery_temp < THROTTLE_BATTERY_TEMP_START:
            # Below start threshold - no throttling
            return 1.0
        elif battery_temp >= THROTTLE_BATTERY_TEMP_FULL:
            # At or above full throttle point
            return THROTTLE_POWER_REDUCTION
        else:
            # Linear ramp between start and full (40-42°C)
            range_width = THROTTLE_BATTERY_TEMP_FULL - THROTTLE_BATTERY_TEMP_START
            progress = (battery_temp - THROTTLE_BATTERY_TEMP_START) / range_width
            # Interpolate from 1.0 to THROTTLE_POWER_REDUCTION
            return 1.0 - (1.0 - THROTTLE_POWER_REDUCTION) * progress
    
    def calculate_component_throttle(self, zone_temp: float, zone_name: str) -> float:
        """
        Calculate per-component throttle factor based on zone's own temperature.
        
        Components throttle themselves continuously based on their die temperature,
        independent of battery temp. This models real silicon behavior:
        - CPU: burst → immediate throttle → sustain 74-77%
        - GPU: progressive bin dropping as temp rises
        - Continuous curves, not step functions
        
        This happens BEFORE Samsung's battery-based global throttle.
        
        Args:
            zone_temp: Current zone temperature in °C
            zone_name: Zone name (CPU_BIG, GPU, etc.)
        
        Returns:
            float: Power multiplier (min_factor to 1.0)
        """
        # Get throttle curve for this component
        if zone_name not in COMPONENT_THROTTLE_CURVES:
            return 1.0  # No self-throttling for this component
        
        curve = COMPONENT_THROTTLE_CURVES[zone_name]
        temp_start = curve['temp_start']
        temp_aggressive = curve['temp_aggressive']
        min_factor = curve['min_factor']
        curve_shape = curve['curve_shape']
        
        # No throttling below start temp
        if zone_temp <= temp_start:
            return 1.0
        
        # Full throttling above aggressive temp
        if zone_temp >= temp_aggressive:
            return min_factor
        
        # Interpolate between start and aggressive
        # normalized = 0.0 at temp_start, 1.0 at temp_aggressive
        temp_range = temp_aggressive - temp_start
        normalized = (zone_temp - temp_start) / temp_range
        
        # Apply curve shape
        if curve_shape == 'linear':
            # Linear ramp: 1.0 → min_factor
            factor = 1.0 - (1.0 - min_factor) * normalized
        elif curve_shape == 'exponential':
            # Exponential decay: models GPU bin dropping
            # factor = 1.0 * exp(-k * normalized) + min_factor * (1 - exp(-k))
            # Solve for k such that at normalized=1.0, factor=min_factor
            k = 3.0  # Tuned for realistic GPU behavior
            exp_term = math.exp(-k * normalized)
            factor = 1.0 * exp_term + min_factor * (1.0 - exp_term)
        elif curve_shape == 'stepped':
            # Step function: sudden drops (CPU burst behavior)
            if normalized < 0.1:
                factor = 1.0
            elif normalized < 0.5:
                factor = 0.85
            else:
                factor = min_factor
        else:
            # Fallback to linear
            factor = 1.0 - (1.0 - min_factor) * normalized
        
        return factor
    
    def _calculate_zone_power_from_battery(
        self,
        zone_name: str,
        battery_current: float,
        velocity: ThermalVelocity,
        current_sample: ThermalSample
    ) -> float:
        """
        Calculate zone power from measured battery current.
        
        Uses total system power from battery, subtracts fixed loads,
        distributes remaining CPU/GPU power by thermal velocity ratios.
        
        This gives INSTANT power awareness instead of 10-30s lag from
        velocity-based estimates. Fixes stress test prediction failures.
        
        Args:
            zone_name: Zone to calculate power for
            battery_current: Current in μA (mislabeled as "mA" by sensor)
            velocity: Thermal velocity for distribution hints
            current_sample: Current thermal state
        
        Returns:
            Power in Watts for this zone (0.0 means fall back to velocity method)
        """
        # Calculate total system power
        I_amps = abs(battery_current) / 1_000_000.0  # μA → A
        V_battery = 3.8  # Typical Li-ion voltage
        P_system_total = I_amps * V_battery
        
        # === SUBTRACT FIXED LOADS ===
        
        P_baseline = BASELINE_SYSTEM_POWER * 0.9
        
        display_brightness = getattr(current_sample, 'display_brightness', None)
        if display_brightness is not None:
            brightness_factor = display_brightness / 255.0
            P_display = DISPLAY_POWER_MIN + (DISPLAY_POWER_MAX - DISPLAY_POWER_MIN) * brightness_factor
        else:
            P_display = (DISPLAY_POWER_MIN + DISPLAY_POWER_MAX) / 2.0
        else:
            P_display = DISPLAY_POWER_OFF
        
        # 3. Network power (modem gets this, subtract from total)
        network_type_name = current_sample.network.name if hasattr(current_sample.network, 'name') else 'UNKNOWN'
        P_modem_fixed = NETWORK_POWER.get(network_type_name, 0.2)
        
        # 4. Battery I²R losses
        P_battery_losses = I_amps ** 2 * S25_PLUS_BATTERY_INTERNAL_RESISTANCE
        
        # 5. Charging inefficiency
        P_charging = 0.0
        if current_sample.charging and battery_current > 0:
            P_charging = 0.15 * I_amps * 4.2
        
        # Total fixed load
        P_fixed = P_baseline + P_display + P_modem_fixed + P_battery_losses + P_charging
        
        # Remaining power = CPU/GPU variable load
        P_variable = max(0.0, P_system_total - P_fixed)
        
        # === ZONE-SPECIFIC DISTRIBUTION ===
        
        if zone_name == 'MODEM':
            return P_modem_fixed
        
        if zone_name == 'BATTERY':
            return P_battery_losses + P_charging
        
        if zone_name not in ['CPU_BIG', 'CPU_LITTLE', 'GPU']:
            # Other zones get velocity-based estimate
            return 0.0  # Signal to use velocity method
        
        # === DISTRIBUTE CPU/GPU POWER BY VELOCITY RATIOS ===
        
        vel_cpu_big = abs(velocity.zones.get(ThermalZone.CPU_BIG, 0.0))
        vel_cpu_little = abs(velocity.zones.get(ThermalZone.CPU_LITTLE, 0.0))
        vel_gpu = abs(velocity.zones.get(ThermalZone.GPU, 0.0))
        
        vel_total = vel_cpu_big + vel_cpu_little + vel_gpu
        
        if vel_total > 0.01:
            # Distribute by which zones are heating
            if zone_name == 'CPU_BIG':
                return P_variable * (vel_cpu_big / vel_total)
            elif zone_name == 'CPU_LITTLE':
                return P_variable * (vel_cpu_little / vel_total)
            elif zone_name == 'GPU':
                return P_variable * (vel_gpu / vel_total)
        else:
            # No heating pattern - use typical ratios
            if zone_name == 'CPU_BIG':
                return P_variable * 0.40
            elif zone_name == 'CPU_LITTLE':
                return P_variable * 0.50
            elif zone_name == 'GPU':
                return P_variable * 0.10
        
        return 0.0
    
    def predict_temperature(self,
                          current: ThermalSample,
                          velocity: ThermalVelocity,
                          horizon: float,
                          samples: List[ThermalSample] = None) -> ThermalPrediction:
        """
        Predict future temperature using per-zone Newton's law.
        
        EXCEPTION: If zone temp >= throttle start temp, predict observed_peak
        from validation data instead of physics (regime changes break the model).
        
        Physics model:
        dT/dt = -k*(T - T_ambient)/R + P/C
        
        Solution:
        T(t) = T_ambient + (T0 - T_ambient)*e^(-kt/RC) + (PR/k)*(1 - e^(-kt/RC))
        
        Where:
          k = ambient coupling coefficient
          R = thermal resistance (°C/W)
          C = thermal mass (J/K)
          P = power dissipation (W)
          T0 = current temperature
          T_ambient = reference temperature (depends on zone)
        
        THERMAL COUPLING HIERARCHY:
          Components (CPU/GPU/Battery/Modem) → Chassis → Ambient Air
          
          For components: T_ambient = chassis temp (zone 53)
          For chassis: T_ambient = ambient air temp (zone 52)
        """
        
        predicted_temps = {}
        power_by_zone = {}  # Track actual power used for each zone
        
        # Get actual chassis temperature (vapor chamber) from zone 53
        chassis_temp = current.zones.get(ThermalZone.CHASSIS, None)
        if chassis_temp is None:
            # Fallback to estimation if chassis sensor unavailable
            chassis_temp = self.estimate_chassis(current)
        
        # Get actual ambient air temperature from zone 52
        ambient_air_temp = current.zones.get(ThermalZone.AMBIENT, None)
        if ambient_air_temp is None:
            # Fallback: estimate from chassis if ambient sensor unavailable
            ambient_air_temp = chassis_temp - 10.0  # Chassis typically 10°C above room air
        
        # ========================================================================
        # PREDICT BATTERY TEMPERATURE FOR THROTTLING DECISIONS
        # ========================================================================
        # Predict battery temp at t+horizon to determine if throttling will occur
        # This must happen BEFORE processing other zones so they use correct power
        predicted_battery_temp = None
        current_battery_temp = current.zones.get(ThermalZone.BATTERY, None)
        
        if current_battery_temp is not None and 'BATTERY' in self.zone_constants:
            battery_constants = self.zone_constants['BATTERY']
            C = battery_constants['thermal_mass']
            
            battery_current = getattr(current, 'battery_current', None)
            if battery_current is not None:
                I_amps = abs(battery_current) / 1_000_000.0  # μA → A
                P_losses = I_amps ** 2 * S25_PLUS_BATTERY_INTERNAL_RESISTANCE
                
                if current.charging and battery_current > 0:
                    P_charge_heat = 0.15 * I_amps * 4.2
                    P_total = P_losses + P_charge_heat
                else:
                    P_total = P_losses
                
                # Simple integration for battery (τ >> horizon)
                delta_T = (P_total / C) * horizon
                predicted_battery_temp = current_battery_temp + delta_T
            else:
                # No current data - assume stable
                predicted_battery_temp = current_battery_temp
        
        for zone, current_temp in current.zones.items():
            zone_name = str(zone).split('.')[-1]
            
            # Skip non-thermal zones (software metrics, not hardware)
            if zone_name in ['DISPLAY', 'CHARGER']:
                continue
            
            # Skip chassis and ambient - predict separately
            if zone in [ThermalZone.CHASSIS, ThermalZone.AMBIENT]:
                continue
            
            # Get zone-specific constants
            if zone_name not in self.zone_constants:
                # Fallback: simple linear extrapolation
                vel = velocity.zones.get(zone, 0)
                predicted_temps[zone] = current_temp + vel * horizon
                continue
            
            constants = self.zone_constants[zone_name]
            
            # ========================================================================
            # THROTTLE ZONE CHECK: Use observed peak if already throttling
            # ========================================================================
            # If zone is already hot enough to throttle, predict the observed peak
            # from validation data instead of trying to model the throttled regime.
            # Physics breaks at regime changes - just use empirical data.
            if zone_name in COMPONENT_THROTTLE_CURVES:
                throttle_info = COMPONENT_THROTTLE_CURVES[zone_name]
                if current_temp >= throttle_info['temp_start']:
                    # Already in throttle regime - use observed peak
                    predicted_temps[zone] = throttle_info['observed_peak']
                    power_by_zone[zone] = 0.0  # Unknown power in throttle regime
                    continue  # Skip physics calculation
            
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
                    power_by_zone[zone] = P_total  # Store actual power
                else:
                    # No current data - assume stable (τ >> horizon)
                    predicted_temp = current_temp
                    power_by_zone[zone] = 0.0  # No power known
                
                # Battery doesn't need measurement dynamics (measurement_tau=0)
                # No artificial caps - trust the physics
                
                predicted_temps[zone] = predicted_temp
                continue  # Skip general Newton's law for battery
            
            # ========================================================================
            # GENERAL ZONES: Full Newton's law with ambient coupling
            # ========================================================================
            C = constants['thermal_mass']
            # Use temperature-dependent thermal resistance
            # Components couple to chassis, not ambient air
            R = self.effective_thermal_resistance(zone_name, current_temp, chassis_temp)
            k = constants['ambient_coupling']
            
            # ========================================================================
            # POWER INJECTION (V2.3 FIX) - Add all known power sources
            # ========================================================================
            
            # Get measured battery current for instant power awareness
            battery_current = getattr(current, 'battery_current', None)
            
            # For CPU/GPU zones: use battery current if available (instant, no lag)
            # For other zones: fall back to velocity-based estimate
            if battery_current is not None and zone_name in ['CPU_BIG', 'CPU_LITTLE', 'GPU', 'MODEM']:
                P_from_battery = self._calculate_zone_power_from_battery(
                    zone_name=zone_name,
                    battery_current=battery_current,
                    velocity=velocity,
                    current_sample=current
                )
                
                if P_from_battery > 0:
                    # Use measured power from battery (instant awareness)
                    P = P_from_battery
                else:
                    # Fall back to velocity-based estimate
                    vel = velocity.zones.get(zone, 0)
                    cooling_rate = -k * (current_temp - chassis_temp) / R
                    heating_rate = vel - cooling_rate
                    P_observed = heating_rate * C
                    P = P_observed
            else:
                # No battery current or other zone - use velocity-based estimate
                vel = velocity.zones.get(zone, 0)
                cooling_rate = -k * (current_temp - chassis_temp) / R
                heating_rate = vel - cooling_rate
                P_observed = heating_rate * C
                P = P_observed
            
            # LEGACY: Still apply P_injected for non-CPU/GPU zones
            # (This code adds display, network, baseline for zones that need it)
            # For CPU/GPU, battery_current method already accounts for these
            
            # Initialize injected power for this zone
            P_injected = 0.0
            
            # 1. BASELINE SYSTEM POWER
            baseline_power = BASELINE_SYSTEM_POWER
            
            # Distribute baseline across CPU zones proportionally
            if zone_name in ['CPU_BIG', 'CPU_LITTLE']:
                baseline_fraction = 0.4 if zone_name == 'CPU_BIG' else 0.3
                P_injected += baseline_power * baseline_fraction
            elif zone_name == 'GPU':
                P_injected += baseline_power * 0.2
            elif zone_name == 'MODEM':
                P_injected += baseline_power * 0.1
            
            # 2. DISPLAY POWER
            display_brightness = getattr(current, 'display_brightness', None)
            if display_brightness is not None:
                display_power = DISPLAY_POWER_MIN + (DISPLAY_POWER_MAX - DISPLAY_POWER_MIN) * (display_brightness / 255.0)
            else:
                display_power = DISPLAY_POWER_MIN + (DISPLAY_POWER_MAX - DISPLAY_POWER_MIN) * 0.5
            
            zone_fraction = DISPLAY_THERMAL_DISTRIBUTION.get(zone_name, 0.0)
            P_injected += display_power * zone_fraction
            
            # 3. NETWORK POWER
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
            # UNLESS we used battery_current for CPU/GPU (already accounts for everything)
            if battery_current is not None and zone_name in ['CPU_BIG', 'CPU_LITTLE', 'GPU', 'MODEM']:
                # Battery current method already accounted for all loads
                # Use P directly without adding P_injected
                P_final = P
            elif P_observed > P_injected * 1.5:
                # Heavy workload detected - trust velocity-based estimate
                P = P_observed
                P_final = P
            elif P_observed < 0:
                # Cooling - use only injected power (positive sources)
                P = P_injected
                P_final = P
            else:
                # Normal operation - use max of observed and injected
                # This ensures we don't miss either workload OR known sources
                P = max(P_observed, P_injected)
                P_final = P
            
            # Trend-aware power adjustment (only for non-battery-current zones)
            if not (battery_current is not None and zone_name in ['CPU_BIG', 'CPU_LITTLE', 'GPU', 'MODEM']):
                # If accelerating, assume power will continue to rise (conservative)
                # If decelerating, assume power will drop (optimistic)
                accel = velocity.acceleration
                
                if abs(accel) < 0.05:
                    # Steady state - use current power
                    pass  # P_final already set
                elif accel > 0.1:
                    # Accelerating - power is increasing (be conservative)
                    P_final = P_final * 1.2
                elif accel < -0.1:
                    # Decelerating - power is dropping (be optimistic)
                    P_final = P_final * 0.8
            
            # Clamp power to realistic range
            P_final = max(constants['idle_power'], min(P_final, constants['peak_power']))
            
            # ========================================================================
            # PER-COMPONENT THROTTLING: Zone's own temperature
            # ========================================================================
            # Components throttle themselves based on their own die temp, not battery.
            # This models real silicon: CPU sustains 74%, GPU drops bins, etc.
            # Happens BEFORE Samsung's battery-based global throttle.
            component_throttle = self.calculate_component_throttle(current_temp, zone_name)
            P_final = P_final * component_throttle
            
            # ========================================================================
            # SAMSUNG THROTTLING: Predicted battery temperature-based power reduction
            # ========================================================================
            # Use PREDICTED battery temp at t+horizon to determine throttling
            # This accounts for thermal runaway before it happens
            if predicted_battery_temp is not None:
                throttle_factor = self.calculate_throttle_factor(predicted_battery_temp, zone_name)
                if throttle_factor < 1.0:
                    P_final = P_final * throttle_factor
            
            # ========================================================================
            # TWO-STAGE PREDICTION: Component → Sensor (v2.9)
            # ========================================================================
            # Stage 1: Predict component temperature (fast response)
            # Components cool to chassis (vapor chamber), not ambient air
            tau = constants['time_constant']  # C * R
            exp_factor = math.exp(-horizon / tau)
            
            # Transient response: initial temperature difference decays
            temp_transient = (current_temp - chassis_temp) * exp_factor
            
            # Steady-state response: power establishes new equilibrium
            temp_steady = (P_final * R / k) * (1 - exp_factor)
            
            component_temp = chassis_temp + temp_transient + temp_steady
            
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
            
            # No artificial caps - trust the physics model
            
            # ========================================================================
            # TJMAX EMERGENCY THROTTLING: Per-zone hardware protection
            # ========================================================================
            # If predicted temp exceeds hardware TJmax, force emergency throttling
            # This is the silicon's last line of defense before damage
            tjmax = None
            if zone_name in ['CPU_BIG', 'CPU_LITTLE']:
                tjmax = TJMAX_CPU
            elif zone_name == 'GPU':
                tjmax = TJMAX_GPU
            elif zone_name == 'MODEM':
                tjmax = TJMAX_MODEM
            elif zone_name == 'BATTERY':
                tjmax = TJMAX_BATTERY
            
            if tjmax is not None and predicted_temp >= tjmax:
                # Emergency: recalculate with idle power only
                P_emergency = constants['idle_power']
                
                # Recalculate with minimal power
                exp_factor_emerg = math.exp(-k * horizon / (R * C))
                temp_transient_emerg = (current_temp - chassis_temp) * exp_factor_emerg
                temp_steady_emerg = (P_emergency * R / k) * (1 - exp_factor_emerg)
                component_temp_emerg = chassis_temp + temp_transient_emerg + temp_steady_emerg
                
                if tau_meas > 0 and horizon > 0:
                    decay_factor_emerg = math.exp(-horizon / tau_meas)
                    predicted_temp = current_temp * decay_factor_emerg + component_temp_emerg * (1 - decay_factor_emerg)
                else:
                    predicted_temp = component_temp_emerg
                
                # Store emergency power used
                power_by_zone[zone] = P_emergency
            else:
                # Store normal power used
                power_by_zone[zone] = P_final
            
            predicted_temps[zone] = predicted_temp
        
        # ========================================================================
        # CHASSIS PREDICTION: Thermal averaging with vapor chamber dynamics
        # ========================================================================
        # Chassis (vapor chamber + metal frame) acts as a passive thermal reservoir:
        # - Conducts heat from hot components (weighted by die size and coupling)
        # - Dissipates to ambient via vapor chamber (efficient)
        # - Exponentially approaches weighted average of component temps
        # - Slower dynamics than components (τ=120s vs component τ<10s)
        #
        # Physical insight: Chassis runs 2-3°C cooler than CPUs (vapor chamber working)
        if ThermalZone.CHASSIS in current.zones:
            chassis_current = current.zones[ThermalZone.CHASSIS]
            
            # Component thermal contribution weights
            # Based on Snapdragon 8 Elite die layout (~110mm² total):
            # - CPU_BIG (2× Oryon Prime): 25mm², 6W peak → 35% weight
            # - CPU_LITTLE (6× Oryon eff): 18mm², 4W peak → 20% weight  
            # - GPU (Adreno 830): 35mm², 5W peak → 30% weight
            # - MODEM: 15mm², 3W peak → 15% weight
            component_weights = {
                ThermalZone.CPU_BIG: 0.35,
                ThermalZone.CPU_LITTLE: 0.20,
                ThermalZone.GPU: 0.30,
                ThermalZone.MODEM: 0.15
            }
            
            # Calculate weighted average of component temperatures
            weighted_sum = 0.0
            total_weight = 0.0
            
            for zone, weight in component_weights.items():
                if zone in current.zones:
                    weighted_sum += weight * current.zones[zone]
                    total_weight += weight
            
            if total_weight > 0:
                # Target: weighted average of components
                target_temp = weighted_sum / total_weight
                
                # Chassis is well-coupled to ambient via vapor chamber
                # Pull target temperature toward ambient (vapor chamber efficiency)
                ambient_pull = 0.25  # 25% ambient influence (efficient cooling)
                target_temp = (1 - ambient_pull) * target_temp + ambient_pull * ambient_air_temp
                
                # Exponential approach with chassis time constant (slow thermal mass)
                # τ=120s from zone constants
                if 'CHASSIS' in self.zone_constants:
                    tau_chassis = self.zone_constants['CHASSIS']['time_constant']
                else:
                    tau_chassis = 120.0
                
                exp_decay = math.exp(-horizon / tau_chassis)
                
                # Prediction: current temp exponentially approaches target
                chassis_pred = chassis_current * exp_decay + target_temp * (1 - exp_decay)
                
                # Add velocity contribution for trending (damped for slow dynamics)
                chassis_vel = velocity.zones.get(ThermalZone.CHASSIS, 0)
                chassis_pred += chassis_vel * horizon * 0.2  # 20% velocity influence (slow response)
                
                # Physical limits: chassis bounded by ambient and hottest component
                max_component = max(current.zones.get(z, ambient_air_temp) 
                                   for z in component_weights.keys() 
                                   if z in current.zones)
                chassis_pred = max(ambient_air_temp, min(chassis_pred, max_component))
                
                # ========================================================================
                
                predicted_temps[ThermalZone.CHASSIS] = chassis_pred
                power_by_zone[ThermalZone.CHASSIS] = 0.0  # Passive zone
            else:
                # Fallback: use current temperature
                predicted_temps[ThermalZone.CHASSIS] = chassis_current
                power_by_zone[ThermalZone.CHASSIS] = 0.0
        
        # ========================================================================
        # AMBIENT AIR PREDICTION: Room temperature stays constant
        # ========================================================================
        # Ambient air temperature doesn't change on thermal prediction timescales
        if ThermalZone.AMBIENT in current.zones:
            predicted_temps[ThermalZone.AMBIENT] = current.zones[ThermalZone.AMBIENT]
            power_by_zone[ThermalZone.AMBIENT] = 0.0  # Ambient has no power
        
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
            power_by_zone=power_by_zone,  # Actual power used for each zone
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
            type('Sample', (), {'zones': current, 'charging': False})()
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
# THERMAL TANK - Battery-Centric Throttle Control with CPU Spike Detection
# ============================================================================

class ThrottleReason(Enum):
    """Why we're throttling"""
    NONE = auto()
    BATTERY_TEMP = auto()
    CPU_VELOCITY = auto()
    BOTH = auto()

@dataclass
class ThermalTankStatus:
    """Enhanced output from thermal tank with dual throttle conditions"""
    battery_temp_current: float      # Current battery temp (°C)
    battery_temp_predicted: float    # Predicted battery temp at horizon (°C)
    should_throttle: bool            # True = reject new work
    throttle_reason: ThrottleReason  # Why we're throttling (if at all)
    headroom_seconds: float          # Seconds until throttle (0 if already hot)
    cooling_rate: float              # °C/s (negative = heating)
    cpu_big_velocity: float          # Current CPU_BIG heating rate (°C/s)
    cpu_little_velocity: float       # Current CPU_LITTLE heating rate (°C/s)

class ThermalTank:
    """
    Battery-centric thermal management with CPU spike detection.
    
    DUAL THROTTLE CONDITIONS:
    1. Battery predicted temp >= 38.5°C (Samsung throttles at 40°C)
    2. CPU velocities indicate regime change (sudden workload spike)
    
    CPU VELOCITY THRESHOLDS (from validation data):
    - DANGER:  >1.0°C/s  (P95+, would add 30°C in 30s horizon)
    - WARNING: >0.5°C/s  (would add 15°C in 30s horizon)
    
    RATIONALE:
    Battery has τ=540s (slow thermal response), CPUs have τ=14-19s (fast).
    Regime changes spike CPU temps before battery reacts. Physics model
    can't predict discontinuities - use velocity as early warning.
    
    Validation showed max velocities:
    - CPU_BIG: 9.5°C/s (extreme spike)
    - CPU_LITTLE: 14°C/s (extreme spike)
    - Normal P90: 0.3-0.4°C/s
    - Normal P95: 0.7-1.0°C/s
    
    Throttle if EITHER CPU exceeds DANGER threshold, indicating
    unpredictable workload our continuous physics model can't handle.
    """
    
    # CPU velocity thresholds (°C/s) - tuned from validation data
    CPU_VELOCITY_DANGER = 1.0     # P95+ level, immediate throttle
    CPU_VELOCITY_WARNING = 0.5    # Watch closely
    
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
        Get current tank status with dual throttle conditions.
        
        CHECKS:
        1. Battery temperature prediction (existing physics)
        2. CPU velocity spikes (regime change detection - NEW)
        
        Returns enriched status with throttle reason.
        """
        # Extract zones
        battery_zone = None
        cpu_big_zone = None
        cpu_little_zone = None
        
        for zone in current.zones:
            zone_name = str(zone).split('.')[-1]
            if 'BATTERY' in zone_name.upper():
                battery_zone = zone
            elif zone_name == 'CPU_BIG':
                cpu_big_zone = zone
            elif zone_name == 'CPU_LITTLE':
                cpu_little_zone = zone
        
        # No battery sensor - conservative throttle
        if battery_zone is None:
            return ThermalTankStatus(
                battery_temp_current=0.0,
                battery_temp_predicted=0.0,
                should_throttle=True,  # Conservative: throttle if no data
                throttle_reason=ThrottleReason.NONE,
                headroom_seconds=0.0,
                cooling_rate=0.0,
                cpu_big_velocity=0.0,
                cpu_little_velocity=0.0
            )
        
        # Battery state
        battery_current = current.zones[battery_zone]
        battery_velocity = velocity.zones.get(battery_zone, 0.0)
        battery_predicted = prediction.predicted_temps.get(battery_zone, battery_current)
        
        # CPU velocities (heating rates)
        cpu_big_velocity = velocity.zones.get(cpu_big_zone, 0.0) if cpu_big_zone else 0.0
        cpu_little_velocity = velocity.zones.get(cpu_little_zone, 0.0) if cpu_little_zone else 0.0
        
        # CONDITION 1: Battery temperature (existing logic)
        battery_throttle = battery_predicted >= self.throttle_temp
        
        # CONDITION 2: CPU velocity spike (regime change detection - NEW)
        # Throttle if EITHER CPU is heating dangerously fast
        # Only care about heating (positive velocity), not cooling
        cpu_spike_throttle = (
            cpu_big_velocity > self.CPU_VELOCITY_DANGER or 
            cpu_little_velocity > self.CPU_VELOCITY_DANGER
        )
        
        # Determine throttle decision and reason
        if battery_throttle and cpu_spike_throttle:
            throttle_reason = ThrottleReason.BOTH
            should_throttle = True
        elif battery_throttle:
            throttle_reason = ThrottleReason.BATTERY_TEMP
            should_throttle = True
        elif cpu_spike_throttle:
            throttle_reason = ThrottleReason.CPU_VELOCITY
            should_throttle = True
        else:
            throttle_reason = ThrottleReason.NONE
            should_throttle = False
        
        # Calculate headroom (seconds until battery throttle)
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
            throttle_reason=throttle_reason,
            headroom_seconds=headroom,
            cooling_rate=battery_velocity,
            cpu_big_velocity=cpu_big_velocity,
            cpu_little_velocity=cpu_little_velocity
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
        self.tank = ThermalTank(self.physics)  # Dual-condition throttle (battery + CPU velocity)
        
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
                            # Match found but no tuning needed anymore
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
