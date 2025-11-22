#!/usr/bin/env python3
"""
🔥🐧🔥 S25+ Thermal Intelligence System
====================================
Copyright (c) 2025 PNGN-Tec LLC

Physics-based thermal management for Android devices under continuous load.

Multi-zone temperature monitoring with Newton's law of cooling predictions. 
Prevents throttling through proactive thermal budget calculation and workload 
scheduling.

Validation results over 152k predictions (6.25 hours continuous operation):
- Overall: 0.58°C MAE (transients filtered), 0.47°C MAE (steady-state)
- Battery prediction: 0.24°C MAE
- 96.5% of predictions within 5°C (3.5% transients during load changes)
- Stress test: CPUs sustained 95°C+ with 1.23°C MAE recovery tracking

ARCHITECTURE:
- Multi-zone sensor monitoring (CPU, GPU, battery, modem, chassis)
- Newton's law of cooling with measured per-zone thermal constants
- Dual-confidence predictions (physics model × sample-size weighting)
- Samsung throttling awareness (50% power reduction at 42°C battery)
- Thermal tank status for dual-condition throttle decisions (battery temp + CPU velocity)
- 1s sampling, 30s prediction horizon

HARDWARE (Samsung Galaxy S25+ / Snapdragon 8 Elite):
- CPU_BIG (cpuss-1-0 / zone 20): 2× Oryon Prime, τ=50s
- CPU_LITTLE (cpuss-0-0 / zone 13): 6× Oryon efficiency, τ=60s
- GPU (gpuss-0 / zone 23): Adreno 830, τ=95s
- MODEM (mdmss-0 / zone 31): 5G/WiFi, τ=80s
- BATTERY (battery / zone 60): τ=210s (critical for Samsung throttle at 42°C)
- CHASSIS (sys-therm-0 / zone 53): vapor chamber reference sensor, τ=100s
- AMBIENT (sys-therm-5 / zone 52): ambient air temperature, τ=30s

ZONE CORRECTIONS (based on thermal scan):
- BATTERY: 31 → 60 (zone 31 is modem, not battery)
- MODEM: 39 → 31 (zone 39 is neural processor, not modem)
- CHASSIS: 47 → 53 (zone 47 broken, reads 0.0°C; zone 52 is ambient air)

PHYSICS:
T(t) = T_amb + (T₀ - T_amb)·exp(-t/τ) + (P·R/k)·(1 - exp(-t/τ))

Battery simplification (τ >> horizon):
ΔT ≈ (P/C) × Δt

PREDICTION ACCURACY:
~0.6°C MAE at 30s horizon (filtered for transients)
0.47°C MAE during steady-state operation
Battery zone highly predictable at 0.24°C MAE (τ=210s)
"""

import sys
import os
import json
import time
import asyncio
import subprocess
from typing import Dict, List, Optional, Tuple, Callable, Any, Deque, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict, OrderedDict
from enum import Enum, auto, IntEnum
from datetime import datetime, timedelta
import math
import statistics
import logging
from pathlib import Path
import numpy as np

# Configure logging
logger = logging.getLogger('PNGN.S25Thermal')

# ============================================================================
# VALIDATION INFRASTRUCTURE
# ============================================================================

# Validation configuration
MAX_VALIDATION_SAMPLES = 500  # Maximum samples to keep per zone for validation
MAX_PREDICTIONS = 10000  # Maximum predictions to store in numpy array (flush ~every 3-4 hours)
MAX_PENDING_VALIDATIONS = 1000  # Maximum pending validations to prevent memory leak
VALIDATION_WINDOW = 3.0  # seconds - tolerance for validating predictions
VALIDATION_MAX_AGE = 120.0  # seconds - maximum age before discarding stale predictions

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for a zone"""
    count: int = 0
    mae: float = 0.0  # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Squared Error
    max_error: float = 0.0
    mean_error: float = 0.0  # Bias
    std_error: float = 0.0
    within_1C: int = 0
    within_2C: int = 0
    within_3C: int = 0
    last_update: float = 0.0

class BoundedDefaultDict:
    """A defaultdict that automatically bounds list sizes to prevent memory leaks"""
    def __init__(self, default_factory, maxlen=MAX_VALIDATION_SAMPLES):
        self.data = {}
        self.default_factory = default_factory
        self.maxlen = maxlen
    
    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = self.default_factory()
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __contains__(self, key):
        return key in self.data
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def append_to(self, key, subkey, value):
        """Append to nested dict with automatic bounding"""
        if key not in self.data:
            self.data[key] = {}
        if subkey not in self.data[key]:
            self.data[key][subkey] = deque(maxlen=self.maxlen)
        
        if isinstance(self.data[key][subkey], deque):
            self.data[key][subkey].append(value)
        elif isinstance(self.data[key][subkey], list):
            # Migrate old list to bounded deque
            self.data[key][subkey] = deque(self.data[key][subkey], maxlen=self.maxlen)
            self.data[key][subkey].append(value)

# ============================================================================
# TUNING CONSTANTS - ALL CONFIGURABLE PARAMETERS IN ONE PLACE
# ============================================================================

# Default tau for zones without specific constants
TAU_DEFAULT = 30.0  # seconds - fallback time constant

# ============================================================================
# PREDICTION PARAMETERS
# ============================================================================
THERMAL_PREDICTION_HORIZON = 30.0       # seconds ahead to predict
THERMAL_SAMPLE_INTERVAL = 1.0           # 1s uniform sampling
THERMAL_HISTORY_SIZE = 300              # samples to keep
MIN_SAMPLES_FOR_PREDICTIONS = 60        # minimum samples before making predictions (1 min warmup for ambient fitting)
API_UPDATE_INTERVAL = 2.0               # seconds between battery/network/brightness cache updates

# Velocity calculation
VELOCITY_CALCULATION_SAMPLES = 10        # samples for velocity linear regression
VELOCITY_HISTORY_SIZE = 3               # samples for regime change detection

# Confidence scaling
CONFIDENCE_SAFETY_SCALE = 0.5           # prediction safety scaling

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

# ============================================================================
# PREDICTION ACCURACY THRESHOLDS (Zone-Specific)
# ============================================================================
# Different zones have different accuracy expectations based on thermal mass
# and measurement characteristics. Color coding uses PNGN team colors:
# Green (#00FF7F) → Purple (#CC33FF) → Red (#FF1493)

ACCURACY_THRESHOLDS = {
    'BATTERY': {
        'excellent': 1.0,
        'good': 1.5,
        'fair': 2.0,
        'poor': 999.0
    },
    'AMBIENT': {
        'excellent': 1.0,
        'good': 1.5,
        'fair': 2.0,
        'poor': 999.0
    },
    'CHASSIS': {
        'excellent': 1.0,
        'good': 1.5,
        'fair': 2.0,
        'poor': 999.0
    },
    'GPU': {
        'excellent': 2.0,
        'good': 2.5,
        'fair': 3.0,
        'poor': 999.0
    },
    'MODEM': {
        'excellent': 2.0,
        'good': 2.5,
        'fair': 3.0,
        'poor': 999.0
    },
    'CPU_BIG': {
        'excellent': 3.0,
        'good': 3.5,
        'fair': 4.0,
        'poor': 999.0
    },
    'CPU_LITTLE': {
        'excellent': 3.0,
        'good': 3.5,
        'fair': 4.0,
        'poor': 999.0
    },
}

# Color definitions (PNGN_32 palette - from PNGN_32.html)
ACCURACY_COLORS = {
    'excellent': '#00FF7F',      # Radiation Green (index 18) - RGB(0, 255, 127)
    'good': '#00FF00',           # Neon Green (index 16) - RGB(0, 255, 0)
    'fair': '#CC33FF',           # PNGN Purple (index 0) - RGB(204, 51, 255)
    'poor': '#FF1493'            # Deep Pink (index 10) - RGB(255, 20, 147)
}

# Text labels for accuracy ratings
ACCURACY_LABELS = {
    'excellent': 'EXCELLENT',
    'good': 'GOOD',
    'fair': 'FAIR',
    'poor': 'POOR'
}

# ============================================================================
# ADAPTIVE POWER LEARNING
# ============================================================================
POWER_LEARNING_ENABLED = True
POWER_LEARNING_RATE = 0.03                          # EMA alpha
POWER_LEARNING_WINDOW = 3                         # samples to stabilize
POWER_LEARNING_MIN_SAMPLES = 5                     # minimum before applying
POWER_LEARNING_PERSIST_INTERVAL = 86400              # 24 hours
POWER_LEARNING_BACKUP_INTERVAL = 3600               # 1 hour
POWER_LEARNING_BACKUP_RETENTION_DAYS = 7
POWER_LEARNING_FILE = '~/.thermal_power_learned.json'
POWER_LEARNING_BACKUP_DIR = '~/.thermal_power_backups'

# Thermal zone definitions with physically-derived constants
ZONE_THERMAL_CONSTANTS = {
    'CPU_BIG': {
        'thermal_mass': 20.00,  # Updated from 15 based on transient analysis
        'ambient_coupling': 0.0,    # Coupled to CHASSIS not ambient 
        'peak_power': 20.0,
        'idle_power': 0.02,  # Reduced from 0.10 to fix +4.7°C bias
        'time_constant': 50.0,  # Updated from 5s based on cooling measurements
        'measurement_tau': 0.3,
    },
    
    'CPU_LITTLE': {
        'thermal_mass': 40.00,  # Updated from 30 based on transient analysis
        'ambient_coupling': 0.0,    # Coupled to CHASSIS not ambient 
        'peak_power': 20.0,
        'idle_power': 0.04,  # Reduced from 0.10 to fix +2.4°C bias
        'time_constant': 60.0,  # Updated from 10s based on cooling measurements
        'measurement_tau': 0.3,
    },
    
    'GPU': {
        'thermal_mass': 40.00,  # Updated from 30 based on transient analysis
        'ambient_coupling': 0.0,
        'peak_power': 20.0,
        'idle_power': 0.03,  # Reduced from 0.10 to fix +3.6°C bias
        'time_constant': 95.0,  # Updated from 15s based on cooling measurements
        'measurement_tau': 0.3,
    },
    
    'BATTERY': {
        'thermal_mass': 75.00,
        'ambient_coupling': 0.0,    # Coupled to CHASSIS not ambient 
        'peak_power': 0.0,
        'idle_power': 0.0,
        'time_constant': 210.0,
        'measurement_tau': 0.3,
    },
    
    'MODEM': {
        'thermal_mass': 35.00,  # Updated from 30 based on transient analysis
        'ambient_coupling': 0.0,    # Coupled to CHASSIS not ambient 
        'peak_power': 20.0,
        'idle_power': 0.04,  # Reduced from 0.10 to fix +2.5°C bias
        'time_constant': 80.0,  # Updated from 15s based on cooling measurements
        'measurement_tau': 0.3,
    },
    
    'CHASSIS': {
        'thermal_mass': 40.00,
        'ambient_coupling': 0.80,
        'peak_power': 0.0,
        'idle_power': 0.0,
        'time_constant': 100.0,
        'measurement_tau': 0.3,
    },
    
    'AMBIENT': {
        'thermal_mass': 10.0,
        'ambient_coupling': 0.0,
        'peak_power': 0.0,
        'idle_power': 0.0,
        'time_constant': 30.0,         # Phones are mobile and ambient changes quickly
        'measurement_tau': 0.0,
    }
}

# ============================================================================
# COMPONENT THROTTLE CURVES
# ============================================================================
# Real silicon throttles progressively. These model actual behavior.
COMPONENT_THROTTLE_CURVES = {
    'CPU_BIG': {
        'temp_start': 45.0,       # Begin throttling
        'observed_peak': 81.0,  # Updated from 72°C based on max load validation    # Where it plateaus under load
        'temp_aggressive': 65.0,  # Heavy throttle point - lowered from 71°C
        'min_factor': 0.50,       # 50% power at max throttle
        'curve_shape': 'linear',
    },
    'CPU_LITTLE': {
        'temp_start': 48.0,
        'observed_peak': 94.0,
        'temp_aggressive': 85.0,  # Heavy throttle point - lowered from 93°C
        'min_factor': 0.50,
        'curve_shape': 'linear',
    },
    'GPU': {
        'temp_start': 38.0,
        'observed_peak': 61.0,  # Updated from 55°C based on max load validation
        'temp_aggressive': 50.0,  # Heavy throttle point - lowered from 54°C
        'min_factor': 0.50,
        'curve_shape': 'linear',
    },
    'MODEM': {
        'temp_start': 40.0,
        'observed_peak': 62.0,  # Updated from 56°C based on max load validation
        'temp_aggressive': 50.0,  # Heavy throttle point - lowered from 55°C
        'min_factor': 0.50,
        'curve_shape': 'linear',
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

THERMAL_SENSOR_CONFIDENCE_REDUCED = 0.5   # Confidence for out-of-range sensor readings

# Termux / external API configuration
THERMAL_SUBPROCESS_TIMEOUT = 3.0         # Seconds to wait for Termux subprocess calls
THERMAL_NETWORK_TIMEOUT = 3.0           # Seconds to wait for network info commands
THERMAL_NETWORK_AWARENESS_ENABLED = True

TERMUX_BATTERY_STATUS_CMD = ["termux-battery-status"]
TERMUX_WIFI_INFO_CMD = ["termux-wifi-connectioninfo"]
TERMUX_TELEPHONY_INFO_CMD = ["termux-telephony-deviceinfo"]

# Minimum Wi-Fi frequency (MHz) considered to be 5 GHz band
THERMAL_WIFI_5G_FREQ_MIN = 4900.0


# Thermal zone mapping (None = auto-discover from /sys/class/thermal)
THERMAL_ZONES = None

# Battery prediction horizon
BATTERY_PREDICTION_HORIZON = THERMAL_PREDICTION_HORIZON    # seconds (matches main horizon for consistency)

# ============================================================================
# CACHE AGE LIMITS
# ============================================================================
# Prediction context cache ages (expensive or slow-changing data)
PREDICTION_BATTERY_CACHE_AGE = 3.0         # seconds - expensive ground truth for power
PREDICTION_NETWORK_CACHE_AGE = 3.0         # seconds - rare regime changes
PREDICTION_BRIGHTNESS_CACHE_AGE = 3.0      # seconds - slow changes, context only

# ============================================================================
# SAMPLE WINDOW SIZES
# ============================================================================
REGIME_DETECTION_WINDOW = 3                 # samples for regime change detection
MIN_SAMPLES_AMBIENT_FIT = 3                 # minimum samples for ambient curve fitting
MIN_COOLING_SAMPLES = 3                     # minimum cooling samples for exponential fit
MIN_TREND_SAMPLES = 3                       # minimum samples for trend detection
SAMPLE_STALENESS_THRESHOLD = 1.0            # Maximum time difference between actual and predicted

# ============================================================================
# VELOCITY THRESHOLDS
# ============================================================================
HIGH_VELOCITY_THRESHOLD = 2.0               # °C/s - regime change detection
CPU_VELOCITY_DANGER = 3.0                   # °C/s - danger threshold for CPU

# Status display trend indicators (°C/s)
VELOCITY_TREND_RAPID_WARMING = 0.15         # ↑↑ rapid warming
VELOCITY_TREND_WARMING = 0.05               # ↑  warming
VELOCITY_TREND_STABLE_HIGH = 0.05           # →  stable (upper bound)
VELOCITY_TREND_STABLE_LOW = -0.05           # →  stable (lower bound)
VELOCITY_TREND_COOLING = -0.15              # ↓  cooling (upper bound)
VELOCITY_TREND_RAPID_COOLING = -0.15        # ↓↓ rapid cooling (at or below)

# ============================================================================
# TEMPERATURE DISPLAY THRESHOLDS
# ============================================================================
# Color-coded temperature zones for status display
TEMP_DISPLAY_COOL = 35.0                    # °C - green zone
TEMP_DISPLAY_WARM = 38.0                    # °C - light green zone
TEMP_DISPLAY_WARNING = 42.0                 # °C - purple zone
TEMP_DISPLAY_HOT = 45.0                     # °C - pink zone
# Above TEMP_DISPLAY_HOT = red zone

# ============================================================================
# CHARGING THRESHOLDS
# ============================================================================
FAST_CHARGE_CURRENT_UA = 1_000_000          # microamps (1A) - fast charging
NORMAL_CHARGE_CURRENT_UA = 500_000          # microamps (0.5A) - normal charging
FAST_CHARGE_CURRENT_MA = 1500               # milliamps - alternative fast charge detection

# Battery SOC thresholds
BATTERY_SOC_CRITICAL = 15                   # % - critical battery level
BATTERY_SOC_OPTIMAL_MIN = 50                # % - optimal range start
BATTERY_SOC_OPTIMAL_MAX = 80                # % - optimal range end
BATTERY_SOC_HIGH = 85                       # % - high battery level

# ============================================================================
# AMBIENT TEMPERATURE ESTIMATION
# ============================================================================
AMBIENT_TEMP_MIN = 10.0                     # °C - lower bound for ambient estimates
AMBIENT_TEMP_MAX = 70.0                     # °C - upper bound for ambient estimates
AMBIENT_ESTIMATE_OFFSET = 2.5               # °C - offset from coolest sensor
AMBIENT_TEMP_FALLBACK = 25.0                # °C - fallback when no estimate available

# Battery-based ambient offset (varies by charge state)
BASE_OFFSET_FAST_CHARGE = 8.0               # °C - fast charging
BASE_OFFSET_DISCHARGE = 6.0                 # °C - discharging
BASE_OFFSET_IDLE = 5.0                      # °C - idle
BASE_OFFSET_SLOW_CHARGE = 3.0               # °C - slow charging

# Ambient estimation weights
AMBIENT_WEIGHT_BATTERY = 0.7                # Battery-based estimate weight
AMBIENT_WEIGHT_CHASSIS = 0.3                # Chassis-based estimate weight (inverse used when reversed)

# ============================================================================
# THERMAL TIME CONSTANTS (FALLBACKS)
# ============================================================================
# Used when zone constants not available
TAU_BATTERY_FALLBACK = 210.0                # seconds - battery thermal time constant
TAU_CHASSIS_FALLBACK = 100.0                   # seconds - chassis thermal time constant
TAU_CHASSIS_SYNTHETIC = 100.0                 # seconds - synthetic for coupled predictions

# ============================================================================
# DISPLAY BRIGHTNESS
# ============================================================================
BRIGHTNESS_MIN = 0                          # minimum brightness value
BRIGHTNESS_MAX = 255                        # maximum brightness value
BRIGHTNESS_FALLBACK = 128                   # fallback brightness when unavailable

# ============================================================================
# CURVE FITTING
# ============================================================================
CURVE_FIT_MAX_ITERATIONS = 1000             # maximum iterations for scipy curve_fit

# ============================================================================
# THERMAL BUDGET
# ============================================================================
THERMAL_BUDGET_UNLIMITED = 999.0            # seconds - sentinel for unlimited thermal budget
MAX_THERMAL_BUDGET = 600.0                  # seconds - maximum reasonable thermal budget
MAX_RECOMMENDED_DELAY = 10.0                # seconds - maximum recommended delay

# ============================================================================
# PHYSICS CONSTANTS
# ============================================================================
BATTERY_NOMINAL_VOLTAGE = 3.8               # V - typical Li-ion voltage
GPU_THERMAL_COEFFICIENT = 3.0               # tuned coefficient for GPU thermal behavior
MIN_THERMAL_POWER = 0.1                     # W - floor for thermal power calculations
EPSILON_VELOCITY = 0.01                     # °C/s - small epsilon to prevent divide-by-zero
SECONDS_PER_DAY = 86400                     # seconds in a day

# ============================================================================
# POWER VALIDATION
# ============================================================================
MIN_POWER_SANITY = 0.5                      # W - minimum realistic power
MAX_POWER_SANITY = 20.0                     # W - maximum realistic power for validation

# ============================================================================
# THROTTLE PREDICTION
# ============================================================================
THROTTLE_PREDICTION_BUFFER = 1.5            # °C - predict throttle this far ahead of actual temp

# ============================================================================
# CONFIDENCE WEIGHTS (Zone-Specific)
# ============================================================================
# Zone velocity weights for overall velocity calculation
VELOCITY_WEIGHTS = {
    'CPU_BIG': 2.0,
    'GPU': 1.5,
    'CPU_LITTLE': 1.0,
    'BATTERY': 0.5,
    'MODEM': 0.8,
}
VELOCITY_WEIGHT_DEFAULT = 1.0
CPU_BIG_VELOCITY_WEIGHT = 2.0  # Weight for CPU_BIG in velocity calculation
CPU_THROTTLE_STEP = 0.85        # Throttle step at medium power

# Zone confidence for predictions
PREDICTION_CONFIDENCE = {
    'BATTERY': 0.95,      # Measured current = ground truth
    'CPU_BIG': 0.85,      # Component throttling understood
    'CPU_LITTLE': 0.85,   # Component throttling understood
    'GPU': 0.75,          # Workload variability
    'MODEM': 0.70,        # Network unpredictable
    'CHASSIS': 0.80,      # Damped response
}
PREDICTION_CONFIDENCE_DEFAULT = 0.60
PREDICTION_CONFIDENCE_FALLBACK = 0.7        # Used in status when no prediction available

# Thermal contribution weights for coupled prediction
THERMAL_CONTRIBUTION_WEIGHTS = {
    'CPU_BIG': 0.40,
    'CPU_LITTLE': 0.20,
    'GPU': 0.20,
    'BATTERY': 0.00,
    'MODEM': 0.20
}

# ============================================================================
# SMOOTHING PARAMETERS
# ============================================================================
VELOCITY_SMOOTHING_ALPHA = 0.3              # EMA alpha for velocity smoothing
VELOCITY_SMOOTHING_WEIGHT = 0.1             # Weight for current vs recent average
VELOCITY_JUMP_THRESHOLD_DEFAULT = 1.0       # °C/s - default regime change threshold
VELOCITY_JUMP_THRESHOLD_GPU = 0.5           # °C/s - GPU more sensitive
TEMP_SCALE_ALPHA = 0.02                     # Temperature scaling alpha
TEMP_SCALE_MIN = 0.1                        # Minimum temperature scale factor
TEMP_SCALE_REF = 30.0                       # °C - reference temperature

# ============================================================================
# AMBIENT INFLUENCE
# ============================================================================
AMBIENT_PULL_EFFICIENT_COOLING = 0.25       # fraction - efficient cooling ambient influence

# ============================================================================
# DISPLAY THERMAL CONVERSION
# ============================================================================
DISPLAY_BRIGHTNESS_DIVISOR = 255.0          # convert brightness (0-255) to fraction

# ============================================================================
# SENSOR FAILURE TRACKING
# ============================================================================
MAX_SENSOR_FAILURES = 5                     # consecutive failures before degrading confidence

# ============================================================================
# THERMAL CONVERSIONS
# ============================================================================
MILLIDEGREE_TO_DEGREE = 1000.0              # conversion factor for kernel thermal zones

# ============================================================================
# BASELINE POWER FRACTIONS
# ============================================================================
BASELINE_FRACTION_CPU_BIG = 0.4             # Baseline power attribution to CPU_BIG
BASELINE_FRACTION_OTHER = 0.3               # Baseline power attribution to other zones

# ============================================================================
# AMBIENT ESTIMATION THRESHOLDS
# ============================================================================
# Battery-to-coolest sensor differential thresholds
BATTERY_COOLEST_DIFF_HIGH = 8.0             # °C - battery much warmer than coolest
BATTERY_COOLEST_DIFF_LOW = 3.0              # °C - battery close to coolest

# ============================================================================
# REGIME DETECTION THRESHOLDS
# ============================================================================
MAX_VELOCITY_THRESHOLD = 5.0                # °C/s - maximum velocity before regime change
VELOCITY_CHANGE_THRESHOLD = 3.0             # °C/s - velocity delta for regime shift
REGIME_SUSTAINED_THRESHOLD = 2.0            # °C/s - threshold for sustained regime

# ============================================================================
# VELOCITY SMOOTHING THRESHOLDS
# ============================================================================
JUMP_THRESHOLD_DEFAULT = 0.8                # °C/s - default jump detection
VELOCITY_EPSILON = 0.01                     # °C/s - minimum meaningful velocity
ACCELERATION_STABLE_THRESHOLD = 0.05        # °C/s² - acceleration considered stable
ACCELERATION_SIGNIFICANT_THRESHOLD = 0.1    # °C/s² - significant acceleration

# ============================================================================
# THROTTLE FACTOR THRESHOLDS
# ============================================================================
THROTTLE_FACTOR_ACTIVE = 1.0                # No throttling active

# ============================================================================
# POWER NORMALIZATION THRESHOLDS
# ============================================================================
POWER_NORMALIZED_LOW = 0.1                  # Normalized power - low threshold
POWER_NORMALIZED_MED = 0.5                  # Normalized power - medium threshold

# ============================================================================
# BATTERY VELOCITY THRESHOLDS
# ============================================================================
BATTERY_VELOCITY_HEATING = 0.001            # °C/s - battery heating threshold

# ============================================================================
# FEATURE FLAGS
# ============================================================================
THERMAL_PREDICTION_ENABLED = True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_accuracy_rating(zone_name: str, error: float) -> Tuple[str, str, str]:
    """
    Get accuracy rating, color, and label for a prediction error.
    
    Args:
        zone_name: Zone name (e.g. 'CPU_BIG', 'BATTERY')
        error: Absolute prediction error in °C
    
    Returns:
        (rating, color_hex, label) tuple
        - rating: 'excellent', 'good', 'fair', or 'poor'
        - color_hex: Hex color from PNGN_32 palette
        - label: Display label ('EXCELLENT', 'GOOD', 'FAIR', 'POOR')
    
    Examples:
        >>> get_accuracy_rating('BATTERY', 0.3)
        ('excellent', '#00FF7F', 'EXCELLENT')
        >>> get_accuracy_rating('CPU_BIG', 2.5)
        ('good', '#00FF00', 'GOOD')
        >>> get_accuracy_rating('GPU', 3.0)
        ('fair', '#CC33FF', 'FAIR')
    """
    # Get zone thresholds, fall back to CPU_BIG for unknown zones
    thresholds = ACCURACY_THRESHOLDS.get(zone_name, ACCURACY_THRESHOLDS['CPU_BIG'])
    
    # Determine rating based on error magnitude
    if error < thresholds['excellent']:
        rating = 'excellent'
    elif error < thresholds['good']:
        rating = 'good'
    elif error < thresholds['fair']:
        rating = 'fair'
    else:
        rating = 'poor'
    
    return rating, ACCURACY_COLORS[rating], ACCURACY_LABELS[rating]

# ============================================================================
# SHARED TYPES
# ============================================================================

class ThermalState(Enum):
    """Device thermal state classification for throttling decisions"""
    COLD = auto()
    OPTIMAL = auto()
    WARM = auto()
    HOT = auto()
    CRITICAL = auto()
    UNKNOWN = auto()

class ThermalZone(Enum):
    """
    Hardware thermal zones on Samsung S25+ (Snapdragon 8 Elite).
    Values are /sys/class/thermal/thermal_zone{N} indices.
    """
    CPU_BIG = 20       # cpuss-1-0: 2× Oryon Prime aggregate
    CPU_LITTLE = 13    # cpuss-0-0: 6× Oryon efficiency aggregate  
    GPU = 23           # gpuss-0: Adreno 830
    BATTERY = 60       # battery thermistor
    MODEM = 31         # mdmss-0: 5G/WiFi modem
    CHASSIS = 53       # sys-therm-0: chassis/vapor chamber (CORRECTED from 52)
    AMBIENT = 52       # sys-therm-5: ambient air temperature (NEW)

class ThermalTrend(Enum):
    """Temperature rate of change classification for prediction confidence"""
    RAPID_COOLING = auto()
    COOLING = auto()
    STABLE = auto()
    WARMING = auto()
    RAPID_WARMING = auto()

class NetworkType(Enum):
    """Network connection type for power modeling"""
    UNKNOWN = auto()
    OFFLINE = auto()
    WIFI_2G = auto()
    WIFI_5G = auto()
    MOBILE_3G = auto()
    MOBILE_4G = auto()
    MOBILE_5G = auto()

class MemoryPressureLevel(IntEnum):
    """System memory pressure classification for resource management"""
    NORMAL = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ThermalSample:
    """
    Temperature measurement at single point in time.
    
    Attributes:
        timestamp: Unix timestamp of sample
        zones: Temperature readings per thermal zone (°C)
        confidence: Confidence score per zone (0.0-1.0)
        chassis: Chassis/ambient temperature estimate (°C)
        network: Current network type for power modeling
        charging: Whether device is charging
        workload_hash: Optional workload identifier
        cache_hit_rate: Vapor chamber efficiency estimate
        display_brightness: Screen brightness level (0-255)
        battery_current: Battery current in μA (negative = discharge)
        battery_voltage: Battery voltage in mV (uses nominal 3.8V if unavailable)
        battery_temp: Battery temperature in °C
        battery_pct: Battery percentage (0-100)
    """
    timestamp: float
    zones: Dict[ThermalZone, float]
    confidence: Dict[ThermalZone, float]
    chassis: Optional[float] = None
    network: NetworkType = NetworkType.UNKNOWN
    charging: bool = False
    workload_hash: Optional[str] = None
    cache_hit_rate: float = S25_PLUS_VAPOR_CHAMBER_EFFICIENCY
    display_brightness: Optional[int] = None
    battery_current: Optional[float] = None
    battery_voltage: Optional[int] = None  # mV
    battery_temp: Optional[float] = None   # °C
    battery_pct: Optional[int] = None      # percentage

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
    workload_correlation: float = THERMAL_SENSOR_CONFIDENCE_REDUCED
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
        Collect complete thermal sample from cache.
        Battery/network/brightness are updated by independent background loop.
        
        Populates battery_current, battery_temp, battery_pct from Termux API cache.
        Battery voltage uses nominal 3.8V (3800mV) as Termux doesn't expose actual voltage.
        
        Args:
            for_prediction: If True, uses prediction cache (0.5s freshness)
                          If False, uses UI cache (5s freshness)
        
        Returns:
            ThermalSample with temperature readings and battery metrics
        """
        timestamp = time.time()
        
        # Determine which cache to use
        if for_prediction:
            battery_cache = self.pred_battery
            network_cache = self.pred_network
            brightness_cache = self.pred_brightness
        else:
            battery_cache = self.ui_battery
            network_cache = self.ui_network
            brightness_cache = self.ui_brightness
        
        # Read thermal zones (fast, local filesystem)
        zones, confidence = await self._read_thermal_zones_batch()
        
        # Ambient temperature (use AMBIENT zone or estimate)
        chassis_temp = zones.get(ThermalZone.CHASSIS)
        if chassis_temp is None:
            # Estimate ambient from battery (slowest changing zone)
            chassis_temp = zones.get(ThermalZone.BATTERY, AMBIENT_TEMP_FALLBACK) - 5.0
        
        # Extract data from cache
        charging = battery_cache.get('plugged', False) if battery_cache else False
        display_brightness = brightness_cache
        
        battery_current = None
        battery_voltage = None
        battery_temp = None
        battery_pct = None
        
        if battery_cache:
            battery_current = battery_cache.get('current', None)
            battery_temp = battery_cache.get('temperature', None)
            battery_pct = battery_cache.get('percentage', None)
            # Termux doesn't provide voltage, use nominal
            if battery_current is not None:
                battery_voltage = int(BATTERY_NOMINAL_VOLTAGE * 1000)  # Convert to mV
        
        cache_hit_rate = S25_PLUS_VAPOR_CHAMBER_EFFICIENCY
        
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
            battery_current=battery_current,
            battery_voltage=battery_voltage,
            battery_temp=battery_temp,
            battery_pct=battery_pct
        )
    
    async def _read_thermal_zones_batch(self) -> Tuple[Dict[ThermalZone, float], Dict[ThermalZone, float]]:
        """
        Read all thermal zones efficiently in batch.
        Reads zone temp files directly; no external shell command.
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
                    temp = float(f.read().strip()) / MILLIDEGREE_TO_DEGREE
                    
                    # Validate temperature
                    if THERMAL_SENSOR_TEMP_MIN <= temp <= THERMAL_SENSOR_TEMP_MAX:
                        zones[zone_enum] = temp
                        confidence[zone_enum] = VELOCITY_WEIGHT_DEFAULT
                    else:
                        # Out of range - reduced confidence
                        zones[zone_enum] = temp
                        confidence[zone_enum] = THERMAL_SENSOR_CONFIDENCE_REDUCED
                    
                self.read_failures[zone_enum] = 0
            except Exception as e:
                self.read_failures[zone_enum] += 1
                if self.read_failures[zone_enum] < MAX_SENSOR_FAILURES:
                    logger.debug(f"Failed to read {zone_enum.name}: {e}")
        
        return zones, confidence
    
    async def collect_sample_for_display(self) -> ThermalSample:
        """
        Convenience method for UI/display data.
        Uses 5s cache (efficient, can tolerate slight staleness).
        """
        return await self.collect_sample(for_prediction=False)
    
    async def _read_battery_status(self) -> Optional[Dict]:
        """
        Read battery status from Termux API.
        
        Returns:
            Dict with keys: 'plugged' (bool), 'percentage' (int), 
            'temperature' (float in °C), 'current' (float in μA)
            Returns None if read fails.
        """
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *TERMUX_BATTERY_STATUS_CMD,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), 
                timeout=THERMAL_SUBPROCESS_TIMEOUT
            )
            
            if proc.returncode == 0:
                data = json.loads(stdout.decode())
                return {
                    'plugged': data.get('plugged', 'UNPLUGGED') != 'UNPLUGGED',
                    'percentage': data.get('percentage', 0),
                    'temperature': data.get('temperature', AMBIENT_TEMP_FALLBACK),
                    'current': data.get('current', 0)  # mA (+ charging, - discharging)
                }
        except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
            pass
        finally:
            # Ensure process is cleaned up
            if proc and proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except:
                    pass
        
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
                return max(BRIGHTNESS_MIN, min(brightness, BRIGHTNESS_MAX))
        
        except Exception as e:
            logger.debug(f"Brightness detection failed: {e}")
        
        # Fallback: assume medium brightness
        return 128
    
    def estimate_display_power(self, brightness: Optional[int] = None) -> float:
        """Estimate display power based on brightness"""
        if brightness is None:
            brightness = BRIGHTNESS_FALLBACK
        
        brightness_fraction = brightness / DISPLAY_BRIGHTNESS_DIVISOR
        return DISPLAY_POWER_MIN + (DISPLAY_POWER_MAX - DISPLAY_POWER_MIN) * brightness_fraction


# ============================================================================
# ZONE-SPECIFIC PHYSICS ENGINE
# ============================================================================

class ZonePhysicsEngine:
    """
    Per-zone thermal physics using Newton's law of cooling.
    Each zone has unique thermal mass, resistance, and time constant.
    """
    
    def __init__(self, power_learning: Optional['PowerLearningSystem'] = None):
        self.zone_constants = ZONE_THERMAL_CONSTANTS
        self.power_learning = power_learning
        
        logger.info(f"Physics engine initialized with {len(self.zone_constants)} zone models")
    
    def estimate_chassis(self, sample: ThermalSample) -> float:
        """
        Dynamically estimate ambient temperature from system state.
        
        Features:
        - Dynamic offset based on battery-to-coolest delta
        - Sun exposure detection (battery cooler than active zones)
        - Wide valid range (10-70°C)
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
                    return max(AMBIENT_TEMP_MIN, min(avg_cool - AMBIENT_ESTIMATE_OFFSET, AMBIENT_TEMP_MAX))
                elif temps:
                    return max(10.0, min(min(temps) - AMBIENT_ESTIMATE_OFFSET, 70.0))
            return sample.chassis if sample.chassis else AMBIENT_TEMP_FALLBACK
        
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
                if battery_current and battery_current > FAST_CHARGE_CURRENT_MA:  # Fast charge
                    base_offset = BASE_OFFSET_FAST_CHARGE
                else:
                    base_offset = BASE_OFFSET_DISCHARGE
            elif battery_current and battery_current < -FAST_CHARGE_CURRENT_MA:
                # Heavy discharge
                base_offset = BASE_OFFSET_IDLE
            else:
                # Light discharge/idle
                base_offset = BASE_OFFSET_SLOW_CHARGE
            
            # Adjust offset based on battery-to-coolest delta
            # Larger delta = more active system = larger offset
            battery_to_coolest = battery_temp - coolest
            if battery_to_coolest > BATTERY_COOLEST_DIFF_HIGH:
                # Very hot battery relative to CPU - reduce offset
                # (ambient is probably also hot)
                offset = base_offset * 0.7
            elif battery_to_coolest < BATTERY_COOLEST_DIFF_LOW:
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
        ambient_est = max(AMBIENT_TEMP_MIN, min(ambient_est, AMBIENT_TEMP_MAX))
        
        return ambient_est
    
    def fit_ambient_temperature(self, samples: List[ThermalSample]) -> Optional[float]:
        """
        Simplified ambient estimation - avoids scipy/numpy allocations.
        Uses simple min-based estimation with physics offsets.
        """
        if len(samples) < MIN_SAMPLES_FOR_PREDICTIONS:
            return None
        
        # Use last few samples
        recent_samples = samples[-MIN_SAMPLES_FOR_PREDICTIONS:]
        
        # Extract battery and chassis temps
        battery_temps = []
        chassis_temps = []
        
        for sample in recent_samples:
            batt_temp = sample.zones.get(ThermalZone.BATTERY)
            chas_temp = sample.zones.get(ThermalZone.CHASSIS)
            
            if batt_temp is not None and chas_temp is not None:
                battery_temps.append(batt_temp)
                chassis_temps.append(chas_temp)
        
        if len(battery_temps) < MIN_SAMPLES_AMBIENT_FIT:
            return None
        
        # Simple min-based estimation with physics offsets
        T_amb_from_battery = min(battery_temps) - 2.0  # Battery runs ~2°C above ambient
        T_amb_from_chassis = min(chassis_temps) - 4.0  # Chassis runs ~4°C above ambient
        
        # Weight by stability (lower variance = more reliable)
        if len(battery_temps) >= 3:
            batt_std = statistics.stdev(battery_temps)
            chas_std = statistics.stdev(chassis_temps)
            
            if batt_std < chas_std:
                T_ambient = 0.7 * T_amb_from_battery + 0.3 * T_amb_from_chassis
            else:
                T_ambient = 0.3 * T_amb_from_battery + 0.7 * T_amb_from_chassis
        else:
            T_ambient = (T_amb_from_battery + T_amb_from_chassis) / 2.0
        
        # Sanity: ambient must be below all observed temps
        absolute_min = min(min(battery_temps), min(chassis_temps))
        if T_ambient > absolute_min:
            T_ambient = absolute_min - 1.0
        
        # Clamp to reasonable range
        return max(AMBIENT_TEMP_MIN, min(T_ambient, THROTTLE_AGGRESSIVE_TEMP))
        return max(AMBIENT_TEMP_MIN, min(T_ambient, THROTTLE_AGGRESSIVE_TEMP))

    

    def _should_predict(self, zone: ThermalZone, vel: float) -> bool:
        """
        Simple prediction filter without buffer tracking.
        Always returns True - let physics handle it.
        """
        return True
    
    def _check_velocity_pattern(self, zone: ThermalZone, vel: float) -> str:
        """
        Simple velocity pattern check without history tracking.
        Returns 'normal' always - let throttle curves handle it.
        """
        return 'normal'
    
    def calculate_velocity(self, samples: List[ThermalSample]) -> ThermalVelocity:
        """Calculate dT/dt for each zone using linear regression"""
        if len(samples) < 2:
            return ThermalVelocity(
                zones={},
                overall=0.0,
                trend=ThermalTrend.STABLE,
                acceleration=0.0
            )
        
        # Use last N samples for velocity calculation (noise reduction)
        recent = samples[-VELOCITY_CALCULATION_SAMPLES:] if len(samples) >= VELOCITY_CALCULATION_SAMPLES else samples
        
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
                ThermalZone.CPU_BIG: CPU_BIG_VELOCITY_WEIGHT,
                ThermalZone.GPU: VELOCITY_WEIGHTS['GPU'],
                ThermalZone.CPU_LITTLE: VELOCITY_WEIGHTS['CPU_LITTLE'],
                ThermalZone.BATTERY: VELOCITY_WEIGHTS['BATTERY'],
                ThermalZone.MODEM: VELOCITY_WEIGHTS['MODEM'],
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            for zone, vel in zone_velocities.items():
                w = weights.get(zone, VELOCITY_WEIGHT_DEFAULT)
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
    
    def validate_and_smooth_velocity(self, zone: ThermalZone, velocity: float) -> Tuple[float, bool]:
        """
        Simple velocity validation without history tracking.
        Returns velocity as-is with reliability flag.
        """
        # No history tracking - just return velocity as reliable
        return velocity, True
    
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
            k = GPU_THERMAL_COEFFICIENT  # Tuned for realistic GPU behavior
            exp_term = math.exp(-k * normalized)
            factor = 1.0 * exp_term + min_factor * (1.0 - exp_term)
        elif curve_shape == 'stepped':
            # Step function: sudden drops (CPU burst behavior)
            if normalized < POWER_NORMALIZED_LOW:
                factor = 1.0
            elif normalized < POWER_NORMALIZED_MED:
                factor = CPU_THROTTLE_STEP
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
        V_battery = BATTERY_NOMINAL_VOLTAGE  # Typical Li-ion voltage
        P_system_total = I_amps * V_battery
        
        # === SUBTRACT FIXED LOADS ===
        
        P_baseline = BASELINE_SYSTEM_POWER * 0.9
        
        display_brightness = getattr(current_sample, 'display_brightness', None)
        if display_brightness is not None:
            brightness_factor = display_brightness / DISPLAY_BRIGHTNESS_DIVISOR
            P_display = DISPLAY_POWER_MIN + (DISPLAY_POWER_MAX - DISPLAY_POWER_MIN) * brightness_factor
        else:
            P_display = (DISPLAY_POWER_MIN + DISPLAY_POWER_MAX) / 2.0
        
        # 3. Network power (modem gets this, subtract from total)
        network_type_name = current_sample.network.name if hasattr(current_sample.network, 'name') else 'UNKNOWN'
        P_modem_fixed = NETWORK_POWER.get(network_type_name, 0.2)
        
        # 4. Battery I²R losses
        P_battery_losses = I_amps ** 2 * S25_PLUS_BATTERY_INTERNAL_RESISTANCE
        
        # 5. Charging inefficiency
        P_charging = 0.0
        if current_sample.charging and battery_current > 0:
            P_charging = 0.20 * I_amps * 4.2
        
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
        
        if vel_total > VELOCITY_EPSILON:
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
        
        # Get TRUE ambient from exponential fitting of battery/chassis
        ambient_air_temp = self.fit_ambient_temperature(samples) if samples else None
        if ambient_air_temp is None:
            # Cannot predict without fitted ambient - return empty prediction
            return ThermalPrediction(
                timestamp=current.timestamp + horizon,
                horizon=horizon,
                predicted_temps={},
                thermal_budget=THERMAL_BUDGET_UNLIMITED,
                confidence=0.0,
                confidence_by_zone={},
                power_by_zone={}
            )
        
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
            
            # Apply component throttling - this models actual hardware behavior
            throttle_factor = self.calculate_component_throttle(current_temp, zone_name)
            
            # If approaching throttle temps, predict plateau at observed_peak
            if zone_name in COMPONENT_THROTTLE_CURVES:
                curve = COMPONENT_THROTTLE_CURVES[zone_name]
                if current_temp > curve['temp_start']:
                    # Temperature will plateau near observed_peak under sustained load
                    observed_peak = curve['observed_peak']
                    # Blend physics prediction with observed peak based on how throttled we are
                    physics_weight = throttle_factor
                    plateau_weight = 1.0 - throttle_factor
            
            # ========================================================================
            # BATTERY SPECIAL CASE (): Integration of measured power
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
            k = constants['ambient_coupling']
            
            # Get velocity first
            vel = velocity.zones.get(zone, 0)
            
            # Validate velocity for regime changes (sudden jumps break physics model)
            vel_smoothed, vel_reliable = self.validate_and_smooth_velocity(zone, vel)
            
            # Use smoothed velocity for predictions
            # This prevents regime change discontinuities from breaking Newton's law
            vel = vel_smoothed
            
            # Check if data is clean enough to predict
            if not self._should_predict(zone, vel):
                # Bad data - skip this zone, defer to next cycle
                continue
            
            # Get tau for this zone
            tau_base = constants.get('time_constant', TAU_DEFAULT)
            
            # Determine reference temperature based on zone type
            # CHASSIS couples to ambient, others couple to chassis
            if zone_name == 'CHASSIS':
                ref_temp = ambient_air_temp
            else:
                ref_temp = chassis_temp
            
            # Calculate cooling rate based on coupling type
            if k > 0:
                # Ambient-coupled (CHASSIS only): cooling rate includes k
                cooling_rate = -k * (current_temp - ref_temp) / tau_base
            else:
                # Chassis-coupled (CPU, GPU, etc): k=0, cooling through tau only
                cooling_rate = -(current_temp - ref_temp) / tau_base
            
            # Equilibrium detection threshold (10% of cooling rate)
            equilibrium_threshold = 0.1 * abs(cooling_rate)
            
            if abs(vel) < equilibrium_threshold:
                # Near equilibrium: velocity is measurement noise
                # Power equals steady-state cooling rate
                if k > 0:
                    # Ambient-coupled: P = k*C*(T - T_ref)/τ
                    P_observed = k * C * (current_temp - ref_temp) / tau_base
                else:
                    # Chassis-coupled: P = C*(T - T_ref)/τ
                    P_observed = C * (current_temp - ref_temp) / tau_base
            else:
                # Transient: velocity measurement is meaningful
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
                baseline_fraction = BASELINE_FRACTION_CPU_BIG if zone_name == 'CPU_BIG' else BASELINE_FRACTION_OTHER
                P_injected += baseline_power * baseline_fraction
            elif zone_name == 'GPU':
                P_injected += baseline_power * 0.2
            elif zone_name == 'MODEM':
                P_injected += baseline_power * 0.1
            
            # 2. DISPLAY POWER
            display_brightness = getattr(current, 'display_brightness', None)
            if display_brightness is not None:
                display_power = DISPLAY_POWER_MIN + (DISPLAY_POWER_MAX - DISPLAY_POWER_MIN) * (display_brightness / DISPLAY_BRIGHTNESS_DIVISOR)
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
                    if battery_current > FAST_CHARGE_CURRENT_UA:  # > 1A (fast charging)
                        base_charging_power = CHARGING_POWER_FAST
                    elif battery_current > NORMAL_CHARGE_CURRENT_UA:  # > 0.5A (normal charging)
                        base_charging_power = CHARGING_POWER_NORMAL
                    else:
                        base_charging_power = CHARGING_POWER_NORMAL
                    
                    # Apply SoC-dependent efficiency curve
                    if battery_percent is not None:
                        if battery_percent < BATTERY_SOC_CRITICAL:
                            # Cold battery at low SoC, poor efficiency
                            charging_power = base_charging_power * 1.35
                        elif battery_percent > BATTERY_SOC_HIGH:
                            # CV phase at high SoC, more heat
                            charging_power = base_charging_power * 1.25
                        elif BATTERY_SOC_OPTIMAL_MIN <= battery_percent <= BATTERY_SOC_OPTIMAL_MAX:
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
            
            
            # ====================================================================
            # POWER COMBINATION: Observed vs Known Sources ()
            # ====================================================================
            # P_observed = velocity-based power (what we're actually seeing)
            # P_injected = known sources (baseline, display, network)
            #
            # For CPU/GPU: P_observed already captures everything, don't double-count
            # For other zones: combine intelligently
            
            if zone_name in ['CPU_BIG', 'CPU_LITTLE', 'GPU', 'MODEM']:
                # Velocity method already captures all thermal dissipation
                # Don't add P_injected (would double-count)
                P_final = P_observed
            elif P_observed > P_injected * 1.5:
                # Heavy workload detected - trust velocity
                P_final = P_observed
            elif P_observed < 0:
                # Cooling - use only known positive sources
                P_final = P_injected
            else:
                # Normal - use max of observed and known
                P_final = max(P_observed, P_injected)
            
            # Trend-aware power adjustment
            accel = velocity.acceleration
            
            if abs(accel) < ACCELERATION_STABLE_THRESHOLD:
                # Steady state - use current power
                pass
            elif accel > ACCELERATION_SIGNIFICANT_THRESHOLD:
                # Accelerating - be conservative
                P_final = P_final * 1.2
            elif accel < -0.1:
                # Decelerating - be optimistic
                P_final = P_final * 0.8
            
            # ====================================================================
            # POWER VALIDATION (): Trust measurements, warn on outliers
            # ====================================================================
            # Measurements already capture reality: burst behavior, throttling,
            # duty cycles, and overhead. Only prevent numerical instability.
            # 
            # peak_power is a sanity check, not a hard limit.
            if P_final > constants['peak_power'] * 1.2:
                logger.debug(
                    f"{zone_name}: Measured power {P_final:.2f}W exceeds spec "
                    f"({constants['peak_power']:.1f}W × 1.2 = {constants['peak_power'] * 1.2:.1f}W). "
                    f"Possible measurement error or undocumented boost mode."
                )
            
            # Floor only: prevent negative power (breaks physics) and near-zero (numerical issues)
            P_final = max(0.1, P_final)
            
            # ====================================================================
            # ADAPTIVE POWER LEARNING CALIBRATION ()
            # ====================================================================
            # Apply learned calibration to velocity-based predictions
            # Calibration corrects systematic bias while preserving transient response
            if self.power_learning and zone_name in ['CPU_BIG', 'CPU_LITTLE', 'GPU', 'MODEM']:
                cal_factor = self.power_learning.get_calibration_factor(zone_name)
                P_final *= cal_factor
            
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
                if throttle_factor < THROTTLE_FACTOR_ACTIVE:
                    P_final = P_final * throttle_factor
            
            # ========================================================================
            # TWO-STAGE PREDICTION: Component → Sensor ()
            # ========================================================================
            # Stage 1: Predict component temperature (fast response)
            # Components cool to chassis (vapor chamber), not ambient air
            tau_base = constants['time_constant']  # C * R (at reference temp)
            
            # Apply temperature-dependent scaling to tau (same as R scaling)
            # Since τ = R × C, and R(T) = R_base × temp_scale, then τ(T) = τ_base × temp_scale
            T_ref = TEMP_SCALE_REF
            alpha = constants.get('r_temp_alpha', 0.0)
            temp_scale = max(TEMP_SCALE_MIN, 1.0 - alpha * (current_temp - T_ref))
            tau = tau_base * temp_scale
            
            # Exponential factor depends on coupling type
            if k > 0:
                # Ambient-coupled: includes k in exponential
                exp_factor = math.exp(-k * horizon / tau)
            else:
                # Chassis-coupled: no k in exponential
                exp_factor = math.exp(-horizon / tau)
            
            # Transient response: initial temperature difference decays
            temp_transient = (current_temp - ref_temp) * exp_factor
            
            # Steady-state response: power establishes new equilibrium
            C = constants.get('thermal_mass', 1.0)
            if k > 0:
                # Ambient-coupled: P*tau/(k*C)
                temp_steady = (P_final * tau / (k * C)) * (1 - exp_factor)
            else:
                # Chassis-coupled: P*tau/C (no k term)
                temp_steady = (P_final * tau / C) * (1 - exp_factor)
            
            component_temp = ref_temp + temp_transient + temp_steady
            
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
            
            # ========================================================================
            # REGIME SWITCH: Plateau at max load
            # ========================================================================
            # Below throttle start: Newton's law runs normally
            # At/above throttle start: Check if heading to plateau
            if zone_name in COMPONENT_THROTTLE_CURVES:
                curve = COMPONENT_THROTTLE_CURVES[zone_name]
                
                if current_temp >= curve['temp_start']:
                    zone_velocity = velocity.zones.get(zone, 0)
                    
                    # Debug logging
                    if current_temp > 60:
                        logger.debug(f"{zone_name}: T0={current_temp:.1f}°C, vel={zone_velocity:.3f}°C/s, "
                                   f"temp_aggressive={curve['temp_aggressive']}°C")
                    
                    # Plateau if:
                    # 1. Heating rapidly (>1.0°C/s) toward max load, OR
                    # 2. Already heavily throttled (>temp_aggressive)
                    if zone_velocity > 1.0 or current_temp >= curve['temp_aggressive']:
                        predicted_temp = curve['observed_peak']
                        if current_temp > 60:
                            logger.debug(f"{zone_name}: PLATEAU TRIGGERED → {predicted_temp}°C")
                    # else: stable at moderate throttle, let physics run
            
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
                
                # Recalculate with minimal power using tau and C
                C = constants.get('thermal_mass', 1.0)
                
                # Handle k=0 vs k>0 for emergency path
                if k > 0:
                    exp_factor_emerg = math.exp(-k * horizon / tau)
                    temp_transient_emerg = (current_temp - ref_temp) * exp_factor_emerg
                    temp_steady_emerg = (P_emergency * tau / (k * C)) * (1 - exp_factor_emerg)
                else:
                    exp_factor_emerg = math.exp(-horizon / tau)
                    temp_transient_emerg = (current_temp - ref_temp) * exp_factor_emerg
                    temp_steady_emerg = (P_emergency * tau / C) * (1 - exp_factor_emerg)
                
                component_temp_emerg = ref_temp + temp_transient_emerg + temp_steady_emerg
                
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
                ThermalZone.CPU_BIG: THERMAL_CONTRIBUTION_WEIGHTS['CPU_BIG'],
                ThermalZone.CPU_LITTLE: THERMAL_CONTRIBUTION_WEIGHTS['CPU_LITTLE'],
                ThermalZone.GPU: THERMAL_CONTRIBUTION_WEIGHTS['GPU'],
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
                ambient_pull = AMBIENT_PULL_EFFICIENT_COOLING  # 25% ambient influence (efficient cooling)
                target_temp = (1 - ambient_pull) * target_temp + ambient_pull * ambient_air_temp
                
                # Exponential approach with chassis time constant (slow thermal mass)
                # τ=120s from zone constants
                if 'CHASSIS' in self.zone_constants:
                    tau_chassis = self.zone_constants['CHASSIS']['time_constant']
                else:
                    tau_chassis = TAU_CHASSIS_SYNTHETIC
                
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
            'BATTERY': PREDICTION_CONFIDENCE['BATTERY'],  # Measured current = ground truth
            'CPU_BIG': 0.85,      # Fast response, well-understood
            'CPU_LITTLE': 0.85,
            'GPU': PREDICTION_CONFIDENCE['GPU'],  # Workload variability
            'MODEM': PREDICTION_CONFIDENCE['MODEM'],  # Network unpredictable
            'CHASSIS': PREDICTION_CONFIDENCE['CHASSIS'],  # Damped response
        }.get(zone_name, PREDICTION_CONFIDENCE_DEFAULT)
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
            return THERMAL_BUDGET_UNLIMITED
        
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
            
            # Get sensor confidence (defaults to 1.0, could optionally use current.confidence dict)
            sensor_conf = 1.0
            
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
            return THERMAL_BUDGET_UNLIMITED
        
        # Time until hit HOT threshold
        budget = (THERMAL_TEMP_HOT - worst_case_temp) / velocity.overall
        
        # Clamp to reasonable range
        return max(0, min(budget, MAX_THERMAL_BUDGET))
    
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
                delay = (max_predicted - THERMAL_TEMP_WARM) / (abs(velocity.overall) + EPSILON_VELOCITY)
                return min(delay, MAX_RECOMMENDED_DELAY)  # cap at 10s
        
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
            tau = constants.get('time_constant', TAU_DEFAULT)
            k = constants['ambient_coupling']
            
            # Temperature change
            dT = sample.zones[zone] - previous.zones[zone]
            
            # Power = C * dT/dt + cooling_power
            heating_power = C * (dT / dt)
            
            chassis = sample.chassis or AMBIENT_TEMP_FALLBACK
            # Use tau to approximate cooling (tau = R*C, so R ≈ tau/C)
            cooling_power = k * (sample.zones[zone] - chassis) * C / tau
            
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
    cooling_rate: float              # °C/s (positive = heating, negative = cooling)
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
    CPU_VELOCITY_DANGER = CPU_VELOCITY_DANGER
    
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
        # Use 3-sample burst detection - don't throttle on transient spikes
        # Only care about heating (positive velocity), not cooling
        
        # Check CPU_BIG velocity pattern
        cpu_big_pattern = 'normal'
        if cpu_big_zone:
            cpu_big_pattern = self.physics._check_velocity_pattern(cpu_big_zone, cpu_big_velocity)
        
        # Check CPU_LITTLE velocity pattern
        cpu_little_pattern = 'normal'
        if cpu_little_zone:
            cpu_little_pattern = self.physics._check_velocity_pattern(cpu_little_zone, cpu_little_velocity)
        
        # Throttle logic:
        # - 'burst': Only throttle if critically dangerous (will hit hardware limit)
        # - 'sustained': Throttle at normal threshold
        # - 'normal': No throttle
        
        cpu_spike_throttle = False
        
        if cpu_big_pattern == 'burst' or cpu_little_pattern == 'burst':
            # Burst detected - only throttle if about to hit critical temps
            # Use predicted temps if available
            cpu_big_predicted = prediction.predicted_temps.get(cpu_big_zone, 0.0) if cpu_big_zone else 0.0
            cpu_little_predicted = prediction.predicted_temps.get(cpu_little_zone, 0.0) if cpu_little_zone else 0.0
            
            # Critical threshold: would exceed observed peak from validation
            if cpu_big_predicted > TJMAX_CPU or cpu_little_predicted > TJMAX_CPU:
                cpu_spike_throttle = True
        
        elif cpu_big_pattern == 'sustained' or cpu_little_pattern == 'sustained':
            # Sustained high velocity - throttle at normal threshold
            if cpu_big_velocity > self.CPU_VELOCITY_DANGER or cpu_little_velocity > self.CPU_VELOCITY_DANGER:
                cpu_spike_throttle = True
        
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
        if battery_velocity > BATTERY_VELOCITY_HEATING:  # Heating
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
# ADAPTIVE POWER LEARNING - Calibration System
# ============================================================================

class PowerLearningSystem:
    """
    Adaptive power learning for velocity-based temperature predictions.
    
    Learns actual power distribution from battery measurements during discharge,
    then generates calibration factors to bias-correct velocity-based predictions.
    
    APPROACH:
    - Measure total system power via battery voltage × current
    - Distribute to zones based on thermal velocity (hot zones get more power)
    - Learn average power per zone using exponential moving average
    - Calculate calibration: learned_power / velocity_predicted_power
    - Apply calibration to velocity-based predictions
    
    CRITICAL: Does NOT replace velocity predictions - only calibrates them.
    Velocity gives transient response, learned power corrects systematic bias.
    """
    
    def __init__(self):
        # Learned power per zone (EMA)
        self.power_ema: Dict[str, float] = {
            'CPU_BIG': 0.0,
            'CPU_LITTLE': 0.0,
            'GPU': 0.0,
            'MODEM': 0.0
        }
        
        # Calibration factors (learned / velocity-predicted)
        self.calibration_factors: Dict[str, float] = {
            'CPU_BIG': 1.0,
            'CPU_LITTLE': 1.0,
            'GPU': 1.0,
            'MODEM': 1.0
        }
        
        # Learning state
        self.sample_count = 0
        self.last_persist_time = time.time()
        self.last_backup_time = time.time()
        
        # Load persisted state
        self._load_from_disk()
        
        logger.info(f"Power learning initialized with {self.sample_count} samples")
    
    def update(self, 
               actual_power_watts: float, 
               zones: Dict, 
               zone_velocities: Dict) -> None:
        """
        Learn from actual power measurement.
        
        Args:
            actual_power_watts: Total system power from battery (voltage × current)
            zones: Dict[ThermalZone, float] - current temperatures
            zone_velocities: Dict[ThermalZone, float] - heating rates (°C/s)
        """
        # Validate power range (0.5W to 20W is reasonable for S25+)
        if not (MIN_POWER_SANITY <= actual_power_watts <= MAX_POWER_SANITY):
            logger.debug(f"Power measurement {actual_power_watts:.2f}W out of range, skipping")
            return
        
        # Subtract non-compute power components
        # Display: ~1-3W (assume 2W average)
        # Charging overhead: 0W (only learning during discharge)
        # Modem baseline: ~0.5W (always on)
        display_power = 2.0
        modem_baseline = 0.5
        
        thermal_power = actual_power_watts - display_power - modem_baseline
        thermal_power = max(MIN_THERMAL_POWER, thermal_power)  # Floor at 0.1W
        
        # Calculate total thermal velocity (sum of heating zones)
        total_velocity = 0.0
        zone_names = ['CPU_BIG', 'CPU_LITTLE', 'GPU', 'MODEM']
        
        for zone, vel in zone_velocities.items():
            zone_name = str(zone).split('.')[-1]
            if zone_name in zone_names and vel > 0:  # Only heating zones
                total_velocity += vel
        
        if total_velocity < 0.01:
            # System at equilibrium or cooling - can't distribute power
            return
        
        # Distribute power proportionally to velocity
        for zone, vel in zone_velocities.items():
            zone_name = str(zone).split('.')[-1]
            if zone_name not in zone_names:
                continue
            
            if vel > 0:
                # This zone's fraction of total heating
                fraction = vel / total_velocity
                zone_power = thermal_power * fraction
            else:
                # Cooling or stable - minimal power
                zone_power = 0.1
            
            # Update EMA
            alpha = POWER_LEARNING_RATE
            if self.sample_count == 0:
                # First sample - initialize
                self.power_ema[zone_name] = zone_power
            else:
                # EMA update
                self.power_ema[zone_name] = (
                    alpha * zone_power + (1 - alpha) * self.power_ema[zone_name]
                )
        
        self.sample_count += 1
        
        # Persist periodically
        current_time = time.time()
        if current_time - self.last_persist_time > POWER_LEARNING_PERSIST_INTERVAL:
            self._persist()
            self.last_persist_time = current_time
        
        if current_time - self.last_backup_time > POWER_LEARNING_BACKUP_INTERVAL:
            self._backup()
            self.last_backup_time = current_time
        
        # Update calibration factors if we have enough samples
        if self.sample_count >= POWER_LEARNING_MIN_SAMPLES:
            self._update_calibration_factors()
    
    def _update_calibration_factors(self) -> None:
        """
        Calculate calibration factors from learned vs velocity-based power.
        
        Calibration = learned_power_avg / velocity_power_avg
        
        Applied to velocity predictions: P_calibrated = P_velocity × calibration
        """
        # For now, use simple heuristic:
        # If learned power is stable (sample_count > window), use it to calibrate
        if self.sample_count < POWER_LEARNING_WINDOW:
            return
        
        for zone_name in self.calibration_factors.keys():
            learned = self.power_ema[zone_name]
            
            # Estimate what velocity-based method would predict
            # From ZONE_THERMAL_CONSTANTS, peak_power is now 8.0W
            # Assume velocity-based gives ~50% of peak on average
            velocity_predicted = ZONE_THERMAL_CONSTANTS[zone_name]['peak_power'] * 0.5
            
            if velocity_predicted > 0.1 and learned > 0.1:
                calibration = learned / velocity_predicted
                # Clamp to reasonable range (0.5x to 2.0x)
                calibration = max(0.5, min(calibration, 2.0))
                self.calibration_factors[zone_name] = calibration
    
    def get_calibration_factor(self, zone_name: str) -> float:
        """
        Get calibration factor for a zone.
        
        Returns 1.0 if learning not yet stable.
        """
        if self.sample_count < POWER_LEARNING_MIN_SAMPLES:
            return 1.0
        
        return self.calibration_factors.get(zone_name, 1.0)
    
    def _persist(self) -> None:
        """Persistence disabled - memory only"""
        pass
    
    def _backup(self) -> None:
        """Backup disabled - memory only"""
        pass
    
    def _cleanup_old_backups(self) -> None:
        """Remove backups older than retention period"""
        try:
            backup_dir = Path(POWER_LEARNING_BACKUP_DIR).expanduser()
            if not backup_dir.exists():
                return
            
            cutoff_time = time.time() - (POWER_LEARNING_BACKUP_RETENTION_DAYS * SECONDS_PER_DAY)
            
            for backup_file in backup_dir.glob('power_learned_*.json'):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    logger.debug(f"Removed old backup: {backup_file.name}")
        
        except Exception as e:
            logger.debug(f"Backup cleanup failed: {e}")
    
    def _load_from_disk(self) -> None:
        """Load persisted state on init"""
        if not POWER_LEARNING_ENABLED:
            return
        
        try:
            path = Path(POWER_LEARNING_FILE).expanduser()
            if not path.exists():
                return
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.power_ema = data.get('power_ema', self.power_ema)
            self.calibration_factors = data.get('calibration_factors', self.calibration_factors)
            self.sample_count = data.get('sample_count', 0)
            
            logger.info(f"Loaded power learning state: {self.sample_count} samples, "
                       f"calibrations: {self.calibration_factors}")
        
        except Exception as e:
            logger.error(f"Failed to load power learning: {e}")


# ============================================================================

class ThermalIntelligenceSystem:
    """
    Main thermal intelligence coordinator with hardware-accurate physics.
    Collects data, learns patterns, provides predictions.
    """
    
    def __init__(self):
        # Core components only
        self.telemetry = ThermalTelemetryCollector()
        
        # Power learning (must be created before physics engine)
        self.power_learning = PowerLearningSystem() if POWER_LEARNING_ENABLED else None
        
        self.physics = ZonePhysicsEngine(power_learning=self.power_learning)
        self.tank = ThermalTank(self.physics)  # Dual-condition throttle (battery + CPU velocity)
        
        # Data storage
        self.samples: Deque[ThermalSample] = deque(maxlen=THERMAL_HISTORY_SIZE)
        self.predictions: Deque[ThermalPrediction] = deque(maxlen=3)  # Only need last few for validation
        
        # State
        self.current_state = ThermalState.UNKNOWN
        self.last_update = 0
        self.update_interval = THERMAL_SAMPLE_INTERVAL
        
        # Monitoring
        self.monitor_task = None
        self.battery_update_task = None
        self.running = False
        
        # Validation tracking
        self.validation_enabled = False  # Enable with enable_validation()
        self.pending_validations: List[Tuple[float, ThermalZone, float, float]] = []  # (target_time, zone, predicted_temp, confidence)
        self.validation_metrics: Dict[str, ValidationMetrics] = {}
        self.validation_errors: Dict[str, deque] = {}  # Zone name -> deque of errors
        self.total_validated = 0
        self.validation_start_time = 0.0
        
        # Numpy array storage for predictions (26 columns like check_zones)
        self.predictions_array = np.zeros((MAX_PREDICTIONS, 26), dtype=np.float32)
        self.prediction_count = 0
        
        # Initialize validation metrics for each zone
        for zone in ThermalZone:
            zone_name = str(zone).split('.')[-1]
            if zone_name not in ['DISPLAY', 'CHARGER']:
                self.validation_metrics[zone_name] = ValidationMetrics()
                self.validation_errors[zone_name] = deque(maxlen=MAX_VALIDATION_SAMPLES)
        
        # Callback mechanism for piggybacking on poll cycle
        self.poll_callbacks: List[Callable] = []
        
        logger.info("Thermal Intelligence System  - Die backdate removed")
    
    async def start(self):
        """Start thermal monitoring and prediction"""
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.battery_update_task = asyncio.create_task(self._battery_update_loop())
        
        logger.info("Thermal monitoring started (monitor + battery loops)")
    
    async def stop(self):
        """Stop thermal monitoring"""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.battery_update_task:
            self.battery_update_task.cancel()
            try:
                await self.battery_update_task
            except asyncio.CancelledError:
                pass
        
        # Flush any remaining validation data before shutdown
        if self.validation_enabled and self.prediction_count > 0:
            try:
                self._flush_predictions()
                logger.info("Flushed validation data on shutdown")
            except Exception as e:
                logger.error(f"Failed to flush on shutdown: {e}")
        
        logger.info("Thermal monitoring stopped")
    
    def register_poll_callback(self, callback: Callable):
        """
        Register a callback to be invoked after each thermal poll cycle.
        
        This allows other systems (like the launcher supervisor) to piggyback
        on the thermal system's 30s polling cadence instead of running their
        own independent loops.
        
        Args:
            callback: Async callable that takes (sample, velocity, prediction, tank_status)
        """
        if callback not in self.poll_callbacks:
            self.poll_callbacks.append(callback)
            logger.info(f"Registered poll callback: {callback.__name__ if hasattr(callback, '__name__') else callback}")
            
            # Limit callback list to prevent unbounded growth
            if len(self.poll_callbacks) > 10:
                logger.warning(f"Poll callbacks exceeded 10 - removing oldest")
                self.poll_callbacks.pop(0)
    
    def unregister_poll_callback(self, callback: Callable):
        """Unregister a callback to prevent memory leaks"""
        if callback in self.poll_callbacks:
            self.poll_callbacks.remove(callback)
            logger.info(f"Unregistered poll callback: {callback.__name__ if hasattr(callback, '__name__') else callback}")
    
    def unregister_poll_callback(self, callback: Callable):
        """Remove a callback from the poll cycle"""
        if callback in self.poll_callbacks:
            self.poll_callbacks.remove(callback)
            logger.info(f"Unregistered poll callback: {callback.__name__ if hasattr(callback, '__name__') else callback}")
    
    def clear_poll_callbacks(self):
        """Clear all callbacks to prevent memory leaks"""
        self.poll_callbacks.clear()
        logger.info("Cleared all poll callbacks")
    
    # ============================================================================
    # VALIDATION METHODS
    # ============================================================================
    
    def enable_validation(self):
        """Enable prediction validation tracking"""
        self.validation_enabled = True
        self.validation_start_time = time.time()
        logger.info("Validation tracking enabled")
    
    def disable_validation(self):
        """Disable prediction validation tracking"""
        self.validation_enabled = False
        logger.info("Validation tracking disabled")
    
    def clear_validation_data(self):
        """Clear validation data to free memory"""
        self.pending_validations.clear()
        self.validation_errors.clear()
        self.predictions_array = np.zeros((MAX_PREDICTIONS, 26), dtype=np.float32)
        self.prediction_count = 0
        # Keep metrics but reset counts
        for metrics in self.validation_metrics.values():
            metrics.count = 0
        logger.info("Validation data cleared")
    
    def export_and_clear_validation(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """Export validation data then clear to free memory"""
        if not self.validation_enabled or self.total_validated == 0:
            return None
        
        path = self.export_validation_data(output_path)
        if path:
            self.clear_validation_data()
            logger.info(f"Exported and cleared validation data to {path}")
        return path
    
    def _store_pending_validation(self, prediction: ThermalPrediction):
        """
        Store prediction for later validation against actual temperatures.
        
        Args:
            prediction: The prediction to validate later
        """
        if not self.validation_enabled:
            return
        
        target_time = prediction.timestamp + prediction.horizon
        
        # Get current state for context
        current = self.samples[-1] if self.samples else None
        velocity = self.physics.calculate_velocity(list(self.samples)) if len(self.samples) >= 2 else None
        
        for zone, predicted_temp in prediction.predicted_temps.items():
            confidence = prediction.confidence_by_zone.get(zone, prediction.confidence)
            
            # Collect full context for numpy storage
            zone_name = str(zone).split('.')[-1]
            T0 = current.zones.get(zone, predicted_temp) if current else predicted_temp
            velocity_zone = velocity.zones.get(zone, 0.0) if velocity else 0.0
            
            # Get physics constants
            if zone_name in ZONE_THERMAL_CONSTANTS:
                constants = ZONE_THERMAL_CONSTANTS[zone_name]
                k_used = constants['ambient_coupling']
                C_used = constants['thermal_mass']
                tau_used = constants['time_constant']
                R_used = tau_used / C_used if C_used > 0 else 0.0
            else:
                R_used = k_used = C_used = tau_used = 0.0
            
            P_used = prediction.power_by_zone.get(zone, 0.0)
            
            # Get thermal state
            state_map = {'COLD': 0, 'OPTIMAL': 1, 'WARM': 2, 'HOT': 3, 'CRITICAL': 4, 'UNKNOWN': 5}
            trend_map = {'COOLING_FAST': 0, 'COOLING': 1, 'STABLE': 2, 'WARMING': 3, 'WARMING_FAST': 4, 'UNKNOWN': 5}
            state_id = state_map.get(self.current_state.name, 5)
            trend_id = trend_map.get(velocity.trend.name if velocity else 'UNKNOWN', 5)
            
            # Get chassis and ambient
            T_chassis = current.chassis if current else T0
            T_ambient = self.physics.fit_ambient_temperature(list(self.samples)) if len(self.samples) >= 3 else T0 - 7.0
            
            # Store with full context
            self.pending_validations.append((
                target_time, zone, predicted_temp, confidence,
                prediction.timestamp, prediction.horizon,
                state_id, trend_id, T0, T_chassis, T_ambient,
                velocity_zone, R_used, k_used, C_used, tau_used, P_used
            ))
        
        # CRITICAL: Enforce maximum size to prevent memory leak
        # Remove oldest entries if we exceed the limit
        if len(self.pending_validations) > MAX_PENDING_VALIDATIONS:
            # Remove oldest entries (FIFO)
            excess = len(self.pending_validations) - MAX_PENDING_VALIDATIONS
            self.pending_validations = self.pending_validations[excess:]
            logger.warning(f"Pending validations exceeded {MAX_PENDING_VALIDATIONS}, removed {excess} oldest entries")
    
    def _validate_pending(self, current_sample: ThermalSample):
        """
        Validate pending predictions against current actual temperatures.
        Also cleans up stale entries to prevent memory leaks.
        
        Extracts battery metrics (voltage, current, temp, percentage) from current_sample
        and calculates instantaneous power draw for discharge events. All data logged
        to validation array for analysis.
        
        Args:
            current_sample: The current temperature sample to validate against,
                          includes battery metrics from Termux API cache
        """
        if not self.validation_enabled or not self.pending_validations:
            return
        
        now = current_sample.timestamp
        validated_indices = []
        
        # Zone name mapping
        zone_names = ['CPU_BIG', 'CPU_LITTLE', 'GPU', 'BATTERY', 'MODEM', 'CHASSIS', 'AMBIENT']
        zone_to_id = {name: i for i, name in enumerate(zone_names)}
        
        for i, validation_data in enumerate(self.pending_validations):
            # Unpack full context
            (target_time, zone, predicted_temp, confidence,
             t_predict, horizon, state_id, trend_id, T0, T_chassis, T_ambient,
             velocity_zone, R_used, k_used, C_used, tau_used, P_used) = validation_data
            
            # CRITICAL: Remove stale entries to prevent memory leak
            # If prediction is older than VALIDATION_MAX_AGE, discard it
            age = now - t_predict
            if age > VALIDATION_MAX_AGE:
                validated_indices.append(i)
                continue
            
            # Check if prediction window has passed (within tolerance)
            if now >= target_time - VALIDATION_WINDOW:
                # Get actual temperature for this zone
                actual_temp = current_sample.zones.get(zone)
                
                # CRITICAL: Remove entry even if zone is missing to prevent leak
                if actual_temp is None:
                    # Zone missing from sample - discard this pending validation
                    validated_indices.append(i)
                    continue
                
                zone_name = str(zone).split('.')[-1]
                zone_id = zone_to_id.get(zone_name, -1)
                
                if zone_id < 0:
                    validated_indices.append(i)
                    continue
                
                # Calculate error
                error = predicted_temp - actual_temp
                abs_error = abs(error)
                
                # Store in numpy array if space available
                if self.prediction_count < MAX_PREDICTIONS:
                    # Extract battery metrics from current sample
                    battery_voltage = getattr(current_sample, 'battery_voltage', 0)
                    battery_current = getattr(current_sample, 'battery_current', 0)
                    battery_temp = getattr(current_sample, 'battery_temp', 0.0)
                    battery_pct = getattr(current_sample, 'battery_pct', 0)
                    
                    # Calculate power if we have the data (during discharge)
                    power_w = 0.0
                    if battery_voltage and battery_current and battery_current < 0:
                        # Current is negative during discharge
                        # mV * μA / 1,000,000,000 = W
                        power_w = (battery_voltage * abs(battery_current)) / 1_000_000_000
                    
                    # 26 columns: [t_predict, t_target, t_actual, horizon, zone_id,
                    #             temp_pred, temp_actual, error, abs_error, confidence,
                    #             state, trend, T0, T_chassis, T_ambient, velocity_zone,
                    #             R_used, k_used, C_used, tau_used, P_used,
                    #             power_w, voltage_mv, current_ua, battery_temp, battery_pct]
                    self.predictions_array[self.prediction_count] = [
                        t_predict, target_time, now, horizon, zone_id,
                        predicted_temp, actual_temp, error, abs_error, confidence,
                        state_id, trend_id, T0, T_chassis, T_ambient, velocity_zone,
                        R_used, k_used, C_used, tau_used, P_used,
                        power_w, battery_voltage or 0, battery_current or 0, battery_temp or 0.0, battery_pct or 0
                    ]
                    self.prediction_count += 1
                    
                    # Auto-flush when array fills to prevent memory leak
                    if self.prediction_count >= MAX_PREDICTIONS:
                        self._flush_predictions()
                
                # Store error for this zone
                if zone_name in self.validation_errors:
                    self.validation_errors[zone_name].append(abs_error)
                
                # Update metrics
                self._update_validation_metrics(zone_name, error, abs_error)
                
                self.total_validated += 1
                validated_indices.append(i)
        
        # Remove validated predictions (reverse order to maintain indices)
        for i in reversed(validated_indices):
            del self.pending_validations[i]
    
    def _flush_predictions(self):
        """
        Flush predictions array to disk and reset counter.
        
        Writes compressed numpy archive (.npz) containing:
        - predictions: 26-column array with thermal predictions, errors, physics parameters,
                      and battery metrics (voltage, current, temp, percentage, power)
        - zone_names: Thermal zone identifiers
        - column_names: Column labels for predictions array
        - total_validated: Cumulative prediction count
        - runtime: Total validation runtime in seconds
        
        Output directory: validation_data/validation_flush_{timestamp}.npz
        """
        if self.prediction_count == 0:
            return
        
        # Create validation_data directory if it doesn't exist
        output_dir = Path("validation_data")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"validation_flush_{timestamp}.npz"
        
        zone_names = ['CPU_BIG', 'CPU_LITTLE', 'GPU', 'BATTERY', 'MODEM', 'CHASSIS', 'AMBIENT']
        column_names = [
            't_predict', 't_target', 't_actual', 'horizon', 'zone_id',
            'temp_pred', 'temp_actual', 'error', 'abs_error', 'confidence',
            'state', 'trend', 'T0', 'T_chassis', 'T_ambient', 'velocity_zone',
            'R_used', 'k_used', 'C_used', 'tau_used', 'P_used',
            'power_w', 'voltage_mv', 'current_ua', 'battery_temp', 'battery_pct'
        ]
        
        np.savez_compressed(
            output_path,
            predictions=self.predictions_array[:self.prediction_count],
            zone_names=zone_names,
            column_names=column_names,
            total_validated=self.total_validated,
            runtime=time.time() - self.validation_start_time if self.validation_start_time > 0 else 0
        )
        
        logger.info(f"Flushed {self.prediction_count} predictions to {output_path}")
        
        # Reset array
        self.prediction_count = 0
    
    def _update_validation_metrics(self, zone_name: str, error: float, abs_error: float):
        """
        Update validation metrics for a zone.
        
        Args:
            zone_name: Name of the thermal zone
            error: Signed error (predicted - actual)
            abs_error: Absolute error
        """
        if zone_name not in self.validation_metrics:
            return
        
        metrics = self.validation_metrics[zone_name]
        n = metrics.count
        
        # Update count
        metrics.count += 1
        
        # Update running statistics using online algorithms
        # Mean error (bias)
        metrics.mean_error = (metrics.mean_error * n + error) / (n + 1)
        
        # MAE
        metrics.mae = (metrics.mae * n + abs_error) / (n + 1)
        
        # For RMSE, we need to track sum of squared errors
        old_sse = (metrics.rmse ** 2) * n if n > 0 else 0
        new_sse = old_sse + (error ** 2)
        metrics.rmse = math.sqrt(new_sse / (n + 1))
        
        # Max error
        metrics.max_error = max(metrics.max_error, abs_error)
        
        # Within thresholds
        if abs_error <= 1.0:
            metrics.within_1C += 1
        if abs_error <= 2.0:
            metrics.within_2C += 1
        if abs_error <= 3.0:
            metrics.within_3C += 1
        
        # Standard deviation - compute from stored errors
        if zone_name in self.validation_errors and len(self.validation_errors[zone_name]) > 1:
            errors_list = list(self.validation_errors[zone_name])
            metrics.std_error = statistics.stdev(errors_list) if len(errors_list) > 1 else 0.0
        
        metrics.last_update = time.time()
    
    def get_validation_report(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive validation report with per-zone metrics.
        
        Returns:
            Dictionary with validation metrics:
            {
                'total_validated': int,
                'runtime_seconds': float,
                'zones': {
                    'CPU_BIG': {
                        'count': int,
                        'mae': float,
                        'rmse': float,
                        'max_error': float,
                        'mean_error': float,
                        'std_error': float,
                        'within_1C': int,
                        'within_2C': int,
                        'within_3C': int,
                        'accuracy_pct': float,
                        'quality': str
                    },
                    ...
                },
                'overall_mae': float,
                'overall_quality': str
            }
        """
        if not self.validation_enabled or self.total_validated == 0:
            return None
        
        runtime = time.time() - self.validation_start_time if self.validation_start_time > 0 else 0
        
        zones_report = {}
        total_mae = 0.0
        zone_count = 0
        
        for zone_name, metrics in self.validation_metrics.items():
            if metrics.count == 0:
                continue
            
            # Calculate accuracy percentage (within 2°C)
            accuracy_pct = (metrics.within_2C / metrics.count * 100) if metrics.count > 0 else 0.0
            
            # Determine quality rating
            if metrics.mae < 0.5:
                quality = 'EXCELLENT'
            elif metrics.mae < 1.0:
                quality = 'GOOD'
            elif metrics.mae < 2.0:
                quality = 'FAIR'
            else:
                quality = 'POOR'
            
            zones_report[zone_name] = {
                'count': metrics.count,
                'mae': metrics.mae,
                'rmse': metrics.rmse,
                'max_error': metrics.max_error,
                'mean_error': metrics.mean_error,
                'std_error': metrics.std_error,
                'within_1C': metrics.within_1C,
                'within_2C': metrics.within_2C,
                'within_3C': metrics.within_3C,
                'accuracy_pct': accuracy_pct,
                'quality': quality
            }
            
            total_mae += metrics.mae
            zone_count += 1
        
        # Overall metrics
        overall_mae = total_mae / zone_count if zone_count > 0 else 0.0
        
        if overall_mae < 0.5:
            overall_quality = 'EXCELLENT'
        elif overall_mae < 1.0:
            overall_quality = 'GOOD'
        elif overall_mae < 2.0:
            overall_quality = 'FAIR'
        else:
            overall_quality = 'POOR'
        
        return {
            'total_validated': self.total_validated,
            'runtime_seconds': runtime,
            'pending_validations': len(self.pending_validations),
            'zones': zones_report,
            'overall_mae': overall_mae,
            'overall_quality': overall_quality
        }
    
    def print_validation_summary(self):
        """Print a formatted validation summary to console"""
        report = self.get_validation_report()
        
        if not report:
            print("Validation not enabled or no data collected")
            return
        
        print(f"\n{'='*80}")
        print(f"🐧 THERMAL PREDICTION VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Validated: {report['total_validated']:,}")
        print(f"Runtime: {report['runtime_seconds']/60:.1f} minutes")
        print(f"Pending: {report['pending_validations']:,}")
        print(f"Overall MAE: {report['overall_mae']:.2f}°C ({report['overall_quality']})")
        print(f"\n{'Zone':<12} {'Count':>7} {'MAE':>7} {'RMSE':>7} {'Max':>7} {'Bias':>7} "
              f"{'<1°C':>7} {'<2°C':>7} {'<3°C':>7} {'Quality'}")
        print('-' * 80)
        
        for zone_name in sorted(report['zones'].keys()):
            metrics = report['zones'][zone_name]
            
            pct_1C = (metrics['within_1C'] / metrics['count'] * 100) if metrics['count'] > 0 else 0
            pct_2C = (metrics['within_2C'] / metrics['count'] * 100) if metrics['count'] > 0 else 0
            pct_3C = (metrics['within_3C'] / metrics['count'] * 100) if metrics['count'] > 0 else 0
            
            print(f"{zone_name:<12} {metrics['count']:>7,} {metrics['mae']:>7.2f} "
                  f"{metrics['rmse']:>7.2f} {metrics['max_error']:>7.2f} {metrics['mean_error']:>+7.2f} "
                  f"{pct_1C:>6.1f}% {pct_2C:>6.1f}% {pct_3C:>6.1f}% {metrics['quality']}")
        
        print(f"{'='*80}\n")
    
    def export_validation_data(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Export validation data to JSON file for analysis.
        
        Args:
            output_path: Optional path for output file. If None, uses default.
        
        Returns:
            Path to exported file, or None if validation not enabled
        """
        if not self.validation_enabled or self.total_validated == 0:
            logger.warning("Cannot export: validation not enabled or no data")
            return None
        
        report = self.get_validation_report()
        
        if not report:
            return None
        
        # Default output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"validation_report_{timestamp}.json")
        
        # Export to JSON
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Validation data exported to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to export validation data: {e}")
            return None
    
    def save_validation_results(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Save validation results to .npz file (same format as check_zones.py).
        
        Args:
            output_path: Optional path for output file. If None, uses default.
        
        Returns:
            Path to saved file, or None if validation not enabled
        """
        if not self.validation_enabled or self.prediction_count == 0:
            logger.warning("Cannot save: validation not enabled or no predictions")
            return None
        
        # Default output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"validation_results_{timestamp}.npz")
        
        try:
            # Zone names for reference
            zone_names = ['CPU_BIG', 'CPU_LITTLE', 'GPU', 'BATTERY', 'MODEM', 'CHASSIS', 'AMBIENT']
            
            # Column names
            column_names = [
                't_predict', 't_target', 't_actual', 'horizon', 'zone_id',
                'temp_pred', 'temp_actual', 'error', 'abs_error', 'confidence',
                'state', 'trend', 'T0', 'T_chassis', 'T_ambient', 'velocity_zone',
                'R_used', 'k_used', 'C_used', 'tau_used', 'P_used',
                'power_w', 'voltage_mv', 'current_ua', 'battery_temp', 'battery_pct'
            ]
            
            # Save compressed
            np.savez_compressed(
                output_path,
                predictions=self.predictions_array[:self.prediction_count],
                zone_names=zone_names,
                column_names=column_names,
                total_validated=self.total_validated,
                runtime=time.time() - self.validation_start_time if self.validation_start_time > 0 else 0
            )
            
            logger.info(f"Validation results saved to {output_path} ({self.prediction_count:,} predictions)")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
            return None
    
    def save_checkpoint(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Save intermediate checkpoint of validation data.
        Alias for save_validation_results for compatibility.
        
        Args:
            output_path: Optional path for output file
        
        Returns:
            Path to saved file, or None if failed
        """
        return self.save_validation_results(output_path)
    
    def get_zone_validation_errors(self, zone_name: str) -> Optional[List[float]]:
        """
        Get raw validation errors for a specific zone.
        
        Args:
            zone_name: Name of the thermal zone (e.g., 'CPU_BIG')
        
        Returns:
            List of absolute errors, or None if zone not found
        """
        if zone_name in self.validation_errors:
            return list(self.validation_errors[zone_name])
        return None
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation system statistics for monitoring memory usage.
        
        Returns:
            Dictionary with validation system stats:
            {
                'enabled': bool,
                'pending_validations': int,
                'total_validated': int,
                'predictions_stored': int,
                'predictions_capacity': int,
                'memory_usage_mb': float
            }
        """
        # Estimate memory usage
        pending_size = len(self.pending_validations) * 17 * 8  # 17 values × 8 bytes each
        array_size = self.predictions_array.nbytes
        errors_size = sum(len(deq) * 8 for deq in self.validation_errors.values())
        total_mb = (pending_size + array_size + errors_size) / (1024 * 1024)
        
        return {
            'enabled': self.validation_enabled,
            'pending_validations': len(self.pending_validations),
            'total_validated': self.total_validated,
            'predictions_stored': self.prediction_count,
            'predictions_capacity': MAX_PREDICTIONS,
            'memory_usage_mb': total_mb,
            'max_pending_limit': MAX_PENDING_VALIDATIONS,
            'max_age_seconds': VALIDATION_MAX_AGE
        }
    
    async def _battery_update_loop(self):
        """
        Independent background loop to update battery/network/brightness cache.
        Runs at API_UPDATE_INTERVAL (5s) to keep caches fresh without blocking monitor loop.
        """
        # Initial sleep to stagger from monitor loop
        await asyncio.sleep(API_UPDATE_INTERVAL)
        
        while self.running:
            try:
                # Update all API caches in parallel
                tasks = [
                    self.telemetry._read_battery_status(),
                    self.telemetry._detect_network_type(),
                    self.telemetry._get_display_brightness()
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                timestamp = time.time()
                
                # Update prediction cache (used by monitor loop)
                if not isinstance(results[0], Exception) and results[0]:
                    self.telemetry.pred_battery = results[0]
                    self.telemetry.pred_battery_time = timestamp
                
                if not isinstance(results[1], Exception):
                    self.telemetry.pred_network = results[1]
                    self.telemetry.pred_network_time = timestamp
                
                if not isinstance(results[2], Exception) and results[2] is not None:
                    self.telemetry.pred_brightness = results[2]
                    self.telemetry.pred_brightness_time = timestamp
                
                await asyncio.sleep(API_UPDATE_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Battery update loop error: {e}")
                await asyncio.sleep(API_UPDATE_INTERVAL)
    
    async def _monitor_loop(self):
        """Main monitoring loop - uniform sampling at THERMAL_SAMPLE_INTERVAL (1s)"""
        
        while self.running:
            try:
                # Collect sample at configured interval (with timeout to prevent hanging)
                try:
                    sample = await asyncio.wait_for(
                        self.telemetry.collect_sample(for_prediction=True),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.error("Sample collection timed out after 10s")
                    await asyncio.sleep(self.update_interval)
                    continue
                
                self.samples.append(sample)
                
                # Validate pending predictions against this new sample
                if self.validation_enabled:
                    self._validate_pending(sample)
                
                velocity = None
                prediction = None
                tank_status = None
                
                # Validate against old predictions if we have any
                if self.predictions and len(self.samples) >= MIN_SAMPLES_FOR_PREDICTIONS:
                    # Check if current sample matches a previous prediction's target time
                    # Prediction horizon is 30s, so look for predictions ~30s ago
                    for pred in list(self.predictions):
                        time_diff = abs(sample.timestamp - (pred.timestamp + THERMAL_PREDICTION_HORIZON))
                        
                        # If within 1s tolerance, this sample validates that prediction
                        if time_diff < SAMPLE_STALENESS_THRESHOLD:
                            # Match found but no tuning needed anymore
                            break  # Only validate one prediction per sample
                
                # Need minimum samples for predictions
                if len(self.samples) >= MIN_SAMPLES_FOR_PREDICTIONS:
                    # Calculate velocity
                    velocity = self.physics.calculate_velocity(list(self.samples))
                    
                    # Update power learning (only during discharge)
                    if self.power_learning and not sample.charging:
                        # Get battery voltage and current from sample
                        battery_voltage = getattr(sample, 'battery_voltage', None)
                        battery_current = getattr(sample, 'battery_current', None)
                        
                        if battery_voltage and battery_current and battery_current < 0:
                            # Current is negative during discharge
                            # Calculate actual power: V × I (current in μA, so divide by 1M)
                            actual_power = (battery_voltage * abs(battery_current)) / 1_000_000
                            
                            # Update learning with actual power and zone velocities
                            self.power_learning.update(actual_power, sample.zones, velocity.zones)
                    
                    # Generate prediction
                    if THERMAL_PREDICTION_ENABLED:
                        prediction = self.physics.predict_temperature(
                            sample, velocity, THERMAL_PREDICTION_HORIZON, list(self.samples)
                        )
                        self.predictions.append(prediction)
                        
                        # Store prediction for validation
                        if self.validation_enabled:
                            self._store_pending_validation(prediction)
                    
                    # Get tank status for callbacks
                    if velocity and prediction:
                        tank_status = self.tank.get_status(sample, velocity, prediction)
                
                # Update thermal state
                self._update_thermal_state(sample)
                
                # Invoke registered callbacks (for launcher supervisor, etc.)
                if self.poll_callbacks:
                    for callback in self.poll_callbacks[:]:  # Copy list to avoid modification during iteration
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(sample, velocity, prediction, tank_status)
                            else:
                                callback(sample, velocity, prediction, tank_status)
                        except Exception as e:
                            logger.error(f"Poll callback error in {callback}: {e}")
                            # Clear reference to exception to prevent holding callback
                            del e
                
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                import traceback
                logger.error(f"Monitor loop traceback:\n{traceback.format_exc()}")
                # Don't delete exception reference - we want the full trace
                await asyncio.sleep(self.update_interval)  # Continue after error
    
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
        
        Color thresholds (PNGN_32 team colors):
        - Green: < 35°C (cool) - #00FF7F Radiation Green
        - Light Green: 35-38°C (warm) - #00FF00 Neon Green
        - Purple: 38-42°C (warning) - #CC33FF PNGN Purple
        - Pink: 42-45°C (hot) - #FF1493 Deep Pink
        - Red: > 45°C (critical) - #FF0080 Neon Red
        
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
            
            # Color based on temperature (purple/green theme from PNGN_32.html)
            if temp < TEMP_DISPLAY_COOL:
                color = '#00FF7F'      # Radiation Green (index 18)
                color_name = 'green'
            elif temp < TEMP_DISPLAY_WARM:
                color = '#00FF00'      # Neon Green (index 16)
                color_name = 'light_green'
            elif temp < TEMP_DISPLAY_WARNING:
                color = '#CC33FF'      # PNGN Purple (index 0)
                color_name = 'purple'
            elif temp < TEMP_DISPLAY_HOT:
                color = '#FF1493'      # Deep Pink (index 10)
                color_name = 'pink'
            else:
                color = '#FF0080'      # Neon Red (index 12)
                color_name = 'red'
            
            # Trend from velocity
            vel = velocity.zones.get(zone, 0.0)
            if vel > VELOCITY_TREND_RAPID_WARMING:
                trend = '↑↑'
            elif vel > VELOCITY_TREND_WARMING:
                trend = '↑'
            elif vel < VELOCITY_TREND_RAPID_COOLING:
                trend = '↓↓'
            elif vel < -VELOCITY_TREND_WARMING:
                trend = '↓'
            else:
                trend = '→'
            
            # Confidence if available
            conf = prediction.confidence_by_zone.get(zone, PREDICTION_CONFIDENCE_FALLBACK) if prediction else PREDICTION_CONFIDENCE_FALLBACK
            
            zones[zone_name] = {
                'temp': temp,
                'color': color,
                'color_name': color_name,
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
    
    def get_accuracy_report(self, actual_sample: Optional[ThermalSample] = None) -> Optional[Dict[str, Any]]:
        """
        Get prediction accuracy report with zone-specific color coding.
        
        Compares the most recent prediction against actual temperature to show
        accuracy metrics with proper PNGN_32 team color coding based on zone type.
        
        Args:
            actual_sample: Optional actual temperature sample to compare against.
                          If None, uses current sample (assumes prediction was for current time)
        
        Returns:
            Dictionary with per-zone accuracy metrics:
            {
                'zones': {
                    'CPU_BIG': {
                        'predicted': 72.5,
                        'actual': 71.8,
                        'error': 0.7,
                        'rating': 'excellent',
                        'color': '#00FF7F',
                        'label': 'EXCELLENT'
                    },
                    ...
                },
                'overall_mae': 1.2,
                'overall_rating': 'good',
                'overall_color': '#00FF00',
                'overall_label': 'GOOD',
                'prediction_horizon': 30.0,
                'timestamp': 1234567890.0
            }
        """
        if not self.predictions or not self.samples:
            return None
        
        # Get the most recent prediction and actual sample
        prediction = self.predictions[-1]
        actual = actual_sample if actual_sample else self.samples[-1]
        
        # Calculate per-zone errors
        zone_errors = {}
        total_error = 0.0
        zone_count = 0
        
        for zone, predicted_temp in prediction.predicted_temps.items():
            zone_name = str(zone).split('.')[-1]
            
            # Skip non-hardware zones
            if zone_name in ['DISPLAY', 'CHARGER']:
                continue
            
            # Get actual temperature for this zone
            actual_temp = actual.zones.get(zone)
            if actual_temp is None:
                continue
            
            # Calculate absolute error
            error = abs(predicted_temp - actual_temp)
            total_error += error
            zone_count += 1
            
            # Get zone-specific accuracy rating and color
            rating, color, label = get_accuracy_rating(zone_name, error)
            
            zone_errors[zone_name] = {
                'predicted': predicted_temp,
                'actual': actual_temp,
                'error': error,
                'rating': rating,
                'color': color,
                'label': label
            }
        
        # Calculate overall metrics
        overall_mae = total_error / zone_count if zone_count > 0 else 0.0
        
        # Determine overall rating (use CPU_BIG as reference for overall)
        overall_rating, overall_color, overall_label = get_accuracy_rating('CPU_BIG', overall_mae)
        
        return {
            'zones': zone_errors,
            'overall_mae': overall_mae,
            'overall_rating': overall_rating,
            'overall_color': overall_color,
            'overall_label': overall_label,
            'prediction_horizon': THERMAL_PREDICTION_HORIZON,
            'timestamp': actual.timestamp
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

logger.info("S25+ Thermal Intelligence System  loaded")
logger.info(f"Hardware constants: {len(ZONE_THERMAL_CONSTANTS)} zones configured")
logger.info("Features: Battery-only backdating | Dual-confidence | 10K history | Damped chassis")
