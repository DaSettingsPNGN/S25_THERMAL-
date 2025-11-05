#!/usr/bin/env python3
"""
🔥🐧🔥 S25+ Thermal Intelligence Configuration
=========================================
Copyright (c) 2025 PNGN-Tec LLC

Configuration constants for Samsung Galaxy S25+ thermal monitoring.
These are defined in s25_thermal.py - this file documents them for reference.

To modify: Edit constants directly in s25_thermal.py (lines 64-241)
"""

# ============================================================================
# CORE PREDICTION PARAMETERS
# ============================================================================

CHASSIS_DAMPING_FACTOR = 0.90           # Chassis thermal inertia
THERMAL_PREDICTION_HORIZON = 30.0       # Seconds ahead (not 60!)
THERMAL_SAMPLE_INTERVAL_MS = 10000      # 10s uniform sampling
THERMAL_HISTORY_SIZE = 300              # 50 min history
MIN_SAMPLES_FOR_PREDICTIONS = 12        # 2 min minimum

# ============================================================================
# ADAPTIVE DAMPING HISTORY SIZES
# ============================================================================

DAMPING_HISTORY_SLOW_ZONES = 10         # Battery: 20 min
DAMPING_HISTORY_FAST_ZONES = 1          # CPU/GPU: 2 min
DAMPING_HISTORY_MEDIUM_ZONES = 2        # Chassis: 4 min

# ============================================================================
# TIMEOUTS
# ============================================================================

THERMAL_SUBPROCESS_TIMEOUT = 2.0        # seconds
THERMAL_NETWORK_TIMEOUT = 3.0           # seconds

# ============================================================================
# NETWORK AWARENESS
# ============================================================================

THERMAL_NETWORK_AWARENESS_ENABLED = True
THERMAL_WIFI_5G_FREQ_MIN = 5000         # MHz

# ============================================================================
# CONFIDENCE
# ============================================================================

CONFIDENCE_SAFETY_SCALE = 0.5
THERMAL_SENSOR_CONFIDENCE_REDUCED = 0.3

# ============================================================================
# SAMSUNG S25+ HARDWARE CONSTANTS
# ============================================================================

S25_PLUS_BATTERY_CAPACITY = 4900        # mAh
S25_PLUS_BATTERY_INTERNAL_RESISTANCE = 0.150  # Ohms
S25_PLUS_VAPOR_CHAMBER_EFFICIENCY = 0.85
S25_PLUS_SCREEN_SIZE = 6.7              # inches

# ============================================================================
# PER-ZONE THERMAL CONSTANTS
# ============================================================================

ZONE_THERMAL_CONSTANTS = {
    'CPU_BIG': {
        'thermal_mass': 0.025,           # J/K
        'thermal_resistance': 2.8,       # °C/W
        'ambient_coupling': 0.80,
        'peak_power': 6.0,               # W
        'idle_power': 0.1,
    },
    'CPU_LITTLE': {
        'thermal_mass': 0.020,
        'thermal_resistance': 3.0,
        'ambient_coupling': 0.75,
        'peak_power': 3.0,
        'idle_power': 0.05,
    },
    'GPU': {
        'thermal_mass': 0.030,
        'thermal_resistance': 3.2,
        'ambient_coupling': 0.75,
        'peak_power': 8.0,
        'idle_power': 0.2,
    },
    'MODEM': {
        'thermal_mass': 0.015,
        'thermal_resistance': 4.0,
        'ambient_coupling': 0.70,
        'peak_power': 4.0,
        'idle_power': 0.5,
    },
    'BATTERY': {
        'thermal_mass': 25.0,
        'thermal_resistance': 10.0,
        'ambient_coupling': 0.30,
        'peak_power': 0.0,
        'idle_power': 0.0,
    },
}

# ============================================================================
# TEMPERATURE THRESHOLDS
# ============================================================================

THERMAL_TEMP_COLD = 20.0                # °C
THERMAL_TEMP_OPTIMAL_MIN = 25.0
THERMAL_TEMP_OPTIMAL_MAX = 38.0
THERMAL_TEMP_WARM = 40.0
THERMAL_TEMP_HOT = 42.0                 # Samsung throttle point
THERMAL_TEMP_CRITICAL = 45.0

# ============================================================================
# HYSTERESIS (PREVENTS STATE FLAPPING)
# ============================================================================

THERMAL_HYSTERESIS_UP = 1.0             # °C
THERMAL_HYSTERESIS_DOWN = 2.0           # °C

# ============================================================================
# VELOCITY THRESHOLDS
# ============================================================================

THERMAL_VELOCITY_RAPID_COOLING = -0.10  # °C/s
THERMAL_VELOCITY_COOLING = -0.02
THERMAL_VELOCITY_WARMING = 0.05
THERMAL_VELOCITY_RAPID_WARMING = 0.15

# ============================================================================
# SENSOR VALIDATION
# ============================================================================

THERMAL_SENSOR_TEMP_MIN = 15.0          # °C sanity check
THERMAL_SENSOR_TEMP_MAX = 75.0          # °C sanity check

# ============================================================================
# FEATURES
# ============================================================================

THERMAL_PREDICTION_ENABLED = True

# ============================================================================
# NOTES
# ============================================================================

# Pattern learning: REMOVED (not in current code)
# Command tracking: REMOVED (not in current code)
# 60s horizon: CHANGED to 30s
# Multi-zone weights: REMOVED (battery-centric now)
