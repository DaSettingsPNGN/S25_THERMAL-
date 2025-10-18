#!/usr/bin/env python3
"""
🐧 S25+ Thermal Intelligence Configuration
=========================================
Copyright (c) 2025 PNGN-Tec LLC
Author: Jesse Vogeler-Wunsch (@DaSettingsPNGN)

Thermal Monitoring Configuration Constants
==========================================
Complete configuration for thermal monitoring, prediction, and pattern
recognition on Android devices.

Configuration Categories:
- Temperature thresholds and hysteresis
- Velocity and acceleration thresholds
- Sampling intervals and history management
- Statistical analysis parameters
- Pattern recognition and learning rates
- Persistence and caching settings
- S25+ thermal characteristics
- Termux API integration

S25+ Thermal Characteristics:
- Thermal mass: 50.0 J/°C (heat capacity)
- Thermal resistance: 5.0°C/W (cooling efficiency)
- Ambient coupling: 0.3 (heat transfer coefficient)
- Max TDP: 15.0 W (Snapdragon 8 Elite thermal design power)

Thermal States:
- COLD: < 35°C (device cool)
- OPTIMAL: 35-45°C (normal operation)
- WARM: 45-55°C (elevated temperature)
- HOT: 55-65°C (throttling may occur)
- CRITICAL: > 60°C (heavy throttling)

Network Thermal Impact:
- WiFi 2.4GHz: +0°C
- WiFi 5GHz: +1°C
- Mobile 3G: +2°C
- Mobile 4G: +3°C
- Mobile 5G: +5°C (significant thermal load)

Usage:
    from config import (
        THERMAL_TEMP_CRITICAL,
        THERMAL_ZONES,
        S25_THERMAL_MASS
    )

Version: 1.0.0
Platform: Android (Termux)
"""

import os

# ============================================================================
# THERMAL ZONES (Android sysfs paths)
# ============================================================================

# Note: Battery path may not exist on all devices - code falls back to Termux API

THERMAL_ZONES = {
    'cpu_big': '/sys/class/thermal/thermal_zone0/temp',
    'cpu_little': '/sys/class/thermal/thermal_zone1/temp',
    'gpu': '/sys/class/thermal/thermal_zone2/temp',
    'battery': '/sys/class/power_supply/battery/temp',
    'skin': '/sys/class/thermal/thermal_zone3/temp',
    'modem': '/sys/class/thermal/thermal_zone4/temp',
    'npu': '/sys/class/thermal/thermal_zone5/temp',
    'camera': '/sys/class/thermal/thermal_zone6/temp',
    'ambient': '/sys/class/thermal/thermal_zone7/temp'
}

THERMAL_ZONE_NAMES = {
    'cpu_big': 'CPU Performance Cores',
    'cpu_little': 'CPU Efficiency Cores',
    'gpu': 'GPU (Adreno 830)',
    'battery': 'Battery',
    'skin': 'Device Surface',
    'modem': '5G Modem',
    'npu': 'AI Engine',
    'camera': 'Camera'
}

THERMAL_ZONE_WEIGHTS = {
    'cpu_big': 0.30,
    'cpu_little': 0.20,
    'gpu': 0.20,
    'battery': 0.15,
    'skin': 0.10,
    'modem': 0.05,
    'npu': 0.00,
    'camera': 0.00
}

# ============================================================================
# SAMPLING
# ============================================================================

THERMAL_SAMPLE_INTERVAL_MS = 10000
THERMAL_HISTORY_SIZE = 1000
THERMAL_PREDICTION_HORIZON = 60.0

# ============================================================================
# NETWORK IMPACT
# ============================================================================

NETWORK_THERMAL_IMPACT = {
    'WIFI_2G': 0.0,
    'WIFI_5G': 1.0,
    'MOBILE_3G': 2.0,
    'MOBILE_4G': 3.0,
    'MOBILE_5G': 5.0
}

# ============================================================================
# STATISTICAL THRESHOLDS
# ============================================================================

THERMAL_PERCENTILE_WINDOWS = [5, 25, 75, 95]
THERMAL_ANOMALY_THRESHOLD = 3.0

# ============================================================================
# PATTERN RECOGNITION
# ============================================================================

THERMAL_SIGNATURE_WINDOW = 300
THERMAL_CORRELATION_THRESHOLD = 0.7

# ============================================================================
# TERMUX API
# ============================================================================

TERMUX_BATTERY_STATUS_CMD = ['termux-battery-status']
TERMUX_WIFI_INFO_CMD = ['termux-wifi-connectioninfo']
TERMUX_TELEPHONY_INFO_CMD = ['termux-telephony-deviceinfo']
TERMUX_SENSORS_CMD = ['termux-sensor', '-s', 'all', '-n', '1']

# ============================================================================
# FEATURES
# ============================================================================

THERMAL_PREDICTION_ENABLED = True
THERMAL_PATTERN_RECOGNITION_ENABLED = True
THERMAL_NETWORK_AWARENESS_ENABLED = True

# ============================================================================
# S25+ THERMAL CHARACTERISTICS
# ============================================================================

S25_THERMAL_MASS = 50.0
S25_THERMAL_RESISTANCE = 5.0
S25_AMBIENT_COUPLING = 0.3
S25_MAX_TDP = 15.0

# ============================================================================
# TEMPERATURE THRESHOLDS
# ============================================================================

THERMAL_TEMP_COLD = 35.0
THERMAL_TEMP_OPTIMAL_MIN = 35.0
THERMAL_TEMP_OPTIMAL_MAX = 45.0
THERMAL_TEMP_WARM = 55.0
THERMAL_TEMP_HOT = 65.0
THERMAL_TEMP_CRITICAL = 60.0

# ============================================================================
# HYSTERESIS
# ============================================================================

THERMAL_HYSTERESIS_UP = 2.0
THERMAL_HYSTERESIS_DOWN = 2.0

# ============================================================================
# TIMEOUTS
# ============================================================================

THERMAL_SUBPROCESS_TIMEOUT = 2.0
THERMAL_TELEMETRY_TIMEOUT = 5.0
THERMAL_SHUTDOWN_TIMEOUT = 5.0
THERMAL_NETWORK_TIMEOUT = 2.0

# ============================================================================
# PROCESSING
# ============================================================================

THERMAL_TELEMETRY_BATCH_SIZE = 100
THERMAL_TELEMETRY_PROCESSING_INTERVAL = 1.0
THERMAL_SIGNATURE_MAX_COUNT = 1000
THERMAL_LEARNING_RATE = 0.1
THERMAL_SIGNATURE_MIN_DELTA = 0.1
THERMAL_COMMAND_HASH_LENGTH = 8

# ============================================================================
# PERSISTENCE
# ============================================================================

THERMAL_PERSISTENCE_INTERVAL = 300
THERMAL_PERSISTENCE_KEY = 'thermal_signatures'
THERMAL_PERSISTENCE_FILE = 'thermal_signatures.json'  # Simplified path for portability

# ============================================================================
# VELOCITY THRESHOLDS
# ============================================================================

THERMAL_VELOCITY_RAPID_COOLING = -2.0 / 60
THERMAL_VELOCITY_COOLING = -0.5 / 60
THERMAL_VELOCITY_WARMING = 0.5 / 60
THERMAL_VELOCITY_RAPID_WARMING = 2.0 / 60

# ============================================================================
# CONFIDENCE CALCULATIONS
# ============================================================================

THERMAL_MIN_SAMPLES_CONFIDENCE = 50
THERMAL_PREDICTION_CONFIDENCE_DECAY = 300.0
THERMAL_SENSOR_TEMP_MIN = -20.0
THERMAL_SENSOR_TEMP_MAX = 100.0
THERMAL_SENSOR_CONFIDENCE_REDUCED = 0.7

# ============================================================================
# NETWORK DETECTION
# ============================================================================

THERMAL_WIFI_5G_FREQ_MIN = 5000

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

THERMAL_BUDGET_WARNING_SECONDS = 60
THERMAL_NETWORK_IMPACT_WARNING = 3.0
THERMAL_CHARGING_IMPACT_WARNING = 5.0
THERMAL_COMMAND_IMPACT_WARNING = 5.0