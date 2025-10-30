#!/usr/bin/env python3
"""
🐧 S25+ Thermal Intelligence Configuration
=========================================
Copyright (c) 2025 PNGN-Tec LLC

Thermal monitoring configuration for Samsung Galaxy S25+ (Snapdragon 8 Elite).
Hardware-specific constants derived from teardowns and thermal testing.
"""

import os

# ============================================================================
# THERMAL ZONES (Android sysfs paths)
# ============================================================================

THERMAL_ZONES = {
    'cpu_big': '/sys/class/thermal/thermal_zone0/temp',
    'cpu_little': '/sys/class/thermal/thermal_zone1/temp',
    'gpu': '/sys/class/thermal/thermal_zone2/temp',
    'battery': '/sys/class/power_supply/battery/temp',
    'modem': '/sys/class/thermal/thermal_zone4/temp',
}

THERMAL_ZONE_NAMES = {
    'cpu_big': 'CPU Performance Cores',
    'cpu_little': 'CPU Efficiency Cores',
    'gpu': 'GPU (Adreno 830)',
    'battery': 'Battery',
    'modem': '5G Modem',
}

THERMAL_ZONE_WEIGHTS = {
    'cpu_big': 0.30,
    'cpu_little': 0.25,
    'gpu': 0.25,
    'battery': 0.15,
    'modem': 0.05,
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
THERMAL_PERSISTENCE_FILE = 'thermal_signatures.json'

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
