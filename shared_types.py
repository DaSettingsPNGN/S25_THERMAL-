#!/usr/bin/env python3
"""
🐧 S25+ Thermal Intelligence Type Definitions
============================================
Copyright (c) 2025 PNGN-Tec LLC

Shared type system for thermal monitoring. Platform-agnostic enums and
dataclasses used across thermal intelligence modules.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum, auto, IntEnum
import time

# ============================================================================
# ENUMS
# ============================================================================

class MemoryPressureLevel(IntEnum):
    """Memory pressure levels for monitoring"""
    NORMAL = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3

class ThermalZone(Enum):
    """Thermal zones monitored on device"""
    CPU_BIG = auto()
    CPU_LITTLE = auto()
    GPU = auto()
    BATTERY = auto()
    MODEM = auto()
    AMBIENT = auto()

class ThermalState(Enum):
    """Thermal states for device monitoring"""
    COLD = auto()
    OPTIMAL = auto()
    WARM = auto()
    HOT = auto()
    CRITICAL = auto()
    UNKNOWN = auto()
    
    @property
    def is_throttling(self) -> bool:
        """Check if thermal state causes throttling"""
        return self in (ThermalState.HOT, ThermalState.CRITICAL)
    
    @property
    def allows_processing(self) -> bool:
        """Check if thermal state allows normal processing"""
        return self not in (ThermalState.CRITICAL,)

class ThermalTrend(Enum):
    """Temperature change trends"""
    RAPID_COOLING = auto()
    COOLING = auto()
    STABLE = auto()
    WARMING = auto()
    RAPID_WARMING = auto()
    
    @property
    def is_improving(self) -> bool:
        """Check if thermal trend is improving"""
        return self in (ThermalTrend.COOLING, ThermalTrend.RAPID_COOLING)

class NetworkType(Enum):
    """Network connection types with thermal impact"""
    UNKNOWN = auto()
    OFFLINE = auto()
    WIFI_2G = auto()
    WIFI_5G = auto()
    MOBILE_3G = auto()
    MOBILE_4G = auto()
    MOBILE_5G = auto()

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
# EXPORT ALL PUBLIC TYPES
# ============================================================================

__all__ = [
    # Enums
    'MemoryPressureLevel',
    'ThermalZone',
    'ThermalState',
    'ThermalTrend',
    'NetworkType',
    
    # Data structures
    'ThermalSample',
    'ThermalVelocity',
    'ThermalPrediction',
    'ThermalSignature',
    'ThermalStatistics',
    'ThermalIntelligence',
    'ThermalEvent',
]
