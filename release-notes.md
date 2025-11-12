# 🔥🐧🔥 v2.25 - Dual-Throttle + Observed Peak Prediction

**Major update:** Dual-condition throttle system (battery temp + CPU velocity) and empirical peak predictions for throttled regimes.

---

## The Problem

**v2.24 had critical issues:**

1. **Physics breaks when throttled** - Model under-predicted by -30 to -60°C when components hit throttle temps
2. **No early warning for regime changes** - Battery (τ=540s) reacts slowly, CPUs (τ=14-19s) spike first
3. **Wrong throttle assumptions** - Thought CPU_BIG throttled at 60-65°C, actually starts at 45°C

**Result:** Catastrophic prediction failures during workload spikes.

---

## The Solution

### 1. Dual-Condition Throttle

**Condition 1: Battery Temperature (existing)**
```python
if battery_predicted >= 38.5°C:
    throttle_reason = BATTERY_TEMP
```

**Condition 2: CPU Velocity Spike (NEW)**
```python
if cpu_big_velocity > 1.0°C/s or cpu_little_velocity > 1.0°C/s:
    throttle_reason = CPU_VELOCITY
```

**Why:** Physics can't predict discontinuities. Velocity >1.0°C/s indicates regime change (workload spike, thermal runaway). Throttle immediately before model breaks.

**Validation data:**
- Normal velocity: <0.4°C/s (P90)
- Danger threshold: >1.0°C/s (P95+)
- Max observed: CPU_BIG 9.5°C/s, CPU_LITTLE 14.0°C/s

### 2. Observed Peak Prediction

**Instead of physics when throttled:**
```python
if current_temp >= throttle_start:
    predicted_temp = observed_peak  # From validation data
    # Skip Newton's law - it's wrong in this regime
```

**Observed peaks from validation:**
- CPU_BIG: 79.1°C (starts throttling at 45°C)
- CPU_LITTLE: 93.4°C (starts throttling at 48°C)
- GPU: 58.1°C (starts throttling at 38°C)
- MODEM: 59.7°C (starts throttling at 40°C)

**Why:** Model assumes constant power. Throttling changes power mid-flight, breaking the model. Use empirical data instead.

### 3. Corrected Throttle Curves

**Reality vs assumptions:**
```
Component    | Old Start | New Start | Observed Peak
-------------|-----------|-----------|---------------
CPU_BIG      | 60-65°C   | 45°C      | 79.1°C
CPU_LITTLE   | 75°C      | 48°C      | 93.4°C
GPU          | 40-55°C   | 38°C      | 58.1°C
MODEM        | 38°C      | 40°C      | 59.7°C
```

**Source:** Validation data + Snapdragon 8 Elite research (mainstream phones throttle to <30% of peak to maintain <44°C touch temp).

---

## Results

**Prediction accuracy:**
- Normal regime: ~1.5°C MAE (unchanged)
- Throttled regime: Use observed peaks → no more -30 to -60°C errors

**Throttle detection:**
- Battery-based: Prevents Samsung's 40-42°C throttle
- Velocity-based: Catches regime changes before physics breaks
- Combined: Proactive protection from two independent failure modes

**Production:** 24/7 Discord bot, 645+ members, Samsung S25+, zero thermal shutdowns.

---

## API Changes

**ThermalTankStatus enhanced:**
```python
@dataclass
class ThermalTankStatus:
    battery_temp_current: float
    battery_temp_predicted: float
    should_throttle: bool
    throttle_reason: ThrottleReason      # NEW: Why throttling
    headroom_seconds: float
    cooling_rate: float
    cpu_big_velocity: float              # NEW: Regime detector
    cpu_little_velocity: float           # NEW: Regime detector
```

**ThrottleReason enum (NEW):**
```python
class ThrottleReason(Enum):
    NONE = auto()           # Safe
    BATTERY_TEMP = auto()   # Battery too hot
    CPU_VELOCITY = auto()   # Regime change
    BOTH = auto()           # Both triggered
```

---

## Quick Start

```bash
pip install numpy
python example_usage.py
```

**Basic usage:**
```python
import asyncio
from s25_thermal import create_thermal_intelligence

async def main():
    thermal = create_thermal_intelligence()
    await thermal.start()
    
    tank = thermal.get_tank_status()
    
    # Dual-condition check
    if tank.should_throttle:
        print(f"Throttling: {tank.throttle_reason.name}")
        
        if tank.throttle_reason == ThrottleReason.CPU_VELOCITY:
            print(f"Regime change: CPU_BIG {tank.cpu_big_velocity:+.3f}°C/s")
        elif tank.throttle_reason == ThrottleReason.BATTERY_TEMP:
            print(f"Battery hot: {tank.battery_temp_current:.1f}°C")
    else:
        print(f"Safe - headroom: {tank.headroom_seconds:.0f}s")
    
    await thermal.stop()

asyncio.run(main())
```

---

## What's Included

- `s25_thermal.py` - Complete system (2798 lines)
- `example_usage.py` - Working demo with dual-throttle
- `README.md` - Full documentation
- `CHANGELOG.md` - Version history

---

## Breaking Changes

**ThermalTankStatus structure changed:**
- Added `throttle_reason: ThrottleReason`
- Added `cpu_big_velocity: float`
- Added `cpu_little_velocity: float`

**If upgrading from v2.24:** Update code accessing `ThermalTankStatus` to handle new fields.

---

## Requirements

- **Python 3.8+**
- **numpy** (for physics)
- **Android device with Termux** (optional)

✅ Tested on Samsung Galaxy S25+ (Snapdragon 8 Elite)

---

## Technical Details

**Why dual-condition?**

Single condition (battery temp) had blind spot: regime changes. Battery τ=540s means it reacts slowly. CPUs τ=14-19s react fast. When workload spikes, CPUs hit 70-90°C before battery reaches 38.5°C. Physics model can't predict this.

Solution: Monitor CPU velocities. >1.0°C/s = regime change = throttle immediately.

**Why observed peaks?**

Physics assumes constant power: `T = T_amb + (P·R/k)·(...)`

Throttling changes P mid-prediction, breaking the model. Instead of fighting this, use empirical data: "When CPU_BIG is throttled, it peaks at 79.1°C."

**Validation:** 2100 predictions, 7 zones, 30s horizon. Found:
- Throttling starts 15-30°C earlier than assumed
- Max errors when hot but stable (already throttled)
- Velocities spike to 9.5-14°C/s during regime changes

---

## Migration Guide

**v2.24 → v2.25:**

Old code:
```python
tank = thermal.get_tank_status()
if tank.should_throttle:
    await asyncio.sleep(30)
```

New code (handles both conditions):
```python
tank = thermal.get_tank_status()
if tank.should_throttle:
    if tank.throttle_reason == ThrottleReason.CPU_VELOCITY:
        # Regime change - wait for stabilization
        await asyncio.sleep(30)
    elif tank.throttle_reason == ThrottleReason.BATTERY_TEMP:
        # Battery hot - wait longer
        await asyncio.sleep(tank.headroom_seconds)
    else:
        # Both conditions - wait for both to clear
        await asyncio.sleep(60)
```

---

## License

MIT License

Copyright (c) 2025 PNGN-Tec LLC  
Author: Jesse Vogeler-Wunsch (@DaSettingsPNGN)

---

**Dual-condition throttle + empirical peak predictions = robust thermal management.** 🔥