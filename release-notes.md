# 🔥🐧🔥 v1.0.0 - Initial Release: S25+ Thermal Intelligence System

**First public release.** Production-tested thermal management for Samsung Galaxy S25+ and compatible Android devices.

---

## The Problem

Running a production Discord bot server on Samsung S25+. Snapdragon 8 Elite throttles at 42°C battery temperature, causing:

- **Performance degradation** - CPU throttles at thermal limits
- **Unpredictable behavior** - No way to anticipate throttling
- **Poor UX** - Bot becomes unresponsive during thermal events

Traditional monitoring only tells you temperature **now**. Need to know what temperature will be in 30 seconds.

## The Solution

Physics-based thermal intelligence system that:

✅ **Monitors** 6 thermal zones (CPU_BIG, CPU_LITTLE, GPU, BATTERY, MODEM, CHASSIS)  
✅ **Predicts** temperatures 30s ahead using Newton's law of cooling  
✅ **Adapts** through prediction error feedback and momentum tuning  
✅ **Prevents** throttling with thermal budget calculation  
✅ **Decides** via simple bool: should_throttle

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
    
    if tank.should_throttle:
        print(f"Too hot - wait {tank.cooldown_needed:.0f}s")
    else:
        print(f"Safe - budget: {tank.thermal_budget:.0f}s")
    
    await thermal.stop()

asyncio.run(main())
```

---

## Results

**Production deployment:** Discord bot serving 645+ members on Samsung S25+

**Metrics:**
- **Prediction accuracy:** ~1.5°C MAE at 30s horizon
- **Thermal stability:** 24/7 operation with zero shutdowns
- **Battery temperature:** Maintained below 42°C throttle point
- **Adaptive improvement:** Accuracy increases over runtime

**Key achievement:** Physics-based prediction prevents throttling proactively, not reactively.

---

## Key Features

### 🔬 Physics-Based Prediction
- Newton's law of cooling per zone
- Measured thermal constants (τ, R, C) from hardware
- 30-second temperature forecasts
- Battery: τ=540s, simplified power integration
- Fast zones: τ<10s, full exponential model

### 🧠 Adaptive Damping
- TransientResponseTuner tracks prediction errors
- Heating/cooling/stable regime classification
- Momentum factor tuning per regime
- History sizes scaled to thermal time constants

### 📊 Thermal Tank Status
- Simple bool throttle decision
- Battery-centric (Samsung throttles at 42°C)
- Thermal budget (seconds until throttle)
- Cooling rate calculation
- Recommended delay for safe operation

### 🌐 Network Awareness
- Detects WiFi vs 5G thermal impact
- Charging state detection
- Screen brightness monitoring (when available)

---

## What's Included

- `s25_thermal.py` - Complete thermal intelligence system (2454 lines)
- `config.py` - Configuration reference
- `example_usage.py` - Working example
- `README.md` - Full documentation

---

## Requirements

- **Python 3.8+**
- **numpy** (for physics calculations)
- **Android device with Termux** (optional - works on any platform)

✅ Tested on Samsung Galaxy S25+ (Snapdragon 8 Elite)

---

## License

MIT License

Copyright (c) 2025 PNGN-Tec LLC  
Author: Jesse Vogeler-Wunsch (@DaSettingsPNGN)

---

**Physics-based thermal management for continuous operation under load.** 🔥
