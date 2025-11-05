# 🔥🐧🔥 S25+ Thermal Intelligence System

**Physics-based thermal management for Android devices**

Multi-zone temperature monitoring with Newton's law of cooling predictions. Prevents throttling through thermal budget forecasting and proactive workload scheduling.

---

## What This Does

Monitors thermal zones on Android devices, applies physics models with measured thermal constants, and predicts temperatures 30s ahead. Provides thermal budget calculations and simple bool throttle decisions through tank status API.

**Production deployment:** Discord bot serving 645+ members on Samsung Galaxy S25+, 24/7 operation with zero thermal shutdowns.

---

## Why Predictive?

**Reactive systems:** Read temp → React when hot → Throttle → Hope it cools

**Predictive systems:** Model thermal physics → Forecast temperature → Schedule work

Result: Continuous operation under load vs constant throttling.

---

## Installation

```bash
git clone https://github.com/DaSettingsPNGN/S25_THERMAL.git
cd S25_THERMAL
pip install numpy
```

**Requirements:**
- Python 3.8+
- numpy ≥1.20.0

**On Termux (Android):**
```bash
pkg install python numpy
pip install numpy
```

---

## Basic Usage

### Initialize

```python
from s25_thermal import create_thermal_intelligence

thermal = create_thermal_intelligence()
await thermal.start()

# Get current reading
sample = thermal.get_current()
if sample:
    battery = sample.zones.get(ThermalZone.BATTERY)
    print(f"Battery: {battery:.1f}°C")
```

### Check Throttle Status

```python
# Primary API - simple bool decision
tank = thermal.get_tank_status()

if tank.should_throttle:
    print(f"⚠️ Too hot - wait {tank.cooldown_needed:.0f}s")
    await asyncio.sleep(tank.cooldown_needed)
else:
    print(f"✅ Safe - budget: {tank.thermal_budget:.0f}s")
    await execute_work()
```

### Get Prediction

```python
prediction = thermal.get_prediction()

if prediction:
    battery_pred = prediction.predicted_temps.get(ThermalZone.BATTERY)
    print(f"Battery (+30s): {battery_pred:.1f}°C")
    print(f"Confidence: {prediction.confidence:.0%}")
```

---

## Core API

### Thermal Tank Status

Primary throttle decision interface:

```python
@dataclass
class ThermalTankStatus:
    battery_temp_current: float      # Current battery °C
    battery_temp_predicted: float    # At 30s horizon
    should_throttle: bool            # Bool decision
    cooldown_needed: float           # Seconds to wait
    thermal_budget: float            # Seconds until throttle
    cooling_rate: float              # °C/s
    peak_temp: float                 # Hottest zone now
    state: ThermalState              # Overall state enum
```

### Thermal Zones

```python
from s25_thermal import ThermalZone

ThermalZone.CPU_BIG      # High-performance cores
ThermalZone.CPU_LITTLE   # Efficiency cores  
ThermalZone.GPU          # Graphics
ThermalZone.BATTERY      # Battery thermistor (critical)
ThermalZone.MODEM        # 5G/WiFi modem
ThermalZone.CHASSIS      # Chassis reference
```

### Predictions

```python
@dataclass
class ThermalPrediction:
    predicted_state: ThermalState
    predicted_temps: Dict[ThermalZone, float]
    horizon: float                    # 30.0s
    time_to_state: float  
    confidence: float                 # 0.0-1.0
    recommended_delay: float
    thermal_budget: float
    timestamp: float
```

---

## Temperature Prediction

### Physics Model

Newton's law of cooling with measured constants:

```python
# Per-zone evolution
T(t) = T_amb + (T₀ - T_amb)·exp(-t/τ) + (P·R/k)·(1 - exp(-t/τ))

# Battery simplified (τ=540s >> 30s horizon)
ΔT ≈ (P/C) × Δt
```

**Measured thermal time constants:**
- CPU_BIG: τ = 6.6s
- CPU_LITTLE: τ = 6.9s
- GPU: τ = 9.1s
- MODEM: τ = 9.0s
- BATTERY: τ = 540s (high thermal mass)

### Adaptive Damping

System tracks prediction errors by regime (heating/cooling/stable) and adjusts momentum factors over time:

```python
# TransientResponseTuner tracks errors
# Separate history sizes per thermal time constant:
# - Battery: 120 samples (20 min)
# - Fast zones: 12 samples (2 min)
# - Chassis: 24 samples (4 min)

# Momentum factors adapt to reduce bias
# Improves accuracy over extended runs
```

### Thermal Budget

```python
tank = thermal.get_tank_status()

# Seconds until Samsung throttles (42°C battery)
budget = tank.thermal_budget

if budget < 60:
    print(f"⚠️ Throttling in {budget:.0f}s")
    await asyncio.sleep(tank.cooldown_needed)
```

---

## Multi-Zone Monitoring

### Zone Readings

```python
sample = thermal.get_current()

if sample:
    for zone, temp in sample.zones.items():
        confidence = sample.confidence.get(zone, 0.0)
        print(f"{zone.name}: {temp:.1f}°C (conf: {confidence:.0%})")
```

**Hardware mapping (Samsung S25+):**
- Zone 20 (cpuss-1-0): CPU_BIG
- Zone 13 (cpuss-0-0): CPU_LITTLE
- Zone 23 (gpuss-0): GPU
- Zone 31 (mdmss-0): MODEM
- Zone 60 (battery): BATTERY (critical)
- Zone 52 (sys-therm-5): CHASSIS

### Thermal States

```python
from s25_thermal import ThermalState

ThermalState.COLD       # Well below throttle
ThermalState.OPTIMAL    # Normal operation
ThermalState.WARM       # Approaching limits
ThermalState.HOT        # Near throttle
ThermalState.CRITICAL   # Immediate action required
ThermalState.UNKNOWN    # Insufficient data
```

---

## Integration Examples

### Discord Bot

```python
import discord
from s25_thermal import create_thermal_intelligence

bot = discord.Bot()
thermal = create_thermal_intelligence()

@bot.event
async def on_ready():
    await thermal.start()

@bot.command()
async def render(ctx):
    tank = thermal.get_tank_status()
    
    if tank.should_throttle:
        await ctx.send(f"⚠️ System hot - cooling {tank.cooldown_needed:.0f}s")
        return
    
    await heavy_operation()
```

### Monitoring Loop

```python
async def monitor():
    thermal = create_thermal_intelligence()
    await thermal.start()
    
    while True:
        tank = thermal.get_tank_status()
        sample = thermal.get_current()
        
        if sample:
            battery = sample.zones.get(ThermalZone.BATTERY)
            print(f"Battery: {battery:.1f}°C | {tank.state.name} | Budget: {tank.thermal_budget:.0f}s")
        
        await asyncio.sleep(10)
```

---

## Configuration

Edit constants in `s25_thermal.py`:

```python
# Core prediction
THERMAL_PREDICTION_HORIZON = 30.0       # seconds ahead
THERMAL_SAMPLE_INTERVAL_MS = 10000      # 10s sampling
THERMAL_HISTORY_SIZE = 300              # 50 min history
MIN_SAMPLES_FOR_PREDICTIONS = 12        # 2 min minimum

# Adaptive damping history
DAMPING_HISTORY_SLOW_ZONES = 10         # Battery: 20 min
DAMPING_HISTORY_FAST_ZONES = 1          # CPU/GPU: 2 min  
DAMPING_HISTORY_MEDIUM_ZONES = 2        # Chassis: 4 min

# Chassis damping
CHASSIS_DAMPING_FACTOR = 0.90           # Thermal inertia
```

### Per-Zone Constants

```python
ZONE_THERMAL_CONSTANTS = {
    'CPU_BIG': {
        'thermal_mass': 0.025,          # J/K
        'thermal_resistance': 2.8,      # °C/W
        'ambient_coupling': 0.80,
        'peak_power': 6.0,              # W
    },
    # ... other zones
}
```

---

## Architecture

**Components (all in s25_thermal.py):**

1. **ThermalTelemetryCollector** - Reads `/sys/class/thermal/` sensors, detects network state
2. **ZonePhysicsEngine** - Newton's law predictions per zone, adaptive damping
3. **TransientResponseTuner** - Tracks prediction errors, tunes momentum factors
4. **ThermalTank** - Simple throttle decision logic, battery-centric
5. **ThermalIntelligenceSystem** - Main orchestration, public API

**Flow:**
- 10s sampling interval (uniform)
- Physics engine predicts 30s ahead
- Prediction validated against actual measurement
- Errors fed to adaptive tuner
- Tank status computed from prediction
- Thermal budget calculated

**Key insight:** Battery τ=540s dominates throttle behavior. Samsung throttles at 42°C battery temperature. System focuses prediction accuracy on battery zone.

---

## Production Metrics

**Deployment:** Discord bot, 645+ members, Samsung S25+

**Prediction accuracy:** ~1.5°C MAE at 30s horizon with 10s sampling

**Observations:**
- Battery zone most predictable (high thermal mass)
- Fast zones (CPU/GPU) harder to predict (low τ)
- Adaptive damping improves over time
- Network state (5G vs WiFi) impacts baseline
- Charging state significantly affects temperature
- Zero thermal shutdowns in production

**Thermal constants measured from step response testing in production workload.**

---

## Troubleshooting

### No Sensors Found

Check Termux permissions:
```bash
termux-setup-storage
ls /sys/class/thermal/thermal_zone*/temp
```

### Low Prediction Accuracy

Wait for adaptive damping calibration:
- Battery: 20 min (120 samples)
- CPU/GPU: 2 min (12 samples)

System improves accuracy over time by tracking prediction errors.

### Battery Temperature Flat

Battery has high thermal mass (τ=540s). Changes are slow. This is expected behavior.

---

## File Structure

```
s25_thermal/
├── s25_thermal.py       # Complete system (2454 lines)
├── README.md           # This file
├── CHANGELOG.md        # Version history
└── LICENSE             # MIT license
```

**Single file architecture:** All components inline. No external dependencies except numpy.

---

## Contact

**Jesse Vogeler-Wunsch** @ PNGN-Tec LLC

Discord: **@DaSettingsPNGN**

---

## License

MIT License. See LICENSE file.

---

*Physics-based thermal management for continuous operation under load.*
