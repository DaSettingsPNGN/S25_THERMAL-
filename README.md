# 🔥🐧🔥 S25+ Thermal Intelligence System

**Physics-based + empirical hybrid thermal management for Android devices**

Multi-zone temperature monitoring with Newton's law predictions and observed peak shortcuts. Dual-condition throttle system: battery temperature prediction + CPU velocity spike detection.

---

## What This Does

Monitors 7 thermal zones, predicts temperatures 30s ahead using physics (or observed peaks when throttled), detects regime changes via CPU velocity spikes, and provides simple bool throttle decisions.

**Production:** Discord bot serving 645+ members on Samsung S25+, 24/7, zero thermal shutdowns.

---

## Why Dual-Condition Throttle?

**Problem 1:** Battery (τ=540s) reacts slowly. CPUs (τ=14-19s) spike first during regime changes. Physics model can't predict discontinuities.

**Solution 1:** CPU velocity spike detection (>1.0°C/s) catches regime changes before physics breaks.

**Problem 2:** Physics model catastrophically under-predicts when throttled (-30 to -60°C errors).

**Solution 2:** When temp >= throttle_start, predict observed_peak from validation data instead of physics.

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

**On Termux:**
```bash
pkg install python numpy
```

---

## Quick Start

### Initialize

```python
from s25_thermal import create_thermal_intelligence, ThermalZone

thermal = create_thermal_intelligence()
await thermal.start()

sample = thermal.get_current()
if sample:
    battery = sample.zones.get(ThermalZone.BATTERY)
    print(f"Battery: {battery:.1f}°C")
```

### Check Throttle Status (Dual-Condition)

```python
tank = thermal.get_tank_status()

# Throttle reason: NONE, BATTERY_TEMP, CPU_VELOCITY, or BOTH
if tank.should_throttle:
    print(f"🛑 Throttling: {tank.throttle_reason.name}")
    print(f"   Battery: {tank.battery_temp_current:.1f}°C")
    print(f"   CPU_BIG vel: {tank.cpu_big_velocity:+.3f}°C/s")
    await asyncio.sleep(30)
else:
    print(f"✅ Safe - headroom: {tank.headroom_seconds:.0f}s")
    await execute_work()
```

### Monitor Velocities (Regime Change Detection)

```python
tank = thermal.get_tank_status()

# Normal: <0.4°C/s, Warning: >0.5°C/s, Danger: >1.0°C/s
if tank.cpu_big_velocity > 1.0 or tank.cpu_little_velocity > 1.0:
    print("⚠️ Regime change detected - workload spike!")
    # Throttle kicks in automatically
```

---

## Core API

### Thermal Tank Status

**Primary interface - dual throttle conditions:**

```python
from s25_thermal import ThermalTankStatus, ThrottleReason

@dataclass
class ThermalTankStatus:
    battery_temp_current: float      # Current battery °C
    battery_temp_predicted: float    # At 30s horizon
    should_throttle: bool            # True = stop work
    throttle_reason: ThrottleReason  # Why throttling (if at all)
    headroom_seconds: float          # Seconds until battery throttle
    cooling_rate: float              # Battery °C/s (negative = heating)
    cpu_big_velocity: float          # CPU_BIG °C/s (regime detector)
    cpu_little_velocity: float       # CPU_LITTLE °C/s (regime detector)
```

**ThrottleReason enum:**
```python
class ThrottleReason(Enum):
    NONE = auto()           # Safe to operate
    BATTERY_TEMP = auto()   # Battery too hot
    CPU_VELOCITY = auto()   # Regime change detected
    BOTH = auto()           # Both conditions triggered
```

### Thermal Zones

```python
ThermalZone.CPU_BIG      # Oryon Prime cores
ThermalZone.CPU_LITTLE   # Oryon efficiency cores
ThermalZone.GPU          # Adreno 830
ThermalZone.BATTERY      # Battery thermistor (critical for throttle)
ThermalZone.MODEM        # 5G/WiFi modem
ThermalZone.CHASSIS      # Vapor chamber reference
ThermalZone.AMBIENT      # Ambient air
```

### Predictions

```python
@dataclass
class ThermalPrediction:
    timestamp: float                        # When prediction made
    horizon: float                          # 30.0s ahead
    predicted_temps: Dict[ThermalZone, float]
    confidence: float                       # 0.0-1.0
    confidence_by_zone: Dict[ThermalZone, float]
    thermal_budget: float                   # Seconds until throttle
    power_by_zone: Dict[ThermalZone, float] # Power used in prediction
```

---

## Temperature Prediction

### Hybrid Approach

**Normal operation (temp < throttle_start):**
```python
# Newton's law of cooling
T(t) = T_amb + (T₀ - T_amb)·exp(-t/τ) + (P·R/k)·(1 - exp(-t/τ))
```

**Throttled regime (temp >= throttle_start):**
```python
# Use observed peak from validation data
if current_temp >= THROTTLE_START:
    predicted_temp = OBSERVED_PEAK  # Empirical, not physics
```

**Why?** Physics breaks at regime changes. Model assumes constant power - throttling changes that assumption mid-flight. Use empirical data instead.

### Observed Peaks (from validation)

```python
'CPU_BIG': 79.1°C      # Starts throttling at 45°C
'CPU_LITTLE': 93.4°C   # Starts throttling at 48°C
'GPU': 58.1°C          # Starts throttling at 38°C
'MODEM': 59.7°C        # Starts throttling at 40°C
'BATTERY': 35.2°C      # Doesn't self-throttle
```

### Thermal Time Constants

**Measured from hardware:**
- CPU_BIG: τ = 18.7s
- CPU_LITTLE: τ = 14.3s
- GPU: τ = 22.3s
- MODEM: τ = 9.0s
- BATTERY: τ = 540s (high thermal mass)
- CHASSIS: τ = 120s (vapor chamber + frame)

### Dual Throttle Conditions

**Condition 1: Battery Temperature**
```python
# Samsung throttles at 40-42°C battery
# We throttle at 38.5°C (safety margin)
if battery_predicted >= 38.5:
    throttle_reason = BATTERY_TEMP
```

**Condition 2: CPU Velocity**
```python
# P95 velocity ~1.0°C/s, normal <0.4°C/s
# >1.0°C/s indicates regime change (workload spike)
if cpu_big_velocity > 1.0 or cpu_little_velocity > 1.0:
    throttle_reason = CPU_VELOCITY
```

**Rationale:** Battery lags (τ=540s), CPUs react fast (τ=14-19s). Regime changes spike CPUs before battery reacts. Velocity catches discontinuities physics can't predict.

---

## Multi-Zone Monitoring

### Zone Readings

```python
sample = thermal.get_current()

if sample:
    for zone, temp in sample.zones.items():
        print(f"{zone.name}: {temp:.1f}°C")
```

**Hardware mapping (Samsung S25+):**
- Zone 20: CPU_BIG (cpuss-1-0)
- Zone 13: CPU_LITTLE (cpuss-0-0)
- Zone 23: GPU (gpuss-0)
- Zone 31: MODEM (mdmss-0)
- Zone 60: BATTERY (battery sensor)
- Zone 53: CHASSIS (sys-therm-0 vapor chamber)
- Zone 52: AMBIENT (sys-therm-5 air temp)

### Thermal States

```python
ThermalState.COLD       # Well below throttle
ThermalState.OPTIMAL    # Normal operation
ThermalState.WARM       # Approaching limits
ThermalState.HOT        # Near throttle
ThermalState.CRITICAL   # Immediate action needed
```

---

## Integration Examples

### Discord Bot with Dual Throttle

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
        if tank.throttle_reason == ThrottleReason.CPU_VELOCITY:
            await ctx.send(f"⚠️ Regime change detected - wait 30s")
        elif tank.throttle_reason == ThrottleReason.BATTERY_TEMP:
            await ctx.send(f"🔥 Battery hot: {tank.battery_temp_current:.1f}°C")
        else:
            await ctx.send(f"🛑 Both conditions triggered!")
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
            
            status = f"Battery: {battery:.1f}°C | "
            status += f"Throttle: {tank.throttle_reason.name} | "
            status += f"Headroom: {tank.headroom_seconds:.0f}s | "
            status += f"CPU_BIG vel: {tank.cpu_big_velocity:+.3f}°C/s"
            
            print(status)
        
        await asyncio.sleep(10)
```

---

## Configuration

Key constants in `s25_thermal.py`:

```python
# Prediction
THERMAL_PREDICTION_HORIZON = 30.0       # seconds ahead
MIN_SAMPLES_FOR_PREDICTIONS = 3         # minimum before predicting

# Throttle thresholds
TANK_THROTTLE_TEMP = 38.5               # Battery °C
CPU_VELOCITY_DANGER = 1.0               # °C/s (P95 level)
CPU_VELOCITY_WARNING = 0.5              # °C/s (watch closely)

# Throttle curves (observed from validation)
COMPONENT_THROTTLE_CURVES = {
    'CPU_BIG': {
        'temp_start': 45.0,
        'observed_peak': 79.1,
        ...
    },
    ...
}
```

---

## Architecture

**Single file (2798 lines):** s25_thermal.py contains everything.

**Components:**
1. **ThermalTelemetryCollector** - Reads sensors, detects network
2. **ZonePhysicsEngine** - Newton's law predictions + observed peak shortcuts
3. **TransientResponseTuner** - Tracks errors, tunes momentum
4. **ThermalTank** - Dual-condition throttle logic
5. **ThermalIntelligenceSystem** - Main orchestration

**Flow:**
- 1s sampling (10s full cycle)
- Check if temp >= throttle_start
  - Yes: predict observed_peak
  - No: use physics (Newton's law)
- Check CPU velocities for regime changes
- Compute throttle decision (battery temp OR CPU velocity)
- Return ThermalTankStatus

---

## Production Metrics

**Deployment:** Discord bot, 645+ members, Samsung S25+

**Prediction accuracy:**
- Normal regime: ~1.5°C MAE at 30s horizon
- Throttled regime: Use observed peaks (avoids -30 to -60°C errors)

**Throttle detection:**
- Normal CPU velocity: <0.4°C/s (P90)
- Warning level: >0.5°C/s
- Danger level: >1.0°C/s (P95+)
- Max observed: CPU_BIG 9.5°C/s, CPU_LITTLE 14.0°C/s

**Result:** Zero thermal shutdowns, proactive throttling before damage.

---

## Troubleshooting

### No Sensors Found

```bash
termux-setup-storage
ls /sys/class/thermal/thermal_zone*/temp
```

### False CPU Velocity Alerts

Velocity threshold (1.0°C/s) might be too sensitive. Increase to 1.5°C/s:

```python
ThermalTank.CPU_VELOCITY_DANGER = 1.5  # Was 1.0
```

### Battery Temperature Stable

Battery has τ=540s. Changes are slow. This is correct behavior.

---

## Contact

**Jesse Vogeler-Wunsch** @ PNGN-Tec LLC  
Discord: **@DaSettingsPNGN**

---

## License

MIT License. See LICENSE file.

---

*Physics-based + empirical hybrid thermal management with dual-condition throttle detection.*