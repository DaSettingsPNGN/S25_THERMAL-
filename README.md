# 🔥🐧🔥 S25+ Thermal Intelligence System

**Physics-based thermal management for Samsung Galaxy S25+**

Multi-zone temperature monitoring with Newton's law predictions and observed peak shortcuts for throttled regimes. Dual-condition throttle system: battery temperature prediction + CPU velocity spike detection.

Validation results over 457k predictions (18.7 hours continuous operation):
- Overall steady-state: 1.22°C MAE, 73% within 1°C
- Battery prediction: 0.375°C MAE, 0.2°C median
- Temperature range validated: 2°C to 95°C (cold boot to near-TJmax)
- 97.8% steady-state, 2.2% transients during regime changes

---

## What This Does

Monitors 7 thermal zones, predicts temperatures 30s ahead using Newton's law (or observed peaks when throttled), detects regime changes via CPU velocity spikes, and provides bool throttle decisions.

**Designed for:** Production workloads on mobile hardware with variable thermal environments.

---

## How This Compares

| System | MAE | Conditions |
|--------|-----|------------|
| **This system (battery)** | **0.375°C** | Production phone, unknown workload, 30s horizon |
| PINN + LSTM (2025) | 0.29°C | Lab, known 2.0C charge rate, same cell |
| FCN-GBM hybrid (2024) | 0.46°C | Lab, 20% training data from test cell |
| RNN benchmark (2024) | 0.15°C | Lab, Bayesian-optimized hyperparameters |
| Crowdsourced phones (2020) | 1.5°C | Estimating ambient from battery |
| Data center ML (2020) | 2.38°C | Gradient boosting, server racks |

The sub-0.3°C systems are doing **interpolation**—trained on the exact battery and conditions they test on. This is **extrapolation** with no training data, predicting 30s into an unknown future on production hardware.

---

## Why Dual-Condition Throttle?

**Problem 1:** Battery (τ=210s) reacts slowly. CPUs (τ=50-60s) spike first during regime changes. Physics model can't predict discontinuities.

**Solution 1:** CPU velocity spike detection (>3.0°C/s) catches regime changes before physics breaks.

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

# Normal: <0.5°C/s, Danger: >3.0°C/s
if tank.cpu_big_velocity > 3.0 or tank.cpu_little_velocity > 3.0:
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
'CPU_BIG': 84°C        # Starts throttling at 45°C
'CPU_LITTLE': 98°C     # Starts throttling at 48°C
'GPU': 66°C            # Starts throttling at 38°C
'MODEM': 68°C          # Starts throttling at 40°C
'BATTERY': 39°C        # Doesn't self-throttle
```

### Thermal Time Constants

**Measured from hardware:**
- CPU_BIG: τ = 25s
- CPU_LITTLE: τ = 35s
- GPU: τ = 30s
- MODEM: τ = 145s
- BATTERY: τ = 210s (high thermal mass)
- CHASSIS: τ = 100s (vapor chamber + frame)

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
# CPU velocity >3.0°C/s indicates regime change (workload spike)
if cpu_big_velocity > 3.0 or cpu_little_velocity > 3.0:
    throttle_reason = CPU_VELOCITY
```

**Rationale:** Battery lags (τ=210s), CPUs react fast (τ=50-60s). Regime changes spike CPUs before battery reacts. Velocity catches discontinuities physics can't predict.

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

## Architecture: Single-File Design

**Why 3,949 lines in one file?**

Mobile constraint: filesystem I/O generates thermal overhead.

**Problem with modular approach:**
```python
# Multiple imports = multiple filesystem reads = thermal spikes
from thermal_telemetry import TelemetryCollector    # Read + parse
from thermal_physics import PhysicsEngine           # Read + parse
from thermal_tank import ThermalTank                # Read + parse
from thermal_validation import ValidationSystem     # Read + parse
# Each import: disk access, parsing, compilation
```

**Single-file approach:**
```python
# One load, stays in memory cache
from s25_thermal import create_thermal_intelligence  # Single read
# Everything cached, zero additional I/O after initial load
```

**Measured impact:**
- Module import: ~10-50ms per file on phone storage
- Filesystem access: thermal sensor spike during I/O
- Python import system: multiple stat() calls per module
- Cold start: 4-6 files = 40-300ms total load time

**On mobile hardware:**
- Storage I/O is thermally expensive (NAND flash heating)
- Every disk access shows up in thermal sensors
- The thermal monitoring system itself must minimize thermal overhead
- Single file loads once, stays cached, zero runtime I/O

**Trade-off accepted:**
Harder to navigate 4k lines vs cleaner thermal profile. For a system designed to *prevent* thermal issues, generating minimal thermal overhead is the priority.

When your thermal monitoring system is running on the same device it's monitoring, constraint-driven design wins.

---

## Configuration

Key constants in `s25_thermal.py`:

```python
# Prediction
THERMAL_PREDICTION_HORIZON = 30.0       # seconds ahead
THERMAL_SAMPLE_INTERVAL = 1.0           # 1s sampling
MIN_SAMPLES_FOR_PREDICTIONS = 60        # 1 min warmup for ambient fitting

# Throttle thresholds
TANK_THROTTLE_TEMP = 38.5               # Battery °C (safety margin from Samsung's 40-42°C)
CPU_VELOCITY_DANGER = 3.0               # °C/s - regime change threshold

# Throttle curves (observed from validation)
COMPONENT_THROTTLE_CURVES = {
    'CPU_BIG': {
        'temp_start': 45.0,
        'observed_peak': 84.0,
        ...
    },
    ...
}
```

---

## Architecture

**Single file (3,949 lines):** s25_thermal.py contains everything.

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

## Validation Metrics

**457k predictions over 18.7 hours:**

**Overall accuracy (steady-state, 97.8% of samples):**
- MAE: 1.22°C
- Bias: -0.016°C (essentially zero)
- Within 1°C: 73.0%
- Within 2°C: 86.4%

**Per-zone accuracy (steady-state):**

| Zone | MAE | Median | Within 1°C | Temp Range |
|------|-----|--------|------------|------------|
| **BATTERY** | **0.375°C** | **0.20°C** | **91.5%** | 2°C – 39°C |
| CHASSIS | 1.00°C | 0.52°C | 73.6% | 4°C – 53°C |
| MODEM | 1.54°C | 0.73°C | 60.9% | 5°C – 65°C |
| GPU | 1.55°C | 0.71°C | 62.2% | 5°C – 64°C |
| CPU_BIG | 1.86°C | 0.82°C | 60.9% | 4°C – 82°C |
| CPU_LITTLE | 2.35°C | 0.96°C | 59.9% | 5°C – 95°C |

**Error percentiles:**
- P50: 0.40°C
- P75: 1.12°C
- P90: 2.53°C
- P95: 4.73°C
- P99: 13.44°C

**Normal operation (excluding stress test periods):**
- MAE: 0.76°C
- Within 1°C: 77.7%
- P50: 0.33°C

**Transient analysis (2.2% of samples):**
- Physics can't predict regime changes—expected limitation
- CPU_BIG transients: 9.96°C MAE
- CPU_LITTLE transients: 6.72°C MAE
- Use CPU velocity detection instead of trusting physics during spikes

---

## Troubleshooting

### No Sensors Found

```bash
termux-setup-storage
ls /sys/class/thermal/thermal_zone*/temp
```

### False CPU Velocity Alerts

Velocity threshold (3.0°C/s) might be too sensitive for your workload. Adjust:

```python
CPU_VELOCITY_DANGER = 4.0  # Was 3.0
```

### Battery Temperature Stable

Battery has τ=210s. Changes are slow. This is correct behavior.

---

## Contact

**Jesse Vogeler-Wunsch** @ PNGN-Tec LLC  
Discord: **@DaSettingsPNGN**

---

## License

MIT License. See LICENSE file.

---

*Physics-based + empirical hybrid thermal management with dual-condition throttle detection.*
