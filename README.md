# 🐧 S25+ Thermal Intelligence System

**Physics-based thermal management for Android devices**

This project provides predictive thermal intelligence through multi-zone temperature monitoring, velocity tracking, and thermal budget forecasting. Built for resource-constrained environments where thermal throttling must be prevented, not managed.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Android-green)

---

## What This Does

Creates predictive thermal models using Newton's law of cooling, thermal mass calculations, and multi-zone sensor fusion. Forecasts temperature changes 30-60 seconds ahead and calculates thermal budgets before executing operations.

The system uses deterministic pattern learning where commands build thermal signatures over time. This makes performance optimization and thermal prediction reproducible.

---

## Why Predictive?

**Reactive systems:** Read temperature → React when hot → Throttle → Hope it cools

**Predictive systems:** Model thermal physics → Forecast temperature → Queue work strategically

The difference: Zero thermal crashes vs constant throttling.

**Production Proof:** Serving 500+ concurrent Discord bot users from a Samsung Galaxy S25+ with zero thermal incidents.

---

## Installation

Install Python dependencies:

```bash
git clone https://github.com/DaSettingsPNGN/S25_THERMAL.git
cd S25_THERMAL
pip install -r requirements.txt
```

**Required**
- numpy (version 1.20.0 or later)
- asyncio (Python 3.8+ standard library)

**Optional**
- psutil (for enhanced system monitoring)

On Termux (Android), install additional packages:

```bash
pkg install python numpy
pip install -r requirements.txt
```

---

## Basic Usage

### Initialize System

```python
from s25_thermal import create_thermal_intelligence

# Create thermal intelligence system
thermal = create_thermal_intelligence()
await thermal.start()

# Get current thermal state
intel = thermal.get_current_intelligence()
print(f"Temperature: {intel.stats.current.zones['battery']:.1f}°C")
print(f"Trend: {intel.stats.velocity.trend.name}")
```

### Predict Future Temperature

```python
# Get prediction for 30 seconds ahead
intel = thermal.get_current_intelligence()

if intel.prediction:
    predicted = intel.prediction.predicted_temps['battery']
    budget = intel.prediction.thermal_budget
    
    print(f"Current: {intel.stats.current.zones['battery']:.1f}°C")
    print(f"Predicted (+30s): {predicted:.1f}°C")
    print(f"Thermal budget: {budget:.0f} seconds")
```

### Track Command Impact

```python
# Track thermal cost of operations
command_hash = "render_animation_123"

thermal.track_command("render", command_hash)
# ... execute your operation ...
thermal.complete_command("render", command_hash)

# Check learned impact
signature = thermal.patterns.get_thermal_impact("render")
if signature:
    print(f"Average temp rise: {signature.avg_delta_temp:.2f}°C")
    print(f"Confidence: {signature.confidence:.0%}")
```

---

## Temperature Prediction

### Physics-Based Model

The system uses Newton's law of cooling for accurate predictions:

```python
# Core prediction equation
predicted_temp = current_temp + (velocity * time) + (0.5 * acceleration * time²)

# Cooling rate based on ambient temperature
cooling_rate = -ambient_coupling * (current_temp - ambient) / thermal_resistance

# Net temperature change
net_rate = velocity + cooling_rate
```

### Thermal Budget Calculation

```python
# How long until throttling?
intel = thermal.get_current_intelligence()

if intel.prediction:
    budget = intel.prediction.thermal_budget
    delay = intel.prediction.recommended_delay
    
    if budget < 60:
        print(f"⚠️ Throttling in {budget:.0f}s - delay operations by {delay:.1f}s")
```

### Velocity and Acceleration

```python
# Temperature isn't just a number - it's a trend
intel = thermal.get_current_intelligence()
velocity = intel.stats.velocity

print(f"Temperature changing at {velocity.overall:+.3f}°C/s")
print(f"Acceleration: {velocity.acceleration:+.4f}°C/s²")
print(f"Trend: {velocity.trend.name}")

# Trends: RAPID_COOLING, COOLING, STABLE, WARMING, RAPID_WARMING
```

---

## Multi-Zone Monitoring

### Available Thermal Zones

```python
intel = thermal.get_current_intelligence()

for zone, temp in intel.stats.current.zones.items():
    print(f"{zone.name}: {temp:.1f}°C")
```

**Monitored zones:**
- CPU Big cores (high-performance)
- CPU Little cores (efficiency)
- GPU
- Battery (critical for throttling)
- Modem
- Ambient temperature

### Zone-Specific Analysis

```python
# Get statistics for each zone
intel = thermal.get_current_intelligence()
stats = intel.stats

for zone in ['battery', 'cpu_big', 'gpu']:
    current = stats.current.zones.get(zone)
    mean = stats.mean.get(zone)
    velocity = stats.velocity.zones.get(zone)
    
    print(f"{zone}: {current:.1f}°C (avg: {mean:.1f}°C, Δ: {velocity:+.3f}°C/s)")
```

---

## Pattern Learning

### Command Thermal Signatures

The system learns thermal patterns from operations:

```python
# Commands automatically build profiles
thermal.track_render("animation_render", thermal_cost_mw=250, duration=2.5)

# After multiple observations, get learned behavior
signature = thermal.patterns.get_thermal_impact("animation_render")

print(f"Average thermal cost: {signature.avg_thermal_cost} mW")
print(f"Average temp rise: {signature.avg_delta_temp:.2f}°C")
print(f"Sample count: {signature.sample_count}")
print(f"Confidence: {signature.confidence:.0%}")
```

### Predict Operation Impact

```python
# Will this sequence cause problems?
commands = ["render", "compress", "upload"]
predicted_impact = thermal.patterns.predict_impact(commands)

print(f"Expected temperature rise: {predicted_impact:.2f}°C")

# Make decision
intel = thermal.get_current_intelligence()
current_temp = intel.stats.current.zones['battery']

if current_temp + predicted_impact > 40.0:
    print("⚠️ Sequence would trigger throttling - queuing for later")
```

### Pattern Persistence

```python
# Signatures are automatically saved
await thermal.save_signatures()

# On next startup, patterns are restored
# Learning continues across restarts
```

---

## Thermal States

### State Classification

```python
intel = thermal.get_current_intelligence()

# State is automatically determined
state = intel.state

if state == ThermalState.OPTIMAL:
    print("✅ System cool - full performance available")
elif state == ThermalState.WARM:
    print("⚠️ System warming - monitor closely")
elif state == ThermalState.HOT:
    print("🔥 System hot - reduce workload")
elif state == ThermalState.CRITICAL:
    print("🚨 Critical temperature - immediate action required")
```

**Thresholds (Samsung S25+):**
- OPTIMAL: < 42°C (normal operation)
- WARM: 42-50°C (increased monitoring)
- HOT: 50-60°C (throttling begins)
- CRITICAL: > 60°C (emergency shutdown)

### Recommendations

```python
intel = thermal.get_current_intelligence()

for recommendation in intel.recommendations:
    print(f"💡 {recommendation}")

# Example output:
# 💡 Thermal budget: 47s
# 💡 Delay: 2.1s
# 💡 5G: +3.2°C
```

---

## Integration Examples

### With Discord Bot

```python
import discord
from s25_thermal import create_thermal_intelligence

bot = discord.Bot()
thermal = create_thermal_intelligence()

@bot.event
async def on_ready():
    await thermal.start()
    print(f"Thermal system initialized")

@bot.command()
async def render(ctx):
    # Check thermal state before heavy operation
    intel = thermal.get_current_intelligence()
    
    if intel.state in [ThermalState.HOT, ThermalState.CRITICAL]:
        await ctx.send("⚠️ System too hot - operation queued")
        return
    
    if intel.prediction and intel.prediction.thermal_budget < 30:
        await ctx.send(f"⏳ Thermal budget low - delaying {intel.prediction.recommended_delay:.0f}s")
        await asyncio.sleep(intel.prediction.recommended_delay)
    
    # Safe to execute
    await heavy_render_operation()
```

### With Performance System

```python
from s25_thermal import create_thermal_intelligence, integrate_with_performance_system
import s25_performance

# Create systems
thermal = create_thermal_intelligence()
performance = s25_performance.accelerate(quality='balanced', thermal_system=thermal)

await thermal.start()

# Thermal data automatically feeds performance decisions
# Performance system uses predictions to optimize scheduling
```

### Monitoring Loop

```python
async def thermal_monitor():
    thermal = create_thermal_intelligence()
    await thermal.start()
    
    while True:
        intel = thermal.get_current_intelligence()
        
        print(f"Battery: {intel.stats.current.zones['battery']:.1f}°C")
        print(f"State: {intel.state.name}")
        print(f"Velocity: {intel.stats.velocity.overall:+.3f}°C/s")
        
        if intel.prediction:
            print(f"Budget: {intel.prediction.thermal_budget:.0f}s")
        
        await asyncio.sleep(10)
```

---

## Configuration

### Thermal Thresholds

Edit `config.py`:

```python
# Temperature thresholds (°C)
THERMAL_TEMP_OPTIMAL_MAX = 42.0
THERMAL_TEMP_WARM = 50.0
THERMAL_TEMP_HOT = 60.0
THERMAL_TEMP_CRITICAL = 70.0

# Hysteresis (prevents state flapping)
THERMAL_HYSTERESIS_UP = 2.0
THERMAL_HYSTERESIS_DOWN = 3.0
```

### Sampling Configuration

```python
# Sampling interval (milliseconds)
THERMAL_SAMPLE_INTERVAL_MS = 1000

# History size (samples to keep)
THERMAL_HISTORY_SIZE = 1000

# Prediction horizon (seconds)
THERMAL_PREDICTION_HORIZON = 30.0
```

### Pattern Learning

```python
# Maximum signatures to store
THERMAL_SIGNATURE_MAX_COUNT = 100

# Learning rate (0-1)
THERMAL_LEARNING_RATE = 0.2

# Minimum temperature change to track
THERMAL_SIGNATURE_MIN_DELTA = 0.1
```

---

## Advanced Features

### Anomaly Detection

```python
intel = thermal.get_current_intelligence()

for timestamp, anomaly in intel.anomalies:
    print(f"⚠️ {anomaly}")

# Example output:
# ⚠️ CPU_BIG anomaly: 65.2°C (expected 52.0±3.5°C)
# ⚠️ GPU temperature unusual: 62.1°C (CPU: 51.3°C)
```

### Network Impact Tracking

```python
# System correlates network type with temperature
intel = thermal.get_current_intelligence()

if intel.stats.network_impact > 3.0:
    print(f"🔥 5G adding {intel.stats.network_impact:.1f}°C")
```

### Charging Detection

```python
# Charging significantly impacts thermal behavior
intel = thermal.get_current_intelligence()

if intel.stats.charging_impact > 5.0:
    print(f"⚡ Charging adding {intel.stats.charging_impact:.1f}°C")
```

### Event Callbacks

```python
async def on_thermal_change(intelligence):
    if intelligence.state == ThermalState.HOT:
        print(f"🔥 System hot: {max(intelligence.stats.current.zones.values()):.1f}°C")

# Register callback
thermal.register_callback(on_thermal_change)
```

---

## File Structure

```
s25_thermal/
├── s25_thermal.py           Main intelligence system
├── config.py                Thermal configuration
├── shared_types.py          Type definitions
├── requirements.txt         Dependencies
├── README.md                This file
├── CHANGELOG.md             Version history
├── LICENSE                  MIT license
├── example_basic.py         Basic usage
├── example_prediction.py    Prediction demo
└── example_monitoring.py    Monitoring setup
```

---

## How It Works

The system operates through several components:

1. **Telemetry Collection** - Multi-zone temperature sensor reading
2. **Physics Engine** - Newton's law of cooling, thermal mass modeling
3. **Pattern Recognition** - Command thermal signature learning
4. **Statistical Analysis** - Trend detection, anomaly identification
5. **Prediction Engine** - Future temperature forecasting
6. **Intelligence Assembly** - Comprehensive thermal state reporting

Temperature changes follow physics laws, not just thresholds. The system models:
- Thermal mass (heat capacity of device)
- Thermal resistance (heat dissipation rate)
- Ambient coupling (environmental heat exchange)
- Zone correlations (how components affect each other)

Pattern learning uses exponential moving averages to build command thermal signatures, enabling predictive workload scheduling.

---

## Production Notes

**Tested on:** Samsung Galaxy S25+ (Snapdragon 8 Elite)

**Production Stats:**
- 500+ concurrent users
- Zero thermal crashes
- Average uptime: 168+ hours
- Thermal-induced queue rate: < 2%

**Key Insights:**
- Battery temperature is the critical throttling threshold (42°C on S25+)
- 5G adds ~3-4°C vs WiFi
- Charging adds ~5-7°C baseline
- Predictive queuing reduces throttling by 95% vs reactive

---

## Troubleshooting

### Sensors Not Found

On Termux, ensure permissions:

```bash
termux-setup-storage
pkg install termux-api
```

Install Termux:API app from F-Droid.

### Prediction Inaccuracy

Increase sample history:

```python
# In config.py
THERMAL_HISTORY_SIZE = 2000
```

Wait 10-15 minutes for accurate ambient baseline.

### Pattern Not Learning

Ensure sufficient samples:

```python
signature = thermal.patterns.get_thermal_impact("command")
if signature:
    print(f"Samples: {signature.sample_count}")
    # Need 10+ samples for high confidence
```

---

## Contact

**Jesse Vogeler-Wunsch** @ PNGN-Tec LLC

Reach me on Discord: **@DaSettingsPNGN**

Part of the PNGN performance systems suite for resource-constrained environments.

---

*Built on a phone. Optimized for mobile-first performance.*

## License

MIT License. See LICENSE file.

---
