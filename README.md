# 🐧 S25+ Thermal Intelligence System

**Copyright (c) 2025 PNGN-Tec LLC**  
**Author: Jesse Vogeler-Wunsch (@DaSettingsPNGN)**

Predictive thermal management for Samsung Galaxy S25+ using physics-based modeling.

---

## 📟 Table of Contents

- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Results](#results)
- [Requirements](#requirements)
- [License](#license)
- [Contact](#contact)

---

## 🤯 The Problem

I'm running a production Discord bot server on my Samsung S25+. The Snapdragon 8 Elite processor throttles at 42°C, which was causing:

- **Performance degradation** - CPU throttles down when hitting thermal limits
- **Unpredictable behavior** - No way to anticipate when throttling would occur
- 🪫 **Battery drain** - Inefficient thermal management wastes power
- **Poor user experience** - Bot becomes unresponsive during thermal events

Traditional monitoring only tells you the temperature **now**. I needed to know:
- What will the temperature be in 60 seconds?
- Which operations cause the most heating?
- How long until I hit thermal throttling?
- Should I delay the next heavy operation?

---

## 💜 The Solution

A complete thermal intelligence system that:

✅ **Monitors** 8 thermal zones in real-time (CPU, GPU, battery, modem, NPU, camera, skin, ambient)  
✅ **Predicts** future temperatures using Newton's law of cooling  
✅ **Learns** thermal signatures of different operations  
✅ **Adapts** workload based on thermal state and trends  
✅ **Alerts** when approaching thermal limits with time-to-throttle estimates  
💾 **Persists** learned patterns across restarts  

---

## 👾 Quick Start

```bash
# Install
pip install numpy
git clone https://github.com/yourusername/s25-thermal-intelligence.git
cd s25-thermal-intelligence

# Run example
python example_usage.py
```

**Basic usage:**

```python
import asyncio
from s25_thermal import create_thermal_intelligence

async def main():
    # Create and start monitoring
    thermal = create_thermal_intelligence()
    await thermal.start()
    
    # Get current intelligence
    intel = thermal.get_current_intelligence()
    
    print(f"Temperature: {intel.temperature:.1f}°C")
    print(f"State: {intel.state.name}")
    print(f"Trend: {intel.trend.name}")
    
    # Check prediction
    if intel.prediction:
        print(f"Predicted (60s): {intel.prediction.predicted_temps['cpu_big']:.1f}°C")
        print(f"Thermal budget: {intel.prediction.thermal_budget:.0f} seconds")
    
    # Stop monitoring
    await thermal.stop()

asyncio.run(main())
```

---

## ☢️ Key Features

### 📟 Physics-Based Prediction

Uses actual thermodynamics to predict future temperatures:

- **Thermal mass**: 50 J/°C (heat capacity of device)
- **Thermal resistance**: 5°C/W (cooling efficiency)
- **Ambient coupling**: 0.3 (heat transfer to environment)
- **Newton's law of cooling** for accurate time-series prediction

```python
# Example prediction
prediction = intel.prediction
print(f"Current: {intel.temperature:.1f}°C")
print(f"In 60s: {prediction.predicted_temps['battery']:.1f}°C")
print(f"Time until throttle: {prediction.thermal_budget:.0f}s")
print(f"Confidence: {prediction.confidence:.1%}")
```

### 🧟 Pattern Recognition

Learns which operations cause heating:

```python
# Track an operation
thermal.track_command("heavy_render", operation_id)

# ... your operation runs ...

# Complete tracking
thermal.complete_command("heavy_render", operation_id)

# Later: check thermal impact
signature = thermal.patterns.get_thermal_impact("heavy_render")
print(f"Average temp rise: {signature.avg_delta_temp:.2f}°C")
print(f"Peak temp rise: {signature.peak_delta_temp:.2f}°C")
print(f"Duration: {signature.duration:.1f}s")
```

### 📟 Statistical Analysis

Real-time statistical monitoring with anomaly detection:

```python
stats = intel.stats

# Current readings
print(f"Mean (1min): {stats.mean_1m['cpu_big']:.1f}°C")
print(f"Max (1min): {stats.max_1m['cpu_big']:.1f}°C")
print(f"Std deviation: {stats.std_dev['cpu_big']:.2f}°C")

# Percentiles
p95 = stats.percentiles[95]['cpu_big']
print(f"95th percentile: {p95:.1f}°C")

# Anomalies (z-score > 3.0)
for timestamp, description in intel.anomalies:
    print(f"[{timestamp}] Anomaly: {description}")
```

### 📟 Network Awareness

Tracks thermal impact of network connectivity:

| Network Type | Thermal Impact |
|-------------|----------------|
| WiFi 2.4GHz | +0°C |
| WiFi 5GHz | +1°C |
| 4G | +3°C |
| 5G | +5°C ☣️ |

```python
# Check current network impact
if intel.stats.network_impact > 3.0:
    print("☣️ 5G causing significant heating!")
```

### 💾 Persistent Learning

Automatically saves and loads learned patterns:

```python
# Patterns are saved every 5 minutes automatically
# And loaded on startup

# Manual save
await thermal.save_signatures()

# Check what's been learned
stats = thermal.get_statistics()
print(f"Patterns learned: {stats['patterns_learned']}")
print(f"High confidence: {stats['persistence']['high_confidence_patterns']}")
```

---

## 🐧 Installation

### Requirements

- **Python 3.11+** (uses modern async features)
- **numpy** (for physics calculations)
- **Android device with Termux** (optional, works on any platform)

### Install Dependencies

```bash
pip install numpy
```

### Clone Repository

```bash
git clone https://github.com/yourusername/s25-thermal-intelligence.git
cd s25-thermal-intelligence
```

### Verify Installation

```bash
python verify_thermal.py
```

You should see:
```
✅ All critical tests passed!
✅ Thermal intelligence system is working
```

---

## 👾 Usage

### Basic Monitoring

```python
import asyncio
from s25_thermal import create_thermal_intelligence

async def monitor():
    thermal = create_thermal_intelligence()
    await thermal.start()
    
    # Monitor for 10 seconds
    for _ in range(10):
        intel = thermal.get_current_intelligence()
        print(f"{intel.temperature:.1f}°C - {intel.state.name}")
        await asyncio.sleep(1)
    
    await thermal.stop()

asyncio.run(monitor())
```

### Check Before Heavy Operation

```python
async def should_run_heavy_task():
    intel = thermal.get_current_intelligence()
    
    # Check current state
    if intel.state == ThermalState.CRITICAL:
        print("❌ Too hot - skip operation")
        return False
    
    # Check prediction
    if intel.prediction:
        if intel.prediction.thermal_budget < 30:
            print("🤬 Less than 30s until throttle - wait")
            return False
    
    print("✅ Safe to proceed")
    return True
```

### Adaptive Workload Management

```python
async def adaptive_task_runner():
    thermal = create_thermal_intelligence()
    await thermal.start()
    
    while True:
        intel = thermal.get_current_intelligence()
        
        if intel.state == ThermalState.OPTIMAL:
            # ✅ Run at full speed
            await run_tasks(batch_size=10)
        
        elif intel.state == ThermalState.WARM:
            # 😈 Reduce load
            await run_tasks(batch_size=5)
            await asyncio.sleep(2)  # Cool down period
        
        elif intel.state == ThermalState.HOT:
            # 🤯 Minimal load
            await run_tasks(batch_size=1)
            await asyncio.sleep(5)  # Longer cool down
        
        else:  # CRITICAL
            # ❌ Stop and wait
            print("🤬 Critical temperature - pausing")
            await asyncio.sleep(10)
```

### Real-time Dashboard

```python
async def thermal_dashboard():
    thermal = create_thermal_intelligence()
    await thermal.start()
    
    try:
        while True:
            intel = thermal.get_current_intelligence()
            
            # Clear screen
            print("\033[2J\033[H")
            
            # Display dashboard
            print("=" * 60)
            print("🐧 S25+ THERMAL DASHBOARD 📟")
            print("=" * 60)
            print()
            
            # Current temperatures
            print("📟 Current Temperatures:")
            for zone, temp in intel.stats.current.zones.items():
                bar = "█" * int(temp / 2)  # Visual bar
                print(f"  {zone.name:12s} {temp:5.1f}°C {bar}")
            
            print()
            
            # State and trend
            state_emoji = {
                ThermalState.OPTIMAL: "✅",
                ThermalState.WARM: "😈",
                ThermalState.HOT: "🤬",
                ThermalState.CRITICAL: "🤯"
            }
            
            print(f"State: {state_emoji.get(intel.state, '❓')} {intel.state.name}")
            print(f"Trend: {intel.trend.name}")
            print(f"Confidence: {intel.confidence:.1%}")
            
            # Prediction
            if intel.prediction:
                print()
                print("👾 Prediction (60s):")
                pred = intel.prediction.predicted_temps
                print(f"  CPU: {pred.get(ThermalZone.CPU_BIG, 0):.1f}°C")
                print(f"  GPU: {pred.get(ThermalZone.GPU, 0):.1f}°C")
                print(f"  Battery: {pred.get(ThermalZone.BATTERY, 0):.1f}°C")
                print(f"  Thermal budget: {intel.prediction.thermal_budget:.0f}s")
            
            # Recommendations
            if intel.recommendations:
                print()
                print("💜 Recommendations:")
                for rec in intel.recommendations[:3]:
                    print(f"  • {rec}")
            
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        await thermal.stop()

asyncio.run(thermal_dashboard())
```

---

## 🧸 Architecture

```
ThermalIntelligenceSystem (Coordinator)
│
├── ThermalTelemetryCollector
│   ├── Reads from /sys/class/thermal/thermal_zone*/temp
│   ├── Reads from /sys/class/power_supply/battery/temp
│   ├── Calls Termux API (termux-battery-status)
│   └── Enriches with network/charging/screen state
│
├── ThermalPhysicsEngine
│   ├── Calculates temperature velocity (°C/s)
│   ├── Calculates acceleration (°C/s²)
│   ├── Predicts future temperatures (Newton's law)
│   └── Estimates thermal budget (time to throttle)
│
├── ThermalPatternEngine
│   ├── Learns command thermal signatures
│   ├── Correlates workload with temperature changes
│   ├── Builds thermal impact database
│   └── Provides predictive recommendations
│
└── ThermalStatisticalAnalyzer
    ├── Computes rolling statistics (mean, median, std)
    ├── Calculates percentiles (5th, 25th, 75th, 95th)
    ├── Detects anomalies (z-score > 3.0)
    └── Tracks thermal cycles and time above thresholds
```

### Data Flow

```
1. Telemetry Collection (every 10s)
   └─> ThermalSample with all zone temperatures
   
2. Physics Analysis
   └─> ThermalVelocity + ThermalPrediction
   
3. Statistical Analysis
   └─> ThermalStatistics with percentiles
   
4. Pattern Matching
   └─> ThermalSignatures for active commands
   
5. Intelligence Assembly
   └─> ThermalIntelligence (complete package)
   
6. 💾 Persistence (every 5 minutes)
   └─> Save learned patterns to thermal_signatures.json
```

---

## ⚙️ Configuration

Edit `config.py` to customize behavior:

### Temperature Thresholds

```python
THERMAL_TEMP_COLD = 35.0
THERMAL_TEMP_OPTIMAL_MIN = 35.0
THERMAL_TEMP_OPTIMAL_MAX = 45.0
THERMAL_TEMP_WARM = 55.0
THERMAL_TEMP_HOT = 65.0
THERMAL_TEMP_CRITICAL = 60.0
```

### Physics Constants (S25+ Tuned)

```python
S25_THERMAL_MASS = 50.0          # J/°C
S25_THERMAL_RESISTANCE = 5.0     # °C/W
S25_AMBIENT_COUPLING = 0.3       # coefficient
S25_MAX_TDP = 15.0               # Watts
```

### Sampling

```python
THERMAL_SAMPLE_INTERVAL_MS = 10000   # Sample every 10 seconds
THERMAL_HISTORY_SIZE = 1000          # Keep 1000 samples (~3 hours)
THERMAL_PREDICTION_HORIZON = 60.0    # Predict 60 seconds ahead
```

### Pattern Recognition

```python
THERMAL_SIGNATURE_WINDOW = 300       # 5 minute window
THERMAL_CORRELATION_THRESHOLD = 0.7  # Min correlation
THERMAL_LEARNING_RATE = 0.1          # Learning rate
```

---

## 🦄 Technical Details

### Thermal Zones Monitored

| Zone | Path | Description |
|------|------|-------------|
| cpu_big | `/sys/class/thermal/thermal_zone0/temp` | Cortex-X925 performance cores |
| cpu_little | `/sys/class/thermal/thermal_zone1/temp` | Efficiency cores |
| gpu | `/sys/class/thermal/thermal_zone2/temp` | Adreno 830 GPU |
| battery 🔋 | `/sys/class/power_supply/battery/temp` | Battery sensor (via Termux API) |
| skin | `/sys/class/thermal/thermal_zone3/temp` | Device surface |
| modem | `/sys/class/thermal/thermal_zone4/temp` | 5G modem |
| npu | `/sys/class/thermal/thermal_zone5/temp` | AI Engine |
| camera | `/sys/class/thermal/thermal_zone6/temp` | Camera module |
| ambient | `/sys/class/thermal/thermal_zone7/temp` | Environmental |

### Thermal States

| State | Range | Behavior | Emoji |
|-------|-------|----------|-------|
| COLD | < 35°C | Device is cool | 😴 |
| OPTIMAL | 35-45°C | Normal operation, no throttling | ✅ |
| WARM | 45-55°C | Elevated temperature, light throttling | 😈 |
| HOT | 55-65°C | High temperature, noticeable throttling | 🤬 |
| CRITICAL | > 60°C | Dangerous temperature, heavy throttling | 🤯 |

### Velocity Classification

| Trend | Rate | Meaning | Direction |
|-------|------|---------|-----------|
| RAPID_COOLING | < -0.033°C/s | Cooling fast | ⬅️ |
| COOLING | -0.033 to -0.008°C/s | Cooling gradually | ⬅️ |
| STABLE | -0.008 to +0.008°C/s | Temperature stable | 🐧 |
| WARMING | +0.008 to +0.033°C/s | Heating gradually | ➡️ |
| RAPID_WARMING | > +0.033°C/s | Heating fast | ➡️ |

### Physics Equations

**Temperature prediction (Newton's law of cooling):**

```
T(t) = T_ambient + (T_current - T_ambient) * e^(-t / τ)

where:
  τ = thermal_resistance * thermal_mass
  t = time in seconds
```

**Thermal budget calculation:**

```
t_budget = -τ * ln((T_throttle - T_ambient) / (T_current - T_ambient))

where:
  T_throttle = 42°C (Samsung throttle point)
```

---

## 📟 Results

### Before Thermal Intelligence

- Temperature: **45°C** (constant throttling) 🤬
- Performance: **Inconsistent** (spikes and crashes) 🤯
- Battery life: **Poor** (inefficient thermal management) 🪫
- Predictability: **None** (reactive only) ❌

### After Thermal Intelligence

- Temperature: **39.9°C** (stable, below throttle) ✅
- Performance: **Consistent** (adaptive workload) 💜
- Battery life: **Improved** (efficient thermal management) 🔋
- Predictability: **High** (60s lookahead with 75%+ confidence) 👾

**Key Achievement:** Maintains production Discord bot at 39.9°C - just below Samsung's 42°C throttle threshold - while handling real user traffic. 🐧

---

## ☢️ Requirements

### Minimum Requirements

- **Python**: 3.11 or higher
- **numpy**: For physics calculations
- **Operating System**: Any (Linux, macOS, Windows, Android/Termux)

### Recommended for Full Features

- **Android device** with Termux 📟
- **Root not required** (uses standard Android APIs) ✅
- **Thermal zones accessible** (most modern Android devices)
- **8+ GB RAM** (for production workloads)

### Tested On

- ✅ Samsung Galaxy S25+ (Snapdragon 8 Elite)
- ✅ Ubuntu 22.04 (fallback mode)
- ✅ macOS 14 (fallback mode)
- ❌ Windows (limited thermal zone access)

---

## 💜 License

```
MIT License

Copyright (c) 2025 PNGN-Tec LLC
Author: Jesse Vogeler-Wunsch (@DaSettingsPNGN)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🐧 Contact

**Author:** Jesse Vogeler-Wunsch  
**Discord:** @DaSettingsPNGN  
**Company:** PNGN-Tec LLC  
**Project:** Part of the PNGN Bot ecosystem 🐧

**Questions? Issues? Improvements?**  
Open an issue on GitHub or reach out on Discord!

---

## 🦄 Acknowledgments

- Built for running production servers on phones 📟
- Optimized for Samsung Galaxy S25+ (Snapdragon 8 Elite) 💜
- Part of the PNGN-Tec suite of performance tools 👾

**Why run a server on a phone?** Because I can. And because it's awesome. 🐧

---

**Made with love for the terminal life** 💜🐧📟