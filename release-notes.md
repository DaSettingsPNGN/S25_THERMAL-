# 🔥🐧🔥 Release Notes - S25+ Thermal Intelligence

**Physics-based thermal management with dual-condition throttle detection**

Validated over 457k predictions (18.7 hours continuous operation):
- Overall steady-state: 1.22°C MAE, 73% within 1°C
- Battery: 0.375°C MAE, 0.2°C median
- Temperature range: 2°C to 95°C (cold boot to near-TJmax)
- Normal operation: 0.76°C MAE, 77.7% within 1°C

---

## System Design

### Dual-Condition Throttle

**Problem:** Single-condition (battery temp) had blind spot for regime changes.

**Solution:**
1. **Battery temperature** - Predicts heating from sustained load (τ=210s)
2. **CPU velocity** - Detects regime changes (>3.0°C/s indicates workload spike)

Combined approach catches both slow battery heating and fast CPU spikes.

### Observed Peak Prediction

When zones reach throttle temperatures, physics breaks down (model assumes constant power, but throttling changes power mid-flight). Solution: use empirical data.

**Observed peaks from validation:**
- CPU_BIG: 84.0°C (starts throttling at 45°C)
- CPU_LITTLE: 98.0°C (starts throttling at 48°C)  
- GPU: 66.0°C (starts throttling at 38°C)
- MODEM: 68.0°C (starts throttling at 40°C)

When `current_temp >= throttle_start`, predict observed_peak instead of using Newton's law.

---

## Thermal Constants

**Measured from hardware:**
- CPU_BIG: τ=25s, thermal mass=20 J/K
- CPU_LITTLE: τ=35s, thermal mass=40 J/K
- GPU: τ=30s, thermal mass=40 J/K
- BATTERY: τ=210s, thermal mass=75 J/K
- MODEM: τ=145s, thermal mass=35 J/K
- CHASSIS: τ=100s, thermal mass=40 J/K

**Sampling:**
- 1s interval (THERMAL_SAMPLE_INTERVAL = 1.0)
- 30s prediction horizon
- Minimum 60 samples before predictions enabled (1 min warmup)

---

## How This Compares

| System | MAE | Conditions |
|--------|-----|------------|
| **This system (battery)** | **0.375°C** | Production phone, unknown workload, 30s horizon |
| PINN + LSTM (2025) | 0.29°C | Lab, known 2.0C charge rate, same cell |
| FCN-GBM hybrid (2024) | 0.46°C | Lab, 20% training data from test cell |
| RNN benchmark (2024) | 0.15°C | Lab, Bayesian-optimized hyperparameters |

The sub-0.3°C systems are doing **interpolation**—trained on the exact conditions they test on. This is **extrapolation** with no training data.

---

## API

**ThermalTankStatus structure:**
```python
@dataclass
class ThermalTankStatus:
    battery_temp_current: float
    battery_temp_predicted: float
    should_throttle: bool
    throttle_reason: ThrottleReason      # Why throttling
    headroom_seconds: float
    cooling_rate: float
    cpu_big_velocity: float              # Regime detector
    cpu_little_velocity: float           # Regime detector
```

**ThrottleReason enum:**
```python
class ThrottleReason(Enum):
    NONE = auto()           # Safe
    BATTERY_TEMP = auto()   # Battery too hot
    CPU_VELOCITY = auto()   # Regime change (>3.0°C/s)
    BOTH = auto()           # Both triggered
```

**Key constants:**
```python
THERMAL_PREDICTION_HORIZON = 30.0       # seconds
THERMAL_SAMPLE_INTERVAL = 1.0           # 1s sampling
TANK_THROTTLE_TEMP = 38.5               # Battery °C (safety margin)
CPU_VELOCITY_DANGER = 3.0               # °C/s regime change threshold
```

---

## Usage

```python
import asyncio
from s25_thermal import create_thermal_intelligence, ThrottleReason

async def main():
    thermal = create_thermal_intelligence()
    await thermal.start()
    
    tank = thermal.get_tank_status()
    
    if tank.should_throttle:
        if tank.throttle_reason == ThrottleReason.CPU_VELOCITY:
            # Regime change - wait for stabilization
            await asyncio.sleep(30)
        elif tank.throttle_reason == ThrottleReason.BATTERY_TEMP:
            # Battery hot - wait for cooling
            await asyncio.sleep(tank.headroom_seconds)
        else:
            # Both conditions - wait longer
            await asyncio.sleep(60)
    else:
        # Safe to operate
        await execute_work()
    
    await thermal.stop()

asyncio.run(main())
```

---

## Requirements

- Python 3.8+
- numpy ≥1.20.0
- Samsung Galaxy S25+ (Snapdragon 8 Elite) or compatible device

---

## License

MIT License - Copyright (c) 2025 PNGN-Tec LLC

---

**Physics-based thermal management with dual-condition throttle detection.** 🔥
