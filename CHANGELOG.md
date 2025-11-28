# Changelog

## Current State

### Validation Metrics
Validated over 457,177 predictions (18.7 hours continuous operation):
- Overall steady-state: 1.22°C MAE, 73% within 1°C, 86.4% within 2°C
- Battery: 0.375°C MAE, 0.2°C median, 91.5% within 1°C
- Normal operation: 0.76°C MAE, 77.7% within 1°C
- Temperature range validated: 2°C to 95°C (cold boot to near-TJmax)
- 97.8% steady-state, 2.2% transients during regime changes

**Per-zone accuracy (steady-state):**
| Zone | MAE | Median | Within 1°C | Temp Range |
|------|-----|--------|------------|------------|
| BATTERY | 0.375°C | 0.20°C | 91.5% | 2°C – 39°C |
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

### Architecture
**Multi-zone monitoring:**
- CPU_BIG, CPU_LITTLE, GPU, BATTERY, MODEM, CHASSIS, AMBIENT
- 1s sampling interval (THERMAL_SAMPLE_INTERVAL = 1.0)
- 30s prediction horizon
- Per-zone thermal constants from hardware measurement

**Physics engine:**
- Newton's law of cooling with measured τ per zone
- Battery: τ=210s, I²R power from measured current
- Fast zones (CPU/GPU): τ=25-35s, full exponential model
- Dual-confidence system: physics × sample-size weighting
- Observed peak predictions when zones are throttled

**Dual-condition throttle:**
- Battery temperature: Predicted temp vs 38.5°C threshold (safety margin)
- CPU velocity: >3.0°C/s indicates regime change
- Prevents both slow battery heating and fast CPU spikes

### API
**Primary interface:**
```python
thermal.get_tank_status() → ThermalTankStatus
```
Returns battery temps, should_throttle bool, throttle_reason, headroom, cooling rate, CPU velocities.

**Additional methods:**
```python
thermal.get_current() → ThermalSample  # Current readings
thermal.get_prediction() → ThermalPrediction  # 30s forecast
thermal.get_display_status() → Dict  # UI formatting
thermal.get_statistics() → Dict  # Runtime stats
```

### Technical Details
**Thermal constants (measured from step response testing):**
- CPU_BIG: τ=25s, thermal mass=20 J/K
- CPU_LITTLE: τ=35s, thermal mass=40 J/K
- GPU: τ=30s, thermal mass=40 J/K
- BATTERY: τ=210s, thermal mass=75 J/K
- MODEM: τ=145s, thermal mass=35 J/K
- CHASSIS: τ=100s, thermal mass=40 J/K

**Observed peaks (from validation):**
- CPU_BIG: 84°C (starts throttling at 45°C)
- CPU_LITTLE: 98°C (starts throttling at 48°C)
- GPU: 66°C (starts throttling at 38°C)
- MODEM: 68°C (starts throttling at 40°C)

**Sampling:**
- 1s interval (THERMAL_SAMPLE_INTERVAL = 1.0)
- Uniform sampling across all zones
- 60 sample warmup for ambient fitting
- Battery uses I²R from measured current for power estimation

**Storage:**
- Memory-bounded collections (deque maxlen, MAX_PENDING_VALIDATIONS=1000)
- Numpy validation arrays with auto-flush at 10k predictions
- Single file architecture for minimal thermal overhead

### Files
- s25_thermal.py (3949 lines) - Complete system
- Types defined inline (ThermalZone, ThermalState, ThrottleReason, etc.)
- Single file architecture for minimal thermal overhead on mobile

---

## How This Compares

| System | MAE | Conditions |
|--------|-----|------------|
| **This system (battery)** | **0.375°C** | Production phone, unknown workload, 30s horizon |
| PINN + LSTM (2025) | 0.29°C | Lab, known 2.0C charge rate, same cell |
| FCN-GBM hybrid (2024) | 0.46°C | Lab, 20% training data from test cell |
| RNN benchmark (2024) | 0.15°C | Lab, Bayesian-optimized hyperparameters |

The sub-0.3°C systems do **interpolation**—trained on the exact conditions they test on. This does **extrapolation** with no training data, predicting 30s into unknown workloads on production hardware.

---

## Design Philosophy

Physics-based predictions with empirical validation. When physics breaks (throttled regimes, regime changes), use observed data. Dual-condition throttle catches both slow battery heating and fast CPU spikes.

System optimized for production stability on mobile hardware with variable thermal environments.
