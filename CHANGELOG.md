# Changelog

## Current State

### Validation Metrics
Validated over 152,418 predictions (6.25 hours continuous operation):
- Overall: 0.58°C MAE (transients filtered), 1.09°C MAE (raw)
- Steady-state: 0.47°C MAE
- Battery: 0.24°C MAE
- 96.5% within 5°C (3.5% transients during load changes)
- Stress test (95°C+ CPUs): 1.23°C MAE recovery tracking

### Architecture
**Multi-zone monitoring:**
- CPU_BIG, CPU_LITTLE, GPU, BATTERY, MODEM, CHASSIS, AMBIENT
- 1s sampling interval (THERMAL_SAMPLE_INTERVAL = 1.0)
- 30s prediction horizon
- Per-zone thermal constants from hardware measurement

**Physics engine:**
- Newton's law of cooling with measured τ per zone
- Battery: τ=210s, simplified power integration
- Fast zones (CPU/GPU): τ=50-95s, full exponential model
- Dual-confidence system: physics × sample-size weighting
- Observed peak predictions when zones are throttled

**Dual-condition throttle:**
- Battery temperature: Predicted temp vs 38.5°C threshold (2° safety margin)
- CPU velocity: >3.0°C/s indicates regime change
- Prevents both slow battery heating and fast CPU spikes

### API
**Primary interface:**
```python
thermal.get_tank_status() → ThermalTankStatus
```
Returns battery temps, should_throttle bool, thermal budget, cooling rate.

**Additional methods:**
```python
thermal.get_current() → ThermalSample  # Current readings
thermal.get_prediction() → ThermalPrediction  # 30s forecast
thermal.get_display_status() → Dict  # UI formatting
thermal.get_statistics() → Dict  # Runtime stats
```

### Performance
**From 152k prediction validation:**
- Overall: 0.58°C MAE (transients filtered), 1.09°C MAE (raw)
- Steady-state: 0.47°C MAE
- Battery: 0.24°C MAE (τ=210s, most predictable)
- CPUs: 0.83-0.88°C MAE (τ=50-60s)
- GPU: 0.84°C MAE (τ=95s)
- 96.5% of predictions within 5°C

### Technical Details
**Thermal constants:**
- Measured from step response testing
- Per-zone: thermal mass, thermal resistance, ambient coupling
- Hardware-specific (Snapdragon 8 Elite for Galaxy)
- CPU_BIG: τ=50s, CPU_LITTLE: τ=60s, GPU: τ=95s
- BATTERY: τ=210s, MODEM: τ=80s, CHASSIS: τ=100s

**Sampling:**
- 1s interval (THERMAL_SAMPLE_INTERVAL = 1.0)
- Uniform sampling across all zones
- Battery uses simplified power integration due to high τ

**Storage:**
- Memory-mapped persistence for adaptive tuner state
- Thermal samples in-memory deque (50 min history)
- Predictions stored for validation

### Files
- s25_thermal.py (4160 lines) - Complete system
- Types defined inline (ThermalZone, ThermalState, ThrottleReason, etc.)
- Single file architecture for minimal thermal overhead

---

## Design Philosophy

Physics-based predictions with empirical validation. When physics breaks (throttled regimes, regime changes), use observed data. Dual-condition throttle catches both slow battery heating and fast CPU spikes.

System optimized for production stability on mobile hardware with variable thermal environments.
