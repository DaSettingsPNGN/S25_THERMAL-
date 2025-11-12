# Changelog

## [Current State] - 2025-11-05

### Production Status
- Discord bot serving 600+ members
- Samsung Galaxy S25+ deployment  
- 24/7 operation with zero thermal shutdowns
- Physics-based thermal management preventing throttling
- Mobile computing: flagship phone as production infrastructure

### Architecture
**Multi-zone monitoring:**
- CPU_BIG, CPU_LITTLE, GPU, BATTERY, MODEM, CHASSIS
- 10s sampling interval
- 30s prediction horizon
- Per-zone thermal constants from hardware measurement

**Physics engine:**
- Newton's law of cooling with measured τ per zone
- Battery: τ=540s, simplified power integration
- Fast zones (CPU/GPU): τ=14-23s, full exponential model
- Dual-confidence system: physics × sample-size weighting
- Observed peak predictions when components throttled

**Dual-condition throttle:**
- Battery temperature prediction (Samsung throttles at 42°C)
- CPU velocity spike detection (>3.0°C/s indicates regime change)
- Prevents thermal runaway before physics model breaks

**Thermal tank:**
- Simple bool throttle decision (should_throttle)
- Battery-centric with CPU velocity early warning
- Thermal budget calculation (seconds until throttle)
- Cooling rate and recommended delay

### API
**Primary interface:**
```python
thermal.get_tank_status() → ThermalTankStatus
```
Returns battery temps, should_throttle bool, throttle_reason enum, thermal budget, cooling rate, CPU velocities.

**Additional methods:**
```python
thermal.get_current() → ThermalSample  # Current readings
thermal.get_prediction() → ThermalPrediction  # 30s forecast
thermal.get_display_status() → Dict  # UI formatting
thermal.get_statistics() → Dict  # Runtime stats
```

### Performance
**Prediction accuracy:** ~0.5°C MAE at 30s horizon (normal regime)

**Characteristics:**
- Battery zone most predictable (τ=540s, 0.04°C MAE)
- CPU zones accurate despite 20°C/s spikes (0.5-0.6°C MAE)
- Throttled zones use observed peaks (avoids -30 to -60°C errors)
- Network state (5G vs WiFi) affects baseline
- Charging state significantly impacts temperature

### Technical Details
**Thermal constants:**
- Measured from cooling curve analysis (1105 samples)
- Exponential decay fits: T(t) = T_amb + (T₀ - T_amb)·exp(-t/τ)
- Per-zone: thermal mass, thermal resistance, ambient coupling
- Hardware-specific (Snapdragon 8 Elite for Galaxy)

**Sampling:**
- Uniform 1s intervals via termux-api (non-rooted)
- No backdating - all sensors report current temperature
- Battery current measured for power-aware predictions

**Storage:**
- Thermal samples in-memory deque (120 samples = 2 min history)
- Predictions stored for validation
- Single-file architecture optimized for Termux filesystem constraints

### Files
- s25_thermal.py (2478 lines) - Complete system
- Types defined inline (ThermalZone, ThermalState, ThrottleReason, etc.)
- Single file architecture (import overhead matters on mobile)

### Removed Features
- Adaptive damping / TransientResponseTuner (dead code, removed)
- Pattern learning / command thermal signatures (not implemented)
- Complex event callback infrastructure (simplified to tank status)

### Current Focus
System optimized for production stability and accurate temperature prediction to prevent Samsung throttling at 42°C. Dual-condition throttle (battery + velocity) provides comprehensive protection from thermal issues. Mobile-first design for running production workloads on flagship phones instead of paying for cloud servers.

---

## Historical Context

Previous iterations explored:
- Adaptive damping (removed - raw physics was already accurate)
- Command thermal signature learning
- Multi-rate sampling strategies
- Various backdating approaches for die sensors
- Different confidence scoring methods

Current version represents production-validated physics model with measured accuracy metrics (21,973 predictions validated) and proven stability over 24/7 operation on mobile hardware.
