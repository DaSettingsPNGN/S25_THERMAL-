# Changelog

## [Current State] - 2025-11-05

### Production Status
- Discord bot serving 645+ members
- Samsung Galaxy S25+ deployment
- 24/7 operation with zero thermal shutdowns
- Physics-based thermal management preventing throttling

### Architecture
**Multi-zone monitoring:**
- CPU_BIG, CPU_LITTLE, GPU, BATTERY, MODEM, CHASSIS
- 10s sampling interval
- 30s prediction horizon
- Per-zone thermal constants from hardware measurement

**Physics engine:**
- Newton's law of cooling with measured τ per zone
- Battery: τ=540s, simplified power integration
- Fast zones (CPU/GPU): τ<10s, full exponential model
- Dual-confidence system: physics × sample-size weighting

**Adaptive damping:**
- TransientResponseTuner tracks prediction errors by regime
- Heating/cooling/stable regimes with separate tuning
- History sizes scaled to thermal time constants
- Momentum factors adapt to reduce bias over time

**Thermal tank:**
- Simple bool throttle decision (should_throttle)
- Battery-centric (Samsung throttles at 42°C)
- Thermal budget calculation (seconds until throttle)
- Cooling rate and recommended delay

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
**Prediction accuracy:** ~1.5°C MAE at 30s horizon

**Characteristics:**
- Battery zone most predictable (τ=540s)
- Fast zones harder to predict (τ<10s)
- Adaptive damping improves accuracy over time
- Network state (5G vs WiFi) affects baseline
- Charging state significantly impacts temperature

### Technical Details
**Thermal constants:**
- Measured from step response testing
- Per-zone: thermal mass, thermal resistance, ambient coupling
- Hardware-specific (Snapdragon 8 Elite)

**Sampling:**
- Uniform 10s intervals
- Battery: 10s backdate from 30s moving average
- Die sensors: no backdating (report current temp)

**Storage:**
- Memory-mapped persistence for adaptive tuner state
- Thermal samples in-memory deque (50 min history)
- Predictions stored for validation

### Files
- s25_thermal.py (2454 lines) - Complete system
- Types defined inline (ThermalZone, ThermalState, etc.)
- No shared_types dependency
- Single file architecture

### Removed Features
- Pattern learning / command thermal signatures (not in current code)
- Recommendation system (not in current code)
- Complex event callback infrastructure (simplified)

### Current Focus
System optimized for production stability and accurate battery temperature prediction to prevent Samsung throttling at 42°C. Adaptive damping provides continuous improvement in prediction accuracy over extended runtime.

---

## Historical Context

Previous iterations explored:
- Command thermal signature learning
- Multi-rate sampling strategies
- Various backdating approaches for die sensors
- Different confidence scoring methods

Current version represents production-validated physics model with measured accuracy metrics and proven stability over 24/7 operation.
