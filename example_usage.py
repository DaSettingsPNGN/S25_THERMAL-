#!/usr/bin/env python3
"""
🔥🐧🔥 S25+ Thermal Intelligence - Usage Example
===========================================
Copyright (c) 2025 PNGN-Tec LLC

Demonstrates dual-condition throttle system:
- Battery temperature prediction (0.375°C MAE, 0.2°C median)
- CPU velocity spike detection (>3.0°C/s regime change)
- Multi-zone thermal monitoring

Validated over 457k predictions (18.7 hours continuous operation).
"""

import asyncio
from s25_thermal import create_thermal_intelligence, ThermalZone

async def main():
    print("🔥 Initializing S25+ Thermal Intelligence...")
    thermal = create_thermal_intelligence()
    
    await thermal.start()
    print("✅ Monitoring started!\n")
    
    try:
        for i in range(60):
            sample = thermal.get_current()
            tank = thermal.get_tank_status()
            prediction = thermal.get_prediction()
            
            if sample and tank:
                # Get key temperatures
                battery = sample.zones.get(ThermalZone.BATTERY, 0.0)
                cpu_big = sample.zones.get(ThermalZone.CPU_BIG, 0.0)
                cpu_little = sample.zones.get(ThermalZone.CPU_LITTLE, 0.0)
                
                # Display current status
                print(f"[{i+1:2d}s] Battery: {battery:.1f}°C | "
                      f"CPU_BIG: {cpu_big:.1f}°C | "
                      f"CPU_LIT: {cpu_little:.1f}°C")
                
                # Show tank status - dual throttle conditions
                throttle_icon = "🛑" if tank.should_throttle else "✅"
                print(f"      {throttle_icon} Throttle: {tank.should_throttle} | "
                      f"Reason: {tank.throttle_reason.name}")
                
                # Show velocities (regime change detection)
                if abs(tank.cpu_big_velocity) > 0.01 or abs(tank.cpu_little_velocity) > 0.01:
                    big_warn = "⚠️" if tank.cpu_big_velocity > 3.0 else ""
                    lit_warn = "⚠️" if tank.cpu_little_velocity > 3.0 else ""
                    print(f"      📈 CPU_BIG vel: {tank.cpu_big_velocity:+.3f}°C/s {big_warn}")
                    print(f"      📈 CPU_LIT vel: {tank.cpu_little_velocity:+.3f}°C/s {lit_warn}")
                
                # Show headroom
                if tank.headroom_seconds < 300:  # Less than 5 min
                    print(f"      ⚠️  Headroom: {tank.headroom_seconds:.0f}s")
                
                # Show prediction
                if prediction:
                    pred_battery = prediction.predicted_temps.get(ThermalZone.BATTERY, 0.0)
                    print(f"      Predicted (+30s): Battery {pred_battery:.1f}°C | "
                          f"Confidence: {prediction.confidence:.0%}")
                
                print()  # Blank line between updates
            
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        print("\n🛑 Stopping monitoring...")
        await thermal.stop()
        print("✅ Done.\n")
        
        # Final stats
        stats = thermal.get_statistics()
        print("📊 Statistics:")
        print(f"   Samples: {stats.get('samples_collected', 0)}")
        print(f"   Predictions: {stats.get('predictions_made', 0)}")
        print(f"   State: {thermal.current_state.name}")
        
        # Final tank status
        tank = thermal.get_tank_status()
        if tank:
            print(f"\n🔥 Final Tank Status:")
            print(f"   Battery: {tank.battery_temp_current:.1f}°C → {tank.battery_temp_predicted:.1f}°C")
            print(f"   Throttle: {tank.should_throttle} ({tank.throttle_reason.name})")
            print(f"   Cooling rate: {tank.cooling_rate:+.3f}°C/s")
            print(f"   Headroom: {tank.headroom_seconds:.0f}s")

if __name__ == "__main__":
    asyncio.run(main())
