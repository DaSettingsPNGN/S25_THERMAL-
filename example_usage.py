#!/usr/bin/env python3
"""
🔥🐧🔥 S25+ Thermal Intelligence - Usage Example
===========================================
Copyright (c) 2025 PNGN-Tec LLC

Basic thermal monitoring demo showing real-time temperature tracking,
thermal tank status, and predictions with 60-second runtime.
"""

import asyncio
from s25_thermal import create_thermal_intelligence, ThermalZone

async def main():
    # Create thermal monitoring system
    print("🔥 Initializing S25+ Thermal Intelligence...")
    thermal = create_thermal_intelligence()
    
    # Start monitoring
    await thermal.start()
    print("✅ Monitoring started!\n")
    
    try:
        # Monitor for 60 seconds
        for i in range(60):
            # Get current sample
            sample = thermal.get_current()
            
            # Get thermal tank status (primary API)
            tank = thermal.get_tank_status()
            
            # Get prediction
            prediction = thermal.get_prediction()
            
            if sample and tank:
                # Get battery temperature (critical for throttling)
                battery = sample.zones.get(ThermalZone.BATTERY, 0.0)
                
                # Display current status
                print(f"[{i+1:2d}s] Battery: {battery:.1f}°C | "
                      f"State: {tank.state.name:8s} | "
                      f"Throttle: {'YES' if tank.should_throttle else 'NO'}")
                
                # Show thermal budget
                if tank.thermal_budget < 300:  # Less than 5 min
                    print(f"       ⚠️  Budget: {tank.thermal_budget:.0f}s | "
                          f"Cooldown: {tank.cooldown_needed:.0f}s")
                
                # Show prediction if available
                if prediction:
                    pred_battery = prediction.predicted_temps.get(ThermalZone.BATTERY, 0.0)
                    print(f"       Predicted (+30s): {pred_battery:.1f}°C | "
                          f"Confidence: {prediction.confidence:.0%}")
            
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        # Stop monitoring
        print("\n🛑 Stopping monitoring...")
        await thermal.stop()
        print("✅ Monitoring stopped.\n")
        
        # Show statistics
        stats = thermal.get_statistics()
        print("📊 Statistics:")
        print(f"   Runtime: {stats.get('runtime_seconds', 0):.0f}s")
        print(f"   Samples: {stats.get('total_samples', 0)}")
        print(f"   Predictions: {stats.get('total_predictions', 0)}")
        
        # Show final tank status
        tank = thermal.get_tank_status()
        if tank:
            print(f"\n🔥 Final Status:")
            print(f"   Battery: {tank.battery_temp_current:.1f}°C")
            print(f"   Peak: {tank.peak_temp:.1f}°C")
            print(f"   State: {tank.state.name}")
            print(f"   Cooling rate: {tank.cooling_rate:+.3f}°C/s")

if __name__ == "__main__":
    asyncio.run(main())
