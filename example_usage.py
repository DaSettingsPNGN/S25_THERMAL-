#!/usr/bin/env python3
"""
🐧 S25+ Thermal Intelligence - Usage Example
===========================================
Copyright (c) 2025 PNGN-Tec LLC
Author: Jesse Vogeler-Wunsch (@DaSettingsPNGN)

Demonstrates basic usage of the thermal intelligence system including:
- System initialization and startup
- Real-time monitoring loop
- Intelligence data retrieval
- Temperature prediction
- Statistics reporting
- Graceful shutdown

Run this script to see thermal monitoring in action for 60 seconds.
"""

import asyncio
from s25_thermal import create_thermal_intelligence

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
            # Get current intelligence
            intel = thermal.get_current_intelligence()
            
            if intel and intel.stats:
                # Get max temperature across all zones
                temps = intel.stats.current.zones
                if temps:
                    max_temp = max(temps.values())
                    max_zone = max(temps.items(), key=lambda x: x[1])
                    
                    # Display current status
                    print(f"[{i+1:2d}s] {max_temp:.1f}°C ({max_zone[0].name}) | "
                          f"State: {intel.state.name:8s} | "
                          f"Trend: {intel.stats.velocity.trend.name}")
                    
                    # Show prediction if available
                    if intel.prediction:
                        pred_temp = max(intel.prediction.predicted_temps.values())
                        print(f"       Predicted in 60s: {pred_temp:.1f}°C | "
                              f"Budget: {intel.prediction.thermal_budget:.0f}s")
                    
                    # Show recommendations if any
                    if intel.recommendations:
                        print(f"       ⚠️  {intel.recommendations[0]}")
            
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
        print(f"   Samples collected: {stats['samples_collected']}")
        print(f"   Predictions made: {stats['predictions_made']}")
        print(f"   Patterns learned: {stats['patterns_learned']}")
        print(f"   Current state: {stats['current_state']}")
        print(f"   Confidence: {stats['confidence']:.1%}")
        
        # Show persistence stats
        if 'persistence' in stats:
            p = stats['persistence']
            print(f"\n💾 Persistence:")
            print(f"   Command signatures: {p['command_signatures']}")
            print(f"   Telemetry signatures: {p['telemetry_signatures']}")
            print(f"   High confidence patterns: {p['high_confidence_patterns']}")

if __name__ == "__main__":
    asyncio.run(main())