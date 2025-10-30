# Changelog

## [1.0.0] - 2025-10-30

### Changed
- Cleaned all headers to use minimal 🐧 PNGN branding
- Removed 99 lines of version history from s25_thermal.py
- Updated README with verified test statistics only (42,738 predictions, 2.96°C MAE)
- Replaced marketing claims with production deployment facts (645+ Discord members)

### Removed
- DISPLAY and CHARGER zone references (software metrics, not thermal zones)
- Unused zones from config.py (skin, npu, camera)
- Unverifiable performance claims
- Version history commentary from all files

### Technical
- 5 real thermal zones: CPU_BIG, CPU_LITTLE, GPU, BATTERY, MODEM
- AMBIENT derived from battery temperature
- Per-zone Newton's law of cooling with hardware-specific constants
- Pattern learning for Discord bot command thermal signatures

### Performance (Verified)
- 30s prediction horizon
- Battery zone: 2.60°C MAE (best)
- GPU zone: 2.70°C MAE
- CPU zones: 3.3-3.5°C MAE
- Overall: 41% predictions within 2°C
- Production deployment: Samsung S25+ serving Discord bot (645+ members)
