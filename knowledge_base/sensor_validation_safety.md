# Sensor Input Validation & Plausibility Safety

## Multi-Stage Sensor Validation Pipeline
Every incoming raw sensor reading (wheel speed, throttle pedal, steering angle, battery temperature) must pass through a four-stage validation pipeline:
1. **Electrical / ADC Range Check**: Validate raw voltage is between 0.5V and 4.5V (values < 0.2V or > 4.8V indicate short-to-ground or short-to-power).
2. **Physical Boundary Check**: Validate engineering units within allowable physical boundaries (e.g., Coolant Temp: -40°C to 130°C).
3. **Gradient / Rate-of-Change Check**: Check that delta value per unit time does not exceed physical actuator capability ($\Delta v / \Delta t \le MaxGradient$).
4. **Signal Plausibility & Cross-Check**: Compare primary sensor with redundant secondary sensor (e.g., PPS1 vs PPS2). If discrepancy exceeds 0.2V for > 100ms, flag signal as implausible.

## Safe Fallback & Default Value Strategy
When a sensor signal is declared invalid:
- Freeze at last valid plausibility value for up to 200ms.
- If invalidity persists beyond 200ms, substitute calibrated failsafe default value and transition control algorithm to degraded mode.
