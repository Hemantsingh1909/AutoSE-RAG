# ISO 26262 Functional Safety Guidelines

## ASIL Classification & Safety Integrity Levels
Automotive Safety Integrity Levels (ASIL A, B, C, D) define safety requirements to prevent unreasonable residual risk. ASIL D represents the highest integrity requirement. Software components assigned ASIL D must employ defensive programming, formal interface validation, and fault injection testing.

## Safe State Transitions & Fault Tolerance
1. When a safety-critical anomaly is detected, the system shall transition to a predefined safe state within the Fault Tolerant Time Interval (FTTI).
2. The default safe state for electronic control units (ECU) is torque deactivation or graceful degradation to limp-home mode.
3. Software shall maintain watchdog heartbeat timers and assert failsafe interlocks if heartbeat timeouts exceed 50ms.

## Defensive Implementation Constraints
- Memory safety: Avoid uninitialized memory and validate all pointer/index boundaries.
- Numeric bounds: Guard against integer overflow, underflow, and division by zero.
- Deterministic execution: Execution time must be bounded; dynamic recursion is prohibited.
- Single Point of Failure (SPoF) prevention: Redundant sensor cross-checking (e.g., dual potentiometer comparison) must be enforced with discrepancy threshold < 5%.
