# AUTOSAR Diagnostic & Event Management (DEM / DCM)

## Diagnostic Trouble Code (DTC) Handling
When a monitored software or hardware component detects an out-of-range or implausible condition:
1. The diagnostic component shall qualify the fault status (Pending, Confirmed, Aged).
2. The fault is confirmed if the error condition persists for more than $N$ consecutive monitoring cycles (debounce filter).
3. Upon fault confirmation, a standard Diagnostic Trouble Code (DTC, e.g., 0xP0120 for throttle sensor circuit malfunction) is recorded into non-volatile storage.

## Freeze Frame & Environmental Recording
When a diagnostic event is confirmed:
- Snapshot context data: timestamp, vehicle speed, engine RPM, battery voltage, and ambient temperature.
- Expose diagnostic status via standard UDS (Unified Diagnostic Services - ISO 14229) Service 0x19 (Read DTC Information).

## Fault Recovery and Healing
If the fault condition clears and remains valid for a healing debounce window (e.g., 40 consecutive drive cycles without error), the DTC status transitions to healed and warning lamps (MIL) are extinguished.
