# End-to-End (E2E) Communication & Protocol Security

## AUTOSAR E2E Profile Protection
Inter-ECU communication over CAN, CAN-FD, and Ethernet buses must implement AUTOSAR E2E Profile protection mechanisms:
1. **CRC-32 / CRC-8 Checksum**: Ensure message integrity and detect bit-flips in payload data.
2. **Alive Counter (Rolling Sequence Counter)**: 4-bit or 8-bit incrementing sequence number to detect message drops, repeats, and out-of-order delivery.
3. **Data ID / Stream Identifier**: Unique 16-bit identifier mixed into the CRC calculation to prevent misaddressed frames and masquerade attacks.
4. **Timeout Monitoring**: Receiver monitors frame arrival interval; if inter-frame arrival exceeds $1.5 \times \text{CycleTime}$, timeout error is triggered.

## Diagnostic Countermeasures
- If CRC validation fails or sequence gap $> 1$, drop the corrupt packet immediately.
- Increment communication error counter and trigger degraded mode if 3 consecutive packets are corrupted.
