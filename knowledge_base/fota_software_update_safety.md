# Firmware Over-The-Air (FOTA) & Dual-Bank Software Update Safety

## Dual-Bank A/B Partition Management
Automotive ECUs implementing FOTA updates must maintain two redundant non-volatile flash partitions (Bank A and Bank B):
1. **Active vs Inactive Bank**: The ECU executes from the active bank while the new software image is staged into the inactive bank.
2. **Cryptographic Signature Verification**: Before committing the boot vector, the bootloader shall verify the payload integrity using SHA-256 and asymmetric signature verification (e.g., ECDSA / RSA-3072).

## Boot Watchdog & Automatic Rollback
1. Upon activating the new firmware partition, a boot confirmation watchdog timer (e.g., 30 seconds) is started.
2. The newly booted application must complete self-tests, verify internal diagnostics, and issue a formal `confirm_boot_success()` system call.
3. If the application crashes, hangs, or fails self-tests before the watchdog expires, the hardware bootloader shall automatically trigger a failsafe rollback to the previous known-good partition.
