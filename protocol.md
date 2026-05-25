## Three Protocols Between This MCU and the External Device

### A. UART Serial Protocol (primary data channel)

**Physical:** USART1 — PA9 (TX), PA10 (RX). 10-byte framed packets with 8-bit checksum.

**Direction is encoded in the sync byte:**

| Direction | Sync Byte | Meaning |
|-----------|-----------|---------|
| This MCU → External | `Z` (0x5A) | Normal response |
| This MCU → External | `[` (0x5B) | Alternate path |
| External → This MCU | `0xA5` | Normal command |
| External → This MCU | `0xB5` | Alternate / clear |

**What this MCU SENDS (3 command types):**

| Type | Purpose | Key Payload |
|------|---------|-------------|
| **1** | Full status telemetry | Output mode, state flags, charge PWM, ADC readings, OutputCtrl field — 7 data bytes of real-time operating state |
| **2** | Heartbeat / ACK | Just sets byte[2]=1, rest zeros |
| **6** | HW enable check | HwEnable flag, plus 0x0D if disabled |

Type 1 packets carry the live operating data: output sequencer state, charge duty cycle, ADC sensor values, and control register fields. This is how the external MCU monitors what this controller is doing.

**What this MCU RECEIVES (4 command types):**

| Cmd | Name | What It Does |
|-----|------|-------------|
| **1** | CONFIGURE | Mode-dependent config: writes to config bytes, sets 16-bit values, can trigger GPIO output on PB6/PB7. Sub-mode selected by byte[6]==1 (GPIO path) vs !=1 (config+reply path) |
| **2** | HEARTBEAT | Requests a Type 1 status reply — external MCU polls for telemetry |
| **3** | SET_500HZ | Sets PWM timer to 500Hz |
| **4** | RESET_VAL | Clears a 16-bit reference value, triggers Type 4 packet reply |

After dispatching any received command, if an output mode was set, it automatically builds and sends the corresponding serial reply packet.

---

### B. GPIO Charge/Discharge Protocol (real-time mode control)

**Physical:** GPIOA.15 as a dedicated mode-select line from the external MCU.

This is a **separate physical channel** from UART — a single GPIO pin that the external MCU toggles to command charge vs discharge mode in real time:

| PA15 Level | Mode |
|------------|------|
| LOW | **CHARGE** |
| HIGH | **DISCHARGE** |

The `gpio_protocol_fsm()` function implements a state machine with debounced transitions:

**CHARGE sequence:** States 1→5→6→7 with 45-tick timeouts. Stage select (0-4) determines substate. Toggles between states 6 and 7 on timeout.

**DISCHARGE sequence:** States 2→3→4→8→9→11 with 30-tick timeouts. Stage select routes to states 9(TAB), 2, 3, or 4.

Key output states trigger higher-level handlers:
- **State 5 (ENQ)** → `protocol_enquiry_handler`: Toggle-based connect/disconnect. On first rising edge: runs `hw_calibration_init()` (ADC cal, PWM setup), sets output sequence A=1/B=2/C=3. Both edges set status to BUSY (0x0B).
- **State 9 (TAB)** → `protocol_stage9_handler`: If HW is enabled, triggers output sequencer state 2.

---

### C. Output Sequencer (GPIO bit-bang on PB6/PB7)

**Physical:** GPIOB pins 6 and 7 — direct coil/heater driver control signals.

This is a bit-banged output stage with two modes:

- **INIT mode:** Configures PB6/PB7 as outputs, toggles them via BSRR writes, 1µs delays between edges. Used to pulse the coil driver.
- **ACTIVE mode:** Multi-step sequencer (3 steps × 2 phases) with 200-tick per-step timeout. On step completion: reconfigures PA15/PB6/PB7, 50µs delay, resets for next cycle. Processes command buffer — if a `0xA5` command was received via UART, it dispatches it here. Sends serial replies with 100-tick inter-packet spacing, toggling between two commands.

---

### Protocol Stack Summary

```
┌──────────────────────────────────────────┐
│ Layer 4: Application                     │
│  Charge/Discharge FSM, Output Sequencer  │
├──────────────────────────────────────────┤
│ Layer 3: Command                         │
│  THIS→OTHER: 3 cmds (status,beat,check)  │
│  OTHER→THIS: 4 cmds (config,beat,pwm,reset)│
├──────────────────────────────────────────┤
│ Layer 2: Link                            │
│  10-byte packets, checksum8, 30-tick RX  │
│  timeout, sync-byte direction encoding   │
├──────────────────────────────────────────┤
│ Layer 1: Physical                        │
│  USART1(PA9/PA10) + GPIOA.15 + GPIOB.6/7 │
│  3 wires: UART TX, UART RX, Mode Select  │
└──────────────────────────────────────────┘
```

The external MCU uses the GPIO mode-select line for immediate charge/discharge commands and the UART for configuration, telemetry polling, and parameter updates. The output sequencer translates commands into timed GPIO bit patterns that drive the actual coil/heater MOSFETs