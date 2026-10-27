# Hardware: what to buy and what to do when it arrives

## Shopping list (prices checked Oct 2026, USD, before shipping and tax)

| Board | Why | Price | Where (2 sources) |
|---|---|---|---|
| **NUCLEO-F446RE** (STM32, Cortex-M4F, 180 MHz) | the STM32 board | ~19 | [ST eStore](https://estore.st.com/en/nucleo-f446re-cpn.html) 18.66; [Mouser](https://www.mouser.com/en/ProductDetail/STMicroelectronics/NUCLEO-F446RE?qs=PRtH0mD6DWYnuBoPSlbRCA%3D%3D) 18.66 (also DigiKey 18.82) |
| **ESP32-S3-DevKitC-1-N8R8** (Xtensa LX7, 240 MHz) | the ESP32 board | ~15-20 | [TME](https://www.tme.eu/en/details/esp32s3devkc-8r8/development-kits-for-data-transmission/espressif/esp32-s3-devkitc-1-n8r8/) 15.63; [Adafruit](https://www.adafruit.com/product/5336) 19.95 (the -1U version without a PCB antenna also works) |
| **Raspberry Pi Pico 2** (RP2350, Cortex-M33, 150 MHz) | the RP2350 board | 5 | [PiShop.us](https://www.pishop.us/product/raspberry-pi-pico-2/) 5.00; [SparkFun](https://www.sparkfun.com/raspberry-pi-pico-2.html) 5.75 |
| Optional: NUCLEO-H743ZI2 (Cortex-M7, 480 MHz) | a faster STM32, if budget allows | ~27 | ST eStore, Mouser |

Three boards: about **40 USD**. With cables: about **50-55 USD**. With the optional H743ZI2: about 80 USD.

**Cables** (must be *data* cables, not charge-only):
- NUCLEO-F446RE: **USB Mini-B** cable.
- ESP32-S3-DevKitC-1: **USB-C** cable (older v1.0 boards: Micro-USB).
- Pico 2: **Micro-USB** cable.
- NUCLEO-H743ZI2: Micro-USB cable.
- If your Mac only has USB-C ports: buy cables with a USB-C plug on the Mac side, or one small USB-C hub with USB-A ports.

No soldering, no wires, no extra power supply. The USB cable powers each board.

## Before the boards arrive (already done on this Mac)
- `firmware/setup.sh` was run: tools are installed and all boards build.
- Test without hardware: `.venv/bin/python scripts/board_probe.py --dry-run`

## The day the boards arrive

Do one board at a time. All commands run from the `coda` folder.

### 1. Find the serial port
```sh
ls /dev/cu.*          # before plugging in
# plug in the board
ls /dev/cu.*          # the new name is your board
```
Typical names: STM32 and Pico 2 `/dev/cu.usbmodemXXXX`; ESP32-S3 `/dev/cu.usbserial-XXXX`
or `/dev/cu.wchusbserialXXXX` or `/dev/cu.usbmodemXXXX`.

### 2. Flash the firmware
- **NUCLEO-F446RE:** plug in the USB on the ST-LINK part (top of the board). Run
  `firmware/flash.sh f446re`
  (if that fails: drag `firmware/stm32/.pio/build/nucleo_f446re/firmware.bin` onto the
  `NODE_F446RE` drive in Finder.)
- **ESP32-S3:** use the USB port labelled **UART**. Run `firmware/flash.sh esp32s3 /dev/cu.XXXX`
  (if it does not connect: hold **BOOT**, press **RST**, release **BOOT**, run again.
  After flashing, press **RST** once.)
- **Pico 2:** first time only: hold the **BOOTSEL** button while plugging in; an `RP2350` drive
  appears. Run `firmware/flash.sh pico2`
  (or drag `firmware/rp2350/.pio/build/pico2/firmware.uf2` onto the `RP2350` drive).
  The port name can change after flashing: check `ls /dev/cu.*` again.

### 3. Run the probe (one command)
```sh
.venv/bin/python scripts/board_probe.py --port /dev/cu.XXXX --machine machine-1-1 --seed 0 --variant int8-pc
```
It uploads the model to the board, sends the 500 probe windows (~1 min), and prints:
- how many board scores are bit-identical to the host TFLite scores,
- the shift vs FP32, q_hat, and tau_d for each alpha,
- latency (cycles and ms), arena (SRAM) used, model size.

Results go to `results/boards/<board>_<machine>_seed<seed>_<variant>.json`.
Add `--write-tau 0.05` to store tau_d (for alpha = 0.05) on the board.
Repeat with `--variant int8-pt` and `--variant w4a8`; no reflash is needed.

### Flash and RAM size
Printed at the end of every build (`firmware/setup.sh` or `firmware/flash.sh ...`), lines
`RAM:` and `Flash:`. Peak SRAM of the model itself = `arena_used_bytes` in the JSON.

## If something goes wrong
- `no CODA board answering`: wrong port, or press the board's reset button and run again.
- Port busy: close any serial monitor (Arduino IDE, `screen`) first.
- STM32 garbled/timeouts: try a different cable; ST-LINK firmware update (STM32CubeProgrammer) helps on old boards.
- Details of the firmware and protocol: `firmware/README.md`.
