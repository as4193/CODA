"""Host side of the CODA serial protocol (see firmware/README.md and firmware/common/coda_protocol.h)."""
import struct
import time
import zlib

import numpy as np

CMD = dict(INFO=0x01, INFER_F32=0x02, INFER_I8=0x03, SET_TAU=0x04, GET_TAU=0x05, STEP=0x06,
           STREAM_RESET=0x07, MODEL_BEGIN=0x10, MODEL_CHUNK=0x11, MODEL_END=0x12, MODEL_BUILTIN=0x13)
STATUS = {0: "OK", 1: "CRC", 2: "LEN", 3: "CMD", 4: "NO_MODEL", 5: "TOO_BIG", 6: "ALLOC",
          7: "INVOKE", 8: "SHAPE"}
INFO_FMT = "<BBHHHIIIIIfififI24s16s"
INFO_KEYS = ("proto", "model_source", "win_len", "n_feat", "_pad", "cpu_hz", "arena_bytes",
             "arena_used_bytes", "model_bytes", "model_crc32", "in_scale", "in_zero_point",
             "out_scale", "out_zero_point", "tau_d", "model_capacity", "board", "backend")
CHUNK = 4096


class BoardError(RuntimeError):
    pass


class Board:
    def __init__(self, port: str, baud: int = 921600, timeout: float = 5.0):
        import serial
        try:
            s = serial.Serial()
            s.port, s.baudrate, s.timeout = port, baud, timeout
            s.rts = False  # keep ESP32 boards out of reset
            s.dtr = True   # USB-CDC boards (Pico) only send when DTR is set
            s.open()
        except OSError:    # pseudo-terminals (firmware/host simulator) have no modem lines
            s = serial.Serial(port, 115200, timeout=timeout)  # baud is ignored on a pty
        self.ser = s
        self.info = self._hello()

    def close(self):
        self.ser.close()

    # ---- framing ----
    def _send(self, cmd: int, payload: bytes = b""):
        body = struct.pack("<BH", cmd, len(payload)) + payload
        self.ser.write(b"\xC0\xDA" + body + struct.pack("<I", zlib.crc32(body)))

    def _read(self, n: int) -> bytes:
        b = self.ser.read(n)
        if len(b) != n:
            raise BoardError(f"timeout: got {len(b)} of {n} bytes")
        return b

    def _recv(self, cmd: int) -> bytes:
        prev = b""
        while True:  # scan for the response magic (skips boot messages)
            c = self._read(1)
            if prev == b"\xDA" and c == b"\xC0":
                break
            prev = c
        hdr = self._read(4)
        rcmd, status, n = struct.unpack("<BBH", hdr)
        payload = self._read(n)
        (crc,) = struct.unpack("<I", self._read(4))
        if crc != zlib.crc32(hdr + payload):
            raise BoardError("response CRC mismatch")
        if rcmd != cmd:
            raise BoardError(f"response for cmd {rcmd:#x}, expected {cmd:#x}")
        if status != 0:
            raise BoardError(f"board error {STATUS.get(status, status)} for cmd {cmd:#x}")
        return payload

    def call(self, name: str, payload: bytes = b"") -> bytes:
        self._send(CMD[name], payload)
        return self._recv(CMD[name])

    def _hello(self, tries: int = 6) -> dict:
        old = self.ser.timeout
        self.ser.timeout = 1.0
        try:
            for _ in range(tries):
                self.ser.reset_input_buffer()
                try:
                    return self.get_info()
                except BoardError:
                    time.sleep(0.5)
            raise BoardError(f"no CODA board answering on {self.ser.port}")
        finally:
            self.ser.timeout = old

    # ---- commands ----
    @staticmethod
    def _parse_info(p: bytes) -> dict:
        d = dict(zip(INFO_KEYS, struct.unpack(INFO_FMT, p[:struct.calcsize(INFO_FMT)])))
        d.pop("_pad")
        for k in ("board", "backend"):
            d[k] = d[k].split(b"\0")[0].decode()
        d["model_source"] = ["builtin", "uploaded"][d["model_source"]]
        return d

    def get_info(self) -> dict:
        return self._parse_info(self.call("INFO"))

    def load_model(self, tfl: bytes) -> dict:
        """Upload a .tflite model into the board's RAM (skipped if it is already loaded)."""
        crc = zlib.crc32(tfl)
        if self.info["model_crc32"] == crc and self.info["model_bytes"] == len(tfl):
            return self.info
        self.call("MODEL_BEGIN", struct.pack("<I", len(tfl)))
        for off in range(0, len(tfl), CHUNK):
            self.call("MODEL_CHUNK", struct.pack("<I", off) + tfl[off:off + CHUNK])
        self.info = self._parse_info(self.call("MODEL_END", struct.pack("<I", crc)))
        return self.info

    def infer(self, window: np.ndarray, int8: bool = False):
        """One window -> (score, cycles_invoke, cycles_total, alarm)."""
        if int8:
            p = self.call("INFER_I8", np.ascontiguousarray(window, dtype=np.int8).tobytes())
        else:
            p = self.call("INFER_F32", np.ascontiguousarray(window, dtype="<f4").tobytes())
        return struct.unpack("<fIIB", p)

    def step(self, row: np.ndarray):
        """Detection mode: one time step (38 floats) -> (ready, score, cyc, cyc_total, alarm)."""
        return struct.unpack("<BfIIB", self.call("STEP", np.asarray(row, dtype="<f4").tobytes()))

    def set_tau(self, tau: float) -> float:
        return struct.unpack("<f", self.call("SET_TAU", struct.pack("<f", tau)))[0]

    def get_tau(self) -> float:
        return struct.unpack("<f", self.call("GET_TAU"))[0]
