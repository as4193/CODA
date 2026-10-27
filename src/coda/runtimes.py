"""Host stand-ins for heterogeneous deployment toolchains (one Keras model, different quantizers).

TFLite / TFLM / CMSIS-NN / ESP-NN are (designed to be) bit-identical, so they count as one
"device". Real fleet heterogeneity comes from different toolchains, each with its own quantizer:
  ort-int8-pc   ONNX Runtime static PTQ, QDQ, INT8 weights per channel, INT8 activations,
                MinMax calibration; CPU EP runs fused integer kernels (QLinearConv).
  ort-int8-pt   same, one weight scale per tensor.
  espdl-s3      ESP-DL toolchain (esp-ppq) for ESP32-S3: power-of-two scales, symmetric INT8,
                KL calibration, 20-bit bias, round-half-up; ConvTranspose -> InsertZeros + Conv.
                Scores come from esp-ppq's simulator (TorchExecutor), not from the chip.
  espdl-p4      same for ESP32-P4: 32-bit bias, round-half-even.
Every scorer returns reconstruction MSE of the dequantized output vs the float input,
exactly like coda.model.score_from_recon.
"""
import contextlib
import io
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from .model import score_from_recon

RUNTIMES = ("ort-int8-pc", "ort-int8-pt", "espdl-s3", "espdl-p4")
THREADS = 3


class Scorer:
    """Callable windows (N, L, F) float32 -> scores (N,); `model_bytes` = deployed model size."""

    def __init__(self, predict, model_bytes: int, batch: int = 1024):
        self._predict, self.model_bytes, self.batch = predict, model_bytes, batch

    def recon(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        return np.concatenate([self._predict(x[i:i + self.batch]) for i in range(0, len(x), self.batch)])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return score_from_recon(x, self.recon(x))


def to_onnx(model: tf.keras.Model, batch=None, opset: int = 17):
    """Keras -> ONNX (tf2onnx) -> onnxsim. batch=None keeps a dynamic batch dimension."""
    import onnxsim
    import tf2onnx
    spec = (tf.TensorSpec((batch,) + tuple(model.input_shape[1:]), tf.float32, name="x"),)
    fn = tf.function(lambda x: model(x, training=False))
    with contextlib.redirect_stdout(io.StringIO()):
        proto, _ = tf2onnx.convert.from_function(fn, input_signature=spec, opset=opset)
    simp, ok = onnxsim.simplify(proto)
    return simp if ok else proto


# ---------------------------------------------------------------- ONNX Runtime static INT8
def ort_int8(model: tf.keras.Model, representative: np.ndarray, per_channel: bool = True,
             calibrate: str = "MinMax") -> Scorer:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"x": representative[i:i + 1].astype(np.float32)}
                            for i in range(len(representative))])

        def get_next(self):
            return next(self.it, None)

    logging.getLogger("onnxruntime").setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory() as d:
        f32, q8 = Path(d) / "fp32.onnx", Path(d) / "int8.onnx"
        onnx.save(to_onnx(model), f32)
        quantize_static(str(f32), str(q8), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        per_channel=per_channel, calibrate_method=getattr(CalibrationMethod, calibrate))
        blob = q8.read_bytes()
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = THREADS, 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return Scorer(lambda x: sess.run(None, {name: x})[0], len(blob))


# ---------------------------------------------------------------- ESP-DL (esp-ppq)
def espdl(model: tf.keras.Model, representative: np.ndarray, target: str = "esp32s3",
          quant_type: str = "w8a8", export_dir: str | None = None) -> Scorer:
    """Quantize with esp-ppq for `target`, export .espdl, simulate integer inference on the host."""
    import onnx
    import torch
    torch.set_num_threads(THREADS)
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.executor import TorchExecutor

    with tempfile.TemporaryDirectory() as d:
        out = Path(export_dir or d)
        out.mkdir(parents=True, exist_ok=True)
        f32, f_espdl = Path(d) / "fp32.onnx", out / f"model_{target}.espdl"
        onnx.save(to_onnx(model, batch=1), f32)  # ESP-DL export requires batch 1
        calib = [torch.from_numpy(r[None].astype(np.float32)) for r in representative]
        with contextlib.redirect_stdout(io.StringIO()), open(os.devnull, "w") as null, \
                contextlib.redirect_stderr(null):
            graph = espdl_quantize_onnx(
                str(f32), str(f_espdl), calib_dataloader=calib, calib_steps=len(calib),
                input_shape=[1] + list(representative.shape[1:]), target=target,
                quant_type=quant_type, error_report=False, verbose=0)
        nbytes = f_espdl.stat().st_size
    ex = TorchExecutor(graph, device="cpu")

    def predict(x):
        with torch.no_grad():
            return ex.forward(torch.from_numpy(x))[0].numpy()

    return Scorer(predict, nbytes, batch=512)


def build(name: str, model: tf.keras.Model, representative: np.ndarray) -> Scorer:
    if name == "ort-int8-pc":
        return ort_int8(model, representative, per_channel=True)
    if name == "ort-int8-pt":
        return ort_int8(model, representative, per_channel=False)
    if name == "espdl-s3":
        return espdl(model, representative, target="esp32s3")
    if name == "espdl-p4":
        return espdl(model, representative, target="esp32p4")
    raise ValueError(name)
