"""Post-training quantization variants and host-side integer inference.

Each variant stands for one deployment target in a heterogeneous fleet:
  int8-pc  TFLite default: INT8 weights per output channel, INT8 activations.
  int8-pt  INT8 weights per tensor (one scale per layer), as some vendor runtimes use.
  w4a8     4-bit weights per output channel, INT8 activations. Weights are rounded to a
           16-level grid before conversion, so the INT8 model holds only 4-bit values
           (CMSIS-NN and TFLM can store these packed as int4).
TFLite's reference and optimized INT8 kernels are bit-identical on the host, so kernel
choice is not a variant here; real boards are measured separately.
"""
import numpy as np
import tensorflow as tf

from .model import score_from_recon

VARIANTS = ("int8-pc", "int8-pt", "w4a8")


def _round_weights(model: tf.keras.Model, bits: int) -> tf.keras.Model:
    """Copy of `model` with conv kernels rounded to a symmetric per-output-channel grid."""
    q = tf.keras.models.clone_model(model)
    q.set_weights(model.get_weights())
    levels = 2 ** (bits - 1) - 1
    for layer in q.layers:
        if not isinstance(layer, (tf.keras.layers.Conv1D, tf.keras.layers.Conv1DTranspose)):
            continue
        k, b = layer.get_weights()
        # Conv1D kernel: (k, in, out); Conv1DTranspose kernel: (k, out, in).
        axis = 1 if isinstance(layer, tf.keras.layers.Conv1DTranspose) else 2
        red = tuple(i for i in range(3) if i != axis)
        scale = np.abs(k).max(axis=red, keepdims=True) / levels
        scale[scale == 0] = 1.0
        layer.set_weights([np.clip(np.round(k / scale), -levels - 1, levels) * scale, b])
    return q


def convert(model: tf.keras.Model, representative: np.ndarray, variant: str = "int8-pc") -> bytes:
    if variant == "w4a8":
        model = _round_weights(model, 4)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([w[None]] for w in representative)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    if variant == "int8-pt":
        conv._experimental_disable_per_channel = True
    return conv.convert()


def score_int8(tflite_model: bytes, x: np.ndarray) -> np.ndarray:
    """Quantize input, run the integer model, dequantize output, return reconstruction MSE."""
    interp = tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]
    interp.resize_tensor_input(inp["index"], (len(x),) + x.shape[1:])
    interp.allocate_tensors()
    xq = np.clip(np.round(x / s_in) + z_in, -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], xq)
    interp.invoke()
    x_hat = (interp.get_tensor(out["index"]).astype(np.float32) - z_out) * s_out
    return score_from_recon(x, x_hat)
