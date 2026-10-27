"""Full-integer (INT8) post-training quantization and host-side INT8 inference.

On the host, TFLite's reference kernels and its default optimized kernels give two
different integer implementations of the same model, standing in for two "devices"
until the real boards are measured.
"""
import numpy as np
import tensorflow as tf

from .model import score_from_recon

KERNELS = {
    "tflite-ref": tf.lite.experimental.OpResolverType.BUILTIN_REF,
    "tflite-opt": tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
}


def to_int8(model: tf.keras.Model, representative: np.ndarray) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([w[None]] for w in representative)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def score_int8(tflite_model: bytes, x: np.ndarray, kernels: str = "tflite-ref") -> np.ndarray:
    """Quantize input, run the INT8 model, dequantize output, return reconstruction MSE."""
    interp = tf.lite.Interpreter(model_content=tflite_model,
                                 experimental_op_resolver_type=KERNELS[kernels])
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
