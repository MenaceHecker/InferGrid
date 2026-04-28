"""
ONNX Runtime model loader.
Runs inference using the exported classifier.onnx file.
Significantly faster than sklearn at serving time with no Python overhead.
"""

from pathlib import Path

import numpy as np


class OnnxLoader:
    def __init__(self, model_path: str, classes: list[str]) -> None:
        # Lazy import to avoid CI/test import failures when onnxruntime isn't installed
        try:
            import onnxruntime as rt
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is required for ONNX models. Install it with `pip install onnxruntime`"
            ) from e

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        # SessionOptions: single-threaded for predictable latency per request
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

        self.session = rt.InferenceSession(str(path), sess_options=opts)
        self.classes = classes

        # Cache input name to avoid repeated lookups
        self._input_name: str = self.session.get_inputs()[0].name

    def predict(self, text: str) -> tuple[str, float]:
        """
        Returns (predicted_class_label, confidence_score).
        Input must be a 2D string array: shape [1, 1].
        """
        input_array = np.array([[text]])
        label_preds, proba_preds = self.session.run(None, {self._input_name: input_array})

        class_idx = int(label_preds[0])
        confidence = float(np.max(proba_preds[0]))

        return self.classes[class_idx], confidence
