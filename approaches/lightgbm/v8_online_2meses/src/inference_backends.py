"""Common float32 predict_proba interface and backend parity checks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol
import numpy as np

class ProbabilityBackend(Protocol):
    name: str
    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...

def _float32_2d(features) -> np.ndarray:
    matrix = np.ascontiguousarray(features, np.float32)
    if matrix.ndim != 2:
        raise ValueError("predict_proba expects float32[batch, features]")
    return matrix

@dataclass
class LightGBMBackend:
    booster: object
    name: str = "lightgbm_native"
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = _float32_2d(features)
        return np.asarray(self.booster.predict(matrix, num_threads=1), np.float32).reshape(-1)

class ONNXRuntimeBackend:
    name = "onnxruntime_cpu"
    def __init__(self, model_path: str | Path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options,
            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[-1].name
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        outputs = self.session.run(
            [self.output_name], {self.input_name: _float32_2d(features)})
        candidate = outputs[0]
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            return np.asarray([row.get(1, row.get("1")) for row in candidate], np.float32)
        array = np.asarray(candidate)
        if array.ndim == 2 and array.shape[1] >= 2:
            array = array[:, 1]
        return np.asarray(array, np.float32).reshape(-1)

class ONNXRuntimeGPUBackend(ONNXRuntimeBackend):
    """Experimental CUDA/TensorRT EP; never selected unless explicitly enabled."""
    def __init__(self, model_path: str | Path, provider: str = "tensorrt"):
        import onnxruntime as ort
        requested = ("TensorrtExecutionProvider" if provider == "tensorrt"
                     else "CUDAExecutionProvider")
        if requested not in ort.get_available_providers():
            raise RuntimeError(f"ONNX Runtime provider unavailable: {requested}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        fallback = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [requested] + [item for item in fallback if item != requested]
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[-1].name
        self.name = f"onnxruntime_{provider}_experimental"

class TreeliteBackend:
    """Optional runtime. This class is instantiated only when a compiled library exists."""
    name = "treelite"
    def __init__(self, library_path: str | Path):
        try:
            import treelite_runtime
        except ImportError as exc:
            raise RuntimeError("Treelite is optional and not installed") from exc
        self._runtime = treelite_runtime
        self.predictor = treelite_runtime.Predictor(str(library_path), nthread=1)
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = _float32_2d(features)
        batch = self._runtime.Batch.from_npy2d(matrix)
        return np.asarray(self.predictor.predict(batch), np.float32).reshape(-1)

def export_lightgbm_onnx(booster, n_features: int, output_path: str | Path) -> Path:
    """Export ONNX with float32 input. Fails clearly when mandatory deps are absent."""
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError("Install onnxmltools and onnxruntime for mandatory ONNX export") from exc
    model = onnxmltools.convert_lightgbm(
        booster, initial_types=[("features", FloatTensorType([None, n_features]))],
        target_opset=15, zipmap=False)
    _align_float32_tree_thresholds(model, booster)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnxmltools.utils.save_model(model, str(destination))
    return destination


def _align_float32_tree_thresholds(onnx_model, booster) -> int:
    """Preserve LightGBM's <= decisions for float32 inputs at split boundaries.

    ONNX stores tree thresholds as float32. A nearest rounding upward can make
    an input equal to that rounded cutoff take a different branch from LightGBM,
    whose original threshold is float64. Flooring the cutoff to the preceding
    representable float32 value preserves the comparison for every float32 input.
    """
    trees = [node for node in onnx_model.graph.node
             if node.op_type == "TreeEnsembleClassifier"]
    if len(trees) != 1:
        raise ValueError("Expected exactly one ONNX TreeEnsembleClassifier")
    attributes = {attr.name: attr for attr in trees[0].attribute}
    required = ("nodes_treeids", "nodes_nodeids", "nodes_featureids",
                "nodes_values", "nodes_modes", "nodes_truenodeids",
                "nodes_falsenodeids")
    if any(name not in attributes for name in required):
        raise ValueError("ONNX tree is missing node mapping attributes")
    node_index = {(int(tree), int(node)): i for i, (tree, node) in enumerate(
        zip(attributes["nodes_treeids"].ints,
            attributes["nodes_nodeids"].ints))}
    changed = 0

    def visit(original, tree_id: int, node_id: int) -> None:
        nonlocal changed
        index = node_index[(tree_id, node_id)]
        mode = attributes["nodes_modes"].strings[index]
        if "split_feature" not in original:
            if mode != b"LEAF":
                raise ValueError("ONNX and LightGBM tree shapes differ")
            return
        if mode != b"BRANCH_LEQ" or original["decision_type"] != "<=":
            raise ValueError("Unsupported LightGBM/ONNX split mode")
        if int(attributes["nodes_featureids"].ints[index]) != original["split_feature"]:
            raise ValueError("ONNX and LightGBM feature indices differ")
        threshold = float(original["threshold"])
        rounded = np.float32(threshold)
        if float(rounded) > threshold:
            rounded = np.nextafter(rounded, np.float32(-np.inf))
        if float(attributes["nodes_values"].floats[index]) != float(rounded):
            attributes["nodes_values"].floats[index] = float(rounded)
            changed += 1
        visit(original["left_child"], tree_id,
              int(attributes["nodes_truenodeids"].ints[index]))
        visit(original["right_child"], tree_id,
              int(attributes["nodes_falsenodeids"].ints[index]))

    for tree_id, tree in enumerate(booster.dump_model()["tree_info"]):
        visit(tree["tree_structure"], tree_id, 0)
    return changed

def validate_backend(reference: ProbabilityBackend, candidate: ProbabilityBackend,
                     samples: np.ndarray, *, atol: float = 1e-5,
                     decision_threshold: float = 0.5) -> dict:
    expected = reference.predict_proba(samples)
    actual = candidate.predict_proba(samples)
    if expected.shape != actual.shape:
        raise AssertionError(f"{candidate.name}: incompatible output shape")
    maximum = float(np.max(np.abs(expected-actual))) if len(expected) else 0.0
    identical = np.array_equal(expected >= decision_threshold,
                               actual >= decision_threshold)
    if maximum > atol or not identical:
        raise AssertionError(
            f"{candidate.name}: max probability delta={maximum:.3g}, "
            f"identical decisions={identical}")
    return {"backend": candidate.name, "max_probability_delta": maximum,
            "identical_binary_decisions": bool(identical)}

def select_fastest_validated(backends: list[ProbabilityBackend],
                             samples: np.ndarray, *, warmup: int = 2000,
                             atol: float = 1e-5,
                             decision_threshold: float = 0.5):
    if not backends:
        raise ValueError("No inference backends supplied")
    reference = backends[0]
    subset = _float32_2d(samples)
    checks, timings = [], {}
    for backend in backends:
        checks.append(validate_backend(reference, backend, subset, atol=atol,
                                       decision_threshold=decision_threshold))
        for i in range(min(warmup, len(subset))):
            backend.predict_proba(subset[i:i+1])
        started = perf_counter_ns()
        for row in subset:
            backend.predict_proba(row.reshape(1, -1))
        timings[backend.name] = (perf_counter_ns()-started)/max(1, len(subset))
    winner = min(backends, key=lambda item: timings[item.name])
    return winner, {"parity": checks, "nanoseconds_per_sample": timings}
