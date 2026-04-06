from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    SIREN_CONFIDENCE_THRESHOLD,
    SIREN_DETECTOR_MODEL_PATH,
    SIREN_MIN_DURATION_SEC,
    SIREN_SAMPLE_RATE,
)


class SirenDetector:
    """Runtime siren detector backed by a TFLite model.

    The detector accepts raw uploaded audio bytes, decodes WAV when possible,
    normalizes the signal, and feeds it to a TFLite model if available.
    """

    def __init__(
        self,
        model_path: str | Path = SIREN_DETECTOR_MODEL_PATH,
        confidence_threshold: float = SIREN_CONFIDENCE_THRESHOLD,
        sample_rate: int = SIREN_SAMPLE_RATE,
    ) -> None:
        self._model_path = Path(model_path)
        self._confidence_threshold = float(confidence_threshold)
        self._sample_rate = int(sample_rate)
        self._interpreter: Any = None
        self._input_details: list[dict[str, Any]] = []
        self._output_details: list[dict[str, Any]] = []
        self._error: str | None = None
        self._backend = "none"
        self._load_model()

    @property
    def is_loaded(self) -> bool:
        return self._interpreter is not None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_path": str(self._model_path),
            "confidence_threshold": self._confidence_threshold,
            "sample_rate": self._sample_rate,
            "backend": self._backend,
            "error": self._error,
        }

    def detect(self, audio_bytes: bytes | None) -> dict[str, Any]:
        if audio_bytes is None:
            return self._empty_result(mode="missing-audio")
        if not audio_bytes:
            return self._empty_result(mode="empty-audio")

        decoded = self._decode_audio(audio_bytes)
        if decoded is None:
            return self._empty_result(mode="invalid-audio")

        audio_signal, sample_rate = decoded
        if audio_signal.size == 0:
            return self._empty_result(mode="invalid-audio")

        min_required = max(1, int(SIREN_MIN_DURATION_SEC)) * max(1, sample_rate)
        if audio_signal.size < min_required:
            # Keep behavior deterministic for short clips in demos.
            padded = np.zeros(min_required, dtype=np.float32)
            padded[: audio_signal.size] = audio_signal
            audio_signal = padded

        if not self.is_loaded:
            return {
                **self._empty_result(mode="unavailable"),
                "error": self._error,
                "sample_rate": sample_rate,
            }

        try:
            features = self._prepare_model_input(audio_signal)
            self._interpreter.set_tensor(self._input_details[0]["index"], features)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_details[0]["index"])
            confidence, detected = self._parse_output(output)
            return {
                "detected": detected,
                "confidence": confidence,
                "mode": "tflite",
                "sample_rate": sample_rate,
            }
        except Exception as exc:
            return {
                **self._empty_result(mode="inference-error"),
                "error": str(exc),
                "sample_rate": sample_rate,
            }

    def _load_model(self) -> None:
        if not self._model_path.exists():
            self._error = f"model not found: {self._model_path}"
            return

        interpreter_cls = None
        try:
            from tflite_runtime.interpreter import Interpreter

            interpreter_cls = Interpreter
            self._backend = "tflite-runtime"
        except Exception:
            interpreter_cls = None

        if interpreter_cls is None:
            try:
                from tensorflow.lite import Interpreter

                interpreter_cls = Interpreter
                self._backend = "tensorflow-lite"
            except Exception:
                try:
                    from tensorflow.lite.python.interpreter import Interpreter

                    interpreter_cls = Interpreter
                    self._backend = "tensorflow-lite-python"
                except Exception:
                    interpreter_cls = None

        if interpreter_cls is None:
            self._backend = "none"
            self._error = "tflite runtime is not installed"
            return

        try:
            self._interpreter = interpreter_cls(model_path=str(self._model_path))
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            if not self._input_details or not self._output_details:
                raise ValueError("model has missing tensor metadata")
            self._error = None
        except Exception as exc:
            self._interpreter = None
            self._input_details = []
            self._output_details = []
            self._error = str(exc)

    def _decode_audio(self, audio_bytes: bytes) -> tuple[np.ndarray, int] | None:
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                sample_rate = int(wav_file.getframerate())
                sample_width = int(wav_file.getsampwidth())
                channels = int(wav_file.getnchannels())
                frames = wav_file.readframes(wav_file.getnframes())
        except Exception:
            return None

        if sample_rate <= 0 or channels <= 0 or sample_width not in {1, 2, 4}:
            return None

        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
        audio = np.frombuffer(frames, dtype=dtype_map[sample_width])
        if audio.size == 0:
            return None

        if sample_width == 1:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32) / 2147483648.0

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        if sample_rate != self._sample_rate:
            audio = self._resample_linear(audio, sample_rate, self._sample_rate)
            sample_rate = self._sample_rate

        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return audio, sample_rate

    def _resample_linear(
        self,
        signal: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        if source_rate == target_rate or signal.size <= 1:
            return signal.astype(np.float32)

        duration = signal.size / float(source_rate)
        target_size = max(1, int(round(duration * target_rate)))
        source_x = np.linspace(0.0, 1.0, num=signal.size, endpoint=True)
        target_x = np.linspace(0.0, 1.0, num=target_size, endpoint=True)
        return np.interp(target_x, source_x, signal).astype(np.float32)

    def _prepare_model_input(self, signal: np.ndarray) -> np.ndarray:
        detail = self._input_details[0]
        shape = detail.get("shape", [])
        if not isinstance(shape, np.ndarray):
            shape = np.asarray(shape)
        shape = shape.astype(int).tolist()

        # Replace dynamic dims with 1 for shaping calculations.
        normalized_shape = [1 if dim <= 0 else dim for dim in shape]
        target_elems = int(np.prod(normalized_shape)) if normalized_shape else signal.size

        if target_elems <= 0:
            target_elems = signal.size

        if signal.size < target_elems:
            padded = np.zeros(target_elems, dtype=np.float32)
            padded[: signal.size] = signal
            signal = padded
        elif signal.size > target_elems:
            signal = signal[:target_elems]

        if normalized_shape:
            features = signal.reshape(normalized_shape).astype(np.float32)
        else:
            features = signal.astype(np.float32)

        return features

    def _parse_output(self, output: np.ndarray) -> tuple[float, bool]:
        flat = np.asarray(output, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return 0.0, False

        if flat.size == 1:
            confidence = float(np.clip(flat[0], 0.0, 1.0))
        else:
            if flat.size >= 2:
                max_val = float(np.max(flat))
                exp = np.exp(flat - max_val)
                probs = exp / np.sum(exp)
                confidence = float(np.max(probs))
            else:
                confidence = float(np.clip(flat[0], 0.0, 1.0))

        detected = confidence >= self._confidence_threshold
        return confidence, detected

    def _empty_result(self, mode: str) -> dict[str, Any]:
        return {
            "detected": False,
            "confidence": 0.0,
            "mode": mode,
            "sample_rate": None,
        }
