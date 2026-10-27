"""
telemetry.py — Módulo de Medición de Telemetría y Recursos del Sistema
=======================================================================
Monitorea en tiempo real el consumo de CPU (%), memoria RAM (MB / GB),
uso/memoria de GPU (NVIDIA) y tiempos de ejecución por etapa usando `psutil`
y `time.perf_counter_ns()`.

Registra métricas de latencia de inferencia en streaming (Media, P50, P95, P99, Máx)
y throughput (muestras/s). Soporta ciclo de vida completo (start, stop, reset, context manager).
"""

import os
import sys
import time
import json
import logging
import threading
import psutil
import numpy as np
from typing import Dict, List, Optional, Any

# Configuración de logging para telemetría
logger = logging.getLogger("SystemTelemetryTracker")


class SystemTelemetryTracker:
    """
    Rastreador de telemetría de hardware y rendimiento del proceso actual.
    Soporta monitoreo no bloqueante de CPU, RAM, GPU y latencias de inferencia.
    """

    def __init__(self, sample_interval_sec: float = 0.1):
        self.process = psutil.Process(os.getpid())
        self.sample_interval = float(sample_interval_sec)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Buffers de muestras en memoria
        self.cpu_samples: List[float] = []
        self.ram_samples_mb: List[float] = []
        self.gpu_mem_samples_mb: List[float] = []
        self.gpu_util_samples: List[float] = []

        # Tiempos por etapa
        self.stage_durations: Dict[str, float] = {}
        self._stage_starts: Dict[str, float] = {}

        # Métricas de streaming
        self.streaming_metrics: Dict[str, Any] = {}

        # Estado GPU (PyTorch o pynvml)
        self.gpu_available: bool = False
        self.gpu_backend: Optional[str] = None
        self._nvml_handle = None
        self._init_gpu_monitoring()

        self.start_time = time.perf_counter()

    def _init_gpu_monitoring(self):
        """Inicializa soporte GPU de forma segura usando torch.cuda o pynvml."""
        # 1. Probar PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_backend = 'torch'
                logger.info("[Telemetría] Monitoreo GPU activo mediante PyTorch CUDA (%s)", torch.cuda.get_device_name(0))
                return
        except Exception as e:
            logger.debug("[Telemetría] PyTorch CUDA no disponible: %s", e)

        # 2. Probar pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
            self.gpu_backend = 'pynvml'
            logger.info("[Telemetría] Monitoreo GPU activo mediante pynvml")
            return
        except Exception as e:
            logger.warning("[Telemetría Warning] GPU no disponible o pynvml/torch.cuda no activo (%s). GPU MB se registrará como 0.0.", e)
            self.gpu_available = False
            self.gpu_backend = None

    def _sample_gpu(self) -> tuple[float, float]:
        """Toma muestra (memoria_mb, uso_percent) de la GPU sin bloquear."""
        if not self.gpu_available:
            return 0.0, 0.0

        try:
            if self.gpu_backend == 'torch':
                import torch
                mem_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
                return float(mem_mb), 0.0
            elif self.gpu_backend == 'pynvml':
                import pynvml
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem_mb = mem_info.used / (1024.0 * 1024.0)
                util_pct = float(util_info.gpu)
                return float(mem_mb), util_pct
        except Exception as e:
            logger.debug("[Telemetría Error] Fallo al leer muestra GPU: %s", e)

        return 0.0, 0.0

    def reset(self):
        """Limpia los buffers de muestras de hardware y métricas acumuladas."""
        self.stop_monitoring()
        self.cpu_samples.clear()
        self.ram_samples_mb.clear()
        self.gpu_mem_samples_mb.clear()
        self.gpu_util_samples.clear()
        self.stage_durations.clear()
        self._stage_starts.clear()
        self.streaming_metrics.clear()
        self.start_time = time.perf_counter()

    def start_monitoring(self):
        """Inicia el monitoreo continuo de CPU, RAM y GPU en un hilo secundario (daemon)."""
        self.stop_monitoring()
        self._stop_event.clear()
        self.process.cpu_percent(interval=None)

        def _monitor_loop():
            while not self._stop_event.is_set():
                try:
                    cpu = float(self.process.cpu_percent(interval=None))
                    ram_mb = float(self.process.memory_info().rss / (1024.0 * 1024.0))
                    gpu_mb, gpu_util = self._sample_gpu()

                    self.cpu_samples.append(cpu)
                    self.ram_samples_mb.append(ram_mb)
                    self.gpu_mem_samples_mb.append(gpu_mb)
                    self.gpu_util_samples.append(gpu_util)
                except Exception as e:
                    logger.debug("[Telemetría Monitor Loop Error]: %s", e)
                time.sleep(self.sample_interval)

        self._thread = threading.Thread(target=_monitor_loop, daemon=True)
        self._thread.start()

    def stop_monitoring(self):
        """Detiene el hilo secundario de monitoreo."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None

    def start(self):
        """Alias para start_monitoring (gestión de ciclo de vida)."""
        self.start_monitoring()

    def stop(self):
        """Alias para stop_monitoring (gestión de ciclo de vida)."""
        self.stop_monitoring()

    # Soporte para Context Manager (`with SystemTelemetryTracker() as tracker:`)
    def __enter__(self):
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()

    def start_stage(self, stage_name: str):
        """Registra el inicio de una etapa del pipeline."""
        self._stage_starts[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        """Registra el fin de una etapa y devuelve la duración en segundos."""
        if stage_name in self._stage_starts:
            duration = time.perf_counter() - self._stage_starts[stage_name]
            self.stage_durations[stage_name] = round(duration, 4)
            return duration
        return 0.0

    def record_streaming_performance(
        self,
        latencies_ms: List[float],
        total_samples: int,
        total_time_sec: float,
    ) -> Dict[str, float]:
        """
        Calcula y almacena las métricas de rendimiento en inferencia streaming
        incluyendo P50, P95, P99, Media y Máximo en ms.
        """
        if not latencies_ms:
            metrics = {
                'total_samples': 0,
                'total_time_sec': round(total_time_sec, 4),
                'samples_per_second': 0.0,
                'latency_mean_ms': 0.0,
                'latency_p50_ms': 0.0,
                'latency_p95_ms': 0.0,
                'latency_p99_ms': 0.0,
                'latency_max_ms': 0.0,
            }
        else:
            arr = np.asarray(latencies_ms, dtype=float)
            throughput = total_samples / total_time_sec if total_time_sec > 0 else 0.0
            metrics = {
                'total_samples': int(total_samples),
                'total_time_sec': round(float(total_time_sec), 4),
                'samples_per_second': round(float(throughput), 2),
                'latency_mean_ms': round(float(np.mean(arr)), 4),
                'latency_p50_ms': round(float(np.median(arr)), 4),
                'latency_p95_ms': round(float(np.percentile(arr, 95)), 4),
                'latency_p99_ms': round(float(np.percentile(arr, 99)), 4),
                'latency_max_ms': round(float(np.max(arr)), 4),
            }
        self.streaming_metrics = metrics
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Genera un resumen completo de recursos (CPU, RAM, GPU) y tiempos."""
        total_duration = time.perf_counter() - self.start_time

        cpu_avg = float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0
        cpu_peak = float(np.max(self.cpu_samples)) if self.cpu_samples else 0.0

        ram_peak_mb = float(np.max(self.ram_samples_mb)) if self.ram_samples_mb else 0.0
        ram_avg_mb = float(np.mean(self.ram_samples_mb)) if self.ram_samples_mb else 0.0

        gpu_peak_mb = float(np.max(self.gpu_mem_samples_mb)) if self.gpu_mem_samples_mb else 0.0
        gpu_avg_mb = float(np.mean(self.gpu_mem_samples_mb)) if self.gpu_mem_samples_mb else 0.0
        gpu_peak_util = float(np.max(self.gpu_util_samples)) if self.gpu_util_samples else 0.0

        return {
            'system_info': {
                'python_version': sys.version.split()[0],
                'platform': sys.platform,
                'cpu_cores_logical': psutil.cpu_count(logical=True),
                'total_system_ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'gpu_available': self.gpu_available,
                'gpu_backend': self.gpu_backend,
            },
            'execution_times_sec': {
                **self.stage_durations,
                'total_pipeline_sec': round(total_duration, 4),
            },
            'cpu_usage_percent': {
                'avg': round(cpu_avg, 2),
                'peak': round(cpu_peak, 2),
            },
            'memory_ram_mb': {
                'avg_mb': round(ram_avg_mb, 2),
                'peak_mb': round(ram_peak_mb, 2),
                'peak_gb': round(ram_peak_mb / 1024.0, 4),
            },
            'gpu_usage': {
                'avg_mem_mb': round(gpu_avg_mb, 2),
                'peak_mem_mb': round(gpu_peak_mb, 2),
                'peak_util_pct': round(gpu_peak_util, 2),
            },
            'streaming_performance': self.streaming_metrics,
        }

    def export_json(self, filepath: str):
        """Exporta el resumen de telemetría a un archivo JSON."""
        summary = self.get_summary()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"  [Telemetría] Recursos exportados a: {filepath}")
