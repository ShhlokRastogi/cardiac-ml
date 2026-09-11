import os
import json
import time
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
TRACES_FILE = os.path.join(LOGS_DIR, "inference_traces.jsonl")


class LatencyTimer:
    """Timer helper to measure sub-phase latency in milliseconds."""
    def __init__(self):
        self.start_time = time.perf_counter()
        self.latencies = {}

    def measure(self, phase_name: str, start_time_ref: float):
        elapsed_ms = round((time.perf_counter() - start_time_ref) * 1000.0, 2)
        self.latencies[phase_name] = elapsed_ms
        return elapsed_ms

    def total(self) -> float:
        return round((time.perf_counter() - self.start_time) * 1000.0, 2)


class DataDriftDetector:
    """
    Detects biometric data drift & out-of-range physiological anomalies
    compared against the baseline ACDC training distribution.
    """
    PHYSIOLOGICAL_BOUNDS = {
        "LVEF": (10.0, 90.0),            # Left Ventricular Ejection Fraction %
        "RVEF": (10.0, 90.0),            # Right Ventricular Ejection Fraction %
        "LVEDVI": (20.0, 260.0),         # LV End-Diastolic Volume Index mL/m²
        "Max_MYO_Thickness_mm": (3.0, 38.0)  # Max Myocardial Wall Thickness mm
    }

    @classmethod
    def check_biometric_drift(cls, clinical_features: dict) -> dict:
        alerts = []
        is_anomalous = False

        for metric, (min_val, max_val) in cls.PHYSIOLOGICAL_BOUNDS.items():
            val = clinical_features.get(metric)
            if val is not None:
                if val < min_val or val > max_val:
                    is_anomalous = True
                    alerts.append({
                        "metric": metric,
                        "value": round(val, 2),
                        "expected_range": [min_val, max_val],
                        "warning": f"Biometric metric '{metric}' value {val:.2f} is outside baseline ACDC bounds [{min_val}, {max_val}]."
                    })

        return {
            "has_data_drift": is_anomalous,
            "drift_alert_count": len(alerts),
            "alerts": alerts
        }


class JSONLTraceLogger:
    """Persists structured inference trace logs to disk."""
    def __init__(self, log_filepath: str = TRACES_FILE):
        self.log_filepath = log_filepath
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)

    def log_trace(self, trace_data: dict):
        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_data) + "\n")
        except Exception as e:
            print(f"[Observability Warning] Failed to log trace: {e}")

    def read_recent_traces(self, limit: int = 20) -> list:
        if not os.path.exists(self.log_filepath):
            return []
        traces = []
        try:
            with open(self.log_filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in reversed(lines[-limit:]):
                    line = line.strip()
                    if line:
                        traces.append(json.loads(line))
        except Exception as e:
            print(f"[Observability Warning] Failed to read traces: {e}")
        return traces


class MetricsCollector:
    """In-memory metrics collector for API health & monitoring endpoints."""
    def __init__(self):
        self.total_predictions = 0
        self.total_latency_sum_ms = 0.0
        self.diagnosis_counts = {"DCM": 0, "HCM": 0, "MINF": 0, "NOR": 0, "RV": 0}
        self.total_drift_alerts = 0
        self.latency_breakdown_sums = {
            "nifti_io_ms": 0.0,
            "preprocessing_ms": 0.0,
            "stage1_segmentation_ms": 0.0,
            "postprocessing_biometrics_ms": 0.0,
            "stage2_classification_ms": 0.0
        }

    def record_inference(self, predicted_class: str, latency_breakdown: dict, has_drift: bool):
        self.total_predictions += 1
        total_ms = latency_breakdown.get("total_pipeline_ms", 0.0)
        self.total_latency_sum_ms += total_ms

        if predicted_class in self.diagnosis_counts:
            self.diagnosis_counts[predicted_class] += 1
        else:
            self.diagnosis_counts[predicted_class] = 1

        if has_drift:
            self.total_drift_alerts += 1

        for key in self.latency_breakdown_sums:
            self.latency_breakdown_sums[key] += latency_breakdown.get(key, 0.0)

    def get_summary_metrics(self) -> dict:
        avg_latency_ms = (
            round(self.total_latency_sum_ms / self.total_predictions, 2)
            if self.total_predictions > 0 else 0.0
        )
        avg_breakdown = {
            key: round(val / self.total_predictions, 2) if self.total_predictions > 0 else 0.0
            for key, val in self.latency_breakdown_sums.items()
        }

        return {
            "total_predictions": self.total_predictions,
            "avg_total_latency_ms": avg_latency_ms,
            "avg_latency_breakdown_ms": avg_breakdown,
            "diagnosis_class_distribution": self.diagnosis_counts,
            "total_data_drift_alerts": self.total_drift_alerts
        }


# Global Observability Instances
trace_logger = JSONLTraceLogger()
metrics_collector = MetricsCollector()
