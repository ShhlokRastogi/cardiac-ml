import time
import numpy as np
from src.observability import (
    LatencyTimer,
    DataDriftDetector,
    JSONLTraceLogger,
    MetricsCollector
)
from src.evals import (
    compute_dice_per_class,
    compute_hd95_approx,
    EvaluatorEngine
)


def test_latency_timer():
    """Verify sub-phase latency timer measurements."""
    timer = LatencyTimer()
    t0 = time.perf_counter()
    time.sleep(0.02)
    ms = timer.measure("test_phase", t0)
    assert ms >= 15.0, f"Expected latency >= 15ms, got {ms}"
    total = timer.total()
    assert total >= ms, f"Total latency should be >= step latency, got {total} vs {ms}"


def test_data_drift_detector():
    """Verify biometric anomaly & data drift detection."""
    normal_features = {
        "LVEF": 55.0,
        "RVEF": 50.0,
        "LVEDVI": 70.0,
        "Max_MYO_Thickness_mm": 10.0
    }
    res_normal = DataDriftDetector.check_biometric_drift(normal_features)
    assert res_normal["has_data_drift"] is False
    assert res_normal["drift_alert_count"] == 0

    anomalous_features = {
        "LVEF": 5.0,  # Below 10.0% min bound
        "RVEF": 50.0,
        "LVEDVI": 300.0,  # Above 260.0 max bound
        "Max_MYO_Thickness_mm": 45.0  # Above 38.0mm max bound
    }
    res_anomaly = DataDriftDetector.check_biometric_drift(anomalous_features)
    assert res_anomaly["has_data_drift"] is True
    assert res_anomaly["drift_alert_count"] == 3


def test_jsonl_trace_logger_and_metrics(tmp_path):
    """Verify JSONL logging and metrics aggregation."""
    log_path = str(tmp_path / "traces.jsonl")
    logger = JSONLTraceLogger(log_filepath=log_path)

    sample_trace = {
        "trace_id": "12345",
        "predicted_diagnosis": "NOR",
        "latency_ms": 120.5
    }
    logger.log_trace(sample_trace)

    recent = logger.read_recent_traces(limit=5)
    assert len(recent) == 1
    assert recent[0]["trace_id"] == "12345"

    collector = MetricsCollector()
    collector.record_inference(
        predicted_class="NOR",
        latency_breakdown={
            "nifti_io_ms": 10.0,
            "preprocessing_ms": 20.0,
            "stage1_segmentation_ms": 50.0,
            "postprocessing_biometrics_ms": 15.0,
            "stage2_classification_ms": 5.0,
            "total_pipeline_ms": 100.0
        },
        has_drift=False
    )
    summary = collector.get_summary_metrics()
    assert summary["total_predictions"] == 1
    assert summary["avg_total_latency_ms"] == 100.0


def test_evaluator_engine():
    """Verify Dice, HD95, and full evaluation report generation."""
    pred = np.zeros((20, 20), dtype=np.uint8)
    gt = np.zeros((20, 20), dtype=np.uint8)
    pred[5:15, 5:15] = 1
    gt[5:15, 5:15] = 1

    dice = compute_dice_per_class(pred, gt, 1)
    assert round(dice, 2) == 1.0

    hd95 = compute_hd95_approx(pred, gt, 1)
    assert hd95 == 0.0

    report = EvaluatorEngine.generate_full_eval_report()
    assert "stage1_segmentation_eval" in report
    assert "stage2_classification_eval" in report
