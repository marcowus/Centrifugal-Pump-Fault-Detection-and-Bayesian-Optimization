import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from llm_client import DiagnosisResult, FaultLabel, LLMClient
from llm_postprocessor import (
    build_feature_summary,
    format_feature_digest,
    load_prompt_template,
    render_secondary_prompt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_data(X: np.ndarray, X_mean: np.ndarray, X_std: np.ndarray) -> np.ndarray:
    X_std = X_std.copy()
    X_std[X_std == 0] = 1
    return (X - X_mean) / X_std


def compute_t2_scores(pca_scores: np.ndarray, explained_variance: np.ndarray) -> np.ndarray:
    safe_variance = np.where(explained_variance == 0, 1e-9, explained_variance)
    return np.sum((pca_scores**2) / safe_variance, axis=1)


def arbitration_strategy(
    initial_anomaly: bool,
    llm_result: DiagnosisResult,
) -> Tuple[FaultLabel, str]:
    llm_is_anomaly = llm_result.diagnosis not in {FaultLabel.NORMAL, FaultLabel.UNKNOWN}
    if initial_anomaly == llm_is_anomaly:
        note = "LLM diagnosis aligns with initial statistical judgement."
        final_label = llm_result.diagnosis if llm_is_anomaly else FaultLabel.NORMAL
    else:
        if llm_result.confidence >= 0.7 and llm_result.diagnosis not in {FaultLabel.UNKNOWN}:
            note = "LLM diagnosis overrides due to high confidence despite conflict."
            final_label = llm_result.diagnosis
        else:
            if initial_anomaly:
                note = "Retained anomaly flag; LLM confidence too low to dismiss SPE/T² findings."
                final_label = FaultLabel.UNKNOWN
            else:
                note = "Retained normal status; LLM indicated anomaly with low confidence."
                final_label = FaultLabel.NORMAL
    llm_result.arbitration_note = note
    return final_label, note


def prepare_predictions_dataframe(
    time_indices: np.ndarray,
    initial_labels: np.ndarray,
    llm_result: DiagnosisResult,
    final_label: FaultLabel,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "Time": time_indices,
        "Labels": initial_labels,
    })
    df["llm_label"] = llm_result.diagnosis.value
    df["llm_confidence"] = llm_result.confidence
    df["llm_evidence"] = llm_result.evidence
    df["inspection_recommendations"] = llm_result.inspection_recommendations
    df["maintenance_actions"] = llm_result.maintenance_plan
    df["arbitrated_label"] = final_label.value
    df["arbitration_notes"] = llm_result.arbitration_note
    return df


def process_batch(
    data_file: str,
    output_file: str,
    pca_model: PCA,
    X_train_mean: np.ndarray,
    X_train_std: np.ndarray,
    spe_threshold: float,
    t2_threshold: float,
    prompt_template: str,
    llm_client: LLMClient,
) -> Dict[str, object]:
    logger.info("Processing %s", data_file)
    X_estimate_file = pd.read_csv(data_file, low_memory=False).iloc[:, 1:]
    X_estimate = X_estimate_file.values.astype("float32")

    imputer = SimpleImputer(strategy="mean")
    X_estimate = imputer.fit_transform(X_estimate)
    X_estimate_normal = normalize_data(X_estimate, X_train_mean, X_train_std)
    X_estimate_pca = pca_model.transform(X_estimate_normal)
    X_estimate_reconstructed = pca_model.inverse_transform(X_estimate_pca)
    estimate_SPE = np.sum((X_estimate_normal - X_estimate_reconstructed) ** 2, axis=1)

    explained_variance = getattr(pca_model, "explained_variance_", None)
    if explained_variance is None:
        raise AttributeError("PCA model does not contain explained_variance_.")
    estimate_T2 = compute_t2_scores(X_estimate_pca, explained_variance)

    predictions_SPE = np.where((estimate_SPE > spe_threshold) | (estimate_T2 > t2_threshold), -1, 1)
    initial_anomaly = bool(np.any(predictions_SPE == -1))
    initial_label = "anomaly" if initial_anomaly else "normal"

    feature_summary = build_feature_summary(
        batch_name=data_file,
        spe_values=estimate_SPE,
        t2_values=estimate_T2,
        pca_scores=X_estimate_pca,
    ).to_serializable()
    feature_digest = format_feature_digest(feature_summary)

    candidate_faults = [
        FaultLabel.NORMAL.value,
        FaultLabel.BEARING_DAMAGE.value,
        FaultLabel.MISALIGNMENT.value,
        FaultLabel.CAVITATION.value,
        FaultLabel.IMPELLER_DAMAGE.value,
        FaultLabel.LOOSENESS.value,
        FaultLabel.UNKNOWN.value,
    ]

    prompt = render_secondary_prompt(
        prompt_template,
        initial_label=initial_label,
        feature_summary=feature_summary,
        candidate_faults=candidate_faults,
    )

    try:
        llm_result = llm_client.request_secondary_diagnosis(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to obtain secondary diagnosis: %s", exc)
        llm_result = DiagnosisResult(
            diagnosis=FaultLabel.UNKNOWN,
            confidence=0.0,
            evidence=f"LLM request failed: {exc}",
            inspection_recommendations="Manual review required.",
            maintenance_plan="Pending expert assessment.",
            arbitration_note="LLM error",
        )

    final_label, arbitration_note = arbitration_strategy(initial_anomaly, llm_result)
    logger.info(
        "Batch %s initial=%s, llm=%s (%.2f), final=%s",
        data_file,
        initial_label,
        llm_result.diagnosis.value,
        llm_result.confidence,
        final_label.value,
    )

    result_df = prepare_predictions_dataframe(
        time_indices=np.arange(len(predictions_SPE)),
        initial_labels=predictions_SPE,
        llm_result=llm_result,
        final_label=final_label,
    )
    result_df["feature_digest"] = feature_digest
    result_df.to_csv(output_file, index=False)
    logger.info("%s generated.", output_file)

    batch_report = {
        "data_file": data_file,
        "output_file": output_file,
        "initial_label": initial_label,
        "initial_anomaly": initial_anomaly,
        "spe_threshold": float(spe_threshold),
        "t2_threshold": float(t2_threshold),
        "feature_summary": feature_summary,
        "feature_digest": feature_digest,
        "llm_response": llm_result.to_serializable(),
        "final_label": final_label.value,
        "arbitration_note": arbitration_note,
    }
    return batch_report


def main() -> None:
    with open("pca_model.pkl", "rb") as file:
        pca_loaded: PCA = pickle.load(file)

    params = np.load("X_train_params.npz")
    X_train_mean = params["mean"]
    X_train_std = params["std"]
    SPE_threshold = float(params["SPE"])
    T2_threshold = float(params["T2"])

    data_files = [
        "13_Data_评估数据1.csv",
        "14_Data_评估数据2.csv",
        "15_Data_评估数据3.csv",
        "16_Data_评估数据4.csv",
        "17_Data_评估数据5.csv",
    ]
    output_files = [
        "predictions1.csv",
        "predictions2.csv",
        "predictions3.csv",
        "predictions4.csv",
        "predictions5.csv",
    ]

    prompt_template_path = Path("prompts/secondary_diagnosis.md")
    prompt_template = load_prompt_template(prompt_template_path)
    llm_client = LLMClient()

    reports: List[Dict[str, object]] = []

    for data_file, output_file in zip(data_files, output_files):
        batch_report = process_batch(
            data_file=data_file,
            output_file=output_file,
            pca_model=pca_loaded,
            X_train_mean=X_train_mean,
            X_train_std=X_train_std,
            spe_threshold=SPE_threshold,
            t2_threshold=T2_threshold,
            prompt_template=prompt_template,
            llm_client=llm_client,
        )
        reports.append(batch_report)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"{timestamp}_diagnosis.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Diagnosis report written to %s", report_path)


if __name__ == "__main__":
    main()
