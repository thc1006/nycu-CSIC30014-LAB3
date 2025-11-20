"""
Smart Calibration for COVID-19 Detection
Based on visual feature analysis showing COVID-19 vs Virus similarity = 85-90%

Key Insight:
- Model is too conservative on COVID-19 (only 37 training samples)
- Need to boost COVID-19 probability when model shows uncertainty between Virus and COVID-19
"""
import numpy as np
import pandas as pd
from typing import Tuple

def covid_boost_calibration(probs: np.ndarray, boost_factor: float = 1.4,
                            threshold: float = 0.12) -> np.ndarray:
    """
    Boost COVID-19 predictions when model shows high Virus probability.

    Strategy:
    - If COVID-19 prob > threshold AND (COVID-19 prob / Virus prob) > 0.3
      → Boost COVID-19 probability by boost_factor

    Args:
        probs: (N, 4) array of [normal, bacteria, virus, covid-19] probabilities
        boost_factor: Multiplicative boost for COVID-19 (default 1.4 = +40%)
        threshold: Minimum COVID-19 probability to consider boosting

    Returns:
        Calibrated probabilities (re-normalized)
    """
    probs = probs.copy()

    virus_prob = probs[:, 2]   # Virus column
    covid_prob = probs[:, 3]   # COVID-19 column

    # Strategy 1: Boost COVID-19 if it's competing with Virus
    # Rationale: Visual similarity means model is uncertain
    ratio_mask = (covid_prob > threshold) & (covid_prob / (virus_prob + 1e-8) > 0.3)
    probs[ratio_mask, 3] *= boost_factor

    # Strategy 2: Stronger boost for very high COVID-19 probability
    # Rationale: Model is confident but cautious due to limited training data
    high_covid_mask = covid_prob > 0.25
    probs[high_covid_mask, 3] *= 1.2

    # Re-normalize to sum to 1
    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs


def conservative_calibration(probs: np.ndarray,
                             target_covid_rate: float = 0.01) -> np.ndarray:
    """
    Conservative approach: Ensure COVID-19 predictions match expected rate.

    If model predicts too many COVID-19 cases, keep only top N% by confidence.
    If model predicts too few, boost borderline cases.

    Args:
        probs: (N, 4) array of probabilities
        target_covid_rate: Expected COVID-19 rate in test set (1% = 0.01)

    Returns:
        Adjusted probabilities
    """
    probs = probs.copy()
    n_samples = len(probs)
    target_covid_count = int(n_samples * target_covid_rate)

    # Get current predictions
    pred_classes = probs.argmax(axis=1)
    current_covid_count = (pred_classes == 3).sum()

    print(f"[Calibration] Current COVID-19 predictions: {current_covid_count}")
    print(f"[Calibration] Target COVID-19 count: {target_covid_count}")

    # If too many COVID-19 predictions, keep only top-N by confidence
    if current_covid_count > target_covid_count * 1.5:
        covid_mask = pred_classes == 3
        covid_confidences = probs[covid_mask, 3]

        # Keep only top N
        threshold_conf = np.sort(covid_confidences)[-target_covid_count]

        # Convert low-confidence COVID-19 → Virus
        low_conf_mask = (pred_classes == 3) & (probs[:, 3] < threshold_conf)
        probs[low_conf_mask, 2] = probs[low_conf_mask, 3]  # Move to Virus
        probs[low_conf_mask, 3] = 0

    # If too few, boost borderline cases
    elif current_covid_count < target_covid_count * 0.5:
        # Find samples where model is uncertain between Virus and COVID-19
        virus_mask = pred_classes == 2
        covid_prob_in_virus = probs[virus_mask, 3]

        # Boost top N uncertain cases
        n_to_boost = min(target_covid_count - current_covid_count,
                         int(n_samples * 0.02))

        if n_to_boost > 0 and len(covid_prob_in_virus) > 0:
            threshold_boost = np.sort(covid_prob_in_virus)[-n_to_boost]
            boost_mask = virus_mask & (probs[:, 3] > threshold_boost)
            probs[boost_mask, 3] *= 2.0

    # Re-normalize
    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs


def ensemble_calibration(pred_files: list, target_covid_rate: float = 0.01,
                        boost_factor: float = 1.4) -> pd.DataFrame:
    """
    Apply calibration to ensemble predictions.

    Args:
        pred_files: List of CSV file paths with predictions
        target_covid_rate: Expected COVID-19 rate
        boost_factor: Boost factor for COVID-19

    Returns:
        Calibrated submission DataFrame
    """
    # Load all predictions
    dfs = [pd.read_csv(f) for f in pred_files]

    # Ensure all have same files
    assert all(df['new_filename'].equals(dfs[0]['new_filename']) for df in dfs)

    prob_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    # Average probabilities
    probs_avg = np.mean([df[prob_cols].values for df in dfs], axis=0)

    print("\n[Calibration] Before calibration:")
    pred_before = probs_avg.argmax(axis=1)
    print(f"  Normal:    {(pred_before == 0).sum()}")
    print(f"  Bacteria:  {(pred_before == 1).sum()}")
    print(f"  Virus:     {(pred_before == 2).sum()}")
    print(f"  COVID-19:  {(pred_before == 3).sum()}")

    # Apply COVID boost calibration
    probs_calibrated = covid_boost_calibration(probs_avg, boost_factor=boost_factor)

    # Apply conservative calibration
    probs_calibrated = conservative_calibration(probs_calibrated,
                                               target_covid_rate=target_covid_rate)

    print("\n[Calibration] After calibration:")
    pred_after = probs_calibrated.argmax(axis=1)
    print(f"  Normal:    {(pred_after == 0).sum()}")
    print(f"  Bacteria:  {(pred_after == 1).sum()}")
    print(f"  Virus:     {(pred_after == 2).sum()}")
    print(f"  COVID-19:  {(pred_after == 3).sum()}")

    # Convert to one-hot
    pred_classes = probs_calibrated.argmax(axis=1)
    one_hot = np.eye(4)[pred_classes]

    # Create submission DataFrame
    result = dfs[0][['new_filename']].copy()
    result[prob_cols] = one_hot

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smart_calibration.py <pred1.csv> <pred2.csv> ...")
        sys.exit(1)

    pred_files = sys.argv[1:]

    print(f"[Calibration] Processing {len(pred_files)} prediction files...")

    result = ensemble_calibration(pred_files,
                                  target_covid_rate=0.01,
                                  boost_factor=1.4)

    output_path = "submission_calibrated.csv"
    result.to_csv(output_path, index=False)

    print(f"\n[Calibration] Saved to {output_path}")
