"""
AI RiskGuard - Model Bias & Risk Auditor
Core auditing logic: checks a dataset/model for bias, imbalance,
missing values, and drift indicators, and produces a risk score.
"""

import pandas as pd
import numpy as np


class RiskAuditor:
    def __init__(self, df: pd.DataFrame, sensitive_features=None, target_col=None):
        """
        df: input dataset
        sensitive_features: list of column names considered sensitive
                             (e.g. gender, age_group, region)
        target_col: the label/target column, if doing classification
        """
        self.df = df
        self.sensitive_features = sensitive_features or []
        self.target_col = target_col
        self.report = {}

    # ---------- Individual checks ----------

    def check_missing_values(self):
        missing_pct = self.df.isnull().mean() * 100
        flagged = missing_pct[missing_pct > 5].round(2).to_dict()
        self.report["missing_values"] = {
            "flagged_columns": flagged,
            "risk_level": "High" if any(v > 20 for v in flagged.values())
                          else "Medium" if flagged else "Low"
        }
        return self.report["missing_values"]

    def check_class_imbalance(self):
        if not self.target_col or self.target_col not in self.df.columns:
            self.report["class_imbalance"] = {"risk_level": "N/A", "note": "No target column provided"}
            return self.report["class_imbalance"]

        counts = self.df[self.target_col].value_counts(normalize=True) * 100
        imbalance_ratio = counts.max() / counts.min() if len(counts) > 1 else 1

        risk = "Low"
        if imbalance_ratio > 4:
            risk = "High"
        elif imbalance_ratio > 2:
            risk = "Medium"

        self.report["class_imbalance"] = {
            "distribution_pct": counts.round(2).to_dict(),
            "imbalance_ratio": round(imbalance_ratio, 2),
            "risk_level": risk
        }
        return self.report["class_imbalance"]

    def check_sensitive_feature_bias(self):
        results = {}
        for feature in self.sensitive_features:
            if feature not in self.df.columns:
                continue
            if self.target_col and self.target_col in self.df.columns:
                # Outcome rate per group (proxy for demographic parity check)
                group_rates = self.df.groupby(feature)[self.target_col].mean()
                spread = (group_rates.max() - group_rates.min()) if len(group_rates) > 1 else 0
                risk = "High" if spread > 0.2 else "Medium" if spread > 0.1 else "Low"
                results[feature] = {
                    "outcome_rate_by_group": group_rates.round(3).to_dict(),
                    "max_spread": round(float(spread), 3),
                    "risk_level": risk
                }
            else:
                dist = self.df[feature].value_counts(normalize=True) * 100
                results[feature] = {
                    "distribution_pct": dist.round(2).to_dict(),
                    "risk_level": "N/A - no target column to test outcome bias"
                }
        self.report["sensitive_feature_bias"] = results
        return results

    def check_data_drift(self, reference_df: pd.DataFrame):
        """Compare numeric column means/std between reference (training) and current data."""
        drift_flags = {}
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if col not in reference_df.columns:
                continue
            ref_mean, ref_std = reference_df[col].mean(), reference_df[col].std()
            cur_mean = self.df[col].mean()
            if ref_std == 0 or np.isnan(ref_std):
                continue
            z = abs(cur_mean - ref_mean) / ref_std
            if z > 1:
                drift_flags[col] = {"z_shift": round(float(z), 2)}
        self.report["data_drift"] = {
            "flagged_columns": drift_flags,
            "risk_level": "High" if len(drift_flags) > 2 else "Medium" if drift_flags else "Low"
        }
        return self.report["data_drift"]

    # ---------- Aggregate score ----------

    def overall_risk_score(self):
        """Combine individual risk levels into a single 0-100 risk score (higher = riskier)."""
        weights = {"Low": 0, "Medium": 15, "High": 30, "N/A": 0}
        score = 0
        count = 0

        for key in ["missing_values", "class_imbalance", "data_drift"]:
            level = self.report.get(key, {}).get("risk_level")
            if level in weights:
                score += weights[level]
                count += 1

        for feature_result in self.report.get("sensitive_feature_bias", {}).values():
            level = feature_result.get("risk_level")
            if level in weights:
                score += weights[level]
                count += 1

        final_score = min(100, score)
        self.report["overall_risk_score"] = final_score
        self.report["overall_risk_level"] = (
            "High" if final_score >= 45 else "Medium" if final_score >= 20 else "Low"
        )
        return final_score

    def run_full_audit(self, reference_df: pd.DataFrame = None):
        self.check_missing_values()
        self.check_class_imbalance()
        self.check_sensitive_feature_bias()
        if reference_df is not None:
            self.check_data_drift(reference_df)
        self.overall_risk_score()
        return self.report
