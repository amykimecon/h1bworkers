from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LogisticRegression, LogisticRegressionCV, PoissonRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
    from company_shift_share.exposure_event_study import (
        _build_design_matrices,
        _build_leaveout_masks,
        _finalize_interaction_meta,
        _limit_model_features,
        _resolve_lasso_cs,
        _resolve_lasso_cv_folds,
        _standardize_nonbinary_features,
    )
    from company_shift_share.revelio_company_features import load_or_build_company_features
    from company_shift_share.source_exposure_data import load_or_build_source_analysis_panel
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config  # type: ignore[no-redef]
    from company_shift_share.exposure_event_study import (  # type: ignore[no-redef]
        _build_design_matrices,
        _build_leaveout_masks,
        _finalize_interaction_meta,
        _limit_model_features,
        _resolve_lasso_cs,
        _resolve_lasso_cv_folds,
        _standardize_nonbinary_features,
    )
    from company_shift_share.revelio_company_features import load_or_build_company_features  # type: ignore[no-redef]
    from company_shift_share.source_exposure_data import load_or_build_source_analysis_panel  # type: ignore[no-redef]


DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "output" / "opt_index_model_comparison_20260420"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    target_col: str
    estimator: str
    add_interactions: bool = False
    class_weight: Optional[str] = None
    use_probability_metrics: bool = False
    inverse_link: Optional[str] = None
    notes: str = ""


def _fit_ols_least_squares(
    x_train: pd.DataFrame,
    x_all: pd.DataFrame,
    y_train: pd.Series,
    *,
    active_cols: list[str],
    output_index: pd.Index,
) -> tuple[pd.Series, Optional[float], pd.Series]:
    x_train_arr = x_train.to_numpy(dtype=float, copy=False)
    x_all_arr = x_all.to_numpy(dtype=float, copy=False)
    y_train_arr = pd.to_numeric(y_train, errors="coerce").to_numpy(dtype=float, copy=False)

    x_train_fit = np.empty((x_train_arr.shape[0], x_train_arr.shape[1] + 1), dtype=float)
    x_train_fit[:, 0] = 1.0
    x_train_fit[:, 1:] = x_train_arr
    coef, _, _, _ = np.linalg.lstsq(x_train_fit, y_train_arr, rcond=None)

    x_all_fit = np.empty((x_all_arr.shape[0], x_all_arr.shape[1] + 1), dtype=float)
    x_all_fit[:, 0] = 1.0
    x_all_fit[:, 1:] = x_all_arr
    predicted = pd.Series(x_all_fit @ coef, index=output_index)
    intercept_value = float(coef[0]) if len(coef) else None
    weight_series = pd.Series(coef[1:], index=list(active_cols)).sort_values(
        key=lambda s: s.abs(),
        ascending=False,
    )
    return predicted, intercept_value, weight_series


def _safe_auc(y_true: pd.Series, score: pd.Series) -> Optional[float]:
    y_true = pd.to_numeric(y_true, errors="coerce")
    score = pd.to_numeric(score, errors="coerce")
    keep = y_true.notna() & score.notna()
    if keep.sum() == 0:
        return None
    y_true = y_true.loc[keep]
    score = score.loc[keep]
    if y_true.nunique() < 2 or score.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, score))


def _score_bin_table(score: pd.Series, y_binary: pd.Series, *, n_bins: int = 10) -> pd.DataFrame:
    work = pd.DataFrame({"score": pd.to_numeric(score, errors="coerce"), "y_binary": pd.to_numeric(y_binary, errors="coerce")})
    work = work.dropna().copy()
    if work.empty:
        return pd.DataFrame()
    n_eval_bins = max(1, min(int(n_bins), int(work["score"].nunique())))
    if n_eval_bins == 1:
        work["score_bin"] = 1
    else:
        work["score_bin"] = pd.qcut(
            work["score"].rank(method="first"),
            q=n_eval_bins,
            labels=False,
            duplicates="drop",
        ) + 1
    return (
        work.groupby("score_bin", as_index=False)
        .agg(
            n_obs=("y_binary", "size"),
            actual_rate=("y_binary", "mean"),
            score_mean=("score", "mean"),
            score_min=("score", "min"),
            score_max=("score", "max"),
        )
    )


def _spearman_corr(left: pd.Series, right: pd.Series) -> Optional[float]:
    work = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if work.empty or work["left"].nunique() < 2 or work["right"].nunique() < 2:
        return None
    return float(work["left"].corr(work["right"], method="spearman"))


def _build_post_targets(
    panel: pd.DataFrame,
    *,
    x_source_col: str,
    target_year_min: int,
    target_year_max: int,
) -> pd.DataFrame:
    if x_source_col not in panel.columns:
        raise ValueError(f"Analysis panel is missing target source column '{x_source_col}'.")
    target = panel[panel["t"].between(target_year_min, target_year_max)][["c", x_source_col]].copy()
    target[x_source_col] = pd.to_numeric(target[x_source_col], errors="coerce").fillna(0)
    target = (
        target.groupby("c", as_index=False)[x_source_col]
        .sum()
        .rename(columns={x_source_col: "post2016_n_opt"})
    )
    target["post2016_any_opt"] = target["post2016_n_opt"].gt(0).astype(int)
    target["post2016_log1p_n_opt"] = np.log1p(target["post2016_n_opt"])

    all_analysis_firms = panel[["c"]].drop_duplicates().copy()
    out = all_analysis_firms.merge(
        target[["c", "post2016_any_opt", "post2016_n_opt", "post2016_log1p_n_opt"]],
        on="c",
        how="left",
    )
    out["post2016_any_opt"] = out["post2016_any_opt"].fillna(0).astype(int)
    out["post2016_n_opt"] = pd.to_numeric(out["post2016_n_opt"], errors="coerce").fillna(0.0)
    out["post2016_log1p_n_opt"] = pd.to_numeric(out["post2016_log1p_n_opt"], errors="coerce").fillna(0.0)
    return out


def _prepare_model_frame(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    leaveout_enabled: bool,
    leaveout_share: float,
    leaveout_seed: int,
) -> pd.DataFrame:
    model_df = feature_df.merge(target_df, on="c", how="left")
    target_cols = ["post2016_any_opt", "post2016_n_opt", "post2016_log1p_n_opt"]
    for col in target_cols:
        model_df[col] = np.where(
            model_df["preferred_rcid_source"].fillna(0).eq(1),
            pd.to_numeric(model_df[col], errors="coerce").fillna(0),
            0,
        )
    model_df["post2016_any_opt"] = model_df["post2016_any_opt"].astype(int)
    model_df["target_source"] = np.where(
        model_df["preferred_rcid_source"].fillna(0).eq(1),
        "preferred_rcid_source",
        np.where(
            model_df["outside_negative_candidate"].fillna(0).eq(1),
            "outside_negative",
            "other_source",
        ),
    )
    return _build_leaveout_masks(
        model_df,
        leaveout_enabled=leaveout_enabled,
        leaveout_share=leaveout_share,
        leaveout_seed=leaveout_seed,
    )


def _load_existing_lasso_metrics(pred_path: Path, target_df: pd.DataFrame) -> Optional[dict[str, object]]:
    if not pred_path.exists():
        return None
    pred_df = pd.read_parquet(pred_path)
    meta_path = pred_path.with_suffix(pred_path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    pred_df = pred_df.merge(
        target_df[["c", "post2016_n_opt"]],
        on="c",
        how="left",
    )
    pred_df["post2016_n_opt"] = pd.to_numeric(pred_df["post2016_n_opt"], errors="coerce").fillna(0.0)
    eval_df = pred_df.loc[pred_df["event_study_sample"].eq(1)].copy()
    if eval_df.empty:
        return None
    y_binary = eval_df["post2016_any_opt"].astype(int)
    score = pd.to_numeric(eval_df["predicted_prob"], errors="coerce")
    pred_class = pd.to_numeric(eval_df["predicted_class"], errors="coerce").fillna(0).astype(int)
    bin_table = _score_bin_table(score, y_binary)
    top_actual = float(bin_table["actual_rate"].iloc[-1]) if not bin_table.empty else None
    bottom_actual = float(bin_table["actual_rate"].iloc[0]) if not bin_table.empty else None
    n_train_obs = int(pred_df["train_sample"].sum())
    n_eval_obs = int(pred_df["event_study_sample"].sum())
    n_total_obs = int(pred_df["c"].nunique())
    is_full_sample = n_train_obs >= n_total_obs and n_eval_obs >= n_total_obs
    evaluation_design = "full sample" if is_full_sample else "held-out leaveout"
    interaction_count = int(meta.get("n_interaction_columns_added") or 0)
    return {
        "name": "binary_logit_l1_full_sample" if is_full_sample else "binary_logit_l1_holdout",
        "target_col": "post2016_any_opt",
        "estimator": "logit_l1_cv",
        "add_interactions": interaction_count > 0,
        "class_weight": meta.get("logit_class_weight"),
        "evaluation_design": evaluation_design,
        "n_train_obs": n_train_obs,
        "n_eval_obs": n_eval_obs,
        "runtime_sec": np.nan,
        "n_active_features": meta.get("n_active_features"),
        "n_interactions": interaction_count,
        "auc_any_opt": _safe_auc(y_binary, score),
        "brier_any_opt": float(brier_score_loss(y_binary, score)),
        "mean_pred_prob": float(score.mean()),
        "mean_actual_any_opt": float(y_binary.mean()),
        "predicted_class_share": float(pred_class.mean()),
        "top_decile_any_opt_rate": top_actual,
        "bottom_decile_any_opt_rate": bottom_actual,
        "top_bottom_rate_diff": (top_actual - bottom_actual) if top_actual is not None and bottom_actual is not None else None,
        "top_bottom_rate_ratio": (top_actual / bottom_actual) if top_actual is not None and bottom_actual not in (None, 0) else None,
        "rmse_count": np.nan,
        "mae_count": np.nan,
        "r2_count": np.nan,
        "rmse_log1p_count": np.nan,
        "spearman_count_score": _spearman_corr(eval_df["post2016_n_opt"], score),
        "notes": (
            "Loaded from completed production lasso run on the full sample."
            if is_full_sample
            else "Loaded from completed cached lasso rerun."
        ),
    }


def _render_report(
    results: pd.DataFrame,
    *,
    out_dir: Path,
    cfg_path: Path,
    feature_df: pd.DataFrame,
    panel: pd.DataFrame,
    target_df: pd.DataFrame,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "model_comparison_report.md"

    binary_cols = [
        "name",
        "estimator",
        "evaluation_design",
        "add_interactions",
        "class_weight",
        "runtime_sec",
        "n_active_features",
        "auc_any_opt",
        "brier_any_opt",
        "mean_pred_prob",
        "mean_actual_any_opt",
        "predicted_class_share",
        "top_decile_any_opt_rate",
        "bottom_decile_any_opt_rate",
        "top_bottom_rate_diff",
    ]
    count_cols = [
        "name",
        "estimator",
        "evaluation_design",
        "add_interactions",
        "runtime_sec",
        "n_active_features",
        "auc_any_opt",
        "rmse_count",
        "mae_count",
        "r2_count",
        "rmse_log1p_count",
        "spearman_count_score",
        "top_decile_any_opt_rate",
        "bottom_decile_any_opt_rate",
    ]

    binary_table = results.loc[results["target_col"].eq("post2016_any_opt"), binary_cols].copy()
    count_table = results.loc[results["target_col"].ne("post2016_any_opt"), count_cols].copy()
    binary_scopes = sorted(set(binary_table["evaluation_design"].dropna().astype(str))) if not binary_table.empty else []
    mixed_binary_scopes = len(binary_scopes) > 1

    lines = [
        "# OPT Index Model Comparison",
        "",
        "## Setup",
        f"- Config: `{cfg_path}`",
        f"- Feature firms: `{len(feature_df):,}`",
        f"- Panel rows: `{len(panel):,}`",
        f"- Post-period mean any-OPT rate: `{target_df['post2016_any_opt'].mean():.3f}`",
        f"- Post-period mean OPT count: `{target_df['post2016_n_opt'].mean():.3f}`",
        "",
        "## Binary Target Models",
        binary_table.to_markdown(index=False, floatfmt=".4f") if not binary_table.empty else "_No binary-target results._",
        "",
        "Interpretation:",
        "- `auc_any_opt` measures how well the score ranks firms with any post-2016 OPT use above firms with zero post-2016 OPT use.",
        "- `brier_any_opt` is only reported when the model outputs probability-like predictions in `[0,1]`.",
        "- `top_bottom_rate_diff` compares realized any-OPT rates in the top versus bottom score decile on the held-out sample.",
        "- `evaluation_design` indicates whether a row is evaluated on the leave-out holdout sample or on the full sample used for training.",
        "",
        "## Count Target Models",
        count_table.to_markdown(index=False, floatfmt=".4f") if not count_table.empty else "_No count-target results._",
        "",
        "Interpretation:",
        "- `rmse_count` / `mae_count` / `r2_count` are evaluated on post-2016 OPT counts.",
        "- `rmse_log1p_count` evaluates fit on `log1p(post2016_n_opt)` to reduce the influence of the extreme right tail.",
        "- `auc_any_opt` is still reported for count-target models by using the predicted score to rank firms on the binary any-OPT outcome.",
        "",
        "## Notes",
        (
            "- Held-out rows are directly comparable because they use the same leave-out split; full-sample rows should be treated as in-sample diagnostics."
            if mixed_binary_scopes
            else "- All rows use the same leave-out split so the holdout metrics are directly comparable."
        ),
        "- Outside-negative firms are treated as zero-target observations by construction, matching the current production setup.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _fit_and_score_spec(
    spec: ExperimentSpec,
    *,
    model_df: pd.DataFrame,
    x_train: pd.DataFrame,
    x_all: pd.DataFrame,
    active_cols: list[str],
    lasso_cv_folds: Optional[int],
    lasso_n_cs: Optional[int],
) -> dict[str, object]:
    train_mask = model_df["train_sample"].eq(1)
    eval_mask = model_df["event_study_sample"].eq(1)
    if not eval_mask.any():
        eval_mask = model_df["in_analysis_universe"].eq(1)

    y_train = pd.to_numeric(model_df.loc[train_mask, spec.target_col], errors="coerce")
    y_eval_binary = model_df.loc[eval_mask, "post2016_any_opt"].astype(int)
    y_eval_count = pd.to_numeric(model_df.loc[eval_mask, "post2016_n_opt"], errors="coerce").fillna(0.0)
    y_eval_log1p = pd.to_numeric(model_df.loc[eval_mask, "post2016_log1p_n_opt"], errors="coerce").fillna(0.0)

    t0 = time.time()
    predicted_score: pd.Series
    predicted_prob: Optional[pd.Series] = None
    predicted_class: Optional[pd.Series] = None

    if spec.estimator == "ols":
        predicted_raw, _, _ = _fit_ols_least_squares(
            x_train,
            x_all,
            y_train,
            active_cols=active_cols,
            output_index=model_df.index,
        )
        if spec.target_col == "post2016_any_opt":
            predicted_prob = predicted_raw.clip(0, 1)
            predicted_class = (predicted_prob >= 0.5).astype(int)
            predicted_score = predicted_prob
        elif spec.inverse_link == "log1p":
            predicted_score = np.expm1(predicted_raw).clip(lower=0)
        else:
            predicted_score = predicted_raw.clip(lower=0)

    elif spec.estimator == "logit_l2":
        if y_train.nunique() < 2:
            raise ValueError(f"{spec.name} training target has only one class.")
        x_train_fit, x_all_fit, _ = _standardize_nonbinary_features(x_train, x_all)
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight=spec.class_weight,
            max_iter=5000,
            n_jobs=-1,
        )
        model.fit(x_train_fit, y_train.astype(int))
        prob_matrix = model.predict_proba(x_all_fit)
        class_order = list(model.classes_)
        class_one_idx = class_order.index(1) if 1 in class_order else 0
        predicted_prob = pd.Series(prob_matrix[:, class_one_idx], index=model_df.index)
        predicted_class = (predicted_prob >= 0.5).astype(int)
        predicted_score = predicted_prob

    elif spec.estimator == "logit_l1_cv":
        if y_train.nunique() < 2:
            raise ValueError(f"{spec.name} training target has only one class.")
        cv_folds = _resolve_lasso_cv_folds(y_train.astype(int), requested_folds=lasso_cv_folds)
        lasso_cs = _resolve_lasso_cs(lasso_n_cs)
        x_train_fit, x_all_fit, _ = _standardize_nonbinary_features(x_train, x_all)
        model = LogisticRegressionCV(
            Cs=lasso_cs,
            cv=cv_folds,
            penalty="l1",
            solver="saga",
            class_weight=spec.class_weight or "balanced",
            scoring="neg_log_loss",
            max_iter=5000,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train_fit, y_train.astype(int))
        prob_matrix = model.predict_proba(x_all_fit)
        class_order = list(model.classes_)
        class_one_idx = class_order.index(1) if 1 in class_order else 0
        predicted_prob = pd.Series(prob_matrix[:, class_one_idx], index=model_df.index)
        predicted_class = (predicted_prob >= 0.5).astype(int)
        predicted_score = predicted_prob

    elif spec.estimator == "lasso_linear_cv":
        x_train_fit, x_all_fit, _ = _standardize_nonbinary_features(x_train, x_all)
        model = LassoCV(cv=3, random_state=42, max_iter=10000, n_jobs=-1)
        model.fit(x_train_fit, y_train.astype(float))
        predicted_raw = pd.Series(model.predict(x_all_fit), index=model_df.index)
        if spec.inverse_link == "log1p":
            predicted_score = np.expm1(predicted_raw).clip(lower=0)
        else:
            predicted_score = predicted_raw.clip(lower=0)

    elif spec.estimator == "poisson":
        x_train_fit, x_all_fit, _ = _standardize_nonbinary_features(x_train, x_all)
        model = PoissonRegressor(alpha=1.0, max_iter=1000)
        model.fit(x_train_fit, y_train.astype(float))
        predicted_score = pd.Series(model.predict(x_all_fit), index=model_df.index).clip(lower=0)

    else:
        raise ValueError(f"Unknown estimator: {spec.estimator}")

    runtime_sec = float(time.time() - t0)
    score_eval = pd.to_numeric(predicted_score.loc[eval_mask], errors="coerce")
    bin_table = _score_bin_table(score_eval, y_eval_binary)
    top_actual = float(bin_table["actual_rate"].iloc[-1]) if not bin_table.empty else None
    bottom_actual = float(bin_table["actual_rate"].iloc[0]) if not bin_table.empty else None

    result: dict[str, object] = {
        **asdict(spec),
        "evaluation_design": "held-out leaveout",
        "n_train_obs": int(train_mask.sum()),
        "n_eval_obs": int(eval_mask.sum()),
        "runtime_sec": runtime_sec,
        "n_active_features": int(len(active_cols)),
        "n_interactions": int(sum(1 for col in active_cols if str(col).startswith("ix__"))),
        "auc_any_opt": _safe_auc(y_eval_binary, score_eval),
        "top_decile_any_opt_rate": top_actual,
        "bottom_decile_any_opt_rate": bottom_actual,
        "top_bottom_rate_diff": (top_actual - bottom_actual) if top_actual is not None and bottom_actual is not None else None,
        "top_bottom_rate_ratio": (top_actual / bottom_actual) if top_actual is not None and bottom_actual not in (None, 0) else None,
        "rmse_count": float(math.sqrt(mean_squared_error(y_eval_count, score_eval.clip(lower=0)))) if spec.target_col != "post2016_any_opt" else np.nan,
        "mae_count": float(mean_absolute_error(y_eval_count, score_eval.clip(lower=0))) if spec.target_col != "post2016_any_opt" else np.nan,
        "r2_count": float(r2_score(y_eval_count, score_eval.clip(lower=0))) if spec.target_col != "post2016_any_opt" else np.nan,
        "rmse_log1p_count": float(math.sqrt(mean_squared_error(y_eval_log1p, np.log1p(score_eval.clip(lower=0))))) if spec.target_col != "post2016_any_opt" else np.nan,
        "spearman_count_score": _spearman_corr(y_eval_count, score_eval),
    }
    if spec.use_probability_metrics and predicted_prob is not None:
        prob_eval = pd.to_numeric(predicted_prob.loc[eval_mask], errors="coerce").clip(0, 1)
        result.update(
            {
                "brier_any_opt": float(brier_score_loss(y_eval_binary, prob_eval)),
                "mean_pred_prob": float(prob_eval.mean()),
                "mean_actual_any_opt": float(y_eval_binary.mean()),
                "predicted_class_share": float(predicted_class.loc[eval_mask].mean()) if predicted_class is not None else np.nan,
            }
        )
    else:
        result.update(
            {
                "brier_any_opt": np.nan,
                "mean_pred_prob": np.nan,
                "mean_actual_any_opt": float(y_eval_binary.mean()),
                "predicted_class_share": np.nan,
            }
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare alternative opt index targets and model structures on cached firm data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Config path for cached features/panel.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for report outputs.")
    parser.add_argument(
        "--existing-lasso-predictions",
        type=Path,
        default=Path("/home/yk0581/data/out/company_shift_share_apr2026_lasso_rerun_20260420/opt_probability_index.parquet"),
        help="Optional existing cached lasso prediction parquet to append to the comparison if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    exp_cfg = get_cfg_section(cfg, "exposure_event_study")

    feature_year_min = int(exp_cfg.get("feature_year_min", 2010))
    feature_year_max = int(exp_cfg.get("feature_year_max", 2015))
    target_year_min = int(exp_cfg.get("target_year_min", 2016))
    target_year_max = int(exp_cfg.get("target_year_max", 2022))
    data_min_t = int(exp_cfg.get("data_min_t", 2010))
    data_max_t = int(exp_cfg.get("data_max_t", 2022))
    x_source_col = str(exp_cfg.get("x_source_col", "any_opt_hires_correction_aware"))
    leaveout_enabled = bool(exp_cfg.get("leaveout_enabled", True))
    leaveout_share = float(exp_cfg.get("leaveout_share", 0.25))
    leaveout_seed = int(exp_cfg.get("leaveout_seed", 42))
    max_active_features = exp_cfg.get("model_max_active_features")
    if max_active_features is None:
        max_active_features = get_cfg_section(cfg, "testing").get("model_max_active_features")
    max_feature_to_train_ratio = exp_cfg.get("model_max_feature_to_train_ratio")
    if max_feature_to_train_ratio is None:
        max_feature_to_train_ratio = get_cfg_section(cfg, "testing").get("model_max_feature_to_train_ratio")
    feature_sample_seed = int(exp_cfg.get("feature_sample_seed", get_cfg_section(cfg, "testing").get("feature_sample_seed", 42)))
    lasso_cv_folds = exp_cfg.get("lasso_cv_folds")
    lasso_n_cs = exp_cfg.get("lasso_n_cs")

    feature_df, _ = load_or_build_company_features(
        config_path=args.config,
        cfg=cfg,
        feature_year_min=feature_year_min,
        feature_year_max=feature_year_max,
        force_rebuild=False,
    )
    panel, _ = load_or_build_source_analysis_panel(
        config_path=args.config,
        cfg=cfg,
        data_min_t=data_min_t,
        data_max_t=data_max_t,
        force_rebuild=False,
    )
    target_df = _build_post_targets(
        panel,
        x_source_col=x_source_col,
        target_year_min=target_year_min,
        target_year_max=target_year_max,
    )
    model_df = _prepare_model_frame(
        feature_df,
        target_df,
        leaveout_enabled=leaveout_enabled,
        leaveout_share=leaveout_share,
        leaveout_seed=leaveout_seed,
    )
    train_mask = model_df["train_sample"].eq(1)

    specs = [
        ExperimentSpec(
            name="binary_lpm",
            target_col="post2016_any_opt",
            estimator="ols",
            add_interactions=False,
            use_probability_metrics=True,
            notes="Current linear probability baseline.",
        ),
        ExperimentSpec(
            name="binary_logit_l2",
            target_col="post2016_any_opt",
            estimator="logit_l2",
            add_interactions=False,
            class_weight=None,
            use_probability_metrics=True,
            notes="Plain logistic regression on the same design matrix.",
        ),
        ExperimentSpec(
            name="binary_logit_l2_balanced",
            target_col="post2016_any_opt",
            estimator="logit_l2",
            add_interactions=False,
            class_weight="balanced",
            use_probability_metrics=True,
            notes="Balanced logistic to compensate for class imbalance in the training sample.",
        ),
        ExperimentSpec(
            name="binary_lasso_linear",
            target_col="post2016_any_opt",
            estimator="lasso_linear_cv",
            add_interactions=False,
            notes="Linear lasso on the binary target.",
        ),
        ExperimentSpec(
            name="count_ols_raw",
            target_col="post2016_n_opt",
            estimator="ols",
            add_interactions=False,
            notes="Raw count target under least squares.",
        ),
        ExperimentSpec(
            name="count_ols_log1p",
            target_col="post2016_log1p_n_opt",
            estimator="ols",
            add_interactions=False,
            inverse_link="log1p",
            notes="Log-count target under least squares to reduce tail sensitivity.",
        ),
        ExperimentSpec(
            name="count_poisson",
            target_col="post2016_n_opt",
            estimator="poisson",
            add_interactions=False,
            notes="Poisson regression on counts with a log link.",
        ),
        ExperimentSpec(
            name="count_lasso_log1p",
            target_col="post2016_log1p_n_opt",
            estimator="lasso_linear_cv",
            add_interactions=False,
            inverse_link="log1p",
            notes="Linear lasso on the log-count target.",
        ),
    ]

    design_cache: dict[bool, tuple[pd.DataFrame, pd.DataFrame, list[str]]] = {}
    results: list[dict[str, object]] = []

    for spec in specs:
        if spec.add_interactions not in design_cache:
            design_df = model_df.drop(columns=["post2016_n_opt", "post2016_log1p_n_opt"], errors="ignore")
            design_method = "lasso" if spec.add_interactions else "lpm"
            x_train, x_all, _, active_cols, _, _, interaction_meta = _build_design_matrices(
                design_df,
                train_mask,
                model_method=design_method,
            )
            x_train, x_all, active_cols, _ = _limit_model_features(
                x_train,
                x_all,
                active_cols,
                n_train_obs=int(train_mask.sum()),
                max_active_features=max_active_features,
                max_feature_to_train_ratio=max_feature_to_train_ratio,
                random_seed=feature_sample_seed,
            )
            interaction_meta = _finalize_interaction_meta(interaction_meta, active_cols=active_cols)
            design_cache[spec.add_interactions] = (x_train, x_all, active_cols)
            print(
                f"[design] interactions={spec.add_interactions} | "
                f"{len(active_cols):,} active features | "
                f"{int(interaction_meta['n_interaction_columns_added']):,} interaction columns"
            )

        x_train, x_all, active_cols = design_cache[spec.add_interactions]
        print(f"[run] {spec.name} | target={spec.target_col} | estimator={spec.estimator}")
        result = _fit_and_score_spec(
            spec,
            model_df=model_df,
            x_train=x_train,
            x_all=x_all,
            active_cols=active_cols,
            lasso_cv_folds=lasso_cv_folds,
            lasso_n_cs=lasso_n_cs,
        )
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True))

    existing_lasso = _load_existing_lasso_metrics(args.existing_lasso_predictions, target_df)
    if existing_lasso is not None:
        print("[run] Included cached production lasso predictions.")
        results.append(existing_lasso)
    else:
        print(f"[run] Existing lasso predictions not found at {args.existing_lasso_predictions}; skipping.")

    result_df = pd.DataFrame(results).sort_values(["target_col", "auc_any_opt"], ascending=[True, False]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "model_comparison_results.csv"
    json_path = args.out_dir / "model_comparison_results.json"
    result_df.to_csv(csv_path, index=False)
    json_path.write_text(result_df.to_json(orient="records", indent=2), encoding="utf-8")
    report_path = _render_report(
        result_df,
        out_dir=args.out_dir,
        cfg_path=args.config.resolve(),
        feature_df=feature_df,
        panel=panel,
        target_df=target_df,
    )
    print(f"[done] Wrote results to {csv_path}")
    print(f"[done] Wrote report to {report_path}")


if __name__ == "__main__":
    main()
