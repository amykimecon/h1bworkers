from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from company_shift_share.config_loader import load_config
from company_shift_share.exposure_event_study import _select_index_feature_columns
from company_shift_share.matched_exposure_design import (
    DEFAULT_CONFIG_PATH,
    _load_base_configs,
    _parse_args,
    assign_persistent_entry_cohorts,
    build_matching_balance_tables,
    build_treatment_propensity_scores,
    build_instrument_panel_from_transition_shares,
    build_stacked_did_panel,
    build_transition_shares_from_source_flows,
    classify_zct_trajectory,
    match_high_to_low_exposure,
    merge_zct_into_analysis_panel,
    run_common_break_event_study,
    run_matched_exposure_design,
    run_stacked_did,
    summarize_zct_trajectory,
)


class MatchedExposureDesignTests(unittest.TestCase):
    def test_parse_args_ignores_ipykernel_launcher_flags(self) -> None:
        with patch(
            "company_shift_share.matched_exposure_design.sys.argv",
            ["ipykernel_launcher.py", "--f=/tmp/kernel.json"],
        ), patch.dict(
            "company_shift_share.matched_exposure_design.sys.modules",
            {"ipykernel": object()},
            clear=False,
        ):
            parsed = _parse_args()
        self.assertIsNone(parsed.config)

    def test_parse_args_keeps_cli_strict_for_explicit_unknown_args(self) -> None:
        with self.assertRaises(SystemExit):
            _parse_args(["--not-a-real-flag"])

    def test_load_base_configs_applies_shift_window_overrides(self) -> None:
        cfg = {
            "matched_exposure_design": {
                "base_shift_share_config_path": "shift.yaml",
                "base_source_config_path": "source.yaml",
                "school_sample_window_start": 2012,
                "school_sample_window_end": 2018,
                "event_shock_pre_years": 3,
                "event_shock_post_years": 4,
                "share_year_min": 2010,
                "share_year_max": 2013,
                "share_robustness_windows": [[2010, 2011], [2012, 2013], [2010, 2013]],
                "reuse_cached_wrds_universe_only": True,
            }
        }
        with patch(
            "company_shift_share.matched_exposure_design.load_config",
            side_effect=[{"pipeline": {}}, {}],
        ):
            shift_cfg, source_cfg = _load_base_configs(cfg)
        self.assertEqual(int(shift_cfg["pipeline"]["school_sample_window_start"]), 2012)
        self.assertEqual(int(shift_cfg["pipeline"]["school_sample_window_end"]), 2018)
        self.assertEqual(int(shift_cfg["pipeline"]["event_shock_pre_years"]), 3)
        self.assertEqual(int(shift_cfg["pipeline"]["event_shock_post_years"]), 4)
        self.assertEqual(int(shift_cfg["pipeline"]["share_year_min"]), 2010)
        self.assertEqual(int(shift_cfg["pipeline"]["share_year_max"]), 2013)
        self.assertEqual(shift_cfg["pipeline"]["share_robustness_windows"], [[2010, 2011], [2012, 2013], [2010, 2013]])
        self.assertTrue(bool(source_cfg["reuse_cached_wrds_universe_only"]))

    def test_matching_feature_family_excludes_school_and_opt_but_keeps_nonus(self) -> None:
        feature_df = pd.DataFrame(
            {
                "c": [1, 2],
                "in_analysis_universe": [1, 1],
                "preferred_rcid_source": [1, 1],
                "outside_negative_candidate": [0, 0],
                "nonus_educ_share_annual_pre_level": [0.2, 0.3],
                "salary_mean_annual_pre_level": [100.0, 120.0],
                "school_opt_share_new_hire_annual_pre_level": [0.4, 0.1],
                "n_schools_new_hire_annual_pre_level": [10.0, 5.0],
                "opt_hire_rate_annual_pre_level": [0.2, 0.0],
                "masters_opt_hire_rate_annual_pre_level": [0.2, 0.0],
            }
        )
        selected = _select_index_feature_columns(
            feature_df,
            feature_family="matching_fundamentals",
        )
        self.assertIn("nonus_educ_share_annual_pre_level", selected)
        self.assertIn("salary_mean_annual_pre_level", selected)
        self.assertNotIn("school_opt_share_new_hire_annual_pre_level", selected)
        self.assertNotIn("n_schools_new_hire_annual_pre_level", selected)
        self.assertNotIn("opt_hire_rate_annual_pre_level", selected)
        self.assertNotIn("masters_opt_hire_rate_annual_pre_level", selected)

    def test_new_config_defaults_leaveout_to_false(self) -> None:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        design_cfg = cfg["matched_exposure_design"]
        self.assertFalse(bool(design_cfg["leaveout_enabled"]))
        self.assertEqual(str(design_cfg["high_exposure_metric"]), "max_z")
        self.assertEqual(int(design_cfg["source_flow_year_min"]), 2010)
        self.assertEqual(int(design_cfg["share_year_min"]), 2010)
        self.assertEqual(int(design_cfg["share_year_max"]), 2013)
        self.assertEqual(int(design_cfg["matching_progress_every"]), 250)
        self.assertEqual(str(design_cfg["propensity_design"]), "treatment_propensity")
        self.assertEqual(str(design_cfg["matching_algorithm"]), "optimal_max_cardinality_min_cost")
        self.assertEqual(str(design_cfg["matching_users_feature"]), "company_n_users_log1p")
        self.assertEqual(str(design_cfg["matching_coverage_feature"]), "firm_size_annual_pre_n_years")
        self.assertEqual(str(design_cfg["matching_coverage_rule"]), "full_pre_coverage")
        self.assertAlmostEqual(float(design_cfg["matching_users_weight"]), 0.20)
        self.assertTrue(bool(design_cfg["write_analysis_reports"]))
        self.assertEqual(int(design_cfg["analysis_top_balance_rows"]), 20)
        self.assertFalse(bool(design_cfg["reuse_saved_analysis_outputs"]))
        self.assertTrue(bool(design_cfg["reuse_saved_matching_outputs"]))
        self.assertTrue(bool(design_cfg["reuse_cached_wrds_universe_only"]))

    def test_build_transition_shares_from_source_flows_reproduces_share_windows(self) -> None:
        school_flows = pd.DataFrame(
            {
                "c": [1, 1, 1, 1],
                "t": [2008, 2010, 2012, 2016],
                "university_raw": ["school a", "school a", "school b", "school b"],
                "n_transitions": [2.0, 4.0, 2.0, 10.0],
            }
        )
        school_map = pd.DataFrame(
            {
                "university_raw_key": ["school a", "school b"],
                "unitid": ["A", "B"],
            }
        )
        firms = pd.DataFrame({"c": [1]})
        out = build_transition_shares_from_source_flows(
            school_flows,
            school_map,
            firms,
            share_period="pre_window",
            share_base_year=2010,
            share_year_min=2008,
            share_year_max=2013,
            robustness_windows=((2008, 2010), (2011, 2013), (2008, 2013)),
            min_universities_for_share=1,
        ).set_index("k")
        self.assertAlmostEqual(float(out.loc["A", "share_ck"]), 0.75)
        self.assertAlmostEqual(float(out.loc["B", "share_ck"]), 0.25)
        self.assertAlmostEqual(float(out.loc["A", "share_ck_2008_2010"]), 1.0)
        self.assertAlmostEqual(float(out.loc["B", "share_ck_2011_2013"]), 1.0)
        self.assertAlmostEqual(float(out.loc["A", "share_ck_full"]), 6.0 / 18.0)
        self.assertAlmostEqual(float(out.loc["B", "share_ck_full"]), 12.0 / 18.0)

    def test_rebuilt_instrument_panel_and_zero_fill(self) -> None:
        transition_shares = pd.DataFrame(
            {
                "c": [1],
                "k": ["A"],
                "share_ck": [0.75],
                "share_ck_full": [0.75],
            }
        )
        school_growth = pd.DataFrame(
            {
                "k": ["A", "A", "A"],
                "t": [2015, 2016, 2017],
                "g_kt": [0.0, 10.0, 10.0],
            }
        )
        instrument_panel = build_instrument_panel_from_transition_shares(
            transition_shares,
            school_growth,
        )
        row_2016 = instrument_panel.set_index(["c", "t"]).loc[(1, 2016)]
        self.assertAlmostEqual(float(row_2016["z_ct"]), 7.5)

        base_panel = pd.DataFrame(
            {
                "c": [1, 1, 1, 2, 2, 2],
                "t": [2015, 2016, 2017, 2015, 2016, 2017],
                "in_analysis_universe": [1] * 6,
                "preferred_rcid_source": [1, 1, 1, 0, 0, 0],
                "outside_negative_candidate": [0, 0, 0, 1, 1, 1],
                "any_opt_hires_correction_aware": [0, 1, 1, 0, 0, 0],
                "y_cst_lag0": [10.0, 11.0, 12.0, 9.0, 9.0, 9.0],
                "y_cst_foreign_lag0": [4.0, 5.0, 6.0, 1.0, 1.0, 1.0],
                "y_cst_native_lag0": [6.0, 6.0, 6.0, 8.0, 8.0, 8.0],
                "y_new_hires_lag0": [2.0, 3.0, 4.0, 1.0, 1.0, 1.0],
                "y_new_hires_foreign_lag0": [1.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                "y_new_hires_native_lag0": [1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            }
        )
        merged = merge_zct_into_analysis_panel(base_panel, instrument_panel).set_index(["c", "t"])
        self.assertAlmostEqual(float(merged.loc[(1, 2016), "z_ct"]), 7.5)
        self.assertAlmostEqual(float(merged.loc[(2, 2016), "z_ct"]), 0.0)

    def test_classify_zct_trajectory_handles_outside_negatives(self) -> None:
        trajectory = pd.DataFrame(
            {
                "c": [1, 2, 3, 4, 5],
                "cum_z": [10.0, 8.0, 1.0, 0.0, 5.0],
                "max_z": [10.0, 8.0, 1.0, 0.0, 5.0],
                "mean_z": [1.7, 1.3, 0.2, 0.0, 0.8],
                "active_share": [0.6, 0.55, 0.2, 0.0, 0.6],
                "preferred_rcid_source": [1, 1, 1, 0, 0],
                "outside_negative_candidate": [0, 0, 0, 1, 1],
                "in_analysis_universe": [1, 1, 1, 1, 1],
                "trajectory_name": ["pre_policy"] * 5,
                "window_start": [2010] * 5,
                "window_end": [2015] * 5,
            }
        )
        out = classify_zct_trajectory(
            trajectory,
            high_quantile=0.75,
            low_quantile=0.25,
            high_exposure_metric="max_z",
            low_exposure_metric="max_z",
        ).set_index("c")
        self.assertEqual(out.loc[1, "exposure_group"], "high_exposure")
        self.assertEqual(out.loc[3, "exposure_group"], "low_exposure")
        self.assertEqual(out.loc[4, "exposure_group"], "low_exposure")
        self.assertEqual(out.loc[2, "exposure_group"], "middle_exposure")

    def test_match_high_to_low_exposure_respects_exact_naics_and_caliper(self) -> None:
        classified = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "cum_z": [10.0, 0.0, 0.0, 9.0],
                "active_share": [0.6, 0.0, 0.0, 0.6],
                "preferred_rcid_source": [1, 1, 0, 1],
                "outside_negative_candidate": [0, 0, 1, 0],
                "in_analysis_universe": [1, 1, 1, 1],
                "trajectory_name": ["pre_policy"] * 4,
                "window_start": [2010] * 4,
                "window_end": [2015] * 4,
                "high_exposure": [1, 0, 0, 1],
                "low_exposure": [0, 1, 1, 0],
                "exposure_group": ["high_exposure", "low_exposure", "low_exposure", "high_exposure"],
            }
        )
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "naics2": ["54", "54", "54", "52"],
                "company_n_users_log1p": [6.0, 5.95, 5.80, 4.0],
                "company_hq_region": ["West", "West", "South", "South"],
                "firm_size_annual_pre_n_years": [4, 4, 4, 4],
            }
        )
        propensity_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "predicted_prob": [0.781, 0.781, 0.781, 0.20],
                "predicted_index": [0.781, 0.781, 0.781, 0.20],
            }
        )
        pair_df, diagnostics = match_high_to_low_exposure(
            classified,
            feature_df,
            propensity_df,
            include_outside_negative_controls=True,
            caliper_sd_multiplier=0.5,
        )
        self.assertEqual(int(diagnostics["n_pairs"]), 1)
        self.assertEqual(int(pair_df.iloc[0]["treated_c"]), 1)
        self.assertEqual(int(pair_df.iloc[0]["control_c"]), 2)
        self.assertEqual(int(diagnostics["n_treated_after_support"]), 1)

    def test_match_high_to_low_exposure_logs_progress_and_search_space(self) -> None:
        classified = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "cum_z": [10.0, 0.0, 0.0],
                "active_share": [0.5, 0.0, 0.0],
                "preferred_rcid_source": [1, 1, 0],
                "outside_negative_candidate": [0, 0, 1],
                "in_analysis_universe": [1, 1, 1],
                "trajectory_name": ["pre_policy"] * 3,
                "window_start": [2010] * 3,
                "window_end": [2015] * 3,
                "high_exposure": [1, 0, 0],
                "low_exposure": [0, 1, 1],
                "exposure_group": ["high_exposure", "low_exposure", "low_exposure"],
            }
        )
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "naics2": ["54", "54", "54"],
                "company_n_users_log1p": [6.0, 5.95, 5.8],
                "company_hq_region": ["West", "West", "South"],
                "firm_size_annual_pre_n_years": [4, 4, 4],
            }
        )
        propensity_df = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "predicted_prob": [0.78, 0.78, 0.77],
                "predicted_index": [0.78, 0.78, 0.77],
            }
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            _, diagnostics = match_high_to_low_exposure(
                classified,
                feature_df,
                propensity_df,
                include_outside_negative_controls=True,
                caliper_sd_multiplier=0.5,
                progress_every=1,
            )
        output = stdout.getvalue()
        self.assertIn("matching start", output)
        self.assertIn("matching search space", output)
        self.assertIn("matching progress", output)
        self.assertIn("matching done", output)
        self.assertIn("estimated_candidate_pairs", diagnostics)
        self.assertIn("avg_candidates_before_caliper", diagnostics)
        self.assertIn("elapsed_seconds", diagnostics)

    def test_match_high_to_low_exposure_uses_rowwise_size_fallback(self) -> None:
        classified = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "cum_z": [10.0, 0.0, 0.0],
                "active_share": [0.6, 0.0, 0.0],
                "preferred_rcid_source": [1, 1, 1],
                "outside_negative_candidate": [0, 0, 0],
                "in_analysis_universe": [1, 1, 1],
                "trajectory_name": ["pre_policy"] * 3,
                "window_start": [2010] * 3,
                "window_end": [2015] * 3,
                "high_exposure": [1, 0, 0],
                "low_exposure": [0, 1, 1],
                "exposure_group": ["high_exposure", "low_exposure", "low_exposure"],
            }
        )
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "naics2": ["54", "54", "54"],
                "company_hq_region": ["West", "West", "West"],
                "company_n_users_log1p": [6.0, 6.0, 5.2],
                "match_firm_size_pre_level": [np.nan, np.nan, np.nan],
                "firm_size_annual_pre_n_years": [4, 4, 4],
            }
        )
        propensity_df = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "predicted_prob": [0.78, 0.78, 0.78],
                "predicted_index": [0.78, 0.78, 0.78],
            }
        )
        pair_df, _ = match_high_to_low_exposure(
            classified,
            feature_df,
            propensity_df,
            include_outside_negative_controls=True,
            caliper_sd_multiplier=0.5,
        )
        self.assertEqual(int(pair_df.iloc[0]["control_c"]), 2)
        self.assertAlmostEqual(float(pair_df.iloc[0]["effective_treated_size"]), 6.0)
        self.assertAlmostEqual(float(pair_df.iloc[0]["effective_control_size"]), 6.0)
        self.assertAlmostEqual(float(pair_df.iloc[0]["size_distance"]), 0.0)

    def test_match_high_to_low_exposure_drops_low_coverage_controls(self) -> None:
        classified = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "cum_z": [10.0, 0.0, 0.0],
                "active_share": [0.6, 0.0, 0.0],
                "preferred_rcid_source": [1, 1, 1],
                "outside_negative_candidate": [0, 0, 0],
                "in_analysis_universe": [1, 1, 1],
                "trajectory_name": ["pre_policy"] * 3,
                "window_start": [2010] * 3,
                "window_end": [2015] * 3,
                "high_exposure": [1, 0, 0],
                "low_exposure": [0, 1, 1],
                "exposure_group": ["high_exposure", "low_exposure", "low_exposure"],
            }
        )
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "naics2": ["54", "54", "54"],
                "company_hq_region": ["West", "West", "West"],
                "company_n_users_log1p": [6.0, 6.0, 5.9],
                "match_firm_size_pre_level": [6.0, 6.0, 5.9],
                "firm_size_annual_pre_n_years": [4, 3, 4],
            }
        )
        propensity_df = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "predicted_prob": [0.78, 0.78, 0.78],
                "predicted_index": [0.78, 0.78, 0.78],
            }
        )
        pair_df, diagnostics = match_high_to_low_exposure(
            classified,
            feature_df,
            propensity_df,
            include_outside_negative_controls=True,
            caliper_sd_multiplier=0.5,
        )
        self.assertEqual(int(pair_df.iloc[0]["control_c"]), 3)
        self.assertEqual(int(diagnostics["dropped_controls_low_coverage"]), 1)

    def test_match_high_to_low_exposure_requires_observed_growth_when_treated_has_growth(self) -> None:
        classified = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "cum_z": [10.0, 0.0, 0.0],
                "active_share": [0.6, 0.0, 0.0],
                "preferred_rcid_source": [1, 1, 1],
                "outside_negative_candidate": [0, 0, 0],
                "in_analysis_universe": [1, 1, 1],
                "trajectory_name": ["pre_policy"] * 3,
                "window_start": [2010] * 3,
                "window_end": [2015] * 3,
                "high_exposure": [1, 0, 0],
                "low_exposure": [0, 1, 1],
                "exposure_group": ["high_exposure", "low_exposure", "low_exposure"],
            }
        )
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "naics2": ["54", "54", "54"],
                "company_hq_region": ["West", "West", "West"],
                "company_n_users_log1p": [6.0, 6.0, 6.0],
                "match_firm_size_pre_level": [6.0, 6.0, 6.0],
                "match_firm_size_pre_growth": [0.50, np.nan, 0.50],
                "firm_size_annual_pre_n_years": [4, 4, 4],
            }
        )
        propensity_df = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "predicted_prob": [0.78, 0.78, 0.78],
                "predicted_index": [0.78, 0.78, 0.78],
            }
        )
        pair_df, _ = match_high_to_low_exposure(
            classified,
            feature_df,
            propensity_df,
            include_outside_negative_controls=True,
            caliper_sd_multiplier=0.5,
        )
        self.assertEqual(int(pair_df.iloc[0]["control_c"]), 3)
        self.assertAlmostEqual(float(pair_df.iloc[0]["growth_distance"]), 0.0)

    def test_match_high_to_low_exposure_uses_optimal_assignment(self) -> None:
        classified = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "cum_z": [10.0, 9.0, 0.0, 0.0],
                "active_share": [0.6, 0.55, 0.0, 0.0],
                "preferred_rcid_source": [1, 1, 1, 1],
                "outside_negative_candidate": [0, 0, 0, 0],
                "in_analysis_universe": [1, 1, 1, 1],
                "trajectory_name": ["pre_policy"] * 4,
                "window_start": [2010] * 4,
                "window_end": [2015] * 4,
                "high_exposure": [1, 1, 0, 0],
                "low_exposure": [0, 0, 1, 1],
                "exposure_group": ["high_exposure", "high_exposure", "low_exposure", "low_exposure"],
            }
        )
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "naics2": ["54", "54", "54", "54"],
                "company_hq_region": ["West", "West", "West", "West"],
                "company_n_users_log1p": [6.0, 5.0, 5.0, 6.2],
                "match_firm_size_pre_level": [6.0, 5.0, 5.0, 6.2],
                "match_firm_size_pre_growth": [np.nan, 0.50, 0.50, np.nan],
                "firm_size_annual_pre_n_years": [4, 4, 4, 4],
            }
        )
        propensity_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "predicted_prob": [0.78, 0.78, 0.78, 0.78],
                "predicted_index": [0.78, 0.78, 0.78, 0.78],
            }
        )
        pair_df, diagnostics = match_high_to_low_exposure(
            classified,
            feature_df,
            propensity_df,
            include_outside_negative_controls=True,
            caliper_sd_multiplier=0.5,
        )
        matched_controls = dict(
            zip(
                pd.to_numeric(pair_df["treated_c"], errors="coerce").astype(int),
                pd.to_numeric(pair_df["control_c"], errors="coerce").astype(int),
                strict=False,
            )
        )
        self.assertEqual(int(diagnostics["n_pairs"]), 2)
        self.assertEqual(matched_controls, {1: 4, 2: 3})

    def test_build_treatment_propensity_scores_uses_eligible_match_pool(self) -> None:
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "in_analysis_universe": [1, 1, 1, 1],
                "preferred_rcid_source": [1, 1, 0, 1],
                "outside_negative_candidate": [0, 0, 1, 0],
                "company_n_users_log1p": [6.0, 5.8, 5.6, 5.4],
            }
        )
        classified = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "trajectory_name": ["pre_policy"] * 4,
                "high_exposure": [1, 0, 0, 0],
                "low_exposure": [0, 1, 1, 0],
                "preferred_rcid_source": [1, 1, 0, 1],
                "outside_negative_candidate": [0, 0, 1, 0],
            }
        )
        captured: dict[str, pd.DataFrame] = {}

        def fake_fit(feature_input: pd.DataFrame, target_df: pd.DataFrame, **_: object) -> tuple[pd.DataFrame, dict]:
            captured["feature_input"] = feature_input.copy()
            captured["target_df"] = target_df.copy()
            return (
                pd.DataFrame(
                    {
                        "c": feature_input["c"],
                        "predicted_prob": [0.5] * len(feature_input),
                        "predicted_index": [0.5] * len(feature_input),
                    }
                ),
                {"model_method": "logit"},
            )

        with patch(
            "company_shift_share.matched_exposure_design.fit_opt_probability_index",
            side_effect=fake_fit,
        ):
            pred_df, diagnostics = build_treatment_propensity_scores(
                feature_df,
                classified,
                design_cfg={"propensity_model_method": "logit"},
                include_outside_negative_controls=False,
            )

        self.assertEqual(
            captured["feature_input"].set_index("c")["in_analysis_universe"].astype(int).to_dict(),
            {1: 1, 2: 1, 3: 0, 4: 0},
        )
        self.assertEqual(
            captured["target_df"].set_index("c")["post2016_any_opt"].astype(int).to_dict(),
            {1: 1, 2: 0, 3: 0, 4: 0},
        )
        self.assertTrue(pred_df["trajectory_name"].eq("pre_policy").all())
        self.assertTrue(pred_df["propensity_design"].eq("treatment_propensity").all())
        self.assertEqual(str(diagnostics["training_sample_mode"]), "eligible_match_pool")
        self.assertEqual(int(diagnostics["n_eligible_treated"]), 1)
        self.assertEqual(int(diagnostics["n_eligible_controls"]), 1)

    def test_build_treatment_propensity_scores_records_feature_weights(self) -> None:
        feature_df = pd.DataFrame(
            {
                "c": [1, 2],
                "in_analysis_universe": [1, 1],
                "preferred_rcid_source": [1, 1],
                "outside_negative_candidate": [0, 0],
                "company_n_users_log1p": [6.0, 5.8],
                "naics2": ["54", "54"],
            }
        )
        classified = pd.DataFrame(
            {
                "c": [1, 2],
                "trajectory_name": ["pre_policy", "pre_policy"],
                "high_exposure": [1, 0],
                "low_exposure": [0, 1],
                "preferred_rcid_source": [1, 1],
                "outside_negative_candidate": [0, 0],
            }
        )

        def fake_fit(feature_input: pd.DataFrame, target_df: pd.DataFrame, **_: object) -> tuple[pd.DataFrame, dict, dict]:
            return (
                pd.DataFrame(
                    {
                        "c": feature_input["c"],
                        "predicted_prob": [0.8, 0.2],
                        "predicted_index": [0.8, 0.2],
                    }
                ),
                {"model_method": "logit"},
                {
                    "weight_series": pd.Series(
                        {
                            "company_n_users_log1p": 0.70,
                            "naics2_54": -0.25,
                        }
                    ),
                    "feature_columns_raw": ["company_n_users_log1p", "naics2"],
                    "standardized_feature_columns": ["company_n_users_log1p"],
                    "interaction_column_names": [],
                    "intercept": -0.10,
                },
            )

        with patch(
            "company_shift_share.matched_exposure_design.fit_opt_probability_index",
            side_effect=fake_fit,
        ):
            _, diagnostics = build_treatment_propensity_scores(
                feature_df,
                classified,
                design_cfg={"propensity_model_method": "logit"},
                include_outside_negative_controls=True,
            )

        weights = pd.DataFrame(diagnostics["feature_weights"])
        self.assertEqual(weights.iloc[0]["feature"], "company_n_users_log1p")
        self.assertEqual(weights.iloc[1]["raw_feature"], "naics2")
        self.assertAlmostEqual(float(diagnostics["intercept"]), -0.10)
        self.assertEqual(int(diagnostics["n_feature_weights"]), 2)

    def test_build_matching_balance_tables_reports_pre_post_and_sources(self) -> None:
        classified = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "cum_z": [10.0, 0.0, 0.0],
                "active_share": [0.6, 0.0, 0.0],
                "preferred_rcid_source": [1, 1, 0],
                "outside_negative_candidate": [0, 0, 1],
                "in_analysis_universe": [1, 1, 1],
                "trajectory_name": ["pre_policy"] * 3,
                "window_start": [2010] * 3,
                "window_end": [2015] * 3,
                "high_exposure": [1, 0, 0],
                "low_exposure": [0, 1, 1],
                "exposure_group": ["high_exposure", "low_exposure", "low_exposure"],
            }
        )
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "naics2": ["54", "54", "54"],
                "company_hq_region": ["West", "West", "South"],
                "company_n_users_log1p": [6.0, 5.8, 5.6],
                "firm_size_annual_pre_n_years": [4, 4, 4],
                "salary_mean_annual_pre_level": [100.0, 99.0, 95.0],
                "nonus_educ_share_annual_pre_level": [0.30, 0.28, 0.25],
            }
        )
        propensity_df = pd.DataFrame(
            {
                "c": [1, 2, 3],
                "predicted_prob": [0.74, 0.74, 0.74],
                "predicted_index": [0.74, 0.74, 0.74],
            }
        )
        pair_df = pd.DataFrame(
            {
                "pair_id": [1],
                "treated_c": [1],
                "control_c": [2],
                "control_source": ["preferred_low_exposure"],
                "trajectory_name": ["pre_policy"],
            }
        )
        balance_table, balance_summary = build_matching_balance_tables(
            classified,
            feature_df,
            propensity_df,
            pair_df,
            include_outside_negative_controls=True,
        )
        self.assertFalse(balance_table.empty)
        self.assertIn("pre_match", set(balance_table["match_stage"]))
        self.assertIn("post_match", set(balance_table["match_stage"]))
        self.assertIn("overall", set(balance_table["control_source"]))
        self.assertIn("preferred_low_exposure", set(balance_table["control_source"]))
        self.assertIn("outside_negative", set(balance_table["control_source"]))
        summary_index = balance_summary.set_index(["match_stage", "control_source"])
        self.assertIn(("pre_match", "overall"), summary_index.index)
        self.assertIn(("post_match", "preferred_low_exposure"), summary_index.index)

    def test_assign_cohorts_and_build_stacked_panel(self) -> None:
        matched_panel = pd.DataFrame(
            {
                "pair_id": [1] * 14,
                "c": [10] * 7 + [20] * 7,
                "t": list(range(2011, 2018)) * 2,
                "z_ct": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0] + [0.0] * 7,
                "treated": [1] * 7 + [0] * 7,
                "high_exposure": [1] * 7 + [0] * 7,
                "trajectory_name": ["pre_policy"] * 14,
                "y_cst_lag0": [10.0] * 14,
            }
        )
        pair_df = pd.DataFrame({"pair_id": [1], "treated_c": [10], "control_c": [20]})
        cohort_df = assign_persistent_entry_cohorts(
            matched_panel.loc[matched_panel["treated"].eq(1), ["c", "t", "z_ct"]],
            cohort_min_year=2014,
            cohort_max_year=2019,
            pre_years_required=3,
            forward_window_years=4,
            min_positive_years_in_forward_window=2,
        )
        self.assertEqual(int(cohort_df.set_index("c").loc[10, "g"]), 2014)
        stacked_panel, diagnostics = build_stacked_did_panel(
            matched_panel,
            pair_df,
            cohort_df,
            pre_window=3,
            post_window=3,
        )
        self.assertEqual(int(diagnostics["n_stacks"]), 1)
        self.assertListEqual(sorted(stacked_panel["rel_time"].unique().tolist()), [-3, -2, -1, 0, 1, 2, 3])
        self.assertIn("unit_stack_fe", stacked_panel.columns)

    def test_build_stacked_panel_drops_controls_that_turn_high(self) -> None:
        matched_panel = pd.DataFrame(
            {
                "pair_id": [1] * 14,
                "c": [10] * 7 + [20] * 7,
                "t": list(range(2011, 2018)) * 2,
                "z_ct": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0] + [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "treated": [1] * 7 + [0] * 7,
                "high_exposure": [1] * 7 + [0] * 7,
                "trajectory_name": ["pre_policy"] * 14,
                "y_cst_lag0": [10.0] * 14,
            }
        )
        pair_df = pd.DataFrame({"pair_id": [1], "treated_c": [10], "control_c": [20]})
        cohort_df = pd.DataFrame({"c": [10], "g": [2014]})
        stacked_panel, diagnostics = build_stacked_did_panel(
            matched_panel,
            pair_df,
            cohort_df,
            pre_window=3,
            post_window=3,
        )
        self.assertTrue(stacked_panel.empty)
        self.assertEqual(int(diagnostics["dropped_controls_turn_high"]), 1)

    def test_common_break_event_study_recovers_post_break_effect(self) -> None:
        rows: list[dict[str, object]] = []
        for pair_id in range(1, 11):
            treated_c = pair_id
            control_c = 100 + pair_id
            for year in range(2012, 2018):
                year_effect = float(year - 2012)
                rows.append(
                    {
                        "pair_id": pair_id,
                        "c": treated_c,
                        "t": year,
                        "trajectory_name": "pre_policy",
                        "high_exposure": 1,
                        "treated": 1,
                        "y_cst_lag0": 10.0 + pair_id + year_effect + (5.0 if year >= 2015 else 0.0),
                    }
                )
                rows.append(
                    {
                        "pair_id": pair_id,
                        "c": control_c,
                        "t": year,
                        "trajectory_name": "pre_policy",
                        "high_exposure": 0,
                        "treated": 0,
                        "y_cst_lag0": 10.0 + pair_id + year_effect,
                    }
                )
        matched_panel = pd.DataFrame(rows)
        out = run_common_break_event_study(
            matched_panel,
            outcome_cols=["y_cst_lag0"],
            ref_year=2014,
        )
        out = out.set_index("year")
        self.assertAlmostEqual(float(out.loc[2013, "coef"]), 0.0, places=6)
        self.assertGreater(float(out.loc[2015, "coef"]), 4.5)
        self.assertGreater(float(out.loc[2017, "coef"]), 4.5)

    def test_stacked_did_recovers_post_treatment_effect(self) -> None:
        rows: list[dict[str, object]] = []
        for pair_id in range(1, 11):
            for rel_time in range(-3, 4):
                year = 2014 + rel_time
                for treated, firm_id in ((1, pair_id), (0, 100 + pair_id)):
                    effect = 0.0
                    if treated == 1 and rel_time >= 0:
                        effect = 2.0 + rel_time
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "stack_id": pair_id,
                            "c": firm_id,
                            "t": year,
                            "rel_time": rel_time,
                            "treated": treated,
                            "trajectory_name": "pre_policy",
                            "pair_stack_fe": f"{pair_id}__{pair_id}",
                            "year_stack_fe": f"{year}__{pair_id}",
                            "y_cst_lag0": 20.0 + pair_id + 0.5 * rel_time + effect,
                        }
                    )
        stacked_panel = pd.DataFrame(rows)
        out = run_stacked_did(
            stacked_panel,
            outcome_cols=["y_cst_lag0"],
            ref_event_time=-1,
        ).set_index("rel_time")
        self.assertAlmostEqual(float(out.loc[-2, "coef"]), 0.0, places=6)
        self.assertGreater(float(out.loc[0, "coef"]), 1.5)
        self.assertGreater(float(out.loc[3, "coef"]), 4.5)

    def test_stacked_did_removes_constant_level_gap_at_reference_time(self) -> None:
        rows: list[dict[str, object]] = []
        for pair_id in range(1, 11):
            for rel_time in range(-3, 4):
                year = 2014 + rel_time
                base = 20.0 + pair_id + 0.5 * rel_time
                rows.append(
                    {
                        "pair_id": pair_id,
                        "stack_id": pair_id,
                        "c": pair_id,
                        "t": year,
                        "rel_time": rel_time,
                        "treated": 1,
                        "trajectory_name": "pre_policy",
                        "pair_stack_fe": f"{pair_id}__{pair_id}",
                        "year_stack_fe": f"{year}__{pair_id}",
                        "y_cst_lag0": base + 5.0,
                    }
                )
                rows.append(
                    {
                        "pair_id": pair_id,
                        "stack_id": pair_id,
                        "c": 100 + pair_id,
                        "t": year,
                        "rel_time": rel_time,
                        "treated": 0,
                        "trajectory_name": "pre_policy",
                        "pair_stack_fe": f"{pair_id}__{pair_id}",
                        "year_stack_fe": f"{year}__{pair_id}",
                        "y_cst_lag0": base,
                    }
                )
        stacked_panel = pd.DataFrame(rows)
        out = run_stacked_did(
            stacked_panel,
            outcome_cols=["y_cst_lag0"],
            ref_event_time=-1,
        ).set_index("rel_time")
        for rel_time in (-3, -2, 0, 1, 2, 3):
            self.assertAlmostEqual(float(out.loc[rel_time, "coef"]), 0.0, places=6)

    def test_summarize_zct_trajectory_uses_requested_window(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 1, 2, 2, 2],
                "t": [2010, 2011, 2012, 2010, 2011, 2012],
                "z_ct": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "preferred_rcid_source": [1, 1, 1, 0, 0, 0],
                "outside_negative_candidate": [0, 0, 0, 1, 1, 1],
                "in_analysis_universe": [1, 1, 1, 1, 1, 1],
            }
        )
        summary = summarize_zct_trajectory(
            panel,
            window_start=2011,
            window_end=2012,
            trajectory_name="pre_policy",
        ).set_index("c")
        self.assertAlmostEqual(float(summary.loc[1, "cum_z"]), 2.0)
        self.assertAlmostEqual(float(summary.loc[1, "active_share"]), 1.0)
        self.assertAlmostEqual(float(summary.loc[2, "cum_z"]), 0.0)

    def test_run_matched_exposure_design_smoke_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg = {
                "paths": {
                    "out_dir": str(tmp),
                    "transition_shares_out": str(tmp / "transition_shares.parquet"),
                    "school_growth_out": str(tmp / "school_growth.parquet"),
                    "school_shift_sample_out": str(tmp / "school_shift_sample.parquet"),
                    "instrument_panel_out": str(tmp / "instrument_panel.parquet"),
                    "analysis_panel_out": str(tmp / "analysis_panel.parquet"),
                    "matching_features_out": str(tmp / "matching_features.parquet"),
                    "propensity_scores_out": str(tmp / "propensity_scores.parquet"),
                    "trajectory_summary_out": str(tmp / "trajectory_summary.parquet"),
                    "matched_pairs_out": str(tmp / "matched_pairs.parquet"),
                    "balance_table_out": str(tmp / "balance_table.parquet"),
                    "balance_summary_out": str(tmp / "balance_summary.parquet"),
                    "matched_panel_out": str(tmp / "matched_panel.parquet"),
                    "common_break_results_out": str(tmp / "common_break_results.parquet"),
                    "stacked_panel_out": str(tmp / "stacked_panel.parquet"),
                    "stacked_results_out": str(tmp / "stacked_results.parquet"),
                    "diagnostics_out": str(tmp / "diagnostics.json"),
                },
                "matched_exposure_design": {
                    "data_min_t": 2010,
                    "data_max_t": 2022,
                    "source_flow_year_min": 2008,
                    "source_flow_year_max": 2022,
                    "propensity_feature_year_min": 2010,
                    "propensity_feature_year_max": 2013,
                    "propensity_target_year_min": 2016,
                    "propensity_target_year_max": 2022,
                    "leaveout_enabled": False,
                    "include_outside_negative_controls": True,
                    "write_analysis_reports": True,
                    "analysis_top_balance_rows": 10,
                    "reuse_saved_analysis_outputs": False,
                    "trajectory_specs": [
                        {"name": "pre_policy", "start": 2010, "end": 2015, "run_matching": True, "run_regressions": True}
                    ],
                    "outcome_cols": ["y_cst_lag0"],
                    "stacked_min_cohort_year": 2014,
                    "stacked_max_cohort_year": 2019,
                    "stacked_pre_years": 3,
                    "stacked_post_years": 3,
                    "stacked_min_positive_years_in_forward_window": 2,
                },
                "testing": {"enabled": False},
            }

            panel = pd.DataFrame(
                {
                    "c": [1] * 13 + [2] * 13,
                    "t": list(range(2010, 2023)) * 2,
                    "in_analysis_universe": [1] * 26,
                    "preferred_rcid_source": [1] * 13 + [0] * 13,
                    "outside_negative_candidate": [0] * 13 + [1] * 13,
                    "any_opt_hires_correction_aware": [0] * 26,
                    "y_cst_lag0": [10.0] * 13 + [9.0] * 13,
                    "y_cst_foreign_lag0": [4.0] * 26,
                    "y_cst_native_lag0": [6.0] * 26,
                    "y_new_hires_lag0": [2.0] * 26,
                    "y_new_hires_foreign_lag0": [1.0] * 26,
                    "y_new_hires_native_lag0": [1.0] * 26,
                }
            )
            firms = pd.DataFrame(
                {
                    "c": [1, 2],
                    "in_analysis_universe": [1, 1],
                    "preferred_rcid_source": [1, 0],
                    "outside_negative_candidate": [0, 1],
                }
            )
            selected_meta = pd.DataFrame(
                {
                    "c": [1, 2],
                    "naics2": ["54", "54"],
                    "naics4": ["5415", "5415"],
                    "company_state_feature": ["CA", "CA"],
                    "company_metro_feature": ["sf", "sf"],
                    "company_hq_region": ["West", "West"],
                    "year_founded": [2000, 2000],
                    "n_users": [100.0, 90.0],
                }
            )
            workforce = pd.DataFrame({"c": [1, 2], "t": [2010, 2010]})
            school_growth = pd.DataFrame({"k": ["A"], "t": [2014], "g_kt": [10.0]})
            school_shift_sample = pd.DataFrame({"k": ["A"], "selected_for_instrument": [1]})
            transition_shares = pd.DataFrame({"c": [1], "k": ["A"], "share_ck": [1.0]})
            instrument_panel = pd.DataFrame(
                {
                    "c": [1, 1, 1],
                    "t": [2013, 2014, 2015],
                    "z_ct": [5.0, 10.0, 10.0],
                    "n_universities": [1.0, 1.0, 1.0],
                }
            )
            matching_features = pd.DataFrame(
                {
                    "c": [1, 2],
                    "in_analysis_universe": [1, 1],
                    "preferred_rcid_source": [1, 0],
                    "outside_negative_candidate": [0, 1],
                    "naics2": ["54", "54"],
                    "company_hq_region": ["West", "West"],
                    "company_n_users_log1p": [4.6, 4.5],
                    "firm_size_annual_pre_n_years": [4, 4],
                }
            )
            propensity_scores = pd.DataFrame(
                {
                    "c": [1, 2],
                    "predicted_prob": [0.781, 0.781],
                    "predicted_index": [0.781, 0.781],
                }
            )

            with patch(
                "company_shift_share.matched_exposure_design._load_base_configs",
                return_value=({}, {}),
            ), patch(
                "company_shift_share.matched_exposure_design.build_source_analysis_panel_from_inputs",
                return_value=(panel, firms, selected_meta, workforce),
            ), patch(
                "company_shift_share.matched_exposure_design.build_shift_share_components_from_configs",
                return_value=(school_growth, school_shift_sample, transition_shares, instrument_panel),
            ), patch(
                "company_shift_share.matched_exposure_design.build_matching_feature_frame_from_wrds",
                return_value=matching_features,
            ), patch(
                "company_shift_share.matched_exposure_design.build_propensity_scores",
                return_value=(propensity_scores, {"model_method": "logit"}),
            ), patch(
                "company_shift_share.matched_exposure_design.run_common_break_event_study",
                return_value=pd.DataFrame({"trajectory_name": ["pre_policy"], "outcome_col": ["y_cst_lag0"], "year": [2014], "coef": [0.0], "se": [0.0]}),
            ), patch(
                "company_shift_share.matched_exposure_design.assign_persistent_entry_cohorts",
                return_value=pd.DataFrame({"c": [1], "g": [2014]}),
            ), patch(
                "company_shift_share.matched_exposure_design.build_stacked_did_panel",
                return_value=(
                    pd.DataFrame(
                        {
                            "pair_id": [1, 1],
                            "stack_id": [1, 1],
                            "c": [1, 2],
                            "t": [2014, 2014],
                            "rel_time": [0, 0],
                            "treated": [1, 0],
                            "trajectory_name": ["pre_policy", "pre_policy"],
                            "pair_stack_fe": ["1__1", "1__1"],
                            "year_stack_fe": ["2014__1", "2014__1"],
                            "y_cst_lag0": [10.0, 9.0],
                        }
                    ),
                    {"n_stacks": 1},
                ),
            ), patch(
                "company_shift_share.matched_exposure_design.run_stacked_did",
                return_value=pd.DataFrame({"trajectory_name": ["pre_policy"], "outcome_col": ["y_cst_lag0"], "rel_time": [0], "coef": [1.0], "se": [0.1]}),
            ):
                out = run_matched_exposure_design(cfg=cfg)

            self.assertFalse(out["matched_pairs"].empty)
            self.assertFalse(out["balance_table"].empty)
            self.assertFalse(out["balance_summary"].empty)
            self.assertTrue((tmp / "analysis_panel.parquet").exists())
            self.assertTrue((tmp / "matched_pairs.parquet").exists())
            self.assertTrue((tmp / "balance_table.parquet").exists())
            self.assertTrue((tmp / "balance_summary.parquet").exists())
            self.assertTrue((tmp / "analysis_summary.md").exists())
            self.assertTrue((tmp / "tables" / "trajectory_group_counts.csv").exists())
            self.assertTrue((tmp / "tables" / "propensity_model_diagnostics.csv").exists())
            self.assertTrue((tmp / "tables" / "propensity_feature_weights.csv").exists())
            self.assertTrue((tmp / "tables" / "propensity_feature_group_weights.csv").exists())
            self.assertTrue((tmp / "tables" / "matching_diagnostics.csv").exists())
            self.assertTrue((tmp / "tables" / "matching_balance_all_covariates.csv").exists())
            self.assertTrue((tmp / "tables" / "common_break_event_study.csv").exists())
            self.assertTrue((tmp / "tables" / "stacked_did_event_study.csv").exists())
            self.assertGreaterEqual(len(list((tmp / "figures").glob("*.png"))), 3)
            summary_text = (tmp / "analysis_summary.md").read_text()
            self.assertLess(
                summary_text.index("common_break_event_study_pre_policy_y_cst_lag0.png"),
                summary_text.index("common_break_raw_means_pre_policy_y_cst_lag0.png"),
            )
            self.assertLess(
                summary_text.index("common_break_raw_means_pre_policy_y_cst_lag0.png"),
                summary_text.index("stacked_did_pre_policy_y_cst_lag0.png"),
            )
            self.assertLess(
                summary_text.index("stacked_did_pre_policy_y_cst_lag0.png"),
                summary_text.index("stacked_did_raw_means_pre_policy_y_cst_lag0.png"),
            )
            self.assertIn("balance", out["diagnostics"]["trajectory_specs"]["pre_policy"])
            self.assertIn("analysis_reports", out)
            self.assertTrue((tmp / "diagnostics.json").exists())

    def test_run_matched_exposure_design_treatment_propensity_mode(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg = {
                "paths": {
                    "out_dir": str(tmp),
                    "transition_shares_out": str(tmp / "transition_shares.parquet"),
                    "school_growth_out": str(tmp / "school_growth.parquet"),
                    "school_shift_sample_out": str(tmp / "school_shift_sample.parquet"),
                    "instrument_panel_out": str(tmp / "instrument_panel.parquet"),
                    "analysis_panel_out": str(tmp / "analysis_panel.parquet"),
                    "matching_features_out": str(tmp / "matching_features.parquet"),
                    "propensity_scores_out": str(tmp / "propensity_scores.parquet"),
                    "trajectory_summary_out": str(tmp / "trajectory_summary.parquet"),
                    "matched_pairs_out": str(tmp / "matched_pairs.parquet"),
                    "balance_table_out": str(tmp / "balance_table.parquet"),
                    "balance_summary_out": str(tmp / "balance_summary.parquet"),
                    "matched_panel_out": str(tmp / "matched_panel.parquet"),
                    "common_break_results_out": str(tmp / "common_break_results.parquet"),
                    "stacked_panel_out": str(tmp / "stacked_panel.parquet"),
                    "stacked_results_out": str(tmp / "stacked_results.parquet"),
                    "diagnostics_out": str(tmp / "diagnostics.json"),
                },
                "matched_exposure_design": {
                    "data_min_t": 2010,
                    "data_max_t": 2022,
                    "source_flow_year_min": 2008,
                    "source_flow_year_max": 2022,
                    "propensity_feature_year_min": 2010,
                    "propensity_feature_year_max": 2013,
                    "propensity_design": "treatment_propensity",
                    "leaveout_enabled": False,
                    "include_outside_negative_controls": True,
                    "write_analysis_reports": False,
                    "reuse_saved_analysis_outputs": False,
                    "trajectory_specs": [
                        {"name": "pre_policy", "start": 2010, "end": 2015, "run_matching": True, "run_regressions": False}
                    ],
                    "outcome_cols": ["y_cst_lag0"],
                },
                "testing": {"enabled": False},
            }
            panel = pd.DataFrame(
                {
                    "c": [1] * 13 + [2] * 13,
                    "t": list(range(2010, 2023)) * 2,
                    "in_analysis_universe": [1] * 26,
                    "preferred_rcid_source": [1] * 13 + [0] * 13,
                    "outside_negative_candidate": [0] * 13 + [1] * 13,
                    "any_opt_hires_correction_aware": [0] * 26,
                    "y_cst_lag0": [10.0] * 13 + [9.0] * 13,
                }
            )
            firms = pd.DataFrame(
                {
                    "c": [1, 2],
                    "in_analysis_universe": [1, 1],
                    "preferred_rcid_source": [1, 0],
                    "outside_negative_candidate": [0, 1],
                }
            )
            selected_meta = pd.DataFrame(
                {
                    "c": [1, 2],
                    "naics2": ["54", "54"],
                    "naics4": ["5415", "5415"],
                    "company_state_feature": ["CA", "CA"],
                    "company_metro_feature": ["sf", "sf"],
                    "company_hq_region": ["West", "West"],
                    "year_founded": [2000, 2000],
                    "n_users": [100.0, 90.0],
                }
            )
            workforce = pd.DataFrame({"c": [1, 2], "t": [2010, 2010]})
            school_growth = pd.DataFrame({"k": ["A"], "t": [2014], "g_kt": [10.0]})
            school_shift_sample = pd.DataFrame({"k": ["A"], "selected_for_instrument": [1]})
            transition_shares = pd.DataFrame({"c": [1], "k": ["A"], "share_ck": [1.0]})
            instrument_panel = pd.DataFrame(
                {
                    "c": [1, 1, 1],
                    "t": [2013, 2014, 2015],
                    "z_ct": [5.0, 10.0, 10.0],
                    "n_universities": [1.0, 1.0, 1.0],
                }
            )
            matching_features = pd.DataFrame(
                {
                    "c": [1, 2],
                    "in_analysis_universe": [1, 1],
                    "preferred_rcid_source": [1, 0],
                    "outside_negative_candidate": [0, 1],
                    "naics2": ["54", "54"],
                    "company_hq_region": ["West", "West"],
                    "company_n_users_log1p": [4.6, 4.5],
                    "firm_size_annual_pre_n_years": [4, 4],
                }
            )
            treatment_propensity_scores = pd.DataFrame(
                {
                    "c": [1, 2],
                    "trajectory_name": ["pre_policy", "pre_policy"],
                    "predicted_prob": [0.781, 0.781],
                    "predicted_index": [0.781, 0.781],
                    "propensity_design": ["treatment_propensity", "treatment_propensity"],
                }
            )

            with patch(
                "company_shift_share.matched_exposure_design._load_base_configs",
                return_value=({}, {}),
            ), patch(
                "company_shift_share.matched_exposure_design.build_source_analysis_panel_from_inputs",
                return_value=(panel, firms, selected_meta, workforce),
            ), patch(
                "company_shift_share.matched_exposure_design.build_shift_share_components_from_configs",
                return_value=(school_growth, school_shift_sample, transition_shares, instrument_panel),
            ), patch(
                "company_shift_share.matched_exposure_design.build_matching_feature_frame_from_wrds",
                return_value=matching_features,
            ), patch(
                "company_shift_share.matched_exposure_design.build_propensity_scores",
                side_effect=AssertionError("opt_takeup propensity path should be skipped"),
            ), patch(
                "company_shift_share.matched_exposure_design.build_treatment_propensity_scores",
                return_value=(
                    treatment_propensity_scores,
                    {"trajectory_name": "pre_policy", "propensity_design": "treatment_propensity"},
                ),
            ):
                out = run_matched_exposure_design(cfg=cfg)

            self.assertFalse(out["matched_pairs"].empty)
            self.assertTrue(out["propensity_scores"]["trajectory_name"].eq("pre_policy").all())
            self.assertTrue(out["propensity_scores"]["propensity_design"].eq("treatment_propensity").all())
            self.assertEqual(str(out["diagnostics"]["propensity_design"]), "treatment_propensity")
            self.assertIn("propensity", out["diagnostics"]["trajectory_specs"]["pre_policy"])
            self.assertTrue((tmp / "propensity_scores.parquet").exists())

    def test_run_matched_exposure_design_can_reuse_saved_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg = {
                "paths": {
                    "out_dir": str(tmp),
                    "transition_shares_out": str(tmp / "transition_shares.parquet"),
                    "school_growth_out": str(tmp / "school_growth.parquet"),
                    "school_shift_sample_out": str(tmp / "school_shift_sample.parquet"),
                    "instrument_panel_out": str(tmp / "instrument_panel.parquet"),
                    "analysis_panel_out": str(tmp / "analysis_panel.parquet"),
                    "matching_features_out": str(tmp / "matching_features.parquet"),
                    "propensity_scores_out": str(tmp / "propensity_scores.parquet"),
                    "trajectory_summary_out": str(tmp / "trajectory_summary.parquet"),
                    "matched_pairs_out": str(tmp / "matched_pairs.parquet"),
                    "balance_table_out": str(tmp / "balance_table.parquet"),
                    "balance_summary_out": str(tmp / "balance_summary.parquet"),
                    "matched_panel_out": str(tmp / "matched_panel.parquet"),
                    "common_break_results_out": str(tmp / "common_break_results.parquet"),
                    "stacked_panel_out": str(tmp / "stacked_panel.parquet"),
                    "stacked_results_out": str(tmp / "stacked_results.parquet"),
                    "diagnostics_out": str(tmp / "diagnostics.json"),
                },
                "matched_exposure_design": {
                    "data_min_t": 2010,
                    "data_max_t": 2022,
                    "source_flow_year_min": 2008,
                    "source_flow_year_max": 2022,
                    "propensity_feature_year_min": 2010,
                    "propensity_feature_year_max": 2013,
                    "propensity_target_year_min": 2016,
                    "propensity_target_year_max": 2022,
                    "leaveout_enabled": False,
                    "include_outside_negative_controls": True,
                    "write_analysis_reports": True,
                    "analysis_top_balance_rows": 10,
                    "reuse_saved_analysis_outputs": False,
                    "trajectory_specs": [
                        {"name": "pre_policy", "start": 2010, "end": 2015, "run_matching": True, "run_regressions": True}
                    ],
                    "outcome_cols": ["y_cst_lag0"],
                    "stacked_min_cohort_year": 2014,
                    "stacked_max_cohort_year": 2019,
                    "stacked_pre_years": 3,
                    "stacked_post_years": 3,
                    "stacked_min_positive_years_in_forward_window": 2,
                },
                "testing": {"enabled": False},
            }
            panel = pd.DataFrame(
                {
                    "c": [1] * 13 + [2] * 13,
                    "t": list(range(2010, 2023)) * 2,
                    "in_analysis_universe": [1] * 26,
                    "preferred_rcid_source": [1] * 13 + [0] * 13,
                    "outside_negative_candidate": [0] * 13 + [1] * 13,
                    "any_opt_hires_correction_aware": [0] * 26,
                    "y_cst_lag0": [10.0] * 13 + [9.0] * 13,
                    "y_cst_foreign_lag0": [4.0] * 26,
                    "y_cst_native_lag0": [6.0] * 26,
                    "y_new_hires_lag0": [2.0] * 26,
                    "y_new_hires_foreign_lag0": [1.0] * 26,
                    "y_new_hires_native_lag0": [1.0] * 26,
                }
            )
            firms = pd.DataFrame(
                {
                    "c": [1, 2],
                    "in_analysis_universe": [1, 1],
                    "preferred_rcid_source": [1, 0],
                    "outside_negative_candidate": [0, 1],
                }
            )
            selected_meta = pd.DataFrame(
                {
                    "c": [1, 2],
                    "naics2": ["54", "54"],
                    "naics4": ["5415", "5415"],
                    "company_state_feature": ["CA", "CA"],
                    "company_metro_feature": ["sf", "sf"],
                    "company_hq_region": ["West", "West"],
                    "year_founded": [2000, 2000],
                    "n_users": [100.0, 90.0],
                }
            )
            workforce = pd.DataFrame({"c": [1, 2], "t": [2010, 2010]})
            school_growth = pd.DataFrame({"k": ["A"], "t": [2014], "g_kt": [10.0]})
            school_shift_sample = pd.DataFrame({"k": ["A"], "selected_for_instrument": [1]})
            transition_shares = pd.DataFrame({"c": [1], "k": ["A"], "share_ck": [1.0]})
            instrument_panel = pd.DataFrame(
                {
                    "c": [1, 1, 1],
                    "t": [2013, 2014, 2015],
                    "z_ct": [5.0, 10.0, 10.0],
                    "n_universities": [1.0, 1.0, 1.0],
                }
            )
            matching_features = pd.DataFrame(
                {
                    "c": [1, 2],
                    "in_analysis_universe": [1, 1],
                    "preferred_rcid_source": [1, 0],
                    "outside_negative_candidate": [0, 1],
                    "naics2": ["54", "54"],
                    "company_hq_region": ["West", "West"],
                    "company_n_users_log1p": [4.6, 4.5],
                    "firm_size_annual_pre_n_years": [4, 4],
                }
            )
            propensity_scores = pd.DataFrame(
                {
                    "c": [1, 2],
                    "predicted_prob": [0.781, 0.781],
                    "predicted_index": [0.781, 0.781],
                }
            )

            with patch(
                "company_shift_share.matched_exposure_design._load_base_configs",
                return_value=({}, {}),
            ), patch(
                "company_shift_share.matched_exposure_design.build_source_analysis_panel_from_inputs",
                return_value=(panel, firms, selected_meta, workforce),
            ), patch(
                "company_shift_share.matched_exposure_design.build_shift_share_components_from_configs",
                return_value=(school_growth, school_shift_sample, transition_shares, instrument_panel),
            ), patch(
                "company_shift_share.matched_exposure_design.build_matching_feature_frame_from_wrds",
                return_value=matching_features,
            ), patch(
                "company_shift_share.matched_exposure_design.build_propensity_scores",
                return_value=(propensity_scores, {"model_method": "logit"}),
            ), patch(
                "company_shift_share.matched_exposure_design.run_common_break_event_study",
                return_value=pd.DataFrame({"trajectory_name": ["pre_policy"], "outcome_col": ["y_cst_lag0"], "year": [2014], "coef": [0.0], "se": [0.0]}),
            ), patch(
                "company_shift_share.matched_exposure_design.assign_persistent_entry_cohorts",
                return_value=pd.DataFrame({"c": [1], "g": [2014]}),
            ), patch(
                "company_shift_share.matched_exposure_design.build_stacked_did_panel",
                return_value=(
                    pd.DataFrame(
                        {
                            "pair_id": [1, 1],
                            "stack_id": [1, 1],
                            "c": [1, 2],
                            "t": [2014, 2014],
                            "rel_time": [0, 0],
                            "treated": [1, 0],
                            "trajectory_name": ["pre_policy", "pre_policy"],
                            "pair_stack_fe": ["1__1", "1__1"],
                            "year_stack_fe": ["2014__1", "2014__1"],
                            "y_cst_lag0": [10.0, 9.0],
                        }
                    ),
                    {"n_stacks": 1},
                ),
            ), patch(
                "company_shift_share.matched_exposure_design.run_stacked_did",
                return_value=pd.DataFrame({"trajectory_name": ["pre_policy"], "outcome_col": ["y_cst_lag0"], "rel_time": [0], "coef": [1.0], "se": [0.1]}),
            ):
                run_matched_exposure_design(cfg=cfg)

            cfg["matched_exposure_design"]["reuse_saved_analysis_outputs"] = True
            with patch(
                "company_shift_share.matched_exposure_design._load_base_configs",
                return_value=({}, {}),
            ), patch(
                "company_shift_share.matched_exposure_design.build_source_analysis_panel_from_inputs",
                side_effect=AssertionError("source panel rebuild should be skipped"),
            ), patch(
                "company_shift_share.matched_exposure_design.build_shift_share_components_from_configs",
                side_effect=AssertionError("shift-share rebuild should be skipped"),
            ), patch(
                "company_shift_share.matched_exposure_design.build_matching_feature_frame_from_wrds",
                side_effect=AssertionError("feature rebuild should be skipped"),
            ), patch(
                "company_shift_share.matched_exposure_design.build_propensity_scores",
                side_effect=AssertionError("propensity refit should be skipped"),
            ):
                reused = run_matched_exposure_design(cfg=cfg)

            self.assertTrue(bool(reused["diagnostics"]["reused_saved_analysis_outputs"]))
            self.assertFalse(reused["matched_pairs"].empty)
            self.assertTrue((tmp / "analysis_summary.md").exists())

    def test_run_matched_exposure_design_can_reuse_saved_matching_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg = {
                "paths": {
                    "out_dir": str(tmp),
                    "transition_shares_out": str(tmp / "transition_shares.parquet"),
                    "school_growth_out": str(tmp / "school_growth.parquet"),
                    "school_shift_sample_out": str(tmp / "school_shift_sample.parquet"),
                    "instrument_panel_out": str(tmp / "instrument_panel.parquet"),
                    "analysis_panel_out": str(tmp / "analysis_panel.parquet"),
                    "matching_features_out": str(tmp / "matching_features.parquet"),
                    "propensity_scores_out": str(tmp / "propensity_scores.parquet"),
                    "trajectory_summary_out": str(tmp / "trajectory_summary.parquet"),
                    "matched_pairs_out": str(tmp / "matched_pairs.parquet"),
                    "balance_table_out": str(tmp / "balance_table.parquet"),
                    "balance_summary_out": str(tmp / "balance_summary.parquet"),
                    "matched_panel_out": str(tmp / "matched_panel.parquet"),
                    "common_break_results_out": str(tmp / "common_break_results.parquet"),
                    "stacked_panel_out": str(tmp / "stacked_panel.parquet"),
                    "stacked_results_out": str(tmp / "stacked_results.parquet"),
                    "diagnostics_out": str(tmp / "diagnostics.json"),
                },
                "matched_exposure_design": {
                    "data_min_t": 2010,
                    "data_max_t": 2022,
                    "source_flow_year_min": 2008,
                    "source_flow_year_max": 2022,
                    "propensity_feature_year_min": 2010,
                    "propensity_feature_year_max": 2013,
                    "propensity_target_year_min": 2016,
                    "propensity_target_year_max": 2022,
                    "leaveout_enabled": False,
                    "include_outside_negative_controls": True,
                    "write_analysis_reports": False,
                    "reuse_saved_analysis_outputs": False,
                    "reuse_saved_matching_outputs": False,
                    "trajectory_specs": [
                        {"name": "pre_policy", "start": 2010, "end": 2015, "run_matching": True, "run_regressions": True}
                    ],
                    "outcome_cols": ["y_cst_lag0"],
                    "stacked_min_cohort_year": 2014,
                    "stacked_max_cohort_year": 2019,
                    "stacked_pre_years": 3,
                    "stacked_post_years": 3,
                    "stacked_min_positive_years_in_forward_window": 2,
                },
                "testing": {"enabled": False},
            }
            panel = pd.DataFrame(
                {
                    "c": [1] * 13 + [2] * 13,
                    "t": list(range(2010, 2023)) * 2,
                    "in_analysis_universe": [1] * 26,
                    "preferred_rcid_source": [1] * 13 + [0] * 13,
                    "outside_negative_candidate": [0] * 13 + [1] * 13,
                    "any_opt_hires_correction_aware": [0] * 26,
                    "y_cst_lag0": [10.0] * 13 + [9.0] * 13,
                }
            )
            firms = pd.DataFrame(
                {
                    "c": [1, 2],
                    "in_analysis_universe": [1, 1],
                    "preferred_rcid_source": [1, 0],
                    "outside_negative_candidate": [0, 1],
                }
            )
            selected_meta = pd.DataFrame(
                {
                    "c": [1, 2],
                    "naics2": ["54", "54"],
                    "naics4": ["5415", "5415"],
                    "company_state_feature": ["CA", "CA"],
                    "company_metro_feature": ["sf", "sf"],
                    "company_hq_region": ["West", "West"],
                    "year_founded": [2000, 2000],
                    "n_users": [100.0, 90.0],
                }
            )
            workforce = pd.DataFrame({"c": [1, 2], "t": [2010, 2010]})
            school_growth = pd.DataFrame({"k": ["A"], "t": [2014], "g_kt": [10.0]})
            school_shift_sample = pd.DataFrame({"k": ["A"], "selected_for_instrument": [1]})
            transition_shares = pd.DataFrame({"c": [1], "k": ["A"], "share_ck": [1.0]})
            instrument_panel = pd.DataFrame(
                {
                    "c": [1, 1, 1],
                    "t": [2013, 2014, 2015],
                    "z_ct": [5.0, 10.0, 10.0],
                    "n_universities": [1.0, 1.0, 1.0],
                }
            )
            matching_features = pd.DataFrame(
                {
                    "c": [1, 2],
                    "in_analysis_universe": [1, 1],
                    "preferred_rcid_source": [1, 0],
                    "outside_negative_candidate": [0, 1],
                    "naics2": ["54", "54"],
                    "company_hq_region": ["West", "West"],
                    "company_n_users_log1p": [4.6, 4.5],
                    "firm_size_annual_pre_n_years": [4, 4],
                }
            )
            propensity_scores = pd.DataFrame(
                {
                    "c": [1, 2],
                    "predicted_prob": [0.781, 0.781],
                    "predicted_index": [0.781, 0.781],
                }
            )
            initial_common_break = pd.DataFrame(
                {"trajectory_name": ["pre_policy"], "outcome_col": ["y_cst_lag0"], "year": [2014], "coef": [0.0], "se": [0.0]}
            )
            reused_common_break = pd.DataFrame(
                {"trajectory_name": ["pre_policy"], "outcome_col": ["y_cst_lag0"], "year": [2014], "coef": [2.0], "se": [0.2]}
            )
            initial_stacked_panel = pd.DataFrame(
                {
                    "pair_id": [1, 1],
                    "stack_id": [1, 1],
                    "c": [1, 2],
                    "t": [2014, 2014],
                    "rel_time": [0, 0],
                    "treated": [1, 0],
                    "trajectory_name": ["pre_policy", "pre_policy"],
                    "pair_stack_fe": ["1__1", "1__1"],
                    "unit_stack_fe": ["1__1", "2__1"],
                    "year_stack_fe": ["2014__1", "2014__1"],
                    "y_cst_lag0": [10.0, 9.0],
                }
            )
            reused_stacked_panel = initial_stacked_panel.copy()
            reused_stacked_panel["y_cst_lag0"] = [12.0, 9.5]
            initial_stacked_results = pd.DataFrame(
                {"trajectory_name": ["pre_policy"], "outcome_col": ["y_cst_lag0"], "rel_time": [0], "coef": [1.0], "se": [0.1]}
            )
            reused_stacked_results = pd.DataFrame(
                {"trajectory_name": ["pre_policy"], "outcome_col": ["y_cst_lag0"], "rel_time": [0], "coef": [3.0], "se": [0.3]}
            )

            with patch(
                "company_shift_share.matched_exposure_design._load_base_configs",
                return_value=({}, {}),
            ), patch(
                "company_shift_share.matched_exposure_design.build_source_analysis_panel_from_inputs",
                return_value=(panel, firms, selected_meta, workforce),
            ), patch(
                "company_shift_share.matched_exposure_design.build_shift_share_components_from_configs",
                return_value=(school_growth, school_shift_sample, transition_shares, instrument_panel),
            ), patch(
                "company_shift_share.matched_exposure_design.build_matching_feature_frame_from_wrds",
                return_value=matching_features,
            ), patch(
                "company_shift_share.matched_exposure_design.build_propensity_scores",
                return_value=(propensity_scores, {"model_method": "logit"}),
            ), patch(
                "company_shift_share.matched_exposure_design.run_common_break_event_study",
                return_value=initial_common_break,
            ), patch(
                "company_shift_share.matched_exposure_design.assign_persistent_entry_cohorts",
                return_value=pd.DataFrame({"c": [1], "g": [2014]}),
            ), patch(
                "company_shift_share.matched_exposure_design.build_stacked_did_panel",
                return_value=(initial_stacked_panel, {"n_stacks": 1}),
            ), patch(
                "company_shift_share.matched_exposure_design.run_stacked_did",
                return_value=initial_stacked_results,
            ):
                run_matched_exposure_design(cfg=cfg)

            cfg["matched_exposure_design"]["reuse_saved_matching_outputs"] = True
            with patch(
                "company_shift_share.matched_exposure_design._load_base_configs",
                return_value=({}, {}),
            ), patch(
                "company_shift_share.matched_exposure_design.build_source_analysis_panel_from_inputs",
                side_effect=AssertionError("source panel rebuild should be skipped"),
            ), patch(
                "company_shift_share.matched_exposure_design.build_shift_share_components_from_configs",
                side_effect=AssertionError("shift-share rebuild should be skipped"),
            ), patch(
                "company_shift_share.matched_exposure_design.build_matching_feature_frame_from_wrds",
                side_effect=AssertionError("feature rebuild should be skipped"),
            ), patch(
                "company_shift_share.matched_exposure_design.build_propensity_scores",
                side_effect=AssertionError("propensity refit should be skipped"),
            ), patch(
                "company_shift_share.matched_exposure_design.run_common_break_event_study",
                return_value=reused_common_break,
            ), patch(
                "company_shift_share.matched_exposure_design.assign_persistent_entry_cohorts",
                return_value=pd.DataFrame({"c": [1], "g": [2014]}),
            ), patch(
                "company_shift_share.matched_exposure_design.build_stacked_did_panel",
                return_value=(reused_stacked_panel, {"n_stacks": 1}),
            ), patch(
                "company_shift_share.matched_exposure_design.run_stacked_did",
                return_value=reused_stacked_results,
            ):
                reused = run_matched_exposure_design(cfg=cfg)

            self.assertTrue(bool(reused["diagnostics"]["reused_saved_matching_outputs"]))
            self.assertFalse(reused["matched_pairs"].empty)
            self.assertEqual(float(reused["common_break_results"].iloc[0]["coef"]), 2.0)
            self.assertEqual(float(reused["stacked_results"].iloc[0]["coef"]), 3.0)
            self.assertTrue((tmp / "common_break_results.parquet").exists())
            self.assertTrue((tmp / "stacked_results.parquet").exists())


if __name__ == "__main__":
    unittest.main()
