from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import duckdb as ddb
import numpy as np
import pandas as pd

import company_shift_share.revelio_company_features as revelio_company_features
import company_shift_share.source_exposure_data as source_exposure_data
from company_shift_share.exposure_event_study import (
    _ensure_derived_outcome,
    _select_index_analysis_firms,
    _resolve_raw_plot_ntiles,
    _resolve_run_log_path,
    _testing_verbose,
    fit_opt_probability_index,
    validate_opt_probability_config,
)
from company_shift_share.revelio_company_features import (
    build_company_features,
    classify_opt_intensive_schools,
    summarize_pre_period_features,
)
from company_shift_share.source_exposure_data import (
    SourceExposurePaths,
    _build_wrds_task_queue,
    _build_testing_analysis_firm_sample_from_counts,
    build_design3_position_outcomes_from_local_caches,
    build_source_analysis_panel,
    load_or_build_source_firm_universe,
    load_or_build_wrds_company_year_workforce_cache,
    build_wrds_company_year_workforce,
    build_wrds_school_flows,
)


def _interaction_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "c": list(range(1, 13)),
            "in_analysis_universe": [1] * 12,
            "preferred_rcid_source": [1] * 10 + [0, 0],
            "outside_negative_candidate": [0] * 10 + [1, 1],
            "naics2": ["54", "54", "54", "54", "54", "52", "52", "52", "52", "52", "54", "52"],
            "naics4": ["5415", "5415", "5416", "5417", "5418", "5231", "5231", "5239", "5241", "5242", "5415", "5231"],
            "company_state_feature": ["CA", "CA", "WA", "WA", "NY", "TX", "TX", "IL", "IL", "GA", "CA", "TX"],
            "company_metro_feature": [
                "san jose metropolitan area (california)",
                "san francisco metropolitan area",
                "seattle metropolitan area",
                "seattle metropolitan area",
                "new york city metropolitan area",
                "houston metropolitan area",
                "dallas metropolitan area",
                "chicago metropolitan area",
                "chicago metropolitan area",
                "atlanta metropolitan area",
                "san jose metropolitan area (california)",
                "houston metropolitan area",
            ],
            "company_hq_region": [
                "West",
                "West",
                "West",
                "West",
                "Northeast",
                "South",
                "South",
                "Midwest",
                "Midwest",
                "South",
                "West",
                "South",
            ],
            "company_age_feature": [12, 10, 14, 11, 18, 20, 23, 19, 21, 24, 16, 22],
            "company_n_users_log1p": [6.8, 6.6, 6.5, 6.4, 6.2, 5.4, 5.2, 5.1, 5.0, 4.9, 4.7, 4.6],
            "masters_opt_hire_rate_annual_pre_level": [0.82, 0.78, 0.75, 0.72, 0.68, 0.18, 0.15, 0.12, 0.10, 0.08, 0.04, 0.03],
            "school_opt_share_new_hire_masters_annual_pre_level": [0.88, 0.84, 0.82, 0.78, 0.74, 0.24, 0.20, 0.18, 0.14, 0.12, 0.07, 0.05],
            "school_opt_share_tenured_masters_annual_pre_level": [0.74, 0.72, 0.70, 0.68, 0.64, 0.22, 0.19, 0.17, 0.13, 0.11, 0.06, 0.04],
            "firm_size_annual_pre_level": [1200, 1100, 1050, 980, 900, 420, 390, 360, 340, 320, 210, 200],
            "n_new_hires_annual_pre_level": [140, 132, 125, 120, 116, 42, 38, 36, 34, 30, 12, 10],
            "nonus_educ_share_annual_pre_level": [0.44, 0.40, 0.38, 0.35, 0.33, 0.12, 0.10, 0.09, 0.08, 0.07, 0.04, 0.03],
            "race_share_api_annual_pre_level": [0.34, 0.32, 0.30, 0.28, 0.27, 0.14, 0.13, 0.12, 0.11, 0.10, 0.08, 0.07],
            "occ_share_computing_math_annual_pre_level": [0.42, 0.40, 0.38, 0.36, 0.34, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04],
            "occ_share_engineering_annual_pre_level": [0.22, 0.24, 0.21, 0.20, 0.19, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03],
            "opt_hire_rate_annual_pre_level": [0.82, 0.78, 0.75, 0.72, 0.68, 0.18, 0.15, 0.12, 0.10, 0.08, 0.04, 0.03],
            "school_opt_share_new_hire_annual_pre_level": [0.88, 0.84, 0.82, 0.78, 0.74, 0.24, 0.20, 0.18, 0.14, 0.12, 0.07, 0.05],
            "school_opt_share_tenured_annual_pre_level": [0.74, 0.72, 0.70, 0.68, 0.64, 0.22, 0.19, 0.17, 0.13, 0.11, 0.06, 0.04],
            "race_share_asian_annual_pre_level": [0.34, 0.32, 0.30, 0.28, 0.27, 0.14, 0.13, 0.12, 0.11, 0.10, 0.08, 0.07],
            "masters_opt_hire_rate_annual_pre_growth": [0.05, 0.04, 0.03, 0.03, 0.02, -0.01, -0.02, -0.02, -0.03, -0.03, -0.01, -0.01],
            "masters_opt_hire_rate_annual_pre_level_missing_ind": [0] * 12,
            "salary_mean_annual_pre_level": [220, 215, 208, 204, 198, 145, 140, 136, 132, 128, 118, 115],
        }
    )


def _interaction_target_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "c": list(range(1, 11)),
            "post2016_any_opt": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
    )


def _small_lasso_feature_frame() -> pd.DataFrame:
    return _interaction_feature_frame().iloc[:6].copy()


def _small_lasso_target_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "c": [1, 2, 3, 4],
            "post2016_any_opt": [1, 1, 0, 0],
        }
    )


class ExposureIndexTests(unittest.TestCase):
    def test_build_design3_position_outcomes_from_local_caches(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            positions_path = tmp / "positions.parquet"
            users_path = tmp / "users.parquet"
            profile_path = tmp / "profile.parquet"
            pd.DataFrame(
                [
                    {"user_id": 1, "position_id": 1, "rcid": 10, "startdate": "2020-07-01", "enddate": "2022-07-01", "onet_code": "15-1000.00"},
                    {"user_id": 2, "position_id": 2, "rcid": 10, "startdate": "2020-08-01", "enddate": "2021-08-01", "onet_code": "17-1000.00"},
                    {"user_id": 3, "position_id": 3, "rcid": 10, "startdate": "2020-09-01", "enddate": "2021-09-01", "onet_code": "13-1000.00"},
                    {"user_id": 4, "position_id": 4, "rcid": 20, "startdate": "2020-10-01", "enddate": "2021-10-01", "onet_code": "19-1000.00"},
                    {"user_id": 5, "position_id": 5, "rcid": 20, "startdate": "2020-11-01", "enddate": "2021-11-01", "onet_code": "29-1000.00"},
                    {"user_id": 6, "position_id": 6, "rcid": 10, "startdate": "2020-01-01", "enddate": "2020-03-01", "onet_code": "15-1000.00"},
                    {"user_id": 6, "position_id": 7, "rcid": 10, "startdate": "2020-02-01", "enddate": "2020-04-01", "onet_code": "15-1000.00"},
                    {"user_id": 7, "position_id": 8, "rcid": 10, "startdate": "2017-01-01", "enddate": "2017-03-01", "onet_code": "15-1000.00"},
                ]
            ).to_parquet(positions_path, index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": i,
                        "education_number": 1,
                        "ed_startdate": "2018-09-01",
                        "ed_enddate": "2020-06-01",
                        "degree": "Master" if i in {1, 2, 4, 6, 7} else "Bachelor",
                        "degree_raw": "",
                        "field_raw": "",
                        "university_raw": "Example University",
                    }
                    for i in range(1, 8)
                ]
            ).to_parquet(users_path, index=False)
            pd.DataFrame(
                {
                    "user_id": [1, 2, 3, 4, 5, 6, 7],
                    "p_likely_foreign": [1.0, 0.5, 0.0, 0.25, 1.0, 0.75, 0.75],
                    "signal_current_country_nonus": [1, 1, 0, 0, 1, 1, 1],
                }
            ).to_parquet(profile_path, index=False)

            out, meta = build_design3_position_outcomes_from_local_caches(
                selected_positions_path=positions_path,
                users_path=users_path,
                user_profile_path=profile_path,
                year_min=2020,
                year_max=2020,
                firm_ids=[10, 20],
                opt_likely_soc2=["15", "17", "19", "13", "11"],
                top_n_soc2=4,
            )

        self.assertEqual(meta["opt_likely_soc2"], ["11", "13", "15", "17", "19"])
        self.assertEqual(meta["opt_likely_soc2_source"], "fixed_config")
        keyed = out.set_index(["c", "t"])
        self.assertAlmostEqual(float(keyed.loc[(10, 2020), "y_new_hires_foreign_lag0"]), 2.0)
        self.assertAlmostEqual(float(keyed.loc[(10, 2020), "y_new_hires_native_lag0"]), 1.0)
        self.assertAlmostEqual(float(keyed.loc[(20, 2020), "y_new_hires_foreign_lag0"]), 1.0)
        self.assertAlmostEqual(float(keyed.loc[(20, 2020), "y_new_hires_native_lag0"]), 1.0)
        self.assertAlmostEqual(float(keyed.loc[(10, 2020), "y_new_hires_foreign_opt_likely_lag0"]), 2.0)
        self.assertAlmostEqual(float(keyed.loc[(20, 2020), "y_new_hires_foreign_opt_likely_lag0"]), 0.0)
        self.assertAlmostEqual(float(keyed.loc[(10, 2020), "y_new_hires_foreign_masters_lag0"]), 2.0)
        self.assertAlmostEqual(float(keyed.loc[(10, 2020), "y_new_hires_native_masters_lag0"]), 0.0)
        self.assertAlmostEqual(float(keyed.loc[(10, 2020), "y_new_hires_foreign_opt_likely_masters_lag0"]), 2.0)
        self.assertAlmostEqual(float(keyed.loc[(10, 2020), "y_intern_positions_opt_likely_lag0"]), 1.0)
        self.assertAlmostEqual(float(keyed.loc[(10, 2020), "y_intern_positions_foreign_lag0"]), 1.0)
        self.assertAlmostEqual(float(keyed.loc[(10, 2020), "y_intern_positions_opt_likely_foreign_lag0"]), 1.0)
        self.assertTrue(pd.notna(keyed.loc[(10, 2020), "avg_tenure_new_hires_lag0"]))

    def test_summarize_pre_period_features_respects_window(self) -> None:
        annual = pd.DataFrame(
            {
                "c": [1, 1, 1, 1, 2, 2, 2],
                "t": [2009, 2010, 2011, 2012, 2010, 2011, 2012],
                "metric_annual": [100.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0],
            }
        )
        out = summarize_pre_period_features(
            annual,
            ["metric_annual"],
            year_min=2010,
            year_max=2012,
        )
        firm1 = out.set_index("c").loc[1]
        self.assertAlmostEqual(firm1["metric_annual_pre_level"], 2.0)
        self.assertAlmostEqual(firm1["metric_annual_pre_growth"], 1.0)
        self.assertEqual(int(firm1["metric_annual_pre_n_years"]), 3)
        self.assertEqual(int(firm1["metric_annual_pre_growth_missing_ind"]), 0)

    def test_summarize_pre_period_features_coerces_string_numeric_values(self) -> None:
        annual = pd.DataFrame(
            {
                "c": [1, 1, 1, 2, 2, 2],
                "t": [2010, 2011, 2012, 2010, 2011, 2012],
                "metric_annual": ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"],
            }
        )
        out = summarize_pre_period_features(
            annual,
            ["metric_annual"],
            year_min=2010,
            year_max=2012,
        )
        firm1 = out.set_index("c").loc[1]
        self.assertAlmostEqual(float(firm1["metric_annual_pre_level"]), 2.0)
        self.assertAlmostEqual(float(firm1["metric_annual_pre_growth"]), 1.0)

    def test_aggregate_school_features_uses_duckdb_and_preserves_shares(self) -> None:
        school_flows = pd.DataFrame(
            {
                "c": [1, 1, 1],
                "t": [2010, 2010, 2010],
                "university_raw": ["MIT", "Harvard", "Unknown School"],
                "n_transitions": [3.0, 2.0, 1.0],
                "n_emp": [10.0, 5.0, 1.0],
                "total_new_hires": [6.0, 6.0, 6.0],
            }
        )
        school_map = pd.DataFrame(
            {
                "university_raw_key": ["mit", "harvard"],
                "unitid": ["1", "2"],
            }
        )
        school_benchmark = pd.DataFrame(
            {
                "unitid": ["1", "2"],
                "opt_intensive_bachelors": [0, 1],
                "opt_intensive_masters": [1, 0],
                "opt_intensive_phd": [0, 0],
            }
        )
        out = revelio_company_features._aggregate_school_features(
            school_flows,
            school_benchmark,
            school_map,
        )
        row = out.set_index(["c", "t"]).loc[(1, 2010)]
        self.assertAlmostEqual(float(row["total_new_hires"]), 6.0)
        self.assertAlmostEqual(float(row["total_emp"]), 16.0)
        self.assertEqual(int(row["n_schools_new_hire_annual"]), 3)
        self.assertEqual(int(row["n_schools_tenured_annual"]), 3)
        self.assertAlmostEqual(float(row["school_opt_share_new_hire_annual"]), 0.5)
        self.assertAlmostEqual(float(row["school_opt_share_tenured_annual"]), 10.0 / 16.0)
        self.assertAlmostEqual(float(row["school_opt_share_new_hire_bachelors_annual"]), 2.0 / 6.0)

    def test_classify_opt_intensive_schools_respects_window(self) -> None:
        components = pd.DataFrame(
            {
                "k": ["A", "A", "B", "B"],
                "t": [2009, 2010, 2009, 2010],
                "g_kt": [0.9, 0.1, 0.1, 0.8],
            }
        )
        out = classify_opt_intensive_schools(components, 2010, 2010).set_index("k")
        self.assertFalse(bool(out.loc["A", "opt_intensive"]))
        self.assertTrue(bool(out.loc["B", "opt_intensive"]))

    def test_rf_continuous_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_opt_probability_config(
                model_method="random_forest",
                entry_mode="continuous",
                ntiles=2,
                feature_year_min=2010,
                feature_year_max=2015,
                leaveout_enabled=False,
                leaveout_share=0.25,
            )

    def test_rf_ntiles_not_two_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_opt_probability_config(
                model_method="random_forest",
                entry_mode="ntiles",
                ntiles=4,
                feature_year_min=2010,
                feature_year_max=2015,
                leaveout_enabled=False,
                leaveout_share=0.25,
            )

    def test_lasso_continuous_is_valid(self) -> None:
        validate_opt_probability_config(
            model_method="lasso",
            entry_mode="continuous",
            ntiles=4,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=False,
            leaveout_share=0.25,
        )

    def test_resolve_raw_plot_ntiles_defaults_to_quartiles(self) -> None:
        self.assertEqual(_resolve_raw_plot_ntiles({}, 2), 4)
        self.assertEqual(_resolve_raw_plot_ntiles({}, 4), 4)
        self.assertEqual(_resolve_raw_plot_ntiles({"raw_plot_ntiles": 5}, 2), 5)

    def test_resolve_raw_plot_ntiles_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_raw_plot_ntiles({"raw_plot_ntiles": 1}, 2)

    def test_select_index_analysis_firms_can_exclude_outside_negatives(self) -> None:
        pred_df = pd.DataFrame(
            {
                "c": [1, 1, 2, 3, 4],
                "predicted_index": [0.8, 0.8, 0.6, 0.2, 0.1],
                "predicted_class": [1, 1, 1, 0, 0],
                "preferred_rcid_source": [1, 1, 1, 0, 0],
                "outside_negative_candidate": [0, 0, 0, 1, 1],
                "event_study_sample": [1, 1, 1, 1, 0],
            }
        )

        all_firms = _select_index_analysis_firms(pred_df, exclude_outside_negatives=False)
        preferred_only = _select_index_analysis_firms(pred_df, exclude_outside_negatives=True)

        self.assertListEqual(all_firms["c"].tolist(), [1, 2, 3])
        self.assertListEqual(preferred_only["c"].tolist(), [1, 2])
        self.assertTrue(preferred_only["outside_negative_candidate"].fillna(0).eq(0).all())

    def test_fit_opt_probability_index_lpm_and_leaveout(self) -> None:
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "in_analysis_universe": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "preferred_rcid_source": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                "outside_negative_candidate": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                "naics2": ["54", "54", "52", "52", "54", "52", "54", "52", "54", "52"],
                "company_state_feature": ["CA", "CA", "TX", "TX", "WA", "WA", "CA", "TX", "WA", "CA"],
                "company_hq_region": ["West", "West", "South", "South", "West", "West", "West", "South", "West", "West"],
                "company_age_feature": [10, 12, 20, 18, 8, 22, 15, 14, 16, 13],
                "company_n_users_log1p": [6.0, 6.2, 5.0, 5.1, 6.1, 5.2, 4.7, 4.8, 4.9, 4.6],
                "opt_hire_rate_annual_pre_level": [0.8, 0.7, 0.1, 0.2, 0.75, 0.15, 0.02, 0.03, 0.04, 0.01],
                "opt_hire_rate_annual_pre_growth": [0.12, 0.1, -0.05, -0.02, 0.08, -0.03, -0.01, -0.02, 0.0, -0.03],
                "school_opt_share_new_hire_annual_pre_level": [0.85, 0.8, 0.2, 0.25, 0.78, 0.22, 0.05, 0.06, 0.04, 0.03],
                "salary_mean_annual_pre_level": [200, 190, 120, 130, 210, 110, 100, 95, 105, 90],
            }
        )
        target_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4, 5, 6],
                "post2016_any_opt": [1, 1, 0, 0, 1, 0],
            }
        )
        pred_df, diagnostics = fit_opt_probability_index(
            feature_df,
            target_df,
            model_method="lpm",
            entry_mode="ntiles",
            ntiles=3,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=True,
            leaveout_share=0.5,
            leaveout_seed=7,
        )
        self.assertTrue(pred_df["predicted_index"].between(0, 1).all())
        self.assertGreater(int(pred_df["leaveout_training_firm"].sum()), 0)
        overlap = pred_df.loc[
            pred_df["leaveout_training_firm"].eq(1) & pred_df["event_study_sample"].eq(1),
            "c",
        ]
        self.assertTrue(overlap.empty)
        self.assertEqual(
            int(pred_df.loc[pred_df["preferred_rcid_source"].eq(1), "train_sample"].sum()),
            int(pred_df.loc[pred_df["outside_negative_candidate"].eq(1), "train_sample"].sum()),
        )
        self.assertGreater(
            int(pred_df.loc[pred_df["outside_negative_candidate"].eq(1), "event_study_sample"].sum()),
            0,
        )
        self.assertGreater(
            int(pred_df.loc[pred_df["preferred_rcid_source"].eq(1), "event_study_sample"].sum()),
            0,
        )
        self.assertEqual(
            int(diagnostics["n_train_preferred_source"]),
            int(diagnostics["n_train_outside_negative"]),
        )
        self.assertEqual(diagnostics["model_method"], "lpm")

    def test_fit_opt_probability_index_logit_outputs_probabilities(self) -> None:
        pred_df, diagnostics = fit_opt_probability_index(
            _interaction_feature_frame(),
            _interaction_target_frame(),
            model_method="logit",
            entry_mode="continuous",
            ntiles=4,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=False,
            leaveout_share=0.25,
            leaveout_seed=13,
        )
        self.assertTrue(pred_df["predicted_prob"].between(0, 1).all())
        self.assertSetEqual(set(pred_df["predicted_class"].unique()), {0, 1})
        self.assertEqual(diagnostics["model_method"], "logit")
        self.assertEqual(diagnostics["logit_class_weight"], "balanced")
        self.assertGreater(int(diagnostics["n_standardized_features"]), 0)

    def test_fit_opt_probability_index_random_forest_binary_output(self) -> None:
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4, 5, 6, 7, 8],
                "in_analysis_universe": [1, 1, 1, 1, 1, 1, 1, 1],
                "preferred_rcid_source": [1, 1, 1, 1, 1, 1, 0, 0],
                "outside_negative_candidate": [0, 0, 0, 0, 0, 0, 1, 1],
                "naics2": ["54", "54", "52", "52", "54", "52", "54", "52"],
                "company_state_feature": ["CA", "CA", "TX", "TX", "WA", "WA", "CA", "TX"],
                "company_hq_region": ["West", "West", "South", "South", "West", "West", "West", "South"],
                "company_age_feature": [10, 12, 20, 18, 8, 22, 15, 14],
                "company_n_users_log1p": [6.0, 6.2, 5.0, 5.1, 6.1, 5.2, 4.7, 4.8],
                "opt_hire_rate_annual_pre_level": [0.8, 0.7, 0.1, 0.2, 0.75, 0.15, 0.02, 0.03],
                "opt_hire_rate_annual_pre_growth": [0.12, 0.1, -0.05, -0.02, 0.08, -0.03, -0.01, -0.02],
                "school_opt_share_new_hire_annual_pre_level": [0.85, 0.8, 0.2, 0.25, 0.78, 0.22, 0.05, 0.06],
                "salary_mean_annual_pre_level": [200, 190, 120, 130, 210, 110, 100, 95],
            }
        )
        target_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4, 5, 6],
                "post2016_any_opt": [1, 1, 0, 0, 1, 0],
            }
        )
        pred_df, diagnostics = fit_opt_probability_index(
            feature_df,
            target_df,
            model_method="random_forest",
            entry_mode="ntiles",
            ntiles=2,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=False,
            leaveout_share=0.25,
            leaveout_seed=11,
            rf_n_estimators=50,
            rf_min_samples_leaf=1,
            rf_min_samples_split=2,
        )
        self.assertTrue(pred_df["predicted_prob"].between(0, 1).all())
        self.assertSetEqual(set(pred_df["predicted_class"].unique()), {0, 1})
        self.assertEqual(diagnostics["model_method"], "random_forest")

    def test_fit_opt_probability_index_lasso_outputs_probabilities(self) -> None:
        pred_df, diagnostics, artifacts = fit_opt_probability_index(
            _interaction_feature_frame(),
            _interaction_target_frame(),
            model_method="lasso",
            entry_mode="continuous",
            ntiles=4,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=False,
            leaveout_share=0.25,
            leaveout_seed=13,
            return_artifacts=True,
        )
        self.assertTrue(pred_df["predicted_prob"].between(0, 1).all())
        self.assertSetEqual(set(pred_df["predicted_class"].unique()), {0, 1})
        self.assertEqual(diagnostics["model_method"], "lasso")
        self.assertGreater(int(diagnostics["n_interaction_columns_added"]), 0)
        self.assertGreater(int(diagnostics["n_standardized_features"]), 0)
        self.assertIn("top_coefficients", diagnostics)
        self.assertTrue(all(str(col).startswith("ix__") for col in artifacts["interaction_column_names"]))

    def test_lasso_supports_ntiles_entry_mode(self) -> None:
        pred_df, diagnostics = fit_opt_probability_index(
            _interaction_feature_frame(),
            _interaction_target_frame(),
            model_method="lasso",
            entry_mode="ntiles",
            ntiles=3,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=False,
            leaveout_share=0.25,
            leaveout_seed=17,
        )
        self.assertTrue(pred_df["predicted_index"].between(0, 1).all())
        self.assertEqual(diagnostics["model_method"], "lasso")

    def test_interactions_added_for_rf_but_not_lpm(self) -> None:
        feature_df = _interaction_feature_frame()
        target_df = _interaction_target_frame()

        _, lpm_diagnostics, lpm_artifacts = fit_opt_probability_index(
            feature_df,
            target_df,
            model_method="lpm",
            entry_mode="ntiles",
            ntiles=3,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=False,
            leaveout_share=0.25,
            leaveout_seed=7,
            return_artifacts=True,
        )
        self.assertEqual(int(lpm_diagnostics["n_interaction_columns_added"]), 0)
        self.assertFalse(any(str(col).startswith("ix__") for col in lpm_artifacts["active_feature_columns"]))

        _, rf_diagnostics, rf_artifacts = fit_opt_probability_index(
            feature_df,
            target_df,
            model_method="random_forest",
            entry_mode="ntiles",
            ntiles=2,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=False,
            leaveout_share=0.25,
            leaveout_seed=7,
            rf_n_estimators=50,
            rf_min_samples_leaf=1,
            rf_min_samples_split=2,
            return_artifacts=True,
        )
        self.assertGreater(int(rf_diagnostics["n_interaction_columns_added"]), 0)
        self.assertTrue(any(str(col).startswith("ix__") for col in rf_artifacts["active_feature_columns"]))
        self.assertTrue(all("naics4" not in str(col) for col in rf_artifacts["interaction_column_names"]))
        self.assertTrue(all("company_metro_feature" not in str(col) for col in rf_artifacts["interaction_column_names"]))

    def test_missing_interaction_sources_are_reported_without_failing(self) -> None:
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4, 5, 6],
                "in_analysis_universe": [1] * 6,
                "preferred_rcid_source": [1, 1, 1, 1, 0, 0],
                "outside_negative_candidate": [0, 0, 0, 0, 1, 1],
                "naics2": ["54", "54", "52", "52", "54", "52"],
                "company_hq_region": ["West", "West", "South", "South", "West", "South"],
                "company_age_feature": [10, 11, 20, 21, 15, 16],
                "company_n_users_log1p": [6.0, 6.1, 5.0, 5.1, 4.5, 4.6],
                "salary_mean_annual_pre_level": [200, 190, 120, 130, 100, 95],
            }
        )
        target_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "post2016_any_opt": [1, 1, 0, 0],
            }
        )
        _, diagnostics, artifacts = fit_opt_probability_index(
            feature_df,
            target_df,
            model_method="random_forest",
            entry_mode="ntiles",
            ntiles=2,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=False,
            leaveout_share=0.25,
            leaveout_seed=11,
            rf_n_estimators=25,
            rf_min_samples_leaf=1,
            rf_min_samples_split=2,
            return_artifacts=True,
        )
        self.assertIn("masters_opt_hire_rate_annual_pre_level", diagnostics["skipped_interaction_source_columns"])
        self.assertEqual(list(artifacts["interaction_column_names"]), [])

    def test_lasso_cv_fallback_uses_two_folds(self) -> None:
        _, diagnostics = fit_opt_probability_index(
            _small_lasso_feature_frame(),
            _small_lasso_target_frame(),
            model_method="lasso",
            entry_mode="continuous",
            ntiles=2,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=False,
            leaveout_share=0.25,
            leaveout_seed=19,
        )
        self.assertEqual(int(diagnostics["lasso_cv_folds"]), 2)

    def test_lasso_respects_requested_cv_folds_and_c_grid(self) -> None:
        captured: dict[str, object] = {}

        class DummyLogisticRegressionCV:
            def __init__(self, *args, **kwargs) -> None:
                captured["cv"] = kwargs["cv"]
                captured["Cs"] = np.asarray(kwargs["Cs"], dtype=float)

            def fit(self, x_train, y_train):
                self.classes_ = np.array([0, 1])
                self.coef_ = np.zeros((1, x_train.shape[1]), dtype=float)
                self.intercept_ = np.array([0.0], dtype=float)
                self.C_ = np.array([float(np.asarray(captured["Cs"])[0])], dtype=float)
                return self

            def predict_proba(self, x_all):
                positive = np.full(x_all.shape[0], 0.6, dtype=float)
                negative = 1.0 - positive
                return np.column_stack([negative, positive])

        with patch("company_shift_share.exposure_event_study.LogisticRegressionCV", DummyLogisticRegressionCV):
            pred_df, diagnostics = fit_opt_probability_index(
                _interaction_feature_frame(),
                _interaction_target_frame(),
                model_method="lasso",
                entry_mode="continuous",
                ntiles=4,
                feature_year_min=2010,
                feature_year_max=2015,
                leaveout_enabled=False,
                leaveout_share=0.25,
                leaveout_seed=13,
                lasso_cv_folds=3,
                lasso_n_cs=7,
            )

        self.assertTrue(pred_df["predicted_prob"].between(0, 1).all())
        self.assertEqual(int(captured["cv"]), 3)
        self.assertEqual(int(diagnostics["lasso_cv_folds"]), 3)
        self.assertEqual(len(captured["Cs"]), 7)
        self.assertEqual(int(diagnostics["lasso_n_cs"]), 7)

    def test_logit_defaults_to_balanced_class_weight(self) -> None:
        captured: dict[str, object] = {}

        class DummyLogisticRegression:
            def __init__(self, *args, **kwargs) -> None:
                captured["class_weight"] = kwargs.get("class_weight")

            def fit(self, x_train, y_train):
                self.classes_ = np.array([0, 1])
                self.coef_ = np.zeros((1, x_train.shape[1]), dtype=float)
                self.intercept_ = np.array([0.0], dtype=float)
                return self

            def predict_proba(self, x_all):
                positive = np.full(x_all.shape[0], 0.6, dtype=float)
                negative = 1.0 - positive
                return np.column_stack([negative, positive])

        with patch("company_shift_share.exposure_event_study.LogisticRegression", DummyLogisticRegression):
            pred_df, diagnostics = fit_opt_probability_index(
                _interaction_feature_frame(),
                _interaction_target_frame(),
                model_method="logit",
                entry_mode="continuous",
                ntiles=4,
                feature_year_min=2010,
                feature_year_max=2015,
                leaveout_enabled=False,
                leaveout_share=0.25,
                leaveout_seed=13,
            )

        self.assertTrue(pred_df["predicted_prob"].between(0, 1).all())
        self.assertEqual(captured["class_weight"], "balanced")
        self.assertEqual(diagnostics["logit_class_weight"], "balanced")

    def test_outside_negative_targets_forced_to_zero(self) -> None:
        feature_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "in_analysis_universe": [1, 1, 1, 1],
                "preferred_rcid_source": [1, 1, 0, 0],
                "outside_negative_candidate": [0, 0, 1, 1],
                "naics2": ["54", "52", "54", "52"],
                "company_state_feature": ["CA", "TX", "CA", "TX"],
                "company_hq_region": ["West", "South", "West", "South"],
                "company_age_feature": [10, 20, 15, 14],
                "company_n_users_log1p": [6.0, 5.0, 4.7, 4.8],
                "opt_hire_rate_annual_pre_level": [0.8, 0.1, 0.02, 0.03],
                "salary_mean_annual_pre_level": [200, 120, 100, 95],
            }
        )
        target_df = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "post2016_any_opt": [1, 0, 1, 1],
            }
        )
        pred_df, _ = fit_opt_probability_index(
            feature_df,
            target_df,
            model_method="lpm",
            entry_mode="ntiles",
            ntiles=2,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=False,
            leaveout_share=0.25,
            leaveout_seed=7,
        )
        outside = pred_df[pred_df["outside_negative_candidate"].eq(1)]
        self.assertTrue((outside["post2016_any_opt"] == 0).all())

    def test_fit_opt_probability_index_can_downsample_active_features(self) -> None:
        feature_df = pd.DataFrame(
            {
                "c": list(range(1, 11)),
                "in_analysis_universe": [1] * 8 + [0, 0],
                "preferred_rcid_source": [1] * 8 + [0, 0],
                "outside_negative_candidate": [0] * 8 + [1, 1],
                "company_state_feature": ["CA", "CA", "TX", "TX", "WA", "WA", "NY", "NY", "CA", "TX"],
                "company_hq_region": ["West", "West", "South", "South", "West", "West", "Northeast", "Northeast", "West", "South"],
                "company_age_feature": [5, 6, 20, 18, 9, 21, 12, 11, 7, 8],
                "company_n_users_log1p": [5.0, 5.1, 4.4, 4.5, 5.2, 4.6, 4.9, 4.8, 4.0, 4.1],
                "feat1": [0.9, 0.8, 0.1, 0.2, 0.85, 0.15, 0.7, 0.3, 0.01, 0.02],
                "feat2": [10, 9, 2, 3, 11, 4, 8, 5, 1, 1],
                "feat3": [100, 95, 60, 62, 102, 58, 97, 65, 55, 56],
                "feat4": [1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                "feat5": [3.2, 3.0, 1.1, 1.3, 3.1, 1.2, 2.9, 1.4, 0.8, 0.7],
                "feat6": [7, 7, 2, 2, 8, 2, 6, 3, 1, 1],
            }
        )
        target_df = pd.DataFrame(
            {
                "c": list(range(1, 9)),
                "post2016_any_opt": [1, 1, 0, 0, 1, 0, 1, 0],
            }
        )

        pred_df, diagnostics, artifacts = fit_opt_probability_index(
            feature_df,
            target_df,
            model_method="lpm",
            entry_mode="ntiles",
            ntiles=3,
            feature_year_min=2010,
            feature_year_max=2015,
            leaveout_enabled=False,
            leaveout_share=0.25,
            leaveout_seed=7,
            max_feature_to_train_ratio=0.25,
            feature_sample_seed=99,
            return_artifacts=True,
        )

        self.assertTrue(bool(diagnostics["feature_downsampled"]))
        self.assertEqual(int(diagnostics["n_active_features_after_sampling"]), 2)
        self.assertEqual(len(artifacts["active_feature_columns"]), 2)
        self.assertEqual(len(pred_df), len(feature_df))

    def test_testing_sample_prefers_requested_target_mix_when_available(self) -> None:
        counts = pd.DataFrame(
            {
                "c": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
                "t": [2010, 2010, 2010, 2010, 2010, 2010, 2017, 2017, 2017, 2017, 2017, 2017],
                "any_opt_hires_correction_aware": [2, 1, 0, 0, 0, 3, 1, 1, 0, 0, 0, 2],
            }
        )
        preferred = pd.DataFrame({"c": [1, 2, 3, 4, 5, 6]})

        selected, meta = _build_testing_analysis_firm_sample_from_counts(
            counts,
            preferred,
            sample_n=4,
            sample_year_min=2010,
            sample_year_max=2015,
            target_year_min=2016,
            target_year_max=2022,
            random_seed=42,
            min_positive=2,
            min_nonpositive=2,
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(meta["selected_post2016_positive_firms"], 2)
        self.assertEqual(meta["selected_post2016_nonpositive_firms"], 2)

    def test_load_or_build_source_firm_universe_reuses_in_process_cache(self) -> None:
        cfg = {
            "testing": {"enabled": True, "output_suffix": "_testing"},
            "revelio_company_features": {
                "outside_negative_ratio": 2.0,
                "outside_negative_seed": 42,
                "outside_negative_min_n_users": 10,
            },
        }
        preferred = pd.DataFrame({"c": [1, 2]})
        outside = pd.DataFrame({"c": [3, 4], "outside_negative_candidate": [1, 1]})
        company_meta = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "n_users": [100, 100, 80, 90],
                "company_state_feature": ["CA", "CA", "CA", "CA"],
                "naics2": ["54", "54", "54", "54"],
                "size_bucket": ["50_249", "50_249", "50_249", "50_249"],
            }
        )

        with TemporaryDirectory() as tmpdir, patch.dict(
            source_exposure_data._SOURCE_FIRM_UNIVERSE_CACHE,
            {},
            clear=True,
        ):
            tmp = Path(tmpdir)
            paths = SourceExposurePaths(
                foia_corrected=tmp / "foia.parquet",
                employer_crosswalk=tmp / "crosswalk.parquet",
                preferred_rcids=tmp / "preferred.parquet",
                company_mapping=tmp / "company_mapping.parquet",
                f1_inst_unitid_crosswalk=tmp / "inst.parquet",
                revelio_inst_crosswalk=tmp / "revelio_inst.parquet",
                revelio_inst_deterministic_map=None,
                revelio_ref_inst_catalog=None,
                source_opt_counts_out=tmp / "source_opt_counts.parquet",
                school_opt_benchmark_out=tmp / "school_opt_benchmark.parquet",
                opt_exposure_analysis_panel_out=tmp / "analysis_panel.parquet",
                outside_negative_sample_out=tmp / "outside_negative_sample_testing.parquet",
                wrds_company_year_workforce_out=tmp / "wrds_company_year_workforce.parquet",
                wrds_school_flows_out=tmp / "wrds_school_flows.parquet",
            )

            def _fake_company_meta_subset(
                path: Path,
                *,
                selected_firms: pd.DataFrame | None = None,
                min_n_users: int | None = None,
                eligible_only: bool = False,
            ) -> pd.DataFrame:
                if selected_firms is not None:
                    out = selected_firms[["c"]].drop_duplicates().copy()
                    out["selected"] = 1
                    return out
                return company_meta.copy()

            with patch.object(source_exposure_data, "resolve_source_exposure_paths", return_value=paths), patch.object(
                source_exposure_data,
                "select_testing_analysis_firms",
                return_value=(preferred.copy(), {"testing_enabled": True, "selected_sample_n": 2}),
            ), patch.object(
                source_exposure_data,
                "_load_company_meta_subset",
                side_effect=_fake_company_meta_subset,
            ), patch.object(
                source_exposure_data,
                "sample_outside_negative_firms",
                return_value=outside.copy(),
            ) as sample_mock:
                first = load_or_build_source_firm_universe(cfg=cfg, force_rebuild=False)
                second = load_or_build_source_firm_universe(cfg=cfg, force_rebuild=False)

        self.assertEqual(sample_mock.call_count, 1)
        self.assertTrue(first[0].equals(second[0]))
        self.assertTrue(first[1].equals(second[1]))
        self.assertTrue(first[2].equals(second[2]))
        self.assertEqual(first[3], second[3])

    def test_load_or_build_source_firm_universe_reuses_in_process_cache_under_force_rebuild(self) -> None:
        cfg = {
            "testing": {"enabled": True, "output_suffix": "_testing"},
            "revelio_company_features": {
                "outside_negative_ratio": 2.0,
                "outside_negative_seed": 42,
                "outside_negative_min_n_users": 10,
            },
        }
        preferred = pd.DataFrame({"c": [1, 2]})
        outside = pd.DataFrame({"c": [3, 4], "outside_negative_candidate": [1, 1]})
        company_meta = pd.DataFrame(
            {
                "c": [1, 2, 3, 4],
                "n_users": [100, 100, 80, 90],
                "company_state_feature": ["CA", "CA", "CA", "CA"],
                "naics2": ["54", "54", "54", "54"],
                "size_bucket": ["50_249", "50_249", "50_249", "50_249"],
            }
        )

        with TemporaryDirectory() as tmpdir, patch.dict(
            source_exposure_data._SOURCE_FIRM_UNIVERSE_CACHE,
            {},
            clear=True,
        ):
            tmp = Path(tmpdir)
            paths = SourceExposurePaths(
                foia_corrected=tmp / "foia.parquet",
                employer_crosswalk=tmp / "crosswalk.parquet",
                preferred_rcids=tmp / "preferred.parquet",
                company_mapping=tmp / "company_mapping.parquet",
                f1_inst_unitid_crosswalk=tmp / "inst.parquet",
                revelio_inst_crosswalk=tmp / "revelio_inst.parquet",
                revelio_inst_deterministic_map=None,
                revelio_ref_inst_catalog=None,
                source_opt_counts_out=tmp / "source_opt_counts.parquet",
                school_opt_benchmark_out=tmp / "school_opt_benchmark.parquet",
                opt_exposure_analysis_panel_out=tmp / "analysis_panel.parquet",
                outside_negative_sample_out=tmp / "outside_negative_sample_testing.parquet",
                wrds_company_year_workforce_out=tmp / "wrds_company_year_workforce.parquet",
                wrds_school_flows_out=tmp / "wrds_school_flows.parquet",
            )

            def _fake_company_meta_subset(
                path: Path,
                *,
                selected_firms: pd.DataFrame | None = None,
                min_n_users: int | None = None,
                eligible_only: bool = False,
            ) -> pd.DataFrame:
                if selected_firms is not None:
                    out = selected_firms[["c"]].drop_duplicates().copy()
                    out["selected"] = 1
                    return out
                return company_meta.copy()

            with patch.object(source_exposure_data, "resolve_source_exposure_paths", return_value=paths), patch.object(
                source_exposure_data,
                "select_testing_analysis_firms",
                return_value=(preferred.copy(), {"testing_enabled": True, "selected_sample_n": 2}),
            ), patch.object(
                source_exposure_data,
                "_load_company_meta_subset",
                side_effect=_fake_company_meta_subset,
            ), patch.object(
                source_exposure_data,
                "sample_outside_negative_firms",
                return_value=outside.copy(),
            ) as sample_mock:
                first = load_or_build_source_firm_universe(cfg=cfg, force_rebuild=True)
                second = load_or_build_source_firm_universe(cfg=cfg, force_rebuild=True)

        self.assertEqual(sample_mock.call_count, 1)
        self.assertTrue(first[0].equals(second[0]))
        self.assertTrue(first[1].equals(second[1]))
        self.assertTrue(first[2].equals(second[2]))
        self.assertEqual(first[3], second[3])

    def test_load_or_build_source_firm_universe_metadata_tracks_preferred_hash(self) -> None:
        cfg = {
            "testing": {"enabled": True, "output_suffix": "_testing"},
            "revelio_company_features": {
                "outside_negative_ratio": 2.0,
                "outside_negative_seed": 42,
                "outside_negative_min_n_users": 10,
            },
        }
        preferred = pd.DataFrame({"c": [1, 2]})
        outside = pd.DataFrame({"c": [3, 4], "outside_negative_candidate": [1, 1]})
        company_meta = pd.DataFrame({"c": [1, 2, 3, 4]})

        with TemporaryDirectory() as tmpdir, patch.dict(
            source_exposure_data._SOURCE_FIRM_UNIVERSE_CACHE,
            {},
            clear=True,
        ):
            tmp = Path(tmpdir)
            paths = SourceExposurePaths(
                foia_corrected=tmp / "foia.parquet",
                employer_crosswalk=tmp / "crosswalk.parquet",
                preferred_rcids=tmp / "preferred.parquet",
                company_mapping=tmp / "company_mapping.parquet",
                f1_inst_unitid_crosswalk=tmp / "inst.parquet",
                revelio_inst_crosswalk=tmp / "revelio_inst.parquet",
                revelio_inst_deterministic_map=None,
                revelio_ref_inst_catalog=None,
                source_opt_counts_out=tmp / "source_opt_counts.parquet",
                school_opt_benchmark_out=tmp / "school_opt_benchmark.parquet",
                opt_exposure_analysis_panel_out=tmp / "analysis_panel.parquet",
                outside_negative_sample_out=tmp / "outside_negative_sample_testing.parquet",
                wrds_company_year_workforce_out=tmp / "wrds_company_year_workforce.parquet",
                wrds_school_flows_out=tmp / "wrds_school_flows.parquet",
            )

            with patch.object(source_exposure_data, "resolve_source_exposure_paths", return_value=paths), patch.object(
                source_exposure_data,
                "select_testing_analysis_firms",
                return_value=(preferred.copy(), {"testing_enabled": False, "selected_sample_n": 2}),
            ), patch.object(
                source_exposure_data,
                "_load_company_meta_subset",
                return_value=company_meta.copy(),
            ), patch.object(
                source_exposure_data,
                "sample_outside_negative_firms",
                return_value=outside.copy(),
            ):
                _, _, _, meta = load_or_build_source_firm_universe(cfg=cfg, force_rebuild=False)

        self.assertEqual(meta["preferred_firms_hash"], source_exposure_data._selected_firms_hash(preferred[["c"]]))

    def test_wrds_workforce_cache_reuses_in_process_wide_build_under_force_rebuild(self) -> None:
        cfg = {
            "testing": {"enabled": True, "output_suffix": "_testing"},
            "revelio_company_features": {
                "wrds_username": "tester",
                "wrds_rcid_batch_size": 10,
                "query_timeout_minutes": 10,
                "query_max_retries": 1,
            },
        }
        firms = pd.DataFrame(
            {
                "c": [1, 2],
                "in_analysis_universe": [1, 1],
                "preferred_rcid_source": [1, 1],
                "outside_negative_candidate": [0, 0],
            }
        )
        universe_meta = {
            "analysis_universe_method": "preferred_plus_outside_sample_v2",
            "outside_negative_ratio": 2.0,
            "outside_negative_seed": 42,
            "outside_negative_min_n_users": 10,
        }
        built_workforce = pd.DataFrame(
            {
                "c": [1, 1, 2, 2],
                "t": [2010, 2022, 2010, 2022],
                "total_headcount_wrds_annual": [10.0, 11.0, 20.0, 21.0],
            }
        )

        with TemporaryDirectory() as tmpdir, patch.dict(
            source_exposure_data._WRDS_WORKFORCE_CACHE,
            {},
            clear=True,
        ):
            tmp = Path(tmpdir)
            paths = SourceExposurePaths(
                foia_corrected=tmp / "foia.parquet",
                employer_crosswalk=tmp / "crosswalk.parquet",
                preferred_rcids=tmp / "preferred.parquet",
                company_mapping=tmp / "company_mapping.parquet",
                f1_inst_unitid_crosswalk=tmp / "inst.parquet",
                revelio_inst_crosswalk=tmp / "revelio_inst.parquet",
                revelio_inst_deterministic_map=None,
                revelio_ref_inst_catalog=None,
                source_opt_counts_out=tmp / "source_opt_counts.parquet",
                school_opt_benchmark_out=tmp / "school_opt_benchmark.parquet",
                opt_exposure_analysis_panel_out=tmp / "analysis_panel.parquet",
                outside_negative_sample_out=tmp / "outside_negative_sample_testing.parquet",
                wrds_company_year_workforce_out=tmp / "wrds_company_year_workforce_testing.parquet",
                wrds_school_flows_out=tmp / "wrds_school_flows.parquet",
            )

            with patch.object(source_exposure_data, "resolve_source_exposure_paths", return_value=paths), patch.object(
                source_exposure_data,
                "load_or_build_source_firm_universe",
                return_value=(firms.copy(), pd.DataFrame(), pd.DataFrame(), dict(universe_meta)),
            ), patch.object(
                source_exposure_data,
                "build_wrds_company_year_workforce",
                return_value=built_workforce.copy(),
            ) as build_mock:
                first_df, first_meta = load_or_build_wrds_company_year_workforce_cache(
                    cfg=cfg,
                    year_min=2010,
                    year_max=2022,
                    force_rebuild=True,
                )
                second_df, second_meta = load_or_build_wrds_company_year_workforce_cache(
                    cfg=cfg,
                    year_min=2010,
                    year_max=2015,
                    force_rebuild=True,
                )

        self.assertEqual(build_mock.call_count, 1)
        self.assertEqual(int(first_meta["year_max"]), 2022)
        self.assertEqual(int(second_meta["year_max"]), 2022)
        self.assertTrue((second_df["t"] >= 2010).all())
        self.assertTrue((second_df["t"] <= 2015).all())
        self.assertEqual(len(first_df), 4)
        self.assertEqual(len(second_df), 2)

    def test_new_hire_origin_sql_logic_handles_weighted_and_hard_cases(self) -> None:
        signals = pd.DataFrame(
            {
                "label": [
                    "us_only",
                    "nonus_current_only",
                    "nonus_educ_only",
                    "nonus_position_only",
                    "tie_current_wins",
                    "tie_current_loses",
                ],
                "signal_current_country_nonus": [0.0, 1.0, None, None, 1.0, 0.0],
                "signal_nonus_educ": [0.0, None, 1.0, None, 0.0, 1.0],
                "signal_nonus_position": [0.0, None, None, 1.0, None, None],
            }
        )
        prob_expr = source_exposure_data._sql_new_hire_origin_probability_expr(
            [
                "signal_current_country_nonus",
                "signal_nonus_educ",
                "signal_nonus_position",
            ]
        )
        hard_expr = source_exposure_data._sql_new_hire_origin_hard_expr(
            "p_likely_foreign",
            "signal_current_country_nonus",
        )

        con = ddb.connect()
        try:
            con.register("signals", signals)
            out = con.sql(
                f"""
                WITH scored AS (
                    SELECT
                        label,
                        signal_current_country_nonus,
                        signal_nonus_educ,
                        signal_nonus_position,
                        {prob_expr} AS p_likely_foreign
                    FROM signals
                )
                SELECT
                    label,
                    p_likely_foreign,
                    (1.0 - p_likely_foreign) AS p_likely_native,
                    {hard_expr} AS likely_foreign_hard
                FROM scored
                ORDER BY label
                """
            ).df()
        finally:
            con.close()

        result = out.set_index("label")
        self.assertAlmostEqual(float(result.loc["us_only", "p_likely_foreign"]), 0.0)
        self.assertEqual(int(result.loc["us_only", "likely_foreign_hard"]), 0)
        self.assertAlmostEqual(float(result.loc["nonus_current_only", "p_likely_foreign"]), 1.0)
        self.assertEqual(int(result.loc["nonus_current_only", "likely_foreign_hard"]), 1)
        self.assertAlmostEqual(float(result.loc["nonus_educ_only", "p_likely_foreign"]), 1.0)
        self.assertEqual(int(result.loc["nonus_educ_only", "likely_foreign_hard"]), 1)
        self.assertAlmostEqual(float(result.loc["nonus_position_only", "p_likely_foreign"]), 1.0)
        self.assertEqual(int(result.loc["nonus_position_only", "likely_foreign_hard"]), 1)
        self.assertAlmostEqual(float(result.loc["tie_current_wins", "p_likely_foreign"]), 0.5)
        self.assertEqual(int(result.loc["tie_current_wins", "likely_foreign_hard"]), 1)
        self.assertAlmostEqual(float(result.loc["tie_current_loses", "p_likely_foreign"]), 0.5)
        self.assertEqual(int(result.loc["tie_current_loses", "likely_foreign_hard"]), 0)

    def test_local_user_profile_cache_builds_expected_signals(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            users_path = tmp / "wrds_users.parquet"
            positions_path = tmp / "wrds_positions.parquet"
            out_path = tmp / "wrds_user_profile_origin_cache.parquet"

            pd.DataFrame(
                {
                    "user_id": [1, 2, 3, 4, 6],
                    "updated_dt": ["2026-01-01"] * 5,
                    "user_country": ["United States", "India", "United States", "United States", None],
                    "education_number": [1, None, 1, 1, 1],
                    "ed_startdate": ["2000-09-01", None, "2005-09-01", "2004-09-01", "2007-09-01"],
                    "ed_enddate": ["2004-06-01", None, "2009-06-01", "2008-06-01", "2011-06-01"],
                    "degree": ["Bachelor", None, "Bachelor", "Bachelor", "Bachelor"],
                    "degree_raw": [
                        "Bachelor of Science",
                        None,
                        "Bachelor of Arts",
                        "Bachelor of Science",
                        "Bachelor of Arts",
                    ],
                    "field_raw": ["cs", None, "econ", "ee", "history"],
                    "university_raw": [
                        "Stanford University",
                        None,
                        "University of Toronto",
                        "UCLA",
                        "University of Toronto",
                    ],
                    "university_country": [
                        "United States",
                        None,
                        "Canada",
                        "United States",
                        "Canada",
                    ],
                }
            ).to_parquet(users_path, index=False)
            pd.DataFrame(
                {
                    "user_id": [1, 2, 3, 4, 4, 6],
                    "rcid": [10, 10, 10, 10, 99, 10],
                    "country": [
                        "United States",
                        "United States",
                        "United States",
                        "United States",
                        "Canada",
                        "United States",
                    ],
                    "startdate": [
                        "2010-01-01",
                        "2010-01-01",
                        "2010-01-01",
                        "2010-01-01",
                        "2009-01-01",
                        "2010-01-01",
                    ],
                    "enddate": [
                        "2011-12-31",
                        "2011-12-31",
                        "2011-12-31",
                        "2011-12-31",
                        "2009-12-31",
                        "2011-12-31",
                    ],
                }
            ).to_parquet(positions_path, index=False)

            profile_cache_path, meta = source_exposure_data.load_or_build_local_wrds_user_profile_cache(
                selected_firms=pd.DataFrame({"c": [10]}),
                users_path=users_path,
                positions_path=positions_path,
                out_path=out_path,
                year_min=2010,
                year_max=2011,
                force_rebuild=True,
            )
            out = pd.read_parquet(profile_cache_path).set_index("user_id")

        self.assertEqual(
            meta["local_user_profile_cache_method"],
            source_exposure_data.LOCAL_USER_PROFILE_CACHE_METHOD,
        )
        self.assertAlmostEqual(float(out.loc[1, "p_likely_foreign"]), 0.0)
        self.assertEqual(int(out.loc[1, "likely_foreign_hard"]), 0)
        self.assertAlmostEqual(float(out.loc[2, "p_likely_foreign"]), 0.5)
        self.assertEqual(int(out.loc[2, "likely_foreign_hard"]), 1)
        self.assertEqual(int(out.loc[3, "has_nonus_educ"]), 1)
        self.assertAlmostEqual(float(out.loc[3, "p_likely_foreign"]), 1.0 / 3.0)
        self.assertEqual(int(out.loc[4, "signal_nonus_position"]), 1)
        self.assertAlmostEqual(float(out.loc[4, "p_likely_foreign"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(out.loc[6, "p_likely_foreign"]), 0.5)
        self.assertEqual(int(out.loc[6, "likely_foreign_hard"]), 0)

    def test_build_local_company_year_profile_metrics_returns_origin_and_education_summaries(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            positions_path = tmp / "wrds_positions.parquet"
            user_profile_cache_path = tmp / "wrds_user_profile_origin_cache.parquet"

            pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "has_nonus_educ": [0.0, 1.0],
                    "est_yob": [1985, 1990],
                    "p_likely_foreign": [0.0, 0.5],
                    "p_likely_native": [1.0, 0.5],
                    "likely_foreign_hard": [0, 1],
                }
            ).to_parquet(user_profile_cache_path, index=False)
            pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "rcid": [10, 10],
                    "country": ["United States", "United States"],
                    "startdate": ["2010-01-01", "2010-01-01"],
                    "enddate": ["2011-12-31", "2010-12-31"],
                }
            ).to_parquet(positions_path, index=False)

            out = source_exposure_data.build_local_company_year_profile_metrics(
                selected_firms=pd.DataFrame({"c": [10]}),
                positions_path=positions_path,
                user_profile_cache_path=user_profile_cache_path,
                year_min=2010,
                year_max=2011,
                include_education_features=True,
            )

        out = out.set_index(["c", "t"]).sort_index()
        self.assertAlmostEqual(float(out.loc[(10, 2010), "local_total_headcount_wrds_annual"]), 2.0)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "nonus_educ_share_annual"]), 0.5)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "age_share_lt30_annual"]), 1.0)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "total_headcount_foreign_weighted_annual"]), 0.5)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "total_headcount_native_weighted_annual"]), 1.5)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "total_headcount_foreign_hard_annual"]), 1.0)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "local_n_new_hires_wrds_annual"]), 2.0)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "n_new_hires_foreign_weighted_annual"]), 0.5)
        self.assertAlmostEqual(float(out.loc[(10, 2011), "local_n_new_hires_wrds_annual"]), 0.0)

    def test_build_local_wrds_company_year_workforce_returns_full_metrics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            users_path = tmp / "wrds_users.parquet"
            selected_positions_path = tmp / "wrds_selected_us_positions.parquet"
            user_profile_cache_path = tmp / "wrds_user_profile_origin_cache.parquet"

            pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "updated_dt": ["2024-01-01", "2024-01-01"],
                    "female_prob": [0.2, 0.8],
                    "white_prob": [0.6, 0.2],
                    "black_prob": [0.1, 0.1],
                    "api_prob": [0.2, 0.6],
                    "hispanic_prob": [0.05, 0.05],
                    "native_prob": [0.0, 0.0],
                    "multiple_prob": [0.05, 0.05],
                }
            ).to_parquet(users_path, index=False)
            pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "has_nonus_educ": [0.0, 1.0],
                    "est_yob": [1985, 1990],
                    "p_likely_foreign": [0.0, 0.5],
                    "p_likely_native": [1.0, 0.5],
                    "likely_foreign_hard": [0, 1],
                }
            ).to_parquet(user_profile_cache_path, index=False)
            pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "position_id": [101, 102],
                    "position_number": [1, 1],
                    "rcid": [10, 10],
                    "country": ["United States", "United States"],
                    "startdate": ["2010-01-01", "2010-06-01"],
                    "enddate": ["2011-12-31", "2010-12-31"],
                    "onet_code": ["15-1132.00", "11-1011.00"],
                    "seniority_raw": ["Senior", "Manager"],
                    "salary": ["100", "200"],
                    "total_compensation": ["110", "220"],
                }
            ).to_parquet(selected_positions_path, index=False)

            out = source_exposure_data.build_local_wrds_company_year_workforce(
                selected_firms=pd.DataFrame({"c": [10]}),
                users_path=users_path,
                selected_positions_path=selected_positions_path,
                user_profile_cache_path=user_profile_cache_path,
                year_min=2010,
                year_max=2011,
                include_education_features=True,
            )

        out = out.set_index(["c", "t"]).sort_index()
        self.assertAlmostEqual(float(out.loc[(10, 2010), "total_headcount_wrds_annual"]), 2.0)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "total_headcount_foreign_weighted_annual"]), 0.5)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "total_headcount_native_weighted_annual"]), 1.5)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "n_new_hires_wrds_annual"]), 2.0)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "female_share_annual"]), 0.5)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "occ_share_computing_math_annual"]), 0.5)
        self.assertAlmostEqual(float(out.loc[(10, 2010), "occ_share_mgmt_annual"]), 0.5)
        self.assertAlmostEqual(float(out.loc[(10, 2011), "total_headcount_wrds_annual"]), 1.0)

    def test_build_local_wrds_school_flows_returns_expected_rows(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            users_path = tmp / "wrds_users.parquet"
            selected_positions_path = tmp / "wrds_selected_us_positions.parquet"

            pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "education_number": [1, 1],
                    "ed_enddate": ["2009-05-15", "2009-05-15"],
                    "university_raw": ["School A", "School B"],
                }
            ).to_parquet(users_path, index=False)
            pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "position_id": [101, 102],
                    "position_number": [1, 1],
                    "rcid": [10, 10],
                    "country": ["United States", "United States"],
                    "startdate": ["2010-01-01", "2010-03-01"],
                    "enddate": ["2011-12-31", "2011-12-31"],
                    "onet_code": [None, None],
                    "seniority_raw": [None, None],
                    "salary": [None, None],
                    "total_compensation": [None, None],
                }
            ).to_parquet(selected_positions_path, index=False)

            out = source_exposure_data.build_local_wrds_school_flows(
                selected_firms=pd.DataFrame({"c": [10]}),
                selected_positions_path=selected_positions_path,
                users_path=users_path,
                year_min=2010,
                year_max=2011,
                min_position_days=30,
                tenure_min_days=30,
            )

        out = out.sort_values(["university_raw", "c", "t"]).reset_index(drop=True)
        self.assertEqual(len(out), 4)
        self.assertEqual(sorted(out["university_raw"].tolist()), ["School A", "School A", "School B", "School B"])
        self.assertTrue((out["n_transitions"].fillna(0) >= 0).all())
        self.assertTrue((out["total_new_hires"].fillna(0) >= 0).all())

    def test_wrds_workforce_local_extract_cache_builds_outputs(self) -> None:
        cfg = {
            "testing": {"enabled": False},
            "revelio_company_features": {
                "wrds_username": "tester",
                "wrds_rcid_batch_size": 10,
                "wrds_large_firm_n_users_threshold": 75_000,
                "wrds_large_firm_year_span": 1,
                "wrds_workforce_extract_chunk_size": 100,
                "wrds_workforce_extract_max_workers": 3,
                "query_timeout_minutes": 10,
                "wrds_singleton_query_timeout_minutes": 60,
                "query_max_retries": 1,
            },
        }
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            settings = {
                "extract_user_ids_path": tmp / "wrds_workforce_user_ids.parquet",
                "extract_users_path": tmp / "wrds_workforce_users.parquet",
                "extract_positions_path": tmp / "wrds_workforce_positions.parquet",
                "extract_selected_positions_path": tmp / "wrds_workforce_selected_us_positions.parquet",
                "extract_users_chunk_dir": tmp / "wrds_workforce_users_chunks",
                "extract_positions_chunk_dir": tmp / "wrds_workforce_positions_chunks",
                "extract_selected_positions_chunk_dir": tmp / "wrds_workforce_selected_us_positions_chunks",
                "extract_chunk_size": 100,
                "extract_max_workers": 3,
            }

            def _fake_extract_selected_positions(
                *,
                rcids: list[int],
                wrds_username: str,
                selected_positions_chunk_dir: Path,
                rcid_batch_size: int,
                rcid_n_users: pd.DataFrame | dict[int, float] | None,
                large_firm_n_users_threshold: float | None,
                large_firm_year_span: int,
                year_min: int,
                year_max: int,
                query_timeout_ms: int | None,
                singleton_query_timeout_ms: int | None,
                query_max_retries: int,
                overwrite: bool,
            ) -> dict[str, int | bool]:
                selected_positions_chunk_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "user_id": [11, 22],
                        "position_id": [1, 2],
                        "position_number": [1, 1],
                        "rcid": [10, 10],
                        "country": ["United States", "United States"],
                        "startdate": ["2010-01-01", "2010-06-01"],
                        "enddate": ["2011-12-31", "2011-12-31"],
                        "onet_code": [None, None],
                        "seniority_raw": [None, None],
                        "salary": [None, None],
                        "total_compensation": [None, None],
                    }
                ).to_parquet(
                    selected_positions_chunk_dir / "wrds_selected_us_positions_chunk_00000.parquet",
                    index=False,
                )
                return {
                    "wrds_selected_us_position_chunks": 1,
                    "selected_positions_include_seniority": False,
                    "selected_positions_include_onet": False,
                }

            def _fake_write_chunks(
                *,
                user_ids: list[int],
                wrds_username: str,
                users_chunk_dir: Path,
                positions_chunk_dir: Path,
                chunk_size: int,
                max_workers: int,
                overwrite: bool,
            ) -> dict[str, int]:
                users_chunk_dir.mkdir(parents=True, exist_ok=True)
                positions_chunk_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "user_id": user_ids,
                        "user_country": ["United States", "Canada"],
                        "female_prob": [0.6, 0.4],
                    }
                ).to_parquet(users_chunk_dir / "wrds_users_chunk_00000.parquet", index=False)
                pd.DataFrame(
                    {
                        "user_id": user_ids,
                        "country": ["United States", "Canada"],
                    }
                ).to_parquet(positions_chunk_dir / "wrds_positions_chunk_00000.parquet", index=False)
                return {"wrds_query_chunks": 1, "wrds_user_rows": 2, "wrds_position_rows": 2}

            with patch.object(
                source_exposure_data,
                "_extract_wrds_selected_us_positions",
                side_effect=_fake_extract_selected_positions,
            ), patch.object(
                source_exposure_data,
                "_write_wrds_extract_chunks_for_user_ids",
                side_effect=_fake_write_chunks,
            ):
                users_path, positions_path, selected_positions_path, meta = (
                    source_exposure_data.load_or_build_wrds_workforce_local_extracts(
                    cfg=cfg,
                    settings=settings,
                    selected_firms=pd.DataFrame({"c": [10]}),
                    rcid_n_users=pd.DataFrame({"c": [10], "n_users": [100.0]}),
                    year_min=2010,
                    year_max=2011,
                    force_rebuild=True,
                ))
                self.assertTrue(users_path.exists())
                self.assertTrue(positions_path.exists())
                self.assertTrue(selected_positions_path.exists())
                self.assertTrue(settings["extract_user_ids_path"].exists())
                self.assertEqual(
                    meta["workforce_wrds_extract_method"],
                    source_exposure_data.WORKFORCE_WRDS_EXTRACT_METHOD,
                )
                self.assertEqual(int(meta["n_selected_user_ids"]), 2)
                self.assertEqual(int(meta["wrds_selected_us_position_rows"]), 2)
                self.assertEqual(int(meta["wrds_workforce_extract_max_workers"]), 3)

    def test_wrds_workforce_local_extract_reuses_in_process_wide_build_under_force_rebuild(self) -> None:
        cfg = {
            "testing": {"enabled": False},
            "revelio_company_features": {
                "wrds_username": "tester",
                "wrds_rcid_batch_size": 10,
                "wrds_large_firm_n_users_threshold": 75_000,
                "wrds_large_firm_year_span": 1,
                "wrds_workforce_extract_chunk_size": 100,
                "wrds_workforce_extract_max_workers": 3,
                "query_timeout_minutes": 10,
                "wrds_singleton_query_timeout_minutes": 60,
                "query_max_retries": 1,
            },
        }
        with TemporaryDirectory() as tmpdir, patch.dict(
            source_exposure_data._WRDS_WORKFORCE_LOCAL_EXTRACT_CACHE,
            {},
            clear=True,
        ):
            tmp = Path(tmpdir)
            settings = {
                "extract_user_ids_path": tmp / "wrds_workforce_user_ids.parquet",
                "extract_users_path": tmp / "wrds_workforce_users.parquet",
                "extract_positions_path": tmp / "wrds_workforce_positions.parquet",
                "extract_selected_positions_path": tmp / "wrds_workforce_selected_us_positions.parquet",
                "extract_users_chunk_dir": tmp / "wrds_workforce_users_chunks",
                "extract_positions_chunk_dir": tmp / "wrds_workforce_positions_chunks",
                "extract_selected_positions_chunk_dir": tmp / "wrds_workforce_selected_us_positions_chunks",
                "extract_chunk_size": 100,
                "extract_max_workers": 3,
            }

            def _fake_extract_selected_positions(
                *,
                rcids: list[int],
                wrds_username: str,
                selected_positions_chunk_dir: Path,
                rcid_batch_size: int,
                rcid_n_users: pd.DataFrame | dict[int, float] | None,
                large_firm_n_users_threshold: float | None,
                large_firm_year_span: int,
                year_min: int,
                year_max: int,
                query_timeout_ms: int | None,
                singleton_query_timeout_ms: int | None,
                query_max_retries: int,
                overwrite: bool,
            ) -> dict[str, int | bool]:
                selected_positions_chunk_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "user_id": [11, 22],
                        "position_id": [1, 2],
                        "position_number": [1, 1],
                        "rcid": [10, 10],
                        "country": ["United States", "United States"],
                        "startdate": ["2010-01-01", "2010-06-01"],
                        "enddate": ["2022-12-31", "2022-12-31"],
                        "onet_code": [None, None],
                        "seniority_raw": [None, None],
                        "salary": [None, None],
                        "total_compensation": [None, None],
                    }
                ).to_parquet(
                    selected_positions_chunk_dir / "wrds_selected_us_positions_chunk_00000.parquet",
                    index=False,
                )
                return {
                    "wrds_selected_us_position_chunks": 1,
                    "selected_positions_include_seniority": False,
                    "selected_positions_include_onet": False,
                }

            def _fake_write_chunks(
                *,
                user_ids: list[int],
                wrds_username: str,
                users_chunk_dir: Path,
                positions_chunk_dir: Path,
                chunk_size: int,
                max_workers: int,
                overwrite: bool,
            ) -> dict[str, int]:
                users_chunk_dir.mkdir(parents=True, exist_ok=True)
                positions_chunk_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "user_id": user_ids,
                        "user_country": ["United States", "Canada"],
                        "female_prob": [0.6, 0.4],
                    }
                ).to_parquet(users_chunk_dir / "wrds_users_chunk_00000.parquet", index=False)
                pd.DataFrame(
                    {
                        "user_id": user_ids,
                        "country": ["United States", "Canada"],
                    }
                ).to_parquet(positions_chunk_dir / "wrds_positions_chunk_00000.parquet", index=False)
                return {"wrds_query_chunks": 1, "wrds_user_rows": 2, "wrds_position_rows": 2}

            with patch.object(
                source_exposure_data,
                "_extract_wrds_selected_us_positions",
                side_effect=_fake_extract_selected_positions,
            ) as selected_mock, patch.object(
                source_exposure_data,
                "_write_wrds_extract_chunks_for_user_ids",
                side_effect=_fake_write_chunks,
            ) as chunk_mock:
                first = source_exposure_data.load_or_build_wrds_workforce_local_extracts(
                    cfg=cfg,
                    settings=settings,
                    selected_firms=pd.DataFrame({"c": [10]}),
                    rcid_n_users=pd.DataFrame({"c": [10], "n_users": [100.0]}),
                    year_min=2010,
                    year_max=2022,
                    force_rebuild=True,
                )
                second = source_exposure_data.load_or_build_wrds_workforce_local_extracts(
                    cfg=cfg,
                    settings=settings,
                    selected_firms=pd.DataFrame({"c": [10]}),
                    rcid_n_users=pd.DataFrame({"c": [10], "n_users": [100.0]}),
                    year_min=2010,
                    year_max=2015,
                    force_rebuild=True,
                )

        self.assertEqual(selected_mock.call_count, 1)
        self.assertEqual(chunk_mock.call_count, 1)
        self.assertEqual(first[:3], second[:3])
        self.assertEqual(int(first[3]["year_max"]), 2022)
        self.assertEqual(int(second[3]["year_max"]), 2022)

    def test_wrds_workforce_local_extract_reuses_on_disk_under_cached_universe_mode(self) -> None:
        cfg = {
            "testing": {"enabled": False},
            "reuse_cached_wrds_universe_only": True,
            "revelio_company_features": {
                "wrds_username": "tester",
                "wrds_rcid_batch_size": 10,
                "wrds_large_firm_n_users_threshold": -1,
                "wrds_large_firm_year_span": 3,
                "wrds_workforce_extract_chunk_size": 100,
                "wrds_workforce_extract_max_workers": 3,
                "query_timeout_minutes": 10,
                "wrds_singleton_query_timeout_minutes": 60,
                "query_max_retries": 1,
            },
        }
        with TemporaryDirectory() as tmpdir, patch.dict(
            source_exposure_data._WRDS_WORKFORCE_LOCAL_EXTRACT_CACHE,
            {},
            clear=True,
        ):
            tmp = Path(tmpdir)
            settings = {
                "extract_user_ids_path": tmp / "wrds_workforce_user_ids.parquet",
                "extract_users_path": tmp / "wrds_workforce_users.parquet",
                "extract_positions_path": tmp / "wrds_workforce_positions.parquet",
                "extract_selected_positions_path": tmp / "wrds_workforce_selected_us_positions.parquet",
                "extract_users_chunk_dir": tmp / "wrds_workforce_users_chunks",
                "extract_positions_chunk_dir": tmp / "wrds_workforce_positions_chunks",
                "extract_selected_positions_chunk_dir": tmp / "wrds_workforce_selected_us_positions_chunks",
                "extract_chunk_size": 100,
                "extract_max_workers": 3,
            }
            pd.DataFrame({"user_id": [1]}).to_parquet(settings["extract_user_ids_path"], index=False)
            pd.DataFrame({"user_id": [1]}).to_parquet(settings["extract_users_path"], index=False)
            pd.DataFrame({"user_id": [1]}).to_parquet(settings["extract_positions_path"], index=False)
            pd.DataFrame({"user_id": [1]}).to_parquet(settings["extract_selected_positions_path"], index=False)
            meta = {
                "year_min": 2010,
                "year_max": 2022,
                "selected_firms_hash": "oldhash",
                "selected_firms_n": 1,
                "wrds_username": "tester",
                "wrds_rcid_batch_size": 10,
                "wrds_large_firm_n_users_threshold": -1,
                "wrds_large_firm_year_span": 3,
                "wrds_workforce_extract_chunk_size": 100,
                "workforce_wrds_extract_method": source_exposure_data.WORKFORCE_WRDS_EXTRACT_METHOD,
                "new_hire_origin_method": source_exposure_data.NEW_HIRE_ORIGIN_METHOD,
            }
            source_exposure_data._write_metadata(settings["extract_users_path"], meta)

            with patch.object(source_exposure_data, "_extract_wrds_selected_us_positions") as selected_mock, patch.object(
                source_exposure_data,
                "_write_wrds_extract_chunks_for_user_ids",
            ) as chunk_mock:
                out = source_exposure_data.load_or_build_wrds_workforce_local_extracts(
                    cfg=cfg,
                    settings=settings,
                    selected_firms=pd.DataFrame({"c": [999]}),
                    rcid_n_users=None,
                    year_min=2010,
                    year_max=2022,
                    force_rebuild=False,
                )

        self.assertEqual(out[0], settings["extract_users_path"])
        self.assertEqual(selected_mock.call_count, 0)
        self.assertEqual(chunk_mock.call_count, 0)

    def test_wrds_workforce_local_extract_merges_selected_positions_without_distinct(self) -> None:
        cfg = {
            "testing": {"enabled": False},
            "revelio_company_features": {
                "wrds_username": "tester",
                "wrds_rcid_batch_size": 10,
                "wrds_large_firm_n_users_threshold": 75_000,
                "wrds_large_firm_year_span": 1,
                "wrds_workforce_extract_chunk_size": 100,
                "wrds_workforce_extract_max_workers": 3,
                "query_timeout_minutes": 10,
                "wrds_singleton_query_timeout_minutes": 60,
                "query_max_retries": 1,
            },
        }
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            settings = {
                "extract_user_ids_path": tmp / "wrds_workforce_user_ids.parquet",
                "extract_users_path": tmp / "wrds_workforce_users.parquet",
                "extract_positions_path": tmp / "wrds_workforce_positions.parquet",
                "extract_selected_positions_path": tmp / "wrds_workforce_selected_us_positions.parquet",
                "extract_users_chunk_dir": tmp / "wrds_workforce_users_chunks",
                "extract_positions_chunk_dir": tmp / "wrds_workforce_positions_chunks",
                "extract_selected_positions_chunk_dir": tmp / "wrds_workforce_selected_us_positions_chunks",
                "extract_chunk_size": 100,
                "extract_max_workers": 3,
            }
            merge_calls: list[dict[str, object]] = []

            def _fake_extract_selected_positions(
                *,
                rcids: list[int],
                wrds_username: str,
                selected_positions_chunk_dir: Path,
                rcid_batch_size: int,
                rcid_n_users: pd.DataFrame | dict[int, float] | None,
                large_firm_n_users_threshold: float | None,
                large_firm_year_span: int,
                year_min: int,
                year_max: int,
                query_timeout_ms: int | None,
                singleton_query_timeout_ms: int | None,
                query_max_retries: int,
                overwrite: bool,
            ) -> dict[str, int | bool]:
                selected_positions_chunk_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "user_id": [11, 22],
                        "position_id": [1, 2],
                        "position_number": [1, 1],
                        "rcid": [10, 10],
                        "country": ["United States", "United States"],
                        "startdate": ["2010-01-01", "2010-06-01"],
                        "enddate": ["2011-12-31", "2011-12-31"],
                        "onet_code": [None, None],
                        "seniority_raw": [None, None],
                        "salary": [None, None],
                        "total_compensation": [None, None],
                    }
                ).to_parquet(
                    selected_positions_chunk_dir / "wrds_selected_us_positions_chunk_00000.parquet",
                    index=False,
                )
                return {
                    "wrds_selected_us_position_chunks": 1,
                    "selected_positions_include_seniority": False,
                    "selected_positions_include_onet": False,
                }

            def _fake_write_chunks(
                *,
                user_ids: list[int],
                wrds_username: str,
                users_chunk_dir: Path,
                positions_chunk_dir: Path,
                chunk_size: int,
                max_workers: int,
                overwrite: bool,
            ) -> dict[str, int]:
                users_chunk_dir.mkdir(parents=True, exist_ok=True)
                positions_chunk_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "user_id": user_ids,
                        "user_country": ["United States", "Canada"],
                        "female_prob": [0.6, 0.4],
                    }
                ).to_parquet(users_chunk_dir / "wrds_users_chunk_00000.parquet", index=False)
                pd.DataFrame(
                    {
                        "user_id": user_ids,
                        "country": ["United States", "Canada"],
                    }
                ).to_parquet(positions_chunk_dir / "wrds_positions_chunk_00000.parquet", index=False)
                return {"wrds_query_chunks": 1, "wrds_user_rows": 2, "wrds_position_rows": 2}

            def _fake_merge_chunk_dir_to_output(
                *,
                chunk_dir: Path,
                output_parquet: Path,
                empty_columns: list[str],
                overwrite: bool,
                distinct: bool = False,
                log_label: str | None = None,
            ) -> None:
                merge_calls.append(
                    {
                        "chunk_dir": chunk_dir,
                        "output_parquet": output_parquet,
                        "distinct": distinct,
                        "log_label": log_label,
                    }
                )
                chunk_files = sorted(chunk_dir.glob("*.parquet"))
                if chunk_files:
                    pd.concat([pd.read_parquet(path) for path in chunk_files], ignore_index=True).to_parquet(
                        output_parquet,
                        index=False,
                    )
                else:
                    pd.DataFrame(columns=empty_columns).to_parquet(output_parquet, index=False)

            with patch.object(
                source_exposure_data,
                "_extract_wrds_selected_us_positions",
                side_effect=_fake_extract_selected_positions,
            ), patch.object(
                source_exposure_data,
                "_write_wrds_extract_chunks_for_user_ids",
                side_effect=_fake_write_chunks,
            ), patch.object(
                source_exposure_data,
                "_merge_chunk_dir_to_output",
                side_effect=_fake_merge_chunk_dir_to_output,
            ):
                source_exposure_data.load_or_build_wrds_workforce_local_extracts(
                    cfg=cfg,
                    settings=settings,
                    selected_firms=pd.DataFrame({"c": [10]}),
                    rcid_n_users=pd.DataFrame({"c": [10], "n_users": [100.0]}),
                    year_min=2010,
                    year_max=2011,
                    force_rebuild=True,
                )

        self.assertGreaterEqual(len(merge_calls), 3)
        self.assertFalse(bool(merge_calls[0]["distinct"]))
        self.assertEqual(str(merge_calls[0]["log_label"]), "wrds_workforce_extract")

    def test_local_user_profile_cache_reuses_in_process_wide_build_under_force_rebuild(self) -> None:
        with TemporaryDirectory() as tmpdir, patch.dict(
            source_exposure_data._LOCAL_USER_PROFILE_CACHE,
            {},
            clear=True,
        ):
            tmp = Path(tmpdir)
            users_path = tmp / "wrds_users.parquet"
            positions_path = tmp / "wrds_positions.parquet"
            out_path = tmp / "wrds_user_profile_origin_cache.parquet"
            pd.DataFrame({"user_id": [1], "user_country": ["United States"]}).to_parquet(users_path, index=False)
            pd.DataFrame({"user_id": [1], "country": ["United States"]}).to_parquet(positions_path, index=False)

            def _fake_build_local_profile_cache(
                *,
                selected_firms: pd.DataFrame,
                users_path: Path,
                positions_path: Path,
                out_path: Path,
                year_min: int,
                year_max: int,
            ) -> None:
                pd.DataFrame(
                    {
                        "user_id": [1],
                        "has_nonus_educ": [0],
                        "est_yob": [1990],
                        "signal_current_country_nonus": [0.0],
                        "signal_nonus_educ": [0.0],
                        "signal_nonus_position": [0.0],
                        "p_likely_foreign": [0.0],
                        "p_likely_native": [1.0],
                        "likely_foreign_hard": [0],
                    }
                ).to_parquet(out_path, index=False)

            with patch.object(
                source_exposure_data,
                "_build_local_wrds_user_profile_cache",
                side_effect=_fake_build_local_profile_cache,
            ) as build_mock:
                first_path, first_meta = source_exposure_data.load_or_build_local_wrds_user_profile_cache(
                    selected_firms=pd.DataFrame({"c": [10]}),
                    users_path=users_path,
                    positions_path=positions_path,
                    out_path=out_path,
                    year_min=2010,
                    year_max=2022,
                    force_rebuild=True,
                )
                second_path, second_meta = source_exposure_data.load_or_build_local_wrds_user_profile_cache(
                    selected_firms=pd.DataFrame({"c": [10]}),
                    users_path=users_path,
                    positions_path=positions_path,
                    out_path=out_path,
                    year_min=2010,
                    year_max=2015,
                    force_rebuild=True,
                )

        self.assertEqual(build_mock.call_count, 1)
        self.assertEqual(first_path, second_path)
        self.assertEqual(int(first_meta["year_max"]), 2022)
        self.assertEqual(int(second_meta["year_max"]), 2022)

    def test_write_wrds_extract_chunks_for_user_ids_uses_parallel_workers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            users_chunk_dir = tmp / "wrds_workforce_users_chunks"
            positions_chunk_dir = tmp / "wrds_workforce_positions_chunks"
            open_calls: list[str] = []

            class _FakeDb:
                def close(self) -> None:
                    return None

            def _fake_open_wrds_connection(wrds_username: str, query_timeout_ms: int | None) -> _FakeDb:
                open_calls.append(f"{wrds_username}:{query_timeout_ms}")
                return _FakeDb()

            def _fake_build_users_extract_query(
                user_chunk: list[int],
                *,
                user_cols: list[str],
                female_source_col: str | None,
            ) -> tuple[str, tuple[int, ...]]:
                return ("users", tuple(user_chunk))

            def _fake_build_positions_extract_query(
                user_chunk: list[int],
                *,
                position_cols: list[str],
            ) -> tuple[str, tuple[int, ...]]:
                return ("positions", tuple(user_chunk))

            def _fake_run_sql_with_retries(
                db: _FakeDb,
                sql: tuple[str, tuple[int, ...]],
                *,
                wrds_username: str,
                query_timeout_ms: int | None,
                max_retries: int,
                label: str,
            ) -> tuple[pd.DataFrame, _FakeDb]:
                kind, chunk = sql
                if kind == "users":
                    return (
                        pd.DataFrame(
                            {
                                "user_id": list(chunk),
                                "user_country": ["United States"] * len(chunk),
                            }
                        ),
                        db,
                    )
                return (
                    pd.DataFrame(
                        {
                            "user_id": list(chunk),
                            "country": ["United States"] * len(chunk),
                        }
                    ),
                    db,
                )

            with patch.object(
                source_exposure_data,
                "_open_wrds_connection",
                side_effect=_fake_open_wrds_connection,
            ), patch.object(
                source_exposure_data,
                "_resolve_wrds_extract_schema",
                return_value={
                    "user_cols": ["user_id", "user_country"],
                    "female_source_col": None,
                    "position_cols": ["user_id", "country"],
                },
            ), patch.object(
                source_exposure_data,
                "_build_wrds_users_extract_query",
                side_effect=_fake_build_users_extract_query,
            ), patch.object(
                source_exposure_data,
                "_build_wrds_positions_extract_query",
                side_effect=_fake_build_positions_extract_query,
            ), patch.object(
                source_exposure_data,
                "_run_sql_with_retries",
                side_effect=_fake_run_sql_with_retries,
            ):
                stats = source_exposure_data._write_wrds_extract_chunks_for_user_ids(
                    user_ids=[1, 2, 3, 4, 5],
                    wrds_username="tester",
                    users_chunk_dir=users_chunk_dir,
                    positions_chunk_dir=positions_chunk_dir,
                    chunk_size=2,
                    max_workers=3,
                    overwrite=True,
                )

            self.assertEqual(stats["wrds_query_chunks"], 3)
            self.assertEqual(stats["wrds_user_rows"], 5)
            self.assertEqual(stats["wrds_position_rows"], 5)
            self.assertEqual(len(list(users_chunk_dir.glob("*.parquet"))), 3)
            self.assertEqual(len(list(positions_chunk_dir.glob("*.parquet"))), 3)
            self.assertEqual(len(open_calls), 4)

    def test_wrds_workforce_cache_builds_full_local_panel_when_enabled(self) -> None:
        cfg = {
            "testing": {"enabled": False},
            "revelio_company_features": {
                "wrds_username": "tester",
                "wrds_rcid_batch_size": 10,
                "query_timeout_minutes": 10,
                "query_max_retries": 1,
                "wrds_workforce_use_local_user_profile_cache": True,
            },
        }
        firms = pd.DataFrame(
            {
                "c": [1],
                "in_analysis_universe": [1],
                "preferred_rcid_source": [1],
                "outside_negative_candidate": [0],
            }
        )
        selected_meta = pd.DataFrame({"c": [1], "n_users": [100.0]})
        universe_meta = {
            "analysis_universe_method": "preferred_plus_outside_sample_v2",
            "outside_negative_ratio": 2.0,
            "outside_negative_seed": 42,
            "outside_negative_min_n_users": 10,
        }
        built_local = pd.DataFrame(
            {
                "c": [1],
                "t": [2010],
                "total_headcount_wrds_annual": [10.0],
                "total_headcount_foreign_weighted_annual": [3.5],
                "total_headcount_native_weighted_annual": [6.5],
                "total_headcount_foreign_hard_annual": [4.0],
                "total_headcount_native_hard_annual": [6.0],
                "long_term_headcount_wrds_annual": [7.0],
                "n_new_hires_wrds_annual": [4.0],
                "n_new_hires_foreign_weighted_annual": [1.5],
                "n_new_hires_native_weighted_annual": [2.5],
                "n_new_hires_foreign_hard_annual": [2.0],
                "n_new_hires_native_hard_annual": [2.0],
                "salary_mean_annual": [100.0],
                "salary_var_annual": [25.0],
                "total_comp_mean_annual": [120.0],
                "total_comp_var_annual": [30.0],
                "compensation_missing_share_annual": [0.1],
                "nonus_educ_share_annual": [0.2],
                "age_share_lt30_annual": [0.3],
                "age_share_30_39_annual": [0.4],
                "age_share_40_49_annual": [0.2],
                "age_share_50_59_annual": [0.1],
                "age_share_60p_annual": [0.0],
                "female_share_annual": [0.45],
                "race_share_white_annual": [0.5],
                "race_share_black_annual": [0.1],
                "race_share_api_annual": [0.2],
                "race_share_hispanic_annual": [0.1],
                "race_share_native_annual": [0.05],
                "race_share_multiple_annual": [0.05],
                "seniority_mean_annual": [2.5],
                "avg_tenure_years_annual": [1.2],
                "occ_share_mgmt_annual": [0.1],
                "occ_share_business_finance_annual": [0.1],
                "occ_share_computing_math_annual": [0.2],
                "occ_share_engineering_annual": [0.1],
                "occ_share_science_annual": [0.1],
                "occ_share_community_legal_education_annual": [0.1],
                "occ_share_arts_media_annual": [0.0],
                "occ_share_healthcare_annual": [0.0],
                "occ_share_sales_office_annual": [0.1],
                "occ_share_manual_annual": [0.2],
            }
        )

        with TemporaryDirectory() as tmpdir, patch.dict(
            source_exposure_data._WRDS_WORKFORCE_CACHE,
            {},
            clear=True,
        ):
            tmp = Path(tmpdir)
            paths = SourceExposurePaths(
                foia_corrected=tmp / "foia.parquet",
                employer_crosswalk=tmp / "crosswalk.parquet",
                preferred_rcids=tmp / "preferred.parquet",
                company_mapping=tmp / "company_mapping.parquet",
                f1_inst_unitid_crosswalk=tmp / "inst.parquet",
                revelio_inst_crosswalk=tmp / "revelio_inst.parquet",
                revelio_inst_deterministic_map=None,
                revelio_ref_inst_catalog=None,
                source_opt_counts_out=tmp / "source_opt_counts.parquet",
                school_opt_benchmark_out=tmp / "school_opt_benchmark.parquet",
                opt_exposure_analysis_panel_out=tmp / "analysis_panel.parquet",
                outside_negative_sample_out=tmp / "outside_negative_sample.parquet",
                wrds_company_year_workforce_out=tmp / "wrds_company_year_workforce.parquet",
                wrds_school_flows_out=tmp / "wrds_school_flows.parquet",
            )
            local_settings = {
                "requested": True,
                "enabled": True,
                "source_mode": "dedicated_wrds_extract",
                "use_dedicated_extracts": True,
                "snapshot_users_path": None,
                "snapshot_positions_path": None,
                "extract_user_ids_path": tmp / "wrds_workforce_user_ids.parquet",
                "extract_users_path": tmp / "wrds_workforce_users.parquet",
                "extract_positions_path": tmp / "wrds_workforce_positions.parquet",
                "extract_selected_positions_path": tmp / "wrds_workforce_selected_us_positions.parquet",
                "extract_users_chunk_dir": tmp / "wrds_workforce_users_chunks",
                "extract_positions_chunk_dir": tmp / "wrds_workforce_positions_chunks",
                "extract_selected_positions_chunk_dir": tmp / "wrds_workforce_selected_us_positions_chunks",
                "extract_chunk_size": 100,
                "extract_max_workers": 3,
                "cache_path": tmp / "wrds_user_profile_origin_cache.parquet",
            }

            with patch.object(source_exposure_data, "resolve_source_exposure_paths", return_value=paths), patch.object(
                source_exposure_data,
                "load_or_build_source_firm_universe",
                return_value=(firms.copy(), pd.DataFrame(), selected_meta.copy(), dict(universe_meta)),
            ), patch.object(
                source_exposure_data,
                "_resolve_local_user_profile_refactor_settings",
                return_value=local_settings,
            ), patch.object(
                source_exposure_data,
                "load_or_build_wrds_workforce_local_extracts",
                return_value=(
                    tmp / "wrds_workforce_users.parquet",
                    tmp / "wrds_workforce_positions.parquet",
                    tmp / "wrds_workforce_selected_us_positions.parquet",
                    {"extract": "meta"},
                ),
            ), patch.object(
                source_exposure_data,
                "load_or_build_local_wrds_user_profile_cache",
                return_value=(local_settings["cache_path"], {"cache": "meta"}),
            ), patch.object(
                source_exposure_data,
                "build_local_wrds_company_year_workforce",
                return_value=built_local.copy(),
            ):
                out, meta = load_or_build_wrds_company_year_workforce_cache(
                    cfg=cfg,
                    year_min=2010,
                    year_max=2010,
                    force_rebuild=True,
                )

        self.assertTrue(meta["local_user_profile_refactor_enabled"])
        self.assertEqual(meta["local_workforce_panel_method"], source_exposure_data.LOCAL_WORKFORCE_PANEL_METHOD)
        row = out.iloc[0]
        self.assertAlmostEqual(float(row["nonus_educ_share_annual"]), 0.2)
        self.assertAlmostEqual(float(row["total_headcount_foreign_weighted_annual"]), 3.5)
        self.assertAlmostEqual(float(row["n_new_hires_native_weighted_annual"]), 2.5)

    def test_wrds_workforce_cache_carries_integrated_origin_split_columns(self) -> None:
        cfg = {
            "testing": {"enabled": True, "output_suffix": "_testing"},
            "revelio_company_features": {
                "wrds_username": "tester",
                "wrds_rcid_batch_size": 10,
                "query_timeout_minutes": 10,
                "query_max_retries": 1,
            },
        }
        firms = pd.DataFrame(
            {
                "c": [1],
                "in_analysis_universe": [1],
                "preferred_rcid_source": [1],
                "outside_negative_candidate": [0],
            }
        )
        built_workforce = pd.DataFrame(
            {
                "c": [1, 1],
                "t": [2010, 2011],
                "total_headcount_wrds_annual": [10.0, 11.0],
                "total_headcount_foreign_weighted_annual": [3.5, 4.0],
                "total_headcount_native_weighted_annual": [6.5, 7.0],
                "total_headcount_foreign_hard_annual": [4.0, 4.0],
                "total_headcount_native_hard_annual": [6.0, 7.0],
                "n_new_hires_wrds_annual": [4.0, 2.0],
                "n_new_hires_foreign_weighted_annual": [1.5, 0.5],
                "n_new_hires_native_weighted_annual": [2.5, 1.5],
                "n_new_hires_foreign_hard_annual": [2.0, 1.0],
                "n_new_hires_native_hard_annual": [2.0, 1.0],
            }
        )

        with TemporaryDirectory() as tmpdir, patch.dict(
            source_exposure_data._WRDS_WORKFORCE_CACHE,
            {},
            clear=True,
        ):
            tmp = Path(tmpdir)
            paths = SourceExposurePaths(
                foia_corrected=tmp / "foia.parquet",
                employer_crosswalk=tmp / "crosswalk.parquet",
                preferred_rcids=tmp / "preferred.parquet",
                company_mapping=tmp / "company_mapping.parquet",
                f1_inst_unitid_crosswalk=tmp / "inst.parquet",
                revelio_inst_crosswalk=tmp / "revelio_inst.parquet",
                revelio_inst_deterministic_map=None,
                revelio_ref_inst_catalog=None,
                source_opt_counts_out=tmp / "source_opt_counts.parquet",
                school_opt_benchmark_out=tmp / "school_opt_benchmark.parquet",
                opt_exposure_analysis_panel_out=tmp / "analysis_panel.parquet",
                outside_negative_sample_out=tmp / "outside_negative_sample_testing.parquet",
                wrds_company_year_workforce_out=tmp / "wrds_company_year_workforce_testing.parquet",
                wrds_school_flows_out=tmp / "wrds_school_flows.parquet",
            )
            universe_meta = {
                "analysis_universe_method": "preferred_plus_outside_sample_v2",
                "outside_negative_ratio": 2.0,
                "outside_negative_seed": 42,
                "outside_negative_min_n_users": 10,
            }

            with patch.object(source_exposure_data, "resolve_source_exposure_paths", return_value=paths), patch.object(
                source_exposure_data,
                "load_or_build_source_firm_universe",
                return_value=(firms.copy(), pd.DataFrame(), pd.DataFrame(), dict(universe_meta)),
            ), patch.object(
                source_exposure_data,
                "build_wrds_company_year_workforce",
                return_value=built_workforce.copy(),
            ):
                out, meta = load_or_build_wrds_company_year_workforce_cache(
                    cfg=cfg,
                    year_min=2010,
                    year_max=2011,
                    force_rebuild=True,
                )

        self.assertEqual(meta["new_hire_origin_method"], source_exposure_data.NEW_HIRE_ORIGIN_METHOD)
        for col in source_exposure_data.ORIGIN_SPLIT_WORKFORCE_COLUMNS:
            self.assertIn(col, out.columns)
        self.assertTrue(
            np.allclose(
                out["total_headcount_foreign_weighted_annual"]
                + out["total_headcount_native_weighted_annual"],
                out["total_headcount_wrds_annual"],
            )
        )
        self.assertTrue(
            (
                out["total_headcount_foreign_hard_annual"] + out["total_headcount_native_hard_annual"]
                == out["total_headcount_wrds_annual"]
            ).all()
        )

    def test_wrds_school_flows_cache_reuses_on_disk_under_cached_universe_mode(self) -> None:
        cfg = {
            "testing": {"enabled": False},
            "reuse_cached_wrds_universe_only": True,
            "revelio_company_features": {
                "wrds_username": "tester",
                "wrds_workforce_build_local_extracts": False,
                "wrds_workforce_use_local_user_profile_cache": False,
                "min_position_days": 365,
                "tenure_min_days": 365,
            },
        }
        firms = pd.DataFrame(
            {
                "c": [1],
                "in_analysis_universe": [1],
                "preferred_rcid_source": [1],
                "outside_negative_candidate": [0],
            }
        )
        selected_meta = pd.DataFrame({"c": [1], "n_users": [100.0]})
        cached_flows = pd.DataFrame(
            {
                "c": [1],
                "t": [2010],
                "university_raw": ["school a"],
                "n_transitions": [1.0],
                "n_emp": [1.0],
                "total_new_hires": [1.0],
            }
        )
        universe_meta = {
            "analysis_universe_method": "preferred_plus_outside_sample_v2",
            "outside_negative_ratio": 1.0,
            "outside_negative_seed": 42,
            "outside_negative_min_n_users": 10,
            "n_preferred_rcids": 1,
            "preferred_sample_meta": {"testing_enabled": False, "selected_sample_n": 1},
            "n_total_firms": 1,
            "n_outside_negative_candidates": 0,
            "analysis_firms_hash": "newhash",
            "outside_negative_firms_hash": "newoutsidehash",
        }

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            paths = SourceExposurePaths(
                foia_corrected=tmp / "foia.parquet",
                employer_crosswalk=tmp / "crosswalk.parquet",
                preferred_rcids=tmp / "preferred.parquet",
                company_mapping=tmp / "company_mapping.parquet",
                f1_inst_unitid_crosswalk=tmp / "inst.parquet",
                revelio_inst_crosswalk=tmp / "revelio_inst.parquet",
                revelio_inst_deterministic_map=None,
                revelio_ref_inst_catalog=None,
                source_opt_counts_out=tmp / "source_opt_counts.parquet",
                school_opt_benchmark_out=tmp / "school_opt_benchmark.parquet",
                opt_exposure_analysis_panel_out=tmp / "analysis_panel.parquet",
                outside_negative_sample_out=tmp / "outside_negative_sample_testing.parquet",
                wrds_company_year_workforce_out=tmp / "wrds_company_year_workforce.parquet",
                wrds_school_flows_out=tmp / "wrds_school_flows.parquet",
            )
            cached_flows.to_parquet(paths.wrds_school_flows_out, index=False)
            source_exposure_data._write_metadata(
                paths.wrds_school_flows_out,
                {
                    "year_min": 2010,
                    "year_max": 2022,
                    "min_position_days": 365,
                    "tenure_min_days": 365,
                    "local_school_flows_method": None,
                    "workforce_wrds_extract_method": None,
                    "analysis_universe_method": "preferred_plus_outside_sample_v2",
                    "outside_negative_ratio": 1.0,
                    "outside_negative_seed": 42,
                    "outside_negative_min_n_users": 10,
                    "n_preferred_rcids": 1,
                    "preferred_sample_meta": {"testing_enabled": False, "selected_sample_n": 1},
                    "n_total_firms": 1,
                    "n_outside_negative_candidates": 0,
                    "analysis_firms_hash": "oldhash",
                    "outside_negative_firms_hash": "oldoutsidehash",
                },
            )

            with patch.object(source_exposure_data, "resolve_source_exposure_paths", return_value=paths), patch.object(
                source_exposure_data,
                "load_or_build_source_firm_universe",
                return_value=(firms.copy(), pd.DataFrame(), selected_meta.copy(), dict(universe_meta)),
            ), patch.object(
                source_exposure_data,
                "build_wrds_school_flows",
            ) as build_mock:
                out, meta = source_exposure_data.load_or_build_wrds_school_flows_cache(
                    cfg=cfg,
                    year_min=2010,
                    year_max=2022,
                    force_rebuild=False,
                )

        self.assertEqual(build_mock.call_count, 0)
        self.assertEqual(len(out), 1)
        self.assertEqual(meta["analysis_firms_hash"], "oldhash")

    def test_build_source_analysis_panel_adds_origin_outcomes(self) -> None:
        cfg = {"testing": {"enabled": False}}
        firms = pd.DataFrame(
            {
                "c": [1, 2],
                "in_analysis_universe": [1, 1],
                "preferred_rcid_source": [1, 1],
                "outside_negative_candidate": [0, 0],
            }
        )
        counts = pd.DataFrame(
            {
                "c": [1],
                "t": [2010],
                "bachelors_opt_hires_correction_aware": [0],
                "masters_opt_hires_correction_aware": [2],
                "phd_opt_hires_correction_aware": [0],
                "any_opt_hires_correction_aware": [2],
            }
        )
        workforce = pd.DataFrame(
            {
                "c": [1, 2],
                "t": [2010, 2010],
                "total_headcount_wrds_annual": [10.0, 20.0],
                "total_headcount_foreign_weighted_annual": [3.5, 5.0],
                "total_headcount_native_weighted_annual": [6.5, 15.0],
                "total_headcount_foreign_hard_annual": [4.0, 5.0],
                "total_headcount_native_hard_annual": [6.0, 15.0],
                "n_new_hires_wrds_annual": [4.0, 3.0],
                "n_new_hires_foreign_weighted_annual": [1.5, 0.0],
                "n_new_hires_native_weighted_annual": [2.5, 3.0],
                "n_new_hires_foreign_hard_annual": [2.0, 0.0],
                "n_new_hires_native_hard_annual": [2.0, 3.0],
            }
        )

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            paths = SourceExposurePaths(
                foia_corrected=tmp / "foia.parquet",
                employer_crosswalk=tmp / "crosswalk.parquet",
                preferred_rcids=tmp / "preferred.parquet",
                company_mapping=tmp / "company_mapping.parquet",
                f1_inst_unitid_crosswalk=tmp / "inst.parquet",
                revelio_inst_crosswalk=tmp / "revelio_inst.parquet",
                revelio_inst_deterministic_map=None,
                revelio_ref_inst_catalog=None,
                source_opt_counts_out=tmp / "source_opt_counts.parquet",
                school_opt_benchmark_out=tmp / "school_opt_benchmark.parquet",
                opt_exposure_analysis_panel_out=tmp / "analysis_panel.parquet",
                outside_negative_sample_out=tmp / "outside_negative_sample.parquet",
                wrds_company_year_workforce_out=tmp / "wrds_company_year_workforce.parquet",
                wrds_school_flows_out=tmp / "wrds_school_flows.parquet",
            )
            universe_meta = {
                "analysis_universe_method": "preferred_plus_outside_sample_v2",
                "outside_negative_ratio": 1.0,
                "outside_negative_seed": 42,
                "outside_negative_min_n_users": 10,
            }
            workforce_meta = {
                "year_min": 2010,
                "year_max": 2010,
                "new_hire_origin_method": source_exposure_data.NEW_HIRE_ORIGIN_METHOD,
            }

            with patch.object(source_exposure_data, "resolve_source_exposure_paths", return_value=paths), patch.object(
                source_exposure_data,
                "load_or_build_source_firm_universe",
                return_value=(firms.copy(), pd.DataFrame(), pd.DataFrame(), dict(universe_meta)),
            ), patch.object(
                source_exposure_data,
                "load_or_build_source_opt_counts",
                return_value=(counts.copy(), {"opt_count_method": source_exposure_data.OPT_COUNT_METHOD}),
            ), patch.object(
                source_exposure_data,
                "load_or_build_wrds_company_year_workforce_cache",
                return_value=(workforce.copy(), dict(workforce_meta)),
            ):
                panel = build_source_analysis_panel(
                    cfg=cfg,
                    data_min_t=2010,
                    data_max_t=2010,
                    force_rebuild=True,
                )

        for col in (
            "y_cst_foreign_lag0",
            "y_cst_native_lag0",
            "y_cst_foreign_hard_lag0",
            "y_cst_native_hard_lag0",
            "y_new_hires_foreign_lag0",
            "y_new_hires_native_lag0",
            "y_new_hires_foreign_hard_lag0",
            "y_new_hires_native_hard_lag0",
        ):
            self.assertIn(col, panel.columns)
        row = panel.set_index("c").loc[1]
        self.assertAlmostEqual(float(row["y_cst_lag0"]), 10.0)
        self.assertAlmostEqual(float(row["y_cst_foreign_lag0"]), 3.5)
        self.assertAlmostEqual(float(row["y_cst_native_lag0"]), 6.5)
        self.assertAlmostEqual(float(row["y_cst_foreign_hard_lag0"]), 4.0)
        self.assertAlmostEqual(float(row["y_cst_native_hard_lag0"]), 6.0)
        self.assertAlmostEqual(
            float(row["y_cst_foreign_lag0"] + row["y_cst_native_lag0"]),
            float(row["y_cst_lag0"]),
        )
        self.assertAlmostEqual(float(row["y_new_hires_lag0"]), 4.0)
        self.assertAlmostEqual(float(row["y_new_hires_foreign_lag0"]), 1.5)
        self.assertAlmostEqual(float(row["y_new_hires_native_lag0"]), 2.5)
        self.assertAlmostEqual(float(row["y_new_hires_foreign_hard_lag0"]), 2.0)
        self.assertAlmostEqual(float(row["y_new_hires_native_hard_lag0"]), 2.0)
        self.assertAlmostEqual(
            float(row["y_new_hires_foreign_lag0"] + row["y_new_hires_native_lag0"]),
            float(row["y_new_hires_lag0"]),
        )
        panel = panel.copy()
        _ensure_derived_outcome(panel, "log1p_y_cst_foreign_lag0", "any_opt_hires_correction_aware")
        _ensure_derived_outcome(panel, "log1p_y_cst_native_lag0", "any_opt_hires_correction_aware")
        _ensure_derived_outcome(panel, "log1p_y_new_hires_foreign_lag0", "any_opt_hires_correction_aware")
        _ensure_derived_outcome(panel, "log1p_y_new_hires_native_lag0", "any_opt_hires_correction_aware")
        self.assertIn("log1p_y_cst_foreign_lag0", panel.columns)
        self.assertIn("log1p_y_cst_native_lag0", panel.columns)
        self.assertIn("log1p_y_new_hires_foreign_lag0", panel.columns)
        self.assertIn("log1p_y_new_hires_native_lag0", panel.columns)
        self.assertAlmostEqual(
            float(panel.set_index("c").loc[1, "log1p_y_cst_foreign_lag0"]),
            float(np.log1p(3.5)),
        )
        self.assertAlmostEqual(
            float(panel.set_index("c").loc[1, "log1p_y_new_hires_foreign_lag0"]),
            float(np.log1p(1.5)),
        )

    def test_build_company_features_includes_weighted_origin_split_columns(self) -> None:
        cfg = {
            "testing": {"enabled": False},
            "revelio_company_features": {
                "feature_year_min": 2010,
                "feature_year_max": 2010,
                "outside_negative_ratio": 1.0,
                "outside_negative_seed": 42,
                "outside_negative_min_n_users": 10,
                "min_position_days": 365,
                "tenure_min_days": 365,
                "wrds_workforce_include_education_features": True,
            },
        }
        firms = pd.DataFrame(
            {
                "c": [1],
                "in_analysis_universe": [1],
                "preferred_rcid_source": [1],
                "outside_negative_candidate": [0],
            }
        )
        selected_meta = pd.DataFrame(
            {
                "c": [1],
                "n_users": [100.0],
                "naics2": ["54"],
                "naics4": ["5415"],
                "company_state_feature": ["CA"],
                "company_metro_feature": ["san jose metropolitan area (california)"],
                "company_hq_region": ["west"],
                "year_founded": [2000.0],
            }
        )
        wrds_annual = pd.DataFrame(
            {
                "c": [1],
                "t": [2010],
                "total_headcount_wrds_annual": [10.0],
                "total_headcount_foreign_weighted_annual": [3.5],
                "total_headcount_native_weighted_annual": [6.5],
                "total_headcount_foreign_hard_annual": [4.0],
                "total_headcount_native_hard_annual": [6.0],
                "long_term_headcount_wrds_annual": [8.0],
                "n_new_hires_wrds_annual": [4.0],
                "n_new_hires_foreign_weighted_annual": [1.5],
                "n_new_hires_native_weighted_annual": [2.5],
                "n_new_hires_foreign_hard_annual": [2.0],
                "n_new_hires_native_hard_annual": [2.0],
                "salary_mean_annual": [100000.0],
                "salary_var_annual": [400.0],
                "total_comp_mean_annual": [120000.0],
                "total_comp_var_annual": [500.0],
                "compensation_missing_share_annual": [0.0],
                "nonus_educ_share_annual": [0.2],
                "age_share_lt30_annual": [0.3],
                "age_share_30_39_annual": [0.4],
                "age_share_40_49_annual": [0.2],
                "age_share_50_59_annual": [0.1],
                "age_share_60p_annual": [0.0],
                "female_share_annual": [0.45],
                "race_share_white_annual": [0.5],
                "race_share_black_annual": [0.1],
                "race_share_api_annual": [0.2],
                "race_share_hispanic_annual": [0.1],
                "race_share_native_annual": [0.05],
                "race_share_multiple_annual": [0.05],
                "seniority_mean_annual": [2.3],
                "avg_tenure_years_annual": [3.5],
                "occ_share_mgmt_annual": [0.1],
                "occ_share_business_finance_annual": [0.05],
                "occ_share_computing_math_annual": [0.35],
                "occ_share_engineering_annual": [0.2],
                "occ_share_science_annual": [0.05],
                "occ_share_community_legal_education_annual": [0.05],
                "occ_share_arts_media_annual": [0.02],
                "occ_share_healthcare_annual": [0.01],
                "occ_share_sales_office_annual": [0.07],
                "occ_share_manual_annual": [0.1],
                "in_analysis_universe": [1],
                "preferred_rcid_source": [1],
                "outside_negative_candidate": [0],
            }
        )
        opt_counts = pd.DataFrame(
            {
                "c": [1],
                "t": [2010],
                "bachelors_opt_hires_correction_aware": [0.0],
                "masters_opt_hires_correction_aware": [2.0],
                "phd_opt_hires_correction_aware": [0.0],
                "any_opt_hires_correction_aware": [2.0],
            }
        )

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            paths = SourceExposurePaths(
                foia_corrected=tmp / "foia.parquet",
                employer_crosswalk=tmp / "crosswalk.parquet",
                preferred_rcids=tmp / "preferred.parquet",
                company_mapping=tmp / "company_mapping.parquet",
                f1_inst_unitid_crosswalk=tmp / "inst.parquet",
                revelio_inst_crosswalk=tmp / "revelio_inst.parquet",
                revelio_inst_deterministic_map=None,
                revelio_ref_inst_catalog=None,
                source_opt_counts_out=tmp / "source_opt_counts.parquet",
                school_opt_benchmark_out=tmp / "school_opt_benchmark.parquet",
                opt_exposure_analysis_panel_out=tmp / "analysis_panel.parquet",
                outside_negative_sample_out=tmp / "outside_negative_sample.parquet",
                wrds_company_year_workforce_out=tmp / "wrds_company_year_workforce.parquet",
                wrds_school_flows_out=tmp / "wrds_school_flows.parquet",
            )
            school_map = pd.DataFrame(columns=["university_raw_key", "unitid"])
            school_map_meta = {
                "mapping_method": "test",
                "deterministic_triple_map": None,
                "ref_inst_catalog": None,
                "legacy_crosswalk": None,
            }
            universe_meta = {
                "analysis_universe_method": "preferred_plus_outside_sample_v2",
                "outside_negative_ratio": 1.0,
                "outside_negative_seed": 42,
                "outside_negative_min_n_users": 10,
            }

            with patch.object(revelio_company_features, "_resolve_feature_paths", return_value=revelio_company_features.FeaturePaths(
                company_mapping=tmp / "company_mapping.parquet",
                revelio_inst_crosswalk=tmp / "revelio_inst.parquet",
                company_features_out=tmp / "company_features.parquet",
                outside_negative_sample_out=tmp / "outside_negative_sample.parquet",
            )), patch.object(
                revelio_company_features,
                "resolve_source_exposure_paths",
                return_value=paths,
            ), patch.object(
                revelio_company_features,
                "load_revelio_school_map",
                return_value=(school_map, school_map_meta),
            ), patch.object(
                revelio_company_features,
                "load_or_build_source_firm_universe",
                return_value=(firms.copy(), pd.DataFrame(), selected_meta.copy(), dict(universe_meta)),
            ), patch.object(
                revelio_company_features,
                "load_or_build_source_school_opt_benchmark",
                return_value=(pd.DataFrame(columns=["unitid"]), {"school_benchmark_method": "test"}),
            ), patch.object(
                revelio_company_features,
                "load_or_build_source_opt_counts",
                return_value=(opt_counts.copy(), {"opt_count_method": source_exposure_data.OPT_COUNT_METHOD}),
            ), patch.object(
                revelio_company_features,
                "load_or_build_wrds_company_year_workforce_cache",
                return_value=(wrds_annual.copy(), {"new_hire_origin_method": source_exposure_data.NEW_HIRE_ORIGIN_METHOD}),
            ), patch.object(
                revelio_company_features,
                "load_or_build_wrds_school_flows_cache",
                return_value=(pd.DataFrame(columns=["c", "t", "university_raw"]), {}),
            ), patch.object(
                revelio_company_features,
                "_aggregate_school_features",
                return_value=pd.DataFrame(columns=["c", "t"]),
            ):
                feature_df = build_company_features(
                    cfg=cfg,
                    feature_year_min=2010,
                    feature_year_max=2010,
                    force_rebuild=True,
                )

            for col in (
                "total_headcount_foreign_weighted_annual_pre_level",
                "total_headcount_native_weighted_annual_pre_level",
                "n_new_hires_foreign_weighted_annual_pre_level",
                "n_new_hires_native_weighted_annual_pre_level",
            ):
                self.assertIn(col, feature_df.columns)
            row = feature_df.set_index("c").loc[1]
            self.assertAlmostEqual(float(row["total_headcount_foreign_weighted_annual_pre_level"]), 3.5)
            self.assertAlmostEqual(float(row["total_headcount_native_weighted_annual_pre_level"]), 6.5)
            self.assertAlmostEqual(float(row["n_new_hires_foreign_weighted_annual_pre_level"]), 1.5)
            self.assertAlmostEqual(float(row["n_new_hires_native_weighted_annual_pre_level"]), 2.5)
            meta = revelio_company_features._load_metadata(tmp / "company_features.parquet")
            self.assertEqual(meta["new_hire_origin_method"], source_exposure_data.NEW_HIRE_ORIGIN_METHOD)

    def test_build_wrds_company_year_workforce_splits_timed_out_batch(self) -> None:
        class _FakeDb:
            def close(self) -> None:
                return None

        called_batches: list[tuple[int, ...]] = []

        def _fake_query_builder(
            batch: list[int],
            *,
            year_min: int,
            year_max: int,
            user_prob_cols: list[str],
            include_seniority: bool,
            female_col: str | None,
            include_onet: bool,
            include_education_features: bool = True,
            history_year_min: int | None = None,
            history_year_max: int | None = None,
        ) -> str:
            return ",".join(str(v) for v in batch)

        def _fake_run_sql_with_retries(
            db: object,
            sql: str,
            *,
            wrds_username: str,
            query_timeout_ms: int | None,
            max_retries: int,
            label: str,
        ) -> tuple[pd.DataFrame, object]:
            batch = tuple(int(v) for v in sql.split(",") if v)
            called_batches.append(batch)
            if len(batch) == 4:
                raise Exception("canceling statement due to statement timeout")
            return (
                pd.DataFrame(
                    {
                        "c": list(batch),
                        "t": [2010] * len(batch),
                    }
                ),
                db,
            )

        with patch.object(source_exposure_data, "wrds", object()), patch.object(
            source_exposure_data,
            "_open_wrds_connection",
            return_value=_FakeDb(),
        ), patch.object(
            source_exposure_data,
            "_table_columns",
            side_effect=[
                {"seniority", "onet_code"},
                {"f_prob", "white_prob", "black_prob", "api_prob", "hispanic_prob", "native_prob", "multiple_prob"},
            ],
        ), patch.object(
            source_exposure_data,
            "_build_wrds_company_year_workforce_query",
            side_effect=_fake_query_builder,
        ), patch.object(
            source_exposure_data,
            "_run_sql_with_retries",
            side_effect=_fake_run_sql_with_retries,
        ):
            out = build_wrds_company_year_workforce(
                [1, 2, 3, 4],
                wrds_username="tester",
                year_min=2010,
                year_max=2022,
                rcid_batch_size=4,
                query_timeout_ms=600_000,
                query_max_retries=1,
            )

        self.assertEqual(called_batches[0], (1, 2, 3, 4))
        self.assertIn((1, 2), called_batches)
        self.assertIn((3, 4), called_batches)
        self.assertEqual(len(out), 4)
        self.assertEqual(sorted(out["c"].tolist()), [1, 2, 3, 4])

    def test_build_wrds_task_queue_pre_slices_large_firms(self) -> None:
        queue, meta = _build_wrds_task_queue(
            [1, 2, 218],
            label_prefix="workforce",
            rcid_batch_size=2,
            year_min=2010,
            year_max=2011,
            rcid_n_users=pd.DataFrame(
                {
                    "c": [1, 2, 218],
                    "n_users": [100.0, 200.0, 91_628.0],
                }
            ),
            large_firm_n_users_threshold=75_000,
            large_firm_year_span=1,
        )

        tasks = list(queue)
        self.assertEqual(meta["n_regular_batches"], 1)
        self.assertEqual(meta["n_large_firms"], 1)
        self.assertEqual(meta["n_large_firm_tasks"], 2)
        self.assertEqual(tasks[0].rcids, (1, 2))
        self.assertEqual(tasks[0].label, "workforce batch 1/1")
        self.assertEqual((tasks[0].year_min, tasks[0].year_max), (2010, 2011))
        self.assertEqual(tasks[1].rcids, (218,))
        self.assertEqual(tasks[1].label, "workforce large-firm rcid=218 y2010")
        self.assertEqual((tasks[1].year_min, tasks[1].year_max), (2010, 2010))
        self.assertEqual((tasks[1].history_year_min, tasks[1].history_year_max), (2010, 2010))
        self.assertEqual(tasks[2].rcids, (218,))
        self.assertEqual(tasks[2].label, "workforce large-firm rcid=218 y2011")
        self.assertEqual((tasks[2].year_min, tasks[2].year_max), (2011, 2011))
        self.assertEqual((tasks[2].history_year_min, tasks[2].history_year_max), (2010, 2011))

    def test_selected_positions_large_firm_span_uses_full_window(self) -> None:
        self.assertEqual(
            source_exposure_data._selected_positions_extract_large_firm_year_span(2010, 2015),
            6,
        )
        self.assertEqual(
            source_exposure_data._selected_positions_extract_large_firm_year_span(2015, 2015),
            1,
        )

    def test_build_wrds_company_year_workforce_retries_singleton_with_relaxed_timeout(self) -> None:
        class _FakeDb:
            def close(self) -> None:
                return None

        calls: list[tuple[tuple[int, ...], int | None, str, int | None, int | None]] = []

        def _fake_query_builder(
            batch: list[int],
            *,
            year_min: int,
            year_max: int,
            user_prob_cols: list[str],
            include_seniority: bool,
            female_col: str | None,
            include_onet: bool,
            include_education_features: bool = True,
            history_year_min: int | None = None,
            history_year_max: int | None = None,
        ) -> str:
            batch_sql = ",".join(str(v) for v in batch)
            hist_min = "" if history_year_min is None else str(history_year_min)
            hist_max = "" if history_year_max is None else str(history_year_max)
            return f"{batch_sql}|{year_min}|{year_max}|{hist_min}|{hist_max}"

        def _fake_run_sql_with_retries(
            db: object,
            sql: str,
            *,
            wrds_username: str,
            query_timeout_ms: int | None,
            max_retries: int,
            label: str,
        ) -> tuple[pd.DataFrame, object]:
            batch_sql, out_year_min, _out_year_max, hist_min, hist_max = sql.split("|")
            batch = tuple(int(v) for v in batch_sql.split(",") if v)
            calls.append(
                (
                    batch,
                    query_timeout_ms,
                    label,
                    int(hist_min) if hist_min else None,
                    int(hist_max) if hist_max else None,
                )
            )
            if batch == (218,) and query_timeout_ms == 600_000:
                raise Exception("canceling statement due to statement timeout")
            return (
                pd.DataFrame(
                    {
                        "c": [218],
                        "t": [int(out_year_min)],
                    }
                ),
                db,
            )

        with patch.object(source_exposure_data, "wrds", object()), patch.object(
            source_exposure_data,
            "_open_wrds_connection",
            return_value=_FakeDb(),
        ), patch.object(
            source_exposure_data,
            "_table_columns",
            side_effect=[
                {"seniority", "onet_code"},
                {"f_prob", "white_prob", "black_prob", "api_prob", "hispanic_prob", "native_prob", "multiple_prob"},
            ],
        ), patch.object(
            source_exposure_data,
            "_build_wrds_company_year_workforce_query",
            side_effect=_fake_query_builder,
        ), patch.object(
            source_exposure_data,
            "_run_sql_with_retries",
            side_effect=_fake_run_sql_with_retries,
        ):
            out = build_wrds_company_year_workforce(
                [218],
                wrds_username="tester",
                year_min=2010,
                year_max=2011,
                rcid_batch_size=1,
                query_timeout_ms=600_000,
                singleton_query_timeout_ms=3_600_000,
                query_max_retries=1,
            )

        self.assertEqual(calls[0], ((218,), 600_000, "workforce batch 1/1", 2010, 2011))
        self.assertEqual(
            calls[1],
            ((218,), 3_600_000, "workforce batch 1/1 singleton-fallback y2010", 2010, 2010),
        )
        self.assertEqual(
            calls[2],
            ((218,), 3_600_000, "workforce batch 1/1 singleton-fallback y2011", 2010, 2011),
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out["c"].tolist(), [218, 218])
        self.assertEqual(sorted(out["t"].tolist()), [2010, 2011])

    def test_build_wrds_company_year_workforce_query_can_skip_education_enrichment(self) -> None:
        sql = source_exposure_data._build_wrds_company_year_workforce_query(
            [218],
            year_min=2010,
            year_max=2011,
            user_prob_cols=["f_prob", "white_prob"],
            include_seniority=True,
            female_col="f_prob",
            include_onet=True,
            include_education_features=False,
        )

        self.assertIn("educ_raw_dedup AS MATERIALIZED", sql)
        self.assertIn("user_educ AS MATERIALIZED", sql)
        self.assertIn("user_origin AS MATERIALIZED", sql)
        self.assertIn("NULL::DOUBLE PRECISION AS nonus_educ_share_annual", sql)
        self.assertIn("NULL::DOUBLE PRECISION AS age_share_lt30_annual", sql)
        self.assertIn("NULL::DOUBLE PRECISION AS has_nonus_educ", sql)
        self.assertIn("NULL::INTEGER AS est_yob", sql)
        self.assertIn("total_headcount_foreign_weighted_annual", sql)
        self.assertIn("total_headcount_native_hard_annual", sql)
        self.assertIn("n_new_hires_foreign_weighted_annual", sql)
        self.assertIn("n_new_hires_native_hard_annual", sql)

    def test_build_wrds_company_year_workforce_query_base_only_skips_origin_enrichment(self) -> None:
        sql = source_exposure_data._build_wrds_company_year_workforce_query_base_only(
            [218],
            year_min=2010,
            year_max=2011,
            user_prob_cols=["f_prob", "white_prob"],
            include_seniority=True,
            female_col="f_prob",
            include_onet=True,
        )

        self.assertNotIn("user_educ AS MATERIALIZED", sql)
        self.assertNotIn("user_origin AS MATERIALIZED", sql)
        self.assertIn("NULL::DOUBLE PRECISION AS nonus_educ_share_annual", sql)
        self.assertIn("NULL::DOUBLE PRECISION AS total_headcount_foreign_weighted_annual", sql)
        self.assertIn("COUNT(DISTINCT user_id) AS n_new_hires_wrds_annual", sql)

    def test_build_wrds_school_flows_splits_timed_out_batch(self) -> None:
        class _FakeDb:
            def close(self) -> None:
                return None

        called_batches: list[tuple[int, ...]] = []

        def _fake_query_builder(
            batch: list[int],
            *,
            year_min: int,
            year_max: int,
            min_position_days: int,
            tenure_min_days: int,
            history_year_min: int | None = None,
            history_year_max: int | None = None,
        ) -> str:
            return ",".join(str(v) for v in batch)

        def _fake_run_sql_with_retries(
            db: object,
            sql: str,
            *,
            wrds_username: str,
            query_timeout_ms: int | None,
            max_retries: int,
            label: str,
        ) -> tuple[pd.DataFrame, object]:
            batch = tuple(int(v) for v in sql.split(",") if v)
            called_batches.append(batch)
            if len(batch) == 4:
                raise Exception("canceling statement due to statement timeout")
            return (
                pd.DataFrame(
                    {
                        "c": list(batch),
                        "t": [2010] * len(batch),
                        "university_raw": ["school"] * len(batch),
                    }
                ),
                db,
            )

        with patch.object(source_exposure_data, "wrds", object()), patch.object(
            source_exposure_data,
            "_open_wrds_connection",
            return_value=_FakeDb(),
        ), patch.object(
            source_exposure_data,
            "_build_wrds_school_flow_query",
            side_effect=_fake_query_builder,
        ), patch.object(
            source_exposure_data,
            "_run_sql_with_retries",
            side_effect=_fake_run_sql_with_retries,
        ):
            out = build_wrds_school_flows(
                [1, 2, 3, 4],
                wrds_username="tester",
                year_min=2010,
                year_max=2022,
                rcid_batch_size=4,
                query_timeout_ms=600_000,
                query_max_retries=1,
                min_position_days=365,
                tenure_min_days=365,
            )

        self.assertEqual(called_batches[0], (1, 2, 3, 4))
        self.assertIn((1, 2), called_batches)
        self.assertIn((3, 4), called_batches)
        self.assertEqual(len(out), 4)
        self.assertEqual(sorted(out["c"].tolist()), [1, 2, 3, 4])

    def test_build_wrds_school_flows_retries_singleton_with_relaxed_timeout(self) -> None:
        class _FakeDb:
            def close(self) -> None:
                return None

        calls: list[tuple[tuple[int, ...], int | None, str, int | None, int | None]] = []

        def _fake_query_builder(
            batch: list[int],
            *,
            year_min: int,
            year_max: int,
            min_position_days: int,
            tenure_min_days: int,
            history_year_min: int | None = None,
            history_year_max: int | None = None,
        ) -> str:
            batch_sql = ",".join(str(v) for v in batch)
            hist_min = "" if history_year_min is None else str(history_year_min)
            hist_max = "" if history_year_max is None else str(history_year_max)
            return f"{batch_sql}|{year_min}|{year_max}|{hist_min}|{hist_max}"

        def _fake_run_sql_with_retries(
            db: object,
            sql: str,
            *,
            wrds_username: str,
            query_timeout_ms: int | None,
            max_retries: int,
            label: str,
        ) -> tuple[pd.DataFrame, object]:
            batch_sql, out_year_min, _out_year_max, hist_min, hist_max = sql.split("|")
            batch = tuple(int(v) for v in batch_sql.split(",") if v)
            calls.append(
                (
                    batch,
                    query_timeout_ms,
                    label,
                    int(hist_min) if hist_min else None,
                    int(hist_max) if hist_max else None,
                )
            )
            if batch == (218,) and query_timeout_ms == 600_000:
                raise Exception("canceling statement due to statement timeout")
            return (
                pd.DataFrame(
                    {
                        "c": [218],
                        "t": [int(out_year_min)],
                        "university_raw": ["school"],
                    }
                ),
                db,
            )

        with patch.object(source_exposure_data, "wrds", object()), patch.object(
            source_exposure_data,
            "_open_wrds_connection",
            return_value=_FakeDb(),
        ), patch.object(
            source_exposure_data,
            "_build_wrds_school_flow_query",
            side_effect=_fake_query_builder,
        ), patch.object(
            source_exposure_data,
            "_run_sql_with_retries",
            side_effect=_fake_run_sql_with_retries,
        ):
            out = build_wrds_school_flows(
                [218],
                wrds_username="tester",
                year_min=2010,
                year_max=2011,
                rcid_batch_size=1,
                query_timeout_ms=600_000,
                singleton_query_timeout_ms=3_600_000,
                query_max_retries=1,
                min_position_days=365,
                tenure_min_days=365,
            )

        self.assertEqual(calls[0], ((218,), 600_000, "school-flow batch 1/1", 2010, 2011))
        self.assertEqual(
            calls[1],
            ((218,), 3_600_000, "school-flow batch 1/1 singleton-fallback y2010", 2010, 2010),
        )
        self.assertEqual(
            calls[2],
            ((218,), 3_600_000, "school-flow batch 1/1 singleton-fallback y2011", 2010, 2011),
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out["c"].tolist(), [218, 218])
        self.assertEqual(sorted(out["t"].tolist()), [2010, 2011])

    def test_resolve_run_log_path_applies_testing_suffix(self) -> None:
        cfg = {
            "testing": {"enabled": True, "output_suffix": "_testing"},
            "exposure_event_study": {"log_out_path": "{root}/tmp/exposure_run.log"},
        }
        out = _resolve_run_log_path(cfg, cfg["exposure_event_study"], cli_log_file=None)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertTrue(str(out).endswith("exposure_run_testing.log"))

    def test_resolve_run_log_path_expands_brace_config_variable(self) -> None:
        cfg = {
            "company_shift_share_out_dir": "/tmp/company_shift_share_out",
            "testing": {"enabled": True, "output_suffix": "_testing"},
            "exposure_event_study": {
                "log_out_path": "{company_shift_share_out_dir}/exposure_event_study_test_log.txt"
            },
        }
        out = _resolve_run_log_path(cfg, cfg["exposure_event_study"], cli_log_file=None)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(str(out), "/tmp/company_shift_share_out/exposure_event_study_test_log_testing.txt")

    def test_testing_verbose_defaults_true_and_respects_override(self) -> None:
        self.assertTrue(_testing_verbose({"testing": {}}))
        self.assertFalse(_testing_verbose({"testing": {"verbose": False}}))


if __name__ == "__main__":
    unittest.main()
