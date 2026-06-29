from __future__ import annotations

from pathlib import Path
import math
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from company_shift_share.design_comparison_suite import (
    ComparisonDataStore,
    _parse_args,
    aggregate_to_local_market,
    build_final_comparison_table,
    build_prepared_panel_summary,
    detect_largest_jump_events,
    filter_analysis_sample,
    build_comparison_stacked_panel,
    prepare_shift_share_state_panel,
    prepare_shift_share_dynamic_panel,
    prepare_comparison_panels,
    recompute_baseline_size_growth,
    _add_year_interactions,
    _calendar_year_dummy_reference,
    _event_ols_absorb_cols,
    _fe_cols,
    _filter_opt_index_analysis_sample,
    _first_stage_lhs,
    _first_stage_transform,
    _fit_event_conditional_ppml,
    _prefilter_ppml_first_stage_panel,
    _selected_designs,
    _stacked_cluster_col,
    _stacked_exposures_to_run,
    _stacked_fe_cols,
    _stacked_sun_abraham_terms,
    _transform_outcome,
    _x_col,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "c": [1, 1, 2, 2, 3, 3],
            "t": [2010, 2011, 2010, 2011, 2010, 2011],
            "any_opt_hires_correction_aware": [0.0, 1.0, 0.0, 0.0, 2.0, 5.0],
            "y_cst_lag0": [20.0, 22.0, 6.0, 7.0, 50.0, 55.0],
            "y_new_hires_foreign_lag0": [1.0, 2.0, 0.0, 0.0, 3.0, 4.0],
            "z_ct": [0.0, 1.0, 0.0, 0.1, 0.5, 3.0],
        }
    )


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "c": [1, 2, 3],
            "company_metro_feature": ["sf", "sf", "ny"],
            "naics2": ["51", "51", "54"],
            "firm_size_annual_pre_level": [20.0, 5.0, 50.0],
            "firm_size_annual_pre_growth": [1.0, 0.0, 2.0],
            "school_opt_share_new_hire_annual_pre_level": [0.4, 0.1, 0.8],
        }
    )


class DesignComparisonSuiteTests(unittest.TestCase):
    def test_parse_args_ignores_ipykernel_launcher_flags(self) -> None:
        with patch(
            "company_shift_share.design_comparison_suite.sys.argv",
            ["ipykernel_launcher.py", "--f=/run/user/224497/jupyter/runtime/kernel-test.json"],
        ), patch.dict(
            "company_shift_share.design_comparison_suite.sys.modules",
            {"ipykernel": object()},
            clear=False,
        ):
            parsed = _parse_args()
        self.assertFalse(parsed.force_rebuild_base)
        self.assertIsNone(parsed.unit)
        self.assertIsNone(parsed.first_stage_type)

    def test_parse_args_keeps_explicit_unknown_args_strict(self) -> None:
        with self.assertRaises(SystemExit):
            _parse_args(["--not-a-real-flag"])

    def test_selected_designs_allows_stacked_only(self) -> None:
        self.assertEqual(_selected_designs({"designs_to_run": ["stacked_did"]}), ["stacked_did"])
        self.assertEqual(_parse_args(["--designs", "stacked_did"]).designs, "stacked_did")

    def test_parse_args_keeps_stacked_exposures(self) -> None:
        self.assertEqual(
            _parse_args(["--stacked-exposures", "ihmp_share"]).stacked_exposures,
            "ihmp_share",
        )
        self.assertEqual(_stacked_exposures_to_run({"stacked_exposures_to_run": "opt_takeup"}), ["opt_takeup"])
        self.assertEqual(
            _stacked_exposures_to_run({"stacked_exposures_to_run": "international_share"}),
            ["international_share"],
        )

    def test_first_stage_column_override_prefers_configured_column(self) -> None:
        panel = pd.DataFrame(
            {
                "any_opt_hires_correction_aware": [10.0],
                "masters_opt_hires_correction_aware": [2.0],
            }
        )
        self.assertEqual(
            _x_col(panel, {"first_stage_col": "masters_opt_hires_correction_aware"}),
            "masters_opt_hires_correction_aware",
        )
        with self.assertRaises(ValueError):
            _x_col(panel, {"first_stage_col": "missing_first_stage"})

    def test_ihs_outcome_transform_allows_negative_count_values(self) -> None:
        out = _transform_outcome(
            pd.Series([-1.0, 0.0, 3.0]),
            "y_new_hires_foreign_minus_one_lag0",
            {"count_outcome_transform": "ihs", "use_log_outcome": True},
        )
        expected = pd.Series([-math.asinh(1.0), 0.0, math.asinh(3.0)])
        pd.testing.assert_series_equal(out.reset_index(drop=True), expected, check_names=False)

    def test_prepare_comparison_panels_reuses_expensive_loaders_once(self) -> None:
        with TemporaryDirectory() as tmpdir:
            shift_path = Path(tmpdir) / "shift.parquet"
            _panel().to_parquet(shift_path, index=False)
            cfg = {
                "paths": {"shift_share_analysis_panel": str(shift_path)},
                "design_comparison": {
                    "memory_cache": True,
                    "trust_existing_base_caches": False,
                    "unit": "firm",
                    "data_min_t": 2010,
                    "data_max_t": 2011,
                    "min_pre_avg_employment": 10,
                    "outcome_cols": ["y_cst_lag0", "y_new_hires_foreign_lag0"],
                },
            }
            opt_index = pd.DataFrame({"c": [1, 2, 3], "predicted_prob": [0.2, 0.4, 0.8]})
            with patch(
                "company_shift_share.design_comparison_suite.load_or_build_source_analysis_panel",
                return_value=(_panel(), {}),
            ) as source_loader, patch(
                "company_shift_share.design_comparison_suite.load_or_build_company_features",
                return_value=(_features(), {}),
            ) as feature_loader, patch.object(
                ComparisonDataStore,
                "_load_opt_probability_index",
                return_value=opt_index,
            ) as index_loader:
                store = ComparisonDataStore(cfg=cfg, memory_cache=True)
                first = prepare_comparison_panels(store, cfg["design_comparison"])
                second = prepare_comparison_panels(store, cfg["design_comparison"])

            self.assertEqual(source_loader.call_count, 1)
            self.assertEqual(feature_loader.call_count, 1)
            self.assertEqual(index_loader.call_count, 1)
            self.assertIn("predicted_prob", first["event_study"].columns)
            self.assertEqual(len(first["event_study"]), len(second["event_study"]))
            self.assertIn("source_analysis_panel", store.manifest["stages"])
            self.assertTrue(store.manifest["stages"]["source_analysis_panel"]["reused_memory"])

    def test_base_loaders_trust_existing_parquet_without_load_or_build(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / "source_panel.parquet"
            features_path = tmp / "features.parquet"
            _panel().to_parquet(source_path, index=False)
            _features().to_parquet(features_path, index=False)
            cfg = {
                "paths": {},
                "design_comparison": {
                    "source_config_path": str(tmp / "source.yaml"),
                    "force_rebuild_base": False,
                    "trust_existing_base_caches": True,
                },
            }
            source_cfg = {
                "paths": {
                    "opt_exposure_analysis_panel_out": str(source_path),
                    "company_features_out": str(features_path),
                },
                "exposure_event_study": {},
            }
            with patch(
                "company_shift_share.design_comparison_suite.load_config",
                return_value=source_cfg,
            ), patch(
                "company_shift_share.design_comparison_suite.load_or_build_source_analysis_panel",
                side_effect=AssertionError("source panel should be read from parquet"),
            ), patch(
                "company_shift_share.design_comparison_suite.load_or_build_company_features",
                side_effect=AssertionError("features should be read from parquet"),
            ):
                store = ComparisonDataStore(cfg=cfg, memory_cache=True)
                source = store.source_analysis_panel()
                features = store.company_features()

            self.assertEqual(len(source), len(_panel()))
            self.assertEqual(len(features), len(_features()))

    def test_filter_analysis_sample_applies_pre_period_size_cutoff(self) -> None:
        panel = _panel().merge(_features(), on="c", how="left")
        out = filter_analysis_sample(
            panel,
            {
                "data_min_t": 2010,
                "data_max_t": 2011,
                "min_pre_avg_employment": 10,
            },
        )
        self.assertEqual(set(out["c"].unique()), {1, 3})

    def test_filter_analysis_sample_excludes_outside_negative_firms_by_default(self) -> None:
        panel = _panel().merge(_features(), on="c", how="left")
        panel["outside_negative_candidate"] = panel["c"].eq(3).astype(int)
        out = filter_analysis_sample(
            panel,
            {
                "data_min_t": 2010,
                "data_max_t": 2011,
                "min_pre_avg_employment": 0,
            },
        )
        self.assertNotIn(3, set(out["c"].unique()))
        retained = filter_analysis_sample(
            panel,
            {
                "data_min_t": 2010,
                "data_max_t": 2011,
                "min_pre_avg_employment": 0,
                "exclude_outside_negative_firms": False,
            },
        )
        self.assertIn(3, set(retained["c"].unique()))

    def test_filter_opt_index_analysis_sample_respects_leaveout_and_outside_negative_flags(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 2, 2, 3, 3],
                "t": [2010, 2011] * 3,
                "event_study_sample": [1, 1, 0, 0, 1, 1],
                "outside_negative_candidate": [0, 0, 0, 0, 1, 1],
                "predicted_prob": [0.2, 0.2, 0.4, 0.4, 0.1, 0.1],
            }
        )
        out = _filter_opt_index_analysis_sample(
            panel,
            {"exclude_outside_negative_firms": True, "verbose": False},
        )
        self.assertEqual(set(out["c"].unique()), {1})
        retained = _filter_opt_index_analysis_sample(
            panel,
            {"exclude_outside_negative_firms": False, "verbose": False},
        )
        self.assertEqual(set(retained["c"].unique()), {1, 3})

    def test_filter_opt_index_analysis_sample_keeps_fractional_local_market_flags(self) -> None:
        panel = pd.DataFrame(
            {
                "c": ["sf", "sf", "ny", "ny"],
                "t": [2010, 2011] * 2,
                "event_study_sample": [0.75, 0.75, 0.0, 0.0],
                "predicted_prob": [0.2, 0.2, 0.4, 0.4],
            }
        )
        out = _filter_opt_index_analysis_sample(
            panel,
            {"unit": "local_market", "exclude_outside_negative_firms": False, "verbose": False},
        )
        self.assertEqual(set(out["c"].unique()), {"sf"})

    def test_recompute_baseline_size_growth_overwrites_feature_cache_values(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 1, 2, 2, 2],
                "t": [2010, 2011, 2015, 2010, 2011, 2015],
                "y_cst_lag0": [10.0, 20.0, 1000.0, 4.0, 6.0, 1000.0],
                "firm_size_annual_pre_level": [999.0] * 6,
                "firm_size_annual_pre_growth": [999.0] * 6,
            }
        )
        out = recompute_baseline_size_growth(
            panel,
            {"baseline_start": 2010, "baseline_end": 2011},
        )
        levels = out.drop_duplicates("c").set_index("c")["firm_size_annual_pre_level"]
        self.assertAlmostEqual(float(levels.loc[1]), 15.0)
        self.assertAlmostEqual(float(levels.loc[2]), 5.0)
        self.assertTrue((out["firm_size_annual_pre_growth"] != 999.0).all())

    def test_prepare_shift_share_dynamic_panel_matches_start_and_balanced_policy(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 1, 2, 2, 3, 3, 3],
                "t": [2012, 2013, 2014, 2013, 2014, 2012, 2013, 2014],
                "z_ct": [0.0] * 8,
            }
        )
        out = prepare_shift_share_dynamic_panel(
            panel,
            {
                "shift_share_start_t": 2013,
                "data_max_t": 2014,
                "shift_share_enforce_balanced_panel": True,
                "verbose": False,
            },
        )
        self.assertEqual(set(out["t"].unique()), {2013, 2014})
        self.assertEqual(set(out["c"].unique()), {1, 2, 3})
        unbalanced = panel.loc[~((panel["c"].eq(2)) & (panel["t"].eq(2014)))].copy()
        out2 = prepare_shift_share_dynamic_panel(
            unbalanced,
            {
                "shift_share_start_t": 2013,
                "data_max_t": 2014,
                "shift_share_enforce_balanced_panel": True,
                "verbose": False,
            },
        )
        self.assertNotIn(2, set(out2["c"].unique()))

    def test_prepare_shift_share_state_panel_uses_original_helper(self) -> None:
        panel = _panel()
        with patch(
            "company_shift_share.design_comparison_suite.ssa._prepare_first_stage_state_panel",
            return_value=panel.assign(baseline_size_growth_year_fe="called"),
        ) as helper:
            out = prepare_shift_share_state_panel(
                panel,
                {
                    "shift_share_fe_baseline_start": 2008,
                    "shift_share_fe_baseline_end": 2013,
                    "baseline_size_bins": 10,
                    "baseline_growth_bins": 5,
                    "joint_size_growth_bins": 3,
                    "verbose": False,
                },
            )
        helper.assert_called_once()
        self.assertIn("baseline_size_growth_year_fe", out.columns)
        self.assertEqual(set(out["baseline_size_growth_year_fe"]), {"called"})

    def test_exclude_size_year_fe_controls_standard_fe_list(self) -> None:
        panel = _panel().assign(baseline_size_growth_year_fe="cell")
        self.assertEqual(_fe_cols(panel, {"exclude_size_year_fe": True}), ["c", "t"])
        self.assertEqual(
            _fe_cols(panel, {"exclude_size_year_fe": False}),
            ["c", "t", "baseline_size_growth_year_fe"],
        )

    def test_ihs_first_stage_uses_arcsinh_opt_hires(self) -> None:
        work = pd.DataFrame({"x": [0.0, 1.0, 5.0]})
        lhs, estimator = _first_stage_lhs(work, work["x"], {"first_stage_type": "ols_ihs"})
        self.assertEqual(lhs, "first_stage_lhs_ols_ihs")
        self.assertEqual(estimator, "ols")
        self.assertEqual(work[lhs].tolist(), [math.asinh(0.0), math.asinh(1.0), math.asinh(5.0)])
        self.assertEqual(_first_stage_transform(work["x"], {"first_stage_type": "ols_binary"}).tolist(), [0.0, 1.0, 1.0])

    def test_event_study_omits_only_reference_treatment_interaction(self) -> None:
        panel = pd.DataFrame({"t": [2013, 2014, 2015], "z": [1.0, 1.0, 1.0]})
        terms = _add_year_interactions(panel, "z", [2013, 2014, 2015], ref_year=2014)
        self.assertEqual(terms, [(2013, "z_x_year_2013"), (2015, "z_x_year_2015")])
        self.assertIn("t", panel.columns)
        self.assertNotIn("z_x_year_2014", panel.columns)

    def test_event_study_residualized_ols_keeps_calendar_year_fe_with_size_year_fe(self) -> None:
        panel = _panel().assign(baseline_size_growth_year_fe="cell")
        fe_cols = ["c", "t", "baseline_size_growth_year_fe"]
        self.assertEqual(_event_ols_absorb_cols(fe_cols, panel, {}), fe_cols)

    def test_calendar_year_dummy_reference_is_not_treatment_reference_year(self) -> None:
        self.assertNotEqual(_calendar_year_dummy_reference([2010, 2014, 2015], 2014), 2014)
        self.assertEqual(_calendar_year_dummy_reference([2014], 2014), 2014)

    def test_stacked_sun_abraham_uses_pooled_calendar_year_fe(self) -> None:
        panel = _panel().assign(
            unit_stack_fe="u",
            year_stack_fe="y",
            baseline_size_growth_year_fe="cell",
        )
        self.assertEqual(
            _stacked_fe_cols(panel, {"exclude_size_year_fe": True}, "sun_abraham"),
            ["c", "t"],
        )
        self.assertEqual(
            _stacked_fe_cols(panel, {"exclude_size_year_fe": False}, "sun_abraham"),
            ["c", "t"],
        )
        self.assertEqual(_stacked_cluster_col("sun_abraham", {}), "c")

    def test_stacked_sun_abraham_terms_are_cohort_specific(self) -> None:
        panel = pd.DataFrame(
            {
                "treated": [1, 1, 1, 1, 0, 0],
                "g": [2013, 2013, 2014, 2014, 2013, 2014],
                "rel_time": [-1, 0, -1, 0, 0, 0],
            }
        )
        term_to_time, term_meta = _stacked_sun_abraham_terms(panel, ref_event_time=-1)
        self.assertEqual(set(term_to_time.values()), {0})
        self.assertIn("sa_g2013_rt0", term_meta)
        self.assertIn("sa_g2014_rt0", term_meta)
        self.assertNotIn("stack_treated_p0", term_meta)
        self.assertEqual(float(panel["sa_g2013_rt0"].sum()), 1.0)
        self.assertEqual(float(panel["sa_g2014_rt0"].sum()), 1.0)

    def test_aggregate_to_local_market_sums_count_like_columns(self) -> None:
        panel = _panel().merge(_features(), on="c", how="left")
        out = aggregate_to_local_market(panel, {"local_market_col": "company_metro_feature"})
        sf_2010 = out.loc[out["c"].eq("sf") & out["t"].eq(2010)].iloc[0]
        self.assertEqual(float(sf_2010["y_cst_lag0"]), 26.0)
        self.assertAlmostEqual(float(sf_2010["school_opt_share_new_hire_annual_pre_level"]), 0.25)

    def test_detect_largest_jump_events_uses_largest_not_first_jump(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 1, 1, 2, 2, 2, 2],
                "t": [2012, 2013, 2014, 2015, 2012, 2013, 2014, 2015],
                "z_ct": [0.0, 1.0, 1.5, 5.0, 0.0, 0.1, 0.2, 0.3],
            }
        )
        events = detect_largest_jump_events(
            panel,
            exposure_col="z_ct",
            cohort_min_year=2013,
            cohort_max_year=2015,
            min_jump=1.0,
        ).set_index("c")
        self.assertEqual(int(events.loc[1, "g"]), 2015)
        self.assertEqual(int(events.loc[1, "treated"]), 1)
        self.assertEqual(int(events.loc[2, "treated"]), 0)

    def test_detect_largest_jump_events_uses_treated_jump_percentile(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 2, 2, 3, 3, 4, 4],
                "t": [2013, 2014] * 4,
                "z_ct": [0.0, 0.0, 0.0, 0.2, 0.0, 0.4, 0.0, 1.0],
            }
        )
        events = detect_largest_jump_events(
            panel,
            exposure_col="z_ct",
            cohort_min_year=2013,
            cohort_max_year=2014,
            min_jump=0.0,
            treated_jump_percentile=75,
        ).set_index("c")
        self.assertEqual(int(events.loc[1, "treated"]), 0)
        self.assertEqual(set(events.loc[events["treated"].eq(1)].index), {4})
        self.assertAlmostEqual(float(events["treated_jump_threshold"].iloc[0]), 0.55)

    def test_detect_largest_jump_events_quantile_threshold_expands_controls(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 2, 2, 3, 3, 4, 4],
                "t": [2013, 2014] * 4,
                "z_ct": [0.2, 0.3, 0.4, 0.2, 0.5, 0.1, 0.0, 1.0],
            }
        )
        events = detect_largest_jump_events(
            panel,
            exposure_col="z_ct",
            cohort_min_year=2013,
            cohort_max_year=2014,
            min_jump=0.0,
            treated_jump_percentile=75,
            control_min_jump_percentile=25,
        ).set_index("c")
        self.assertEqual(set(events.loc[events["treated"].eq(1)].index), {4})
        self.assertEqual(set(events.loc[events["control_eligible"], :].index), {3})
        self.assertAlmostEqual(float(events["treated_jump_threshold"].iloc[0]), 0.325)
        self.assertAlmostEqual(float(events["control_min_jump_threshold"].iloc[0]), -0.25)

    def test_detect_largest_jump_events_drops_units_with_missing_exposure_in_cohort_window(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 2, 2, 3, 3],
                "t": [2013, 2014] * 3,
                "z_ct": [0.0, 1.0, None, None, 0.0, 0.0],
            }
        )
        events = detect_largest_jump_events(
            panel,
            exposure_col="z_ct",
            cohort_min_year=2013,
            cohort_max_year=2014,
            min_jump=0.0,
            treated_jump_percentile=75,
            control_min_jump_percentile=25,
        )
        self.assertEqual(set(events["c"]), {1, 3})
        self.assertEqual(set(events.loc[events["control_eligible"], "c"]), {3})

    def test_stacked_panel_pair_frame_does_not_collide_with_cohort_g(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 1, 2, 2, 2],
                "t": [2012, 2013, 2014, 2012, 2013, 2014],
                "z_ct": [0.0, 1.0, 2.0, 0.0, 0.0, 0.0],
                "any_opt_hires_correction_aware": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "y_cst_lag0": [10.0, 11.0, 12.0, 20.0, 20.0, 20.0],
            }
        )
        event_df = pd.DataFrame(
            {
                "c": [1, 2],
                "g": [2013.0, float("nan")],
                "largest_jump": [1.0, 0.0],
                "treated": [1, 0],
            }
        )
        stacked = build_comparison_stacked_panel(
            panel,
            event_df,
            {"stacked_pre_years": 1, "stacked_post_years": 1},
            matching_style="unmatched",
        )
        self.assertFalse(stacked.empty)
        self.assertIn("g", stacked.columns)

    def test_stacked_panel_accepts_local_market_string_ids(self) -> None:
        panel = pd.DataFrame(
            {
                "c": ["sf", "sf", "sf", "ny", "ny", "ny"],
                "t": [2012, 2013, 2014, 2012, 2013, 2014],
                "z_ct": [0.0, 1.0, 2.0, 0.0, 0.0, 0.0],
                "any_opt_hires_correction_aware": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "y_cst_lag0": [10.0, 11.0, 12.0, 20.0, 20.0, 20.0],
                "firm_size_annual_pre_level": [11.0, 11.0, 11.0, 20.0, 20.0, 20.0],
                "firm_size_annual_pre_growth": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "naics2": ["54", "54", "54", "54", "54", "54"],
                "company_metro_feature": ["bay", "bay", "bay", "bay", "bay", "bay"],
            }
        )
        event_df = pd.DataFrame(
            {
                "c": ["sf", "ny"],
                "g": [2013.0, float("nan")],
                "largest_jump": [1.0, 0.0],
                "treated": [1, 0],
            }
        )
        stacked = build_comparison_stacked_panel(
            panel,
            event_df,
            {
                "stacked_pre_years": 1,
                "stacked_post_years": 1,
                "baseline_size_bins": 1,
                "verbose": False,
            },
            matching_style="matched",
        )
        self.assertFalse(stacked.empty)
        self.assertEqual(set(stacked["c"].unique()), {"sf", "ny"})

    def test_stacked_panel_old_positive_z_control_filter_is_not_applied(self) -> None:
        panel = pd.DataFrame(
            {
                "c": ["sf", "sf", "sf", "ny", "ny", "ny"],
                "t": [2012, 2013, 2014, 2012, 2013, 2014],
                "z_ct": [0.0, 1.0, 2.0, 0.1, 0.1, 0.1],
                "any_opt_hires_correction_aware": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "y_cst_lag0": [10.0, 11.0, 12.0, 20.0, 20.0, 20.0],
                "firm_size_annual_pre_level": [10.0, 10.0, 10.0, 11.0, 11.0, 11.0],
                "firm_size_annual_pre_growth": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "naics2": ["54", "54", "54", "54", "54", "54"],
                "company_metro_feature": ["bay", "bay", "bay", "bay", "bay", "bay"],
            }
        )
        event_df = pd.DataFrame(
            {
                "c": ["sf", "ny"],
                "g": [2013.0, float("nan")],
                "largest_jump": [1.0, 0.0],
                "treated": [1, 0],
            }
        )
        strict = build_comparison_stacked_panel(
            panel,
            event_df,
            {
                "stacked_pre_years": 1,
                "stacked_post_years": 1,
                "baseline_size_bins": 1,
                "stacked_require_clean_controls": True,
                "verbose": False,
            },
            matching_style="matched",
        )
        self.assertFalse(strict.empty)

    def test_stacked_panel_strict_exact_match_drops_outside_cell_controls(self) -> None:
        panel = pd.DataFrame(
            {
                "c": ["treated", "treated", "treated", "control", "control", "control"],
                "t": [2012, 2013, 2014, 2012, 2013, 2014],
                "z_ct": [0.0] * 6,
                "any_opt_hires_correction_aware": [0.0] * 6,
                "y_cst_lag0": [10.0, 11.0, 12.0, 20.0, 20.0, 20.0],
                "firm_size_annual_pre_level": [10.0] * 6,
                "firm_size_annual_pre_growth": [0.0] * 6,
                "naics2": ["54", "54", "54", "62", "62", "62"],
                "company_metro_feature": ["bay", "bay", "bay", "bay", "bay", "bay"],
            }
        )
        event_df = pd.DataFrame(
            {
                "c": ["treated", "control"],
                "g": [2013.0, float("nan")],
                "largest_jump": [1.0, 0.0],
                "treated": [1, 0],
                "control_eligible": [False, True],
            }
        )
        stacked = build_comparison_stacked_panel(
            panel,
            event_df,
            {
                "stacked_pre_years": 1,
                "stacked_post_years": 1,
                "baseline_size_bins": 1,
                "verbose": False,
            },
            matching_style="matched",
        )
        self.assertTrue(stacked.empty)

    def test_stacked_panel_drops_pairs_with_missing_exposure_in_stack_window(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 1, 2, 2, 2],
                "t": [2012, 2013, 2014, 2012, 2013, 2014],
                "z_ct": [0.0, 1.0, 2.0, 0.0, None, 0.0],
                "any_opt_hires_correction_aware": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "y_cst_lag0": [10.0, 11.0, 12.0, 20.0, 20.0, 20.0],
            }
        )
        event_df = pd.DataFrame(
            {
                "c": [1, 2],
                "g": [2013.0, float("nan")],
                "largest_jump": [1.0, 0.0],
                "treated": [1, 0],
            }
        )
        retained_without_check = build_comparison_stacked_panel(
            panel,
            event_df,
            {"stacked_pre_years": 1, "stacked_post_years": 1},
            matching_style="unmatched",
        )
        dropped_with_check = build_comparison_stacked_panel(
            panel,
            event_df,
            {"stacked_pre_years": 1, "stacked_post_years": 1},
            matching_style="unmatched",
            exposure_col="z_ct",
        )
        self.assertFalse(retained_without_check.empty)
        self.assertTrue(dropped_with_check.empty)

    def test_final_table_reports_first_stage_only(self) -> None:
        coef_df = pd.DataFrame(
            {
                "design": ["shift_share", "shift_share", "shift_share"],
                "spec": ["ihmp", "ihmp", "ihmp"],
                "family": ["first_stage", "reduced_form", "reduced_form"],
                "outcome_col": [
                    "any_opt_hires_correction_aware",
                    "y_new_hires_foreign_lag0",
                    "y_cst_lag0",
                ],
                "horizon": [0, 0, 0],
                "coef": [0.5, 0.2, 0.9],
                "se": [0.1, 0.05, 0.2],
                "f_stat": [25.0, 16.0, 20.25],
                "n_obs": [100, 100, 100],
                "estimator": ["ppml", "ols", "ols"],
            }
        )
        out = build_final_comparison_table(coef_df)
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out.loc[0, "first_stage_coef"]), 0.5)
        self.assertNotIn("foreign_new_hire_rf_coef", out.columns)
        self.assertIn("f_stat", out.columns)

    def test_final_table_uses_horizon_when_event_time_is_missing(self) -> None:
        coef_df = pd.DataFrame(
            {
                "design": ["shift_share", "shift_share", "shift_share"],
                "spec": ["ihmp", "ihmp", "ihmp"],
                "family": ["first_stage", "first_stage", "reduced_form"],
                "outcome_col": [
                    "any_opt_hires_correction_aware",
                    "any_opt_hires_correction_aware",
                    "y_new_hires_foreign_lag0",
                ],
                "event_time": [float("nan"), float("nan"), float("nan")],
                "horizon": [-4, 0, 0],
                "coef": [0.1, 0.5, 0.2],
                "se": [0.1, 0.1, 0.05],
                "f_stat": [1.0, 25.0, 16.0],
                "n_obs": [100, 100, 100],
                "estimator": ["ols", "ols", "ols"],
            }
        )
        out = build_final_comparison_table(coef_df)
        self.assertEqual(float(out.loc[0, "first_stage_coef"]), 0.5)
        self.assertEqual(float(out.loc[0, "f_stat"]), 25.0)

    def test_fast_event_conditional_ppml_drops_all_zero_units(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "t": [2013, 2014, 2015] * 3,
                "x": [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 3.0],
                "z_x_year_2013": [0.2, 0.0, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0, 0.0],
                "z_x_year_2015": [0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 0.0, 0.8],
            }
        )
        fit = _fit_event_conditional_ppml(
            panel,
            "x",
            ["z_x_year_2013", "z_x_year_2015"],
            {"z_x_year_2013": 2013, "z_x_year_2015": 2015},
            {"verbose": False},
            "c",
        )
        self.assertEqual(fit.nobs, 6)
        self.assertIn("z_x_year_2013", fit.params.index)
        self.assertIn("z_x_year_2015", fit.std_errors.index)
        self.assertTrue(pd.notna(fit.params["z_x_year_2013"]))

    def test_ppml_first_stage_prefilter_drops_all_zero_fe_groups(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 2, 2, 3, 3],
                "t": [2010, 2011, 2010, 2011, 2010, 2011],
                "x": [0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
                "z": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )
        out, note = _prefilter_ppml_first_stage_panel(
            panel,
            lhs="x",
            estimator="ppml",
            family="first_stage",
            fe_cols=["c", "t"],
            cmp_cfg={"verbose": False},
            context="test",
        )
        self.assertEqual(set(out["c"].unique()), {1, 3})
        self.assertIsNotNone(note)

    def test_prepared_panel_summary_reports_shape_and_years(self) -> None:
        summary = build_prepared_panel_summary({"shift_share": _panel()})
        row = summary.iloc[0]
        self.assertEqual(int(row["rows"]), 6)
        self.assertEqual(int(row["n_units"]), 3)
        self.assertEqual(int(row["min_t"]), 2010)
        self.assertEqual(int(row["max_t"]), 2011)


if __name__ == "__main__":
    unittest.main()
