from __future__ import annotations

from pathlib import Path
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
    _fe_cols,
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

    def test_detect_largest_jump_events_zero_threshold_requires_positive_jump(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1, 2, 2],
                "t": [2013, 2014, 2013, 2014],
                "z_ct": [0.0, 0.0, 0.0, 1.0],
            }
        )
        events = detect_largest_jump_events(
            panel,
            exposure_col="z_ct",
            cohort_min_year=2013,
            cohort_max_year=2014,
            min_jump=0.0,
        ).set_index("c")
        self.assertEqual(int(events.loc[1, "treated"]), 0)
        self.assertEqual(int(events.loc[2, "treated"]), 1)

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

    def test_final_table_pairs_first_stage_and_foreign_new_hire_rf(self) -> None:
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
        self.assertEqual(float(out.loc[0, "foreign_new_hire_rf_coef"]), 0.2)

    def test_prepared_panel_summary_reports_shape_and_years(self) -> None:
        summary = build_prepared_panel_summary({"shift_share": _panel()})
        row = summary.iloc[0]
        self.assertEqual(int(row["rows"]), 6)
        self.assertEqual(int(row["n_units"]), 3)
        self.assertEqual(int(row["min_t"]), 2010)
        self.assertEqual(int(row["max_t"]), 2011)


if __name__ == "__main__":
    unittest.main()
