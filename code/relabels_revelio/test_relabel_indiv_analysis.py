from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "relabels_revelio" / "relabel_indiv_analysis.py"
spec = importlib.util.spec_from_file_location(
    "relabel_indiv_analysis_test_module",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class RelabelIndivAnalysisTests(unittest.TestCase):
    def _write_fixture_inputs(self, tempdir: Path) -> dict[str, object]:
        stage04 = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "education_number": 10,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450601",
                    "university_raw": "Treated University",
                    "field_clean": "economics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2020-05-15",
                    "school_match_score": 0.97,
                },
                {
                    "user_id": 1,
                    "education_number": 11,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Treated University",
                    "field_clean": "econometrics",
                    "ed_startdate": "2017-09-01",
                    "ed_enddate": "2019-05-15",
                    "school_match_score": 0.96,
                },
                {
                    "user_id": 2,
                    "education_number": 20,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "520201",
                    "university_raw": "Treated University",
                    "field_clean": "business",
                    "ed_startdate": "2019-09-01",
                    "ed_enddate": "2021-05-15",
                    "school_match_score": 0.95,
                },
                {
                    "user_id": 3,
                    "education_number": 30,
                    "unitid": None,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Unknown University",
                    "field_clean": "economics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2020-05-15",
                    "school_match_score": 0.80,
                },
                {
                    "user_id": 4,
                    "education_number": 40,
                    "unitid": 2001,
                    "degree_clean": "MBA",
                    "cip": "45.0603",
                    "university_raw": "Control University",
                    "field_clean": "economics",
                    "ed_startdate": "2017-09-01",
                    "ed_enddate": None,
                    "school_match_score": 0.91,
                },
                {
                    "user_id": 5,
                    "education_number": 50,
                    "unitid": 2001,
                    "degree_clean": "Doctor",
                    "cip": "450699",
                    "university_raw": "Control University",
                    "field_clean": "economics",
                    "ed_startdate": "2016-09-01",
                    "ed_enddate": None,
                    "school_match_score": 0.89,
                },
                {
                    "user_id": 6,
                    "education_number": 60,
                    "unitid": 1001,
                    "degree_clean": "bachelors",
                    "cip": "450699",
                    "university_raw": "Treated University",
                    "field_clean": "economics",
                    "ed_startdate": "2013-09-01",
                    "ed_enddate": None,
                    "school_match_score": 0.88,
                },
                {
                    "user_id": 7,
                    "education_number": 70,
                    "unitid": 2001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Control University",
                    "field_clean": "economics",
                    "ed_startdate": None,
                    "ed_enddate": None,
                    "school_match_score": 0.87,
                },
                {
                    "user_id": 8,
                    "education_number": 80,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Treated University",
                    "field_clean": "econometrics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2019-05-15",
                    "school_match_score": 0.94,
                },
                {
                    "user_id": 8,
                    "education_number": 81,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Treated University",
                    "field_clean": "econometrics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2019-12-31",
                    "school_match_score": 0.94,
                },
                {
                    "user_id": 9,
                    "education_number": 90,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Treated University",
                    "field_clean": "econometrics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2019-06-01",
                    "school_match_score": 0.93,
                },
                {
                    "user_id": 9,
                    "education_number": 91,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Treated University",
                    "field_clean": "econometrics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2019-06-01",
                    "school_match_score": 0.93,
                },
            ]
        )
        stage04_path = tempdir / "rev_match_ready.parquet"
        stage04.to_parquet(stage04_path, index=False)

        stage05 = pd.DataFrame(
            [
                {"person_id": 100, "user_id": 1, "person_match_rank": 1},
                {"person_id": 101, "user_id": 4, "person_match_rank": 1},
                {"person_id": 102, "user_id": 5, "person_match_rank": 2},
                {"person_id": 103, "user_id": 999, "person_match_rank": 1},
            ]
        )
        stage05_path = tempdir / "person_baseline.parquet"
        stage05.to_parquet(stage05_path, index=False)

        positions = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "rcid": 9001,
                    "country": "United States",
                    "startdate": "2020-07-01",
                    "enddate": "2022-06-30",
                    "total_compensation": 120000,
                },
                {
                    "user_id": 4,
                    "rcid": 9004,
                    "country": "United States",
                    "startdate": "2019-07-01",
                    "enddate": "2021-06-30",
                    "total_compensation": 95000,
                },
                {
                    "user_id": 5,
                    "rcid": 9005,
                    "country": "United States",
                    "startdate": "2020-07-01",
                    "enddate": "2022-06-30",
                    "total_compensation": 105000,
                },
            ]
        )
        positions_path = tempdir / "positions.parquet"
        positions.to_parquet(positions_path, index=False)

        relabel_df = pd.DataFrame(
            [
                {
                    "unitid": 1001,
                    "relabel_year": 2020,
                    "relabel_type": "econ_to_econometrics",
                    "event_flag": 1,
                }
            ]
        )
        matched_pairs = pd.DataFrame(
            [
                {
                    "relabel_type": "econ_to_econometrics",
                    "relabel_year": 2020,
                    "treated_unitid": 1001,
                    "control_unitid": 2001,
                }
            ]
        )
        output_panel_path = tempdir / "relabel_panel.parquet"
        output_did_path = tempdir / "relabel_did.parquet"
        output_dir = tempdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "stage04_path": stage04_path,
            "stage05_path": stage05_path,
            "positions_path": positions_path,
            "relabel_df": relabel_df,
            "matched_pairs": matched_pairs,
            "output_panel_path": output_panel_path,
            "output_did_path": output_did_path,
            "output_dir": output_dir,
        }

    def _patch_runtime(
        self,
        fixtures: dict[str, object],
        *,
        event_window: int = 1,
        outcome_horizons: tuple[int, ...] = (1,),
    ) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            patch.object(analysis.cfg, "STAGE04_MERGE_READY_PARQUET", str(fixtures["stage04_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "STAGE05_PERSON_BASELINE_PARQUET", str(fixtures["stage05_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "REV_POS_PARQUET", str(fixtures["positions_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "OUTPUT_PANEL_PARQUET", str(fixtures["output_panel_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "OUTPUT_DID_RESULTS_PARQUET", str(fixtures["output_did_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_SAMPLE_CIP_PREFIXES", ["4506"])
        )
        stack.enter_context(
            patch.object(
                analysis.cfg,
                "BUILD_SAMPLE_VARIANTS",
                ["stage04_all", "foia_linked_person_baseline"],
            )
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_SAMPLE_GRADYEAR_WINDOW", 5)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_EVENT_WINDOW", event_window)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_OUTCOME_HORIZONS", list(outcome_horizons))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_CAP_TO_LATEST_AVAILABLE_YEAR", True)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_RUN_DID", False)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "TESTING_ENABLED", False)
        )
        stack.enter_context(
            patch.object(analysis, "OUTPUT_DIR", fixtures["output_dir"])
        )
        stack.enter_context(
            patch.object(analysis.plt, "show", lambda: None)
        )
        return stack

    def test_step2_prepare_stage04_samples_filters_and_derives_grad_year(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures):
                con = duckdb.connect()
                variants = analysis.step2_prepare_stage04_samples(con)
                self.assertEqual(
                    variants,
                    ["stage04_all", "foia_linked_person_baseline"],
                )
                sample_all = con.sql(
                    """
                    SELECT user_id, unitid, education_number, grad_year
                    FROM stage04_sample_all
                    ORDER BY user_id, education_number
                    """
                ).df()
                self.assertNotIn(2, sample_all["user_id"].tolist())
                self.assertNotIn(3, sample_all["user_id"].tolist())
                self.assertNotIn(7, sample_all["user_id"].tolist())
                grad_years = dict(
                    sample_all.loc[
                        sample_all["education_number"].isin([40, 50, 60]),
                        ["education_number", "grad_year"],
                    ].itertuples(index=False, name=None)
                )
                self.assertEqual(grad_years[40], 2019)
                self.assertEqual(grad_years[50], 2020)
                self.assertEqual(grad_years[60], 2017)

    def test_step3_match_treated_uses_unitid_window_and_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                treated = analysis.step3_match_treated(
                    con,
                    fixtures["relabel_df"],
                    "stage04_sample_all",
                    "stage04_all",
                )
                self.assertEqual(set(treated["user_id"].tolist()), {1, 6, 8, 9})
                self.assertTrue(treated[["user_id", "relabel_year"]].duplicated().sum() == 0)

                picked = treated.set_index("user_id")
                self.assertEqual(int(picked.loc[1, "education_number"]), 10)
                self.assertEqual(int(picked.loc[1, "grad_year"]), 2020)
                self.assertEqual(int(picked.loc[8, "education_number"]), 81)
                self.assertEqual(int(picked.loc[9, "education_number"]), 90)

    def test_foia_linked_variant_only_uses_person_baseline_users(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                foia_users = con.sql(
                    """
                    SELECT DISTINCT user_id
                    FROM stage04_sample_foia_linked_person_baseline
                    ORDER BY user_id
                    """
                ).df()["user_id"].tolist()
                self.assertEqual(foia_users, [1, 4])

                treated = analysis.step3_match_treated(
                    con,
                    fixtures["relabel_df"],
                    "stage04_sample_foia_linked_person_baseline",
                    "foia_linked_person_baseline",
                )
                self.assertEqual(treated["user_id"].tolist(), [1])

    def test_control_matching_uses_same_window_on_stage04_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                control_matches = analysis._match_individuals_to_events(
                    con,
                    "stage04_sample_all",
                    fixtures["matched_pairs"].rename(columns={"control_unitid": "unitid"})[
                        ["unitid", "relabel_year", "relabel_type"]
                    ],
                    treated_ind=0,
                    group_label="control_test",
                )
                self.assertEqual(set(control_matches["user_id"].tolist()), {4, 5})
                self.assertTrue((control_matches["treated_ind"] == 0).all())
                self.assertEqual(
                    dict(control_matches.set_index("user_id")["grad_year"].to_dict()),
                    {4: 2019, 5: 2020},
                )

    def test_step4_uses_post_graduation_horizons(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, outcome_horizons=(1, 3)):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                treated = analysis.step3_match_treated(
                    con,
                    fixtures["relabel_df"],
                    "stage04_sample_all",
                    "stage04_all",
                )
                panel = analysis.step4_build_outcome_panel(
                    con,
                    treated,
                    group_label="treated_stage04_all",
                    analysis_variant="stage04_all",
                )

            user1 = panel[panel["user_id"] == 1].sort_values("horizon_years").reset_index(drop=True)
            self.assertEqual(user1["horizon_years"].tolist(), [1, 3])

            horizon1 = user1[user1["horizon_years"] == 1].iloc[0]
            self.assertEqual(int(horizon1["target_year"]), 2021)
            self.assertEqual(int(horizon1["eval_year"]), 2021)
            self.assertEqual(int(horizon1["target_year_observed"]), 1)
            self.assertEqual(int(horizon1["in_us"]), 1)
            self.assertEqual(int(horizon1["n_pos"]), 1)
            self.assertAlmostEqual(float(horizon1["salary_imputed"]), 120000.0)

            horizon3 = user1[user1["horizon_years"] == 3].iloc[0]
            self.assertEqual(int(horizon3["target_year"]), 2023)
            self.assertEqual(int(horizon3["eval_year"]), 2022)
            self.assertEqual(int(horizon3["target_year_observed"]), 0)
            self.assertEqual(int(horizon3["used_latest_avail"]), 1)

    def test_step8_did_returns_results_and_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, event_window=1) as stack, patch.object(
                analysis.v2,
                "_match_treated_to_untreated_cohorts",
                return_value=fixtures["matched_pairs"],
            ):
                stack.enter_context(patch.object(analysis.cfg, "BUILD_RUN_DID", True))
                stack.enter_context(patch.object(analysis.cfg, "BUILD_DID_MODEL", "panel"))

                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                treated = analysis.step3_match_treated(
                    con,
                    fixtures["relabel_df"],
                    "stage04_sample_all",
                    "stage04_all",
                )
                treated_panel = analysis.step4_build_outcome_panel(
                    con,
                    treated,
                    group_label="treated_stage04_all",
                    analysis_variant="stage04_all",
                )
                control_events = analysis.build_control_events(con, fixtures["relabel_df"])
                control_panel = analysis.step6_control_group(
                    con,
                    "stage04_sample_all",
                    control_events,
                    "stage04_all",
                )
                did_results = analysis.step8_did(
                    treated_panel,
                    control_panel,
                    analysis_variant="stage04_all",
                )

            self.assertFalse(did_results.empty)
            self.assertEqual(set(did_results["did_model"].tolist()), {"simple"})
            self.assertEqual(set(did_results["analysis_variant"].tolist()), {"stage04_all"})
            self.assertTrue(
                {"horizon_years", "cohort_t", "reference_cohort_t", "coef", "se", "pval", "n_entities"}.issubset(
                    did_results.columns
                )
            )
            self.assertEqual(set(did_results["horizon_years"].unique().tolist()), {1})
            self.assertEqual(set(did_results["cohort_t"].unique().tolist()), {0})
            self.assertEqual(set(did_results["reference_cohort_t"].unique().tolist()), {-1})
            written_files = {path.name for path in Path(fixtures["output_dir"]).glob("*.png")}
            self.assertTrue(
                {
                    "did_att_active_us_stage04_all.png",
                    "did_att_active_positions_stage04_all.png",
                }.issubset(written_files)
            )

    def test_plot_did_variant_comparison_offsets_series_and_uses_per_outcome_names(self) -> None:
        results_df = pd.DataFrame(
            [
                {
                    "analysis_variant": "stage04_all",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 3,
                    "cohort_t": 0,
                    "reference_cohort_t": -1,
                    "coef": 0.01,
                    "se": 0.005,
                },
                {
                    "analysis_variant": "stage04_all",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 3,
                    "cohort_t": 1,
                    "reference_cohort_t": -1,
                    "coef": 0.02,
                    "se": 0.006,
                },
                {
                    "analysis_variant": "foia_linked_person_baseline",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 3,
                    "cohort_t": 0,
                    "reference_cohort_t": -1,
                    "coef": 0.03,
                    "se": 0.010,
                },
                {
                    "analysis_variant": "foia_linked_person_baseline",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 3,
                    "cohort_t": 1,
                    "reference_cohort_t": -1,
                    "coef": 0.04,
                    "se": 0.011,
                },
            ]
        )
        saved_names: list[str] = []
        errorbar_x: list[list[float]] = []

        def _capture_save(fig, name, analysis_variant=None):  # type: ignore[no-untyped-def]
            saved_names.append(name)
            analysis.plt.close(fig)
            return Path("/tmp/ignored.png")

        original_errorbar = matplotlib.axes._axes.Axes.errorbar

        def _capture_errorbar(self, x, *args, **kwargs):  # type: ignore[no-untyped-def]
            errorbar_x.append([float(v) for v in x])
            return original_errorbar(self, x, *args, **kwargs)

        with patch.object(analysis, "_save_and_show", side_effect=_capture_save), patch.object(
            analysis.plt, "show", lambda: None
        ), patch.object(matplotlib.axes._axes.Axes, "errorbar", new=_capture_errorbar):
            analysis._plot_did_variant_comparison(results_df)

        self.assertEqual(saved_names, ["did_att_by_variant_active_us"])
        self.assertTrue(any(any(abs(v - round(v)) > 1e-9 for v in xs) for xs in errorbar_x))

    def test_main_writes_did_results_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, event_window=1) as stack, patch.object(
                analysis,
                "step1_relabels",
                return_value=fixtures["relabel_df"],
            ), patch.object(
                analysis.v2,
                "_match_treated_to_untreated_cohorts",
                return_value=fixtures["matched_pairs"],
            ):
                stack.enter_context(patch.object(analysis.cfg, "BUILD_RUN_DID", True))
                stack.enter_context(patch.object(analysis.cfg, "BUILD_DID_MODEL", "simple"))
                analysis.main()

            did = pd.read_parquet(fixtures["output_did_path"])
            self.assertFalse(did.empty)
            self.assertEqual(
                set(did["analysis_variant"].unique().tolist()),
                {"stage04_all"},
            )
            self.assertEqual(set(did["did_model"].unique().tolist()), {"simple"})
            self.assertEqual(set(did["reference_cohort_t"].unique().tolist()), {-1})
            self.assertTrue(set(did["cohort_t"].unique().tolist()).issubset({0}))

    def test_main_runs_both_variants_and_writes_variant_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures), patch.object(
                analysis,
                "step1_relabels",
                return_value=fixtures["relabel_df"],
            ), patch.object(
                analysis.v2,
                "_match_treated_to_untreated_cohorts",
                return_value=fixtures["matched_pairs"],
            ):
                analysis.main()

            panel = pd.read_parquet(fixtures["output_panel_path"])
            self.assertIn("analysis_variant", panel.columns)
            self.assertEqual(
                set(panel["analysis_variant"].unique().tolist()),
                {"stage04_all", "foia_linked_person_baseline"},
            )
            expected_files = {
                "in_us_event_study_treated_stage04_all.png",
                "in_us_event_study_treated_foia_linked_person_baseline.png",
                "in_us_event_study_treated_vs_control_stage04_all.png",
                "in_us_event_study_treated_vs_control_foia_linked_person_baseline.png",
            }
            written_files = {path.name for path in Path(fixtures["output_dir"]).glob("*.png")}
            self.assertTrue(expected_files.issubset(written_files))


if __name__ == "__main__":
    unittest.main()
