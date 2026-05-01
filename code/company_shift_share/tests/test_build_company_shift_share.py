from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

import duckdb
import pandas as pd

from company_shift_share.build_company_shift_share import (
    _build_school_metric_panel,
    _build_school_shift_sample,
    _confirm_matched_school_sample,
    _create_analysis_panel,
    _create_growth_view_for_pipeline,
    _create_instrument_views,
    _create_ipeds_growth_view,
    _create_sampled_school_growth_view,
    _register_school_metric_views,
)


def _register_view(con: duckdb.DuckDBPyConnection, name: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    con.register(f"{name}_df", df)
    con.sql(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {name}_df")


class BuildCompanyShiftShareTests(unittest.TestCase):
    def _register_ipeds(self, con: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
        _register_view(con, "ipeds_raw", rows)

    def _register_foia_school_inputs(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        foia_rows: list[dict],
        cw_rows: list[dict],
    ) -> None:
        _register_view(con, "foia_raw", foia_rows)
        _register_view(con, "f1_inst_unitid_cw", cw_rows)

    def test_ipeds_school_metric_builders_use_year_specific_levels(self) -> None:
        con = duckdb.connect()
        self._register_ipeds(
            con,
            [
                {"unitid": 1, "year": 2014, "cipcode": 10101, "ctotalt": 100.0, "cnralt": 60.0, "share_intl": 0.60},
                {"unitid": 1, "year": 2014, "cipcode": 20202, "ctotalt": 50.0, "cnralt": 10.0, "share_intl": 0.20},
                {"unitid": 1, "year": 2015, "cipcode": 10101, "ctotalt": 100.0, "cnralt": 40.0, "share_intl": 0.40},
                {"unitid": 1, "year": 2015, "cipcode": 20202, "ctotalt": 50.0, "cnralt": 30.0, "share_intl": 0.60},
            ],
        )

        ihmp = _build_school_metric_panel(
            con,
            metric="ihmp_share",
            degree_scope="masters",
        )
        intl = _build_school_metric_panel(
            con,
            metric="international_share",
            degree_scope="masters",
        )

        ihmp_2014 = ihmp.loc[(ihmp["k"] == "1") & (ihmp["t"] == 2014)].iloc[0]
        ihmp_2015 = ihmp.loc[(ihmp["k"] == "1") & (ihmp["t"] == 2015)].iloc[0]
        intl_2014 = intl.loc[(intl["k"] == "1") & (intl["t"] == 2014)].iloc[0]

        self.assertAlmostEqual(ihmp_2014["metric_level"], 100.0)
        self.assertAlmostEqual(ihmp_2014["metric_share"], 100.0 / 150.0)
        self.assertAlmostEqual(ihmp_2015["metric_level"], 50.0)
        self.assertAlmostEqual(ihmp_2015["metric_share"], 50.0 / 150.0)
        self.assertAlmostEqual(intl_2014["metric_level"], 70.0)
        self.assertAlmostEqual(intl_2014["metric_share"], 70.0 / 150.0)

    def test_opt_ihmp_metric_builder_uses_foia_opt_totals(self) -> None:
        con = duckdb.connect()
        self._register_ipeds(
            con,
            [
                {"unitid": 1, "year": 2014, "cipcode": 10101, "ctotalt": 100.0, "cnralt": 40.0, "share_intl": 0.40},
                {"unitid": 1, "year": 2014, "cipcode": 20202, "ctotalt": 50.0, "cnralt": 5.0, "share_intl": 0.10},
            ],
        )
        self._register_foia_school_inputs(
            con,
            foia_rows=[
                {"person_id": 1, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2014-05-30", "major_1_cip_code": "01.0101", "student_edu_level_desc": "Master's", "opt_authorization_start_date": "2014-06-01"},
                {"person_id": 2, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2014-05-30", "major_1_cip_code": "01.0101", "student_edu_level_desc": "Master's", "opt_authorization_start_date": "2014-06-01"},
                {"person_id": 3, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2014-05-30", "major_1_cip_code": "01.0101", "student_edu_level_desc": "Master's", "opt_authorization_start_date": None},
                {"person_id": 4, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2014-05-30", "major_1_cip_code": "01.0101", "student_edu_level_desc": "Master's", "opt_authorization_start_date": None},
                {"person_id": 5, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2014-05-30", "major_1_cip_code": "02.0202", "student_edu_level_desc": "Master's", "opt_authorization_start_date": "2014-06-01"},
                {"person_id": 6, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2014-05-30", "major_1_cip_code": "02.0202", "student_edu_level_desc": "Master's", "opt_authorization_start_date": None},
                {"person_id": 7, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2014-05-30", "major_1_cip_code": "02.0202", "student_edu_level_desc": "Master's", "opt_authorization_start_date": None},
            ],
            cw_rows=[
                {"school_name": "Test U", "f1_city_clean": "test city", "f1_state_clean": "CA", "f1_zip_clean": "94105", "UNITID": 1},
            ],
        )

        panel = _build_school_metric_panel(
            con,
            metric="opt_ihmp_share",
            degree_scope="masters",
            opt_ihmp_ipeds_share_intl_threshold=0.30,
            opt_ihmp_foia_opt_share_threshold=0.50,
            opt_ihmp_min_program_f1_count=2,
        )
        row = panel.iloc[0]

        self.assertEqual(row["k"], "1")
        self.assertEqual(int(row["t"]), 2014)
        self.assertAlmostEqual(row["school_size"], 150.0)
        self.assertAlmostEqual(row["metric_level"], 3.0)
        self.assertAlmostEqual(row["metric_share"], 3.0 / 150.0)
        self.assertAlmostEqual(row["foia_total_students"], 7.0)
        self.assertAlmostEqual(row["foia_total_opt_students"], 3.0)

    def test_opt_share_metric_builder_uses_school_year_ever_opt_share(self) -> None:
        con = duckdb.connect()
        self._register_foia_school_inputs(
            con,
            foia_rows=[
                {"person_id": 1, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2014-05-30", "major_1_cip_code": "01.0101", "student_edu_level_desc": "Master's", "opt_authorization_start_date": "2014-06-01"},
                {"person_id": 2, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2014-05-30", "major_1_cip_code": "01.0101", "student_edu_level_desc": "Master's", "opt_authorization_start_date": "2014-06-01"},
                {"person_id": 3, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2014-05-30", "major_1_cip_code": "02.0202", "student_edu_level_desc": "Master's", "opt_authorization_start_date": None},
                {"person_id": 4, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2015-05-30", "major_1_cip_code": "01.0101", "student_edu_level_desc": "Master's", "opt_authorization_start_date": None},
                {"person_id": 5, "school_name": "Test U", "campus_city": "Test City", "campus_state": "California", "campus_zip_code": "94105", "program_end_date": "2015-05-30", "major_1_cip_code": "02.0202", "student_edu_level_desc": "Master's", "opt_authorization_start_date": "2015-06-01"},
            ],
            cw_rows=[
                {"school_name": "Test U", "f1_city_clean": "test city", "f1_state_clean": "CA", "f1_zip_clean": "94105", "UNITID": 1},
            ],
        )

        panel = _build_school_metric_panel(
            con,
            metric="opt_share",
            degree_scope="masters",
        ).sort_values("t")

        row_2014 = panel.loc[panel["t"] == 2014].iloc[0]
        row_2015 = panel.loc[panel["t"] == 2015].iloc[0]

        self.assertAlmostEqual(row_2014["school_size"], 3.0)
        self.assertAlmostEqual(row_2014["metric_level"], 2.0)
        self.assertAlmostEqual(row_2014["metric_share"], 2.0 / 3.0)
        self.assertAlmostEqual(row_2015["metric_share"], 0.5)

    def test_matched_sample_uses_largest_positive_jump_and_sign_specific_controls(self) -> None:
        metric_panel = pd.DataFrame(
            [
                {"k": "T1", "t": 2014, "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10},
                {"k": "T1", "t": 2015, "school_size": 100.0, "metric_level": 12.0, "metric_share": 0.12},
                {"k": "T1", "t": 2016, "school_size": 100.0, "metric_level": 45.0, "metric_share": 0.45},
                {"k": "T1", "t": 2017, "school_size": 100.0, "metric_level": 46.0, "metric_share": 0.46},
                {"k": "T2", "t": 2014, "school_size": 200.0, "metric_level": 40.0, "metric_share": 0.20},
                {"k": "T2", "t": 2015, "school_size": 200.0, "metric_level": 42.0, "metric_share": 0.21},
                {"k": "T2", "t": 2016, "school_size": 200.0, "metric_level": 44.0, "metric_share": 0.22},
                {"k": "T2", "t": 2017, "school_size": 200.0, "metric_level": 120.0, "metric_share": 0.60},
                {"k": "C1", "t": 2014, "school_size": 100.0, "metric_level": 30.0, "metric_share": 0.30},
                {"k": "C1", "t": 2015, "school_size": 100.0, "metric_level": 29.0, "metric_share": 0.29},
                {"k": "C1", "t": 2016, "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10},
                {"k": "C1", "t": 2017, "school_size": 100.0, "metric_level": 9.0, "metric_share": 0.09},
                {"k": "C2", "t": 2014, "school_size": 200.0, "metric_level": 80.0, "metric_share": 0.40},
                {"k": "C2", "t": 2015, "school_size": 200.0, "metric_level": 82.0, "metric_share": 0.41},
                {"k": "C2", "t": 2016, "school_size": 200.0, "metric_level": 84.0, "metric_share": 0.42},
                {"k": "C2", "t": 2017, "school_size": 200.0, "metric_level": 86.0, "metric_share": 0.43},
            ]
        )
        summary = _build_school_shift_sample(
            metric_panel,
            metric="international_share",
            school_name_lookup=pd.DataFrame(
                [
                    {"k": "T1", "school_name": "T1"},
                    {"k": "T2", "school_name": "T2"},
                    {"k": "C1", "school_name": "C1"},
                    {"k": "C2", "school_name": "C2"},
                ]
            ),
            n_shifted=2,
            control_positive_cap=0.02,
        )
        control_row = summary.loc[summary["k"] == "C1"].iloc[0]
        treated_row = summary.loc[summary["k"] == "T2"].iloc[0]

        self.assertEqual(control_row["control_eligible"], 1)
        self.assertAlmostEqual(control_row["max_positive_annual_change"], 0.0)
        self.assertAlmostEqual(treated_row["treated_score"], 0.38)
        selected_controls = summary.loc[summary["sample_role"] == "control", "k"].tolist()
        self.assertEqual(sorted(selected_controls), ["C1", "C2"])
        self.assertEqual(summary["matched_school_k"].dropna().nunique(), 4)

    def test_opt_share_sampling_excludes_large_f1_drop(self) -> None:
        metric_panel = pd.DataFrame(
            [
                {"k": "DROP", "t": 2014, "school_size": 80.0, "metric_level": 8.0, "metric_share": 0.10},
                {"k": "DROP", "t": 2015, "school_size": 90.0, "metric_level": 18.0, "metric_share": 0.20},
                {"k": "DROP", "t": 2016, "school_size": 40.0, "metric_level": 12.0, "metric_share": 0.30},
                {"k": "DROP", "t": 2017, "school_size": 80.0, "metric_level": 32.0, "metric_share": 0.40},
                {"k": "KEEP", "t": 2014, "school_size": 60.0, "metric_level": 6.0, "metric_share": 0.10},
                {"k": "KEEP", "t": 2015, "school_size": 60.0, "metric_level": 7.0, "metric_share": 0.1167},
                {"k": "KEEP", "t": 2016, "school_size": 60.0, "metric_level": 20.0, "metric_share": 0.3333},
                {"k": "KEEP", "t": 2017, "school_size": 60.0, "metric_level": 21.0, "metric_share": 0.35},
                {"k": "CTRL", "t": 2014, "school_size": 60.0, "metric_level": 12.0, "metric_share": 0.20},
                {"k": "CTRL", "t": 2015, "school_size": 60.0, "metric_level": 12.0, "metric_share": 0.20},
                {"k": "CTRL", "t": 2016, "school_size": 60.0, "metric_level": 11.0, "metric_share": 0.1833},
                {"k": "CTRL", "t": 2017, "school_size": 60.0, "metric_level": 11.0, "metric_share": 0.1833},
            ]
        )
        summary = _build_school_shift_sample(
            metric_panel,
            metric="opt_share",
            n_shifted=1,
            opt_share_min_school_f1_count=50,
            opt_share_max_yoy_drop=0.50,
        )
        drop_row = summary.loc[summary["k"] == "DROP"].iloc[0]
        keep_row = summary.loc[summary["k"] == "KEEP"].iloc[0]

        self.assertEqual(drop_row["fails_large_yoy_drop"], 1)
        self.assertEqual(drop_row["treated_candidate"], 0)
        self.assertEqual(keep_row["selected_for_instrument"], 1)

    def test_school_shift_min_size_is_configurable(self) -> None:
        metric_panel = pd.DataFrame(
            [
                {"k": "T", "t": 2014, "school_size": 90.0, "metric_level": 10.0, "metric_share": 0.10},
                {"k": "T", "t": 2015, "school_size": 90.0, "metric_level": 12.0, "metric_share": 0.12},
                {"k": "T", "t": 2016, "school_size": 90.0, "metric_level": 16.0, "metric_share": 0.16},
                {"k": "T", "t": 2017, "school_size": 90.0, "metric_level": 17.0, "metric_share": 0.17},
                {"k": "C", "t": 2014, "school_size": 90.0, "metric_level": 8.0, "metric_share": 0.08},
                {"k": "C", "t": 2015, "school_size": 90.0, "metric_level": 8.0, "metric_share": 0.08},
                {"k": "C", "t": 2016, "school_size": 90.0, "metric_level": 8.0, "metric_share": 0.08},
                {"k": "C", "t": 2017, "school_size": 90.0, "metric_level": 8.0, "metric_share": 0.08},
            ]
        )
        strict = _build_school_shift_sample(
            metric_panel,
            metric="ihmp_share",
            n_shifted=0,
            min_school_size=100,
        )
        permissive = _build_school_shift_sample(
            metric_panel,
            metric="ihmp_share",
            n_shifted=1,
            min_school_size=90,
        )

        self.assertEqual(int(strict["selected_for_instrument"].sum()), 0)
        self.assertEqual(int(permissive["selected_for_instrument"].sum()), 2)
        self.assertEqual(int(permissive.loc[permissive["k"] == "T", "meets_min_size"].iloc[0]), 1)
        self.assertEqual(int(permissive["min_required_size"].iloc[0]), 90)

    def test_confirm_matched_school_sample_prints_preview_and_accepts_yes(self) -> None:
        sample_summary = pd.DataFrame(
            [
                {
                    "k": "T1",
                    "school_name": "Treated U",
                    "sample_role": "treated",
                    "selected_for_instrument": 1,
                    "matched_school_k": "C1",
                    "matched_school_name": "Control U",
                    "matched_pair_id": 1,
                    "metric_share_2014": 0.10,
                    "metric_share_2015": 0.12,
                    "metric_share_2016": 0.45,
                    "metric_share_2017": 0.46,
                    "school_size_2014": 1000,
                    "school_size_2015": 1000,
                    "school_size_2016": 1000,
                    "school_size_2017": 1000,
                },
                {
                    "k": "C1",
                    "school_name": "Control U",
                    "sample_role": "control",
                    "selected_for_instrument": 1,
                    "matched_school_k": "T1",
                    "matched_school_name": "Treated U",
                    "matched_pair_id": 1,
                    "metric_share_2014": 0.30,
                    "metric_share_2015": 0.29,
                    "metric_share_2016": 0.10,
                    "metric_share_2017": 0.09,
                    "school_size_2014": 500,
                    "school_size_2015": 500,
                    "school_size_2016": 500,
                    "school_size_2017": 500,
                },
            ]
        )

        buf = io.StringIO()
        with patch("builtins.input", return_value="yes"):
            with contextlib.redirect_stdout(buf):
                _confirm_matched_school_sample(
                    sample_summary,
                    metric="international_share",
                    window_start=2014,
                    window_end=2017,
                )

        output = buf.getvalue()
        self.assertIn("Matched-school sample preview", output)
        self.assertIn("Treated U", output)
        self.assertIn("Control U", output)
        self.assertIn("international_share_share_2014", output)
        self.assertIn("school_size_2014", output)

    def test_confirm_matched_school_sample_exits_when_not_confirmed(self) -> None:
        sample_summary = pd.DataFrame(
            [
                {
                    "k": "T1",
                    "school_name": "Treated U",
                    "sample_role": "treated",
                    "selected_for_instrument": 1,
                    "matched_school_k": "C1",
                    "matched_school_name": "Control U",
                    "matched_pair_id": 1,
                    "metric_share_2014": 0.10,
                    "metric_share_2015": 0.12,
                    "metric_share_2016": 0.45,
                    "metric_share_2017": 0.46,
                    "school_size_2014": 1000,
                    "school_size_2015": 1000,
                    "school_size_2016": 1000,
                    "school_size_2017": 1000,
                },
            ]
        )

        with patch("builtins.input", return_value="n"):
            with self.assertRaises(SystemExit):
                _confirm_matched_school_sample(
                    sample_summary,
                    metric="ihmp_share",
                    window_start=2014,
                    window_end=2017,
                )

    def test_no_share_renormalization_after_sampling(self) -> None:
        con = duckdb.connect()
        metric_panel = pd.DataFrame(
            [
                {"k": "A", "t": 2014, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 10.0, "foia_total_students": None, "foia_total_opt_students": None},
            ]
        )
        sample_summary = pd.DataFrame(
            [
                {"k": "A", "school_name": "A", "metric": "ihmp_share", "sample_role": "treated", "selected_for_instrument": 1, "matched_school_k": None, "matched_school_name": None, "matched_pair_id": 1, "treated_rank": 1, "treated_score": 0.3, "control_eligible": 0, "treated_candidate": 1, "has_full_window_coverage": 1, "meets_min_size": 1, "fails_large_yoy_drop": 0, "min_required_size": 100, "control_positive_cap": 0.02, "avg_size_window": 100.0, "log_avg_size_window": 4.605170186, "max_positive_annual_change": 0.3, "max_negative_yoy_size_change": None},
            ]
        )
        _register_school_metric_views(con, metric_panel=metric_panel, sample_summary=sample_summary)
        growth_view = _create_sampled_school_growth_view(con)
        _register_view(
            con,
            "transition_unit_shares",
            [
                {"c": 1, "k": "A", "share_ck": 0.6, "share_ck_base": 0.6, "share_ck_full": 0.6, "n_transitions": 6, "n_transitions_full": 6, "total_new_hires": 10, "total_new_hires_alt": 10, "total_new_hires_full": 10},
                {"c": 1, "k": "B", "share_ck": 0.4, "share_ck_base": 0.4, "share_ck_full": 0.4, "n_transitions": 4, "n_transitions_full": 4, "total_new_hires": 10, "total_new_hires_alt": 10, "total_new_hires_full": 10},
            ],
        )
        _create_instrument_views(con, "transition_unit_shares", growth_view)
        row = con.sql("SELECT z_ct, z_ct_full FROM company_instrument_panel").df().iloc[0]

        self.assertAlmostEqual(row["z_ct"], 6.0)
        self.assertAlmostEqual(row["z_ct_full"], 6.0)

    def test_all_mode_dispatch_matches_legacy_ipeds_growth(self) -> None:
        rows = [
            {"unitid": 1, "year": 2014, "cipcode": 10101, "ctotalt": 100.0, "cnralt": 60.0, "share_intl": 0.60},
            {"unitid": 1, "year": 2014, "cipcode": 20202, "ctotalt": 50.0, "cnralt": 10.0, "share_intl": 0.20},
            {"unitid": 1, "year": 2015, "cipcode": 10101, "ctotalt": 100.0, "cnralt": 40.0, "share_intl": 0.40},
            {"unitid": 1, "year": 2015, "cipcode": 20202, "ctotalt": 50.0, "cnralt": 30.0, "share_intl": 0.60},
        ]
        con_legacy = duckdb.connect()
        self._register_ipeds(con_legacy, rows)
        legacy_view = _create_ipeds_growth_view(con_legacy)
        legacy = con_legacy.sql(f"SELECT * FROM {legacy_view} ORDER BY k, t").df()

        con_dispatch = duckdb.connect()
        self._register_ipeds(con_dispatch, rows)
        dispatched_view, panel_view, sample_view = _create_growth_view_for_pipeline(
            con_dispatch,
            school_sample_mode="all",
            school_shift_metric=None,
            include_bachelors=False,
            use_changes=False,
            use_log_y=False,
            opt_shifts=False,
            opt_shifts_degree_scope="masters",
            opt_shifts_normalization="none",
            opt_shifts_normalize_by_graduates=False,
        )
        dispatched = con_dispatch.sql(f"SELECT * FROM {dispatched_view} ORDER BY k, t").df()

        self.assertIsNone(panel_view)
        self.assertIsNone(sample_view)
        pd.testing.assert_frame_equal(legacy.reset_index(drop=True), dispatched.reset_index(drop=True))

    def test_sampled_mode_still_emits_existing_analysis_panel_columns(self) -> None:
        con = duckdb.connect()
        metric_panel = pd.DataFrame(
            [
                {"k": "A", "t": 2014, "metric": "international_share", "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 10.0, "foia_total_students": None, "foia_total_opt_students": None},
            ]
        )
        sample_summary = pd.DataFrame(
            [
                {"k": "A", "school_name": "A", "metric": "international_share", "sample_role": "treated", "selected_for_instrument": 1, "matched_school_k": None, "matched_school_name": None, "matched_pair_id": 1, "treated_rank": 1, "treated_score": 0.3, "control_eligible": 0, "treated_candidate": 1, "has_full_window_coverage": 1, "meets_min_size": 1, "fails_large_yoy_drop": 0, "min_required_size": 100, "control_positive_cap": 0.02, "avg_size_window": 100.0, "log_avg_size_window": 4.605170186, "max_positive_annual_change": 0.3, "max_negative_yoy_size_change": None},
            ]
        )
        _register_school_metric_views(con, metric_panel=metric_panel, sample_summary=sample_summary)
        growth_view = _create_sampled_school_growth_view(con)
        _register_view(
            con,
            "transition_unit_shares",
            [
                {"c": 1, "k": "A", "share_ck": 0.6, "share_ck_base": 0.6, "share_ck_full": 0.6, "n_transitions": 6, "n_transitions_full": 6, "total_new_hires": 10, "total_new_hires_alt": 10, "total_new_hires_full": 10},
            ],
        )
        _create_instrument_views(con, "transition_unit_shares", growth_view)
        _register_view(
            con,
            "outcomes_wide",
            [
                {"c": 1, "t": 2014, "y_cst_lag0": 1000.0, "y_new_hires_lag0": 100.0},
            ],
        )
        _register_view(
            con,
            "opt_new_hires",
            [
                {"c": 1, "t": 2014, "masters_opt_hires": 5, "valid_masters_opt_hires": 4, "masters_opt_hires_correction_aware": 5, "x_cst_lag0": 5},
            ],
        )
        _register_view(
            con,
            "employer_crosswalk",
            [
                {"preferred_rcid": 1, "f1_state_clean": "CA"},
            ],
        )

        _create_analysis_panel(con, outcome_lag_start=0, outcome_lag_end=0, use_log_y=False)
        panel = con.sql("SELECT * FROM analysis_panel").df()

        for col in [
            "c",
            "t",
            "y_cst_lag0",
            "y_new_hires_lag0",
            "masters_opt_hires",
            "valid_masters_opt_hires",
            "masters_opt_hires_correction_aware",
            "x_cst_lag0",
            "z_ct",
            "z_ct_full",
            "n_universities",
            "company_state",
        ]:
            self.assertIn(col, panel.columns)


if __name__ == "__main__":
    unittest.main()
