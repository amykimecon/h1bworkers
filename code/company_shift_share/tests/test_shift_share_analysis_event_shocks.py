from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd

from company_shift_share.shift_share_analysis import (
    _apply_analysis_sample_restrictions,
    _build_analysis_panel,
    _build_outcomes,
    _build_first_stage_conditioning_specs,
    _build_event_quantity_growth_view,
    _build_interaction_plot_frame,
    _build_instrument,
    _build_regression_variant_headline_summary,
    _build_school_event_summary,
    _build_transition_shares,
    _diagnostic_school_name_lookup,
    _prepare_first_stage_outcomes,
    _prepare_first_stage_state_panel,
    _select_regression_instrument_cols,
    run_falsification_tests,
    run_regression_variants,
)


def _register_view(con: duckdb.DuckDBPyConnection, name: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    con.register(f"_{name}_df", df)
    con.sql(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM _{name}_df")


class ShiftShareEventShockTests(unittest.TestCase):
    def test_build_outcomes_adds_source_new_hire_splits_and_tenure(self) -> None:
        con = duckdb.connect()
        _register_view(con, "matched_rcids", [{"rcid": 1}])
        _register_view(
            con,
            "revelio_headcount",
            [
                {"rcid": 1, "year": 2020, "total_headcount": 10.0},
                {"rcid": 1, "year": 2021, "total_headcount": 12.0},
            ],
        )
        _register_view(
            con,
            "revelio_transitions",
            [{"rcid": 1, "year": 2020, "total_new_hires": 99.0}],
        )
        _register_view(
            con,
            "wrds_company_year_workforce",
            [
                {
                    "c": 1,
                    "t": 2020,
                    "n_new_hires_wrds_annual": 2.0,
                    "n_new_hires_foreign_weighted_annual": 0.5,
                    "n_new_hires_native_weighted_annual": 1.5,
                    "n_new_hires_foreign_hard_annual": 1.0,
                    "n_new_hires_native_hard_annual": 1.0,
                    "avg_tenure_years_annual": 4.0,
                },
                {
                    "c": 1,
                    "t": 2021,
                    "n_new_hires_wrds_annual": 3.0,
                    "n_new_hires_foreign_weighted_annual": 1.0,
                    "n_new_hires_native_weighted_annual": 2.0,
                    "n_new_hires_foreign_hard_annual": 1.0,
                    "n_new_hires_native_hard_annual": 2.0,
                    "avg_tenure_years_annual": 4.5,
                },
            ],
        )

        _build_outcomes(con, lag_start=0, lag_end=1, use_changes=False)

        row = con.sql(
            """
            SELECT y_cst_lag0, y_new_hires_lag0, y_new_hires_foreign_lag0,
                   y_new_hires_native_lag0, y_new_hires_foreign_hard_lag0,
                   y_new_hires_native_hard_lag0, avg_tenure_years_lag0
            FROM outcomes_wide
            WHERE c = 1 AND t = 2020
            """
        ).df().iloc[0]
        self.assertAlmostEqual(float(row["y_cst_lag0"]), 10.0)
        self.assertAlmostEqual(float(row["y_new_hires_lag0"]), 2.0)
        self.assertAlmostEqual(float(row["y_new_hires_foreign_lag0"]), 0.5)
        self.assertAlmostEqual(float(row["y_new_hires_native_lag0"]), 1.5)
        self.assertAlmostEqual(float(row["y_new_hires_foreign_hard_lag0"]), 1.0)
        self.assertAlmostEqual(float(row["y_new_hires_native_hard_lag0"]), 1.0)
        self.assertAlmostEqual(float(row["avg_tenure_years_lag0"]), 4.0)

    def test_school_event_summary_uses_smoothed_pre_post_break(self) -> None:
        metric_panel = pd.DataFrame(
            [
                {"k": "A", "t": 2012, "school_size": 100.0, "metric_share": 0.10},
                {"k": "A", "t": 2013, "school_size": 100.0, "metric_share": 0.10},
                {"k": "A", "t": 2014, "school_size": 100.0, "metric_share": 0.10},
                {"k": "A", "t": 2015, "school_size": 100.0, "metric_share": 0.10},
                {"k": "A", "t": 2016, "school_size": 100.0, "metric_share": 0.40},
                {"k": "A", "t": 2017, "school_size": 100.0, "metric_share": 0.45},
                {"k": "A", "t": 2018, "school_size": 100.0, "metric_share": 0.46},
            ]
        )
        summary = _build_school_event_summary(
            metric_panel=metric_panel,
            window_start=2014,
            window_end=2017,
            event_pre_years=2,
            event_post_years=2,
        )
        row = summary.iloc[0]
        self.assertEqual(int(row["treated_event_year"]), 2016)
        self.assertAlmostEqual(float(row["treated_score"]), 0.325, places=6)
        self.assertAlmostEqual(float(row["event_pre_share"]), 0.10, places=6)
        self.assertAlmostEqual(float(row["event_post_share"]), 0.425, places=6)
        self.assertAlmostEqual(float(row["event_pre_size"]), 100.0, places=6)
        self.assertAlmostEqual(float(row["event_pre_level"]), 10.0, places=6)
        self.assertAlmostEqual(float(row["event_post_level"]), 42.5, places=6)
        self.assertAlmostEqual(float(row["event_level_growth"]), 32.5, places=6)
        self.assertAlmostEqual(float(row["event_level_growth_rate"]), 3.25, places=6)

    def test_event_quantity_growth_view_builds_persistent_and_matched_variants(self) -> None:
        con = duckdb.connect()
        metric_panel = pd.DataFrame(
            [
                {"k": "A", "t": 2011, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 10.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "A", "t": 2012, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 10.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "A", "t": 2013, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 10.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "A", "t": 2014, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 10.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "A", "t": 2015, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 12.0, "metric_share": 0.12, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 12.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "A", "t": 2016, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 40.0, "metric_share": 0.40, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 40.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "A", "t": 2017, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 42.0, "metric_share": 0.42, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 42.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "A", "t": 2018, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 43.0, "metric_share": 0.43, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 43.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "B", "t": 2011, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 10.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "B", "t": 2012, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 10.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "B", "t": 2013, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 10.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "B", "t": 2014, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 10.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "B", "t": 2015, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 10.0, "metric_share": 0.10, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 10.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "B", "t": 2016, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 11.0, "metric_share": 0.11, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 11.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "B", "t": 2017, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 11.0, "metric_share": 0.11, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 11.0, "foia_total_students": None, "foia_total_opt_students": None},
                {"k": "B", "t": 2018, "metric": "ihmp_share", "school_size": 100.0, "metric_level": 11.0, "metric_share": 0.11, "ipeds_total_students": 100.0, "ipeds_total_intl_students": 11.0, "foia_total_students": None, "foia_total_opt_students": None},
            ]
        )
        sample_summary = pd.DataFrame(
            [
                {
                    "k": "A",
                    "school_name": "School A",
                    "selected_for_instrument": 1,
                    "sample_role": "treated",
                    "matched_pair_id": 1,
                    "treated_event_year": 2016,
                    "treated_score": 0.30,
                    "event_pre_share": 0.11,
                    "event_post_share": 0.41,
                    "event_pre_size": 100.0,
                    "event_pre_level": 11.0,
                    "event_post_level": 41.0,
                    "event_level_growth": 30.0,
                    "event_pre_opt_rate": 0.50,
                    "matched_treated_event_year": 2016,
                },
                {
                    "k": "B",
                    "school_name": "School B",
                    "selected_for_instrument": 1,
                    "sample_role": "control",
                    "matched_pair_id": 1,
                    "treated_event_year": pd.NA,
                    "treated_score": 0.0,
                    "event_pre_share": pd.NA,
                    "event_post_share": pd.NA,
                    "event_pre_size": pd.NA,
                    "event_pre_level": pd.NA,
                    "event_post_level": pd.NA,
                    "event_level_growth": pd.NA,
                    "event_pre_opt_rate": pd.NA,
                    "matched_treated_event_year": 2016,
                },
            ]
        )
        opt_share_panel = pd.DataFrame(
            [
                {"k": "A", "t": 2014, "metric_share": 0.50, "metric_level": 50.0},
                {"k": "A", "t": 2015, "metric_share": 0.50, "metric_level": 50.0},
                {"k": "A", "t": 2016, "metric_share": 0.55, "metric_level": 55.0},
                {"k": "B", "t": 2014, "metric_share": 0.20, "metric_level": 20.0},
                {"k": "B", "t": 2015, "metric_share": 0.20, "metric_level": 20.0},
                {"k": "B", "t": 2016, "metric_share": 0.20, "metric_level": 20.0},
            ]
        )
        _build_event_quantity_growth_view(
            con,
            metric_panel=metric_panel,
            sample_summary=sample_summary,
            opt_share_panel=opt_share_panel,
            year_min=2014,
            year_max=2018,
            event_window_start=2014,
            event_window_end=2017,
            event_pre_years=2,
            event_post_years=2,
        )
        out = con.sql("""
            SELECT
                k, t, g_kt, g_kt_v3_broad_predicted_opt, g_kt_v4_matched_step, g_kt_v5_matched_pulse,
                g_kt_v7_matched_pulse_growth_rate, g_kt_ihmp_share, event_level_growth_rate,
                g_kt_falsification_lead4_broad, g_kt_falsification_lead4_matched
            FROM ipeds_unit_growth
            ORDER BY k, t
        """).df()
        a_2014 = out.loc[(out["k"] == "A") & (out["t"] == 2014)].iloc[0]
        a_2015 = out.loc[(out["k"] == "A") & (out["t"] == 2015)].iloc[0]
        a_2016 = out.loc[(out["k"] == "A") & (out["t"] == 2016)].iloc[0]
        a_2018 = out.loc[(out["k"] == "A") & (out["t"] == 2018)].iloc[0]
        b_2018 = out.loc[(out["k"] == "B") & (out["t"] == 2018)].iloc[0]

        self.assertAlmostEqual(float(a_2014["g_kt_falsification_lead4_broad"]), 30.0)
        self.assertAlmostEqual(float(a_2014["g_kt_falsification_lead4_matched"]), 30.0)
        self.assertAlmostEqual(float(a_2015["g_kt"]), 0.0)
        self.assertAlmostEqual(float(a_2016["event_level_growth_rate"]), 30.0 / 11.0)
        self.assertAlmostEqual(float(a_2016["g_kt"]), 30.0 / 11.0)
        self.assertAlmostEqual(float(a_2018["g_kt"]), 0.0)
        self.assertAlmostEqual(float(a_2016["g_kt_v3_broad_predicted_opt"]), 15.0)
        self.assertAlmostEqual(float(a_2016["g_kt_v4_matched_step"]), 30.0)
        self.assertAlmostEqual(float(a_2016["g_kt_v5_matched_pulse"]), 30.0)
        self.assertAlmostEqual(float(a_2016["g_kt_v7_matched_pulse_growth_rate"]), 30.0 / 11.0)
        self.assertAlmostEqual(float(a_2016["g_kt_ihmp_share"]), 0.40)
        self.assertAlmostEqual(float(a_2018["g_kt_v5_matched_pulse"]), 0.0)
        self.assertAlmostEqual(float(b_2018["g_kt"]), 0.0)
        self.assertAlmostEqual(float(b_2018["g_kt_v4_matched_step"]), 0.0)
        self.assertAlmostEqual(float(b_2018["g_kt_v7_matched_pulse_growth_rate"]), 0.0)

    def test_transition_shares_use_pre_window_and_full_window(self) -> None:
        con = duckdb.connect()
        _register_view(
            con,
            "revelio_transitions",
            [
                {"rcid": 1, "university_raw": "school a", "year": 2008, "n_transitions": 2.0},
                {"rcid": 1, "university_raw": "school a", "year": 2010, "n_transitions": 4.0},
                {"rcid": 1, "university_raw": "school b", "year": 2012, "n_transitions": 2.0},
                {"rcid": 1, "university_raw": "school b", "year": 2016, "n_transitions": 10.0},
            ],
        )
        _register_view(con, "matched_rcids", [{"rcid": 1}])
        _register_view(
            con,
            "revelio_inst_cw",
            [
                {"university_raw_norm": "school a", "unitid": "A"},
                {"university_raw_norm": "school b", "unitid": "B"},
            ],
        )

        _build_transition_shares(
            con,
            share_period="pre_window",
            share_base_year=2010,
            share_year_min=2008,
            share_year_max=2013,
            robustness_windows=[(2008, 2010), (2011, 2013)],
            min_universities_for_share=1,
        )
        out = con.sql("""
            SELECT k, share_ck, share_ck_2008_2010, share_ck_2011_2013, share_ck_full
            FROM transition_shares
            ORDER BY k
        """).df()
        row_a = out.loc[out["k"] == "A"].iloc[0]
        row_b = out.loc[out["k"] == "B"].iloc[0]
        self.assertAlmostEqual(float(row_a["share_ck"]), 0.75)
        self.assertAlmostEqual(float(row_b["share_ck"]), 0.25)
        self.assertAlmostEqual(float(row_a["share_ck_2008_2010"]), 1.0)
        self.assertAlmostEqual(float(row_b["share_ck_2011_2013"]), 1.0)
        self.assertAlmostEqual(float(row_a["share_ck_full"]), 6.0 / 18.0)
        self.assertAlmostEqual(float(row_b["share_ck_full"]), 12.0 / 18.0)

    def test_transition_shares_drop_firms_with_fewer_than_two_school_links(self) -> None:
        con = duckdb.connect()
        _register_view(
            con,
            "revelio_transitions",
            [
                {"rcid": 1, "university_raw": "school a", "year": 2010, "n_transitions": 5.0},
                {"rcid": 2, "university_raw": "school a", "year": 2010, "n_transitions": 3.0},
                {"rcid": 2, "university_raw": "school b", "year": 2011, "n_transitions": 2.0},
            ],
        )
        _register_view(con, "matched_rcids", [{"rcid": 1}, {"rcid": 2}])
        _register_view(
            con,
            "revelio_inst_cw",
            [
                {"university_raw_norm": "school a", "unitid": "A"},
                {"university_raw_norm": "school b", "unitid": "B"},
            ],
        )

        _build_transition_shares(
            con,
            share_period="pre_window",
            share_base_year=2010,
            share_year_min=2008,
            share_year_max=2013,
            robustness_windows=[(2008, 2010), (2011, 2013)],
            min_universities_for_share=2,
        )
        kept = con.sql("SELECT DISTINCT c FROM transition_shares ORDER BY c").df()["c"].tolist()
        self.assertEqual(kept, [2])

    def test_instrument_builder_aggregates_variant_columns(self) -> None:
        con = duckdb.connect()
        _register_view(
            con,
            "transition_shares",
            [
                {
                    "c": 1,
                    "k": "A",
                    "share_ck": 0.75,
                    "share_ck_2008_2010": 1.0,
                    "share_ck_2011_2013": 0.0,
                    "share_ck_2008_2013": 0.75,
                    "share_ck_full": 0.33,
                },
                {
                    "c": 1,
                    "k": "B",
                    "share_ck": 0.25,
                    "share_ck_2008_2010": 0.0,
                    "share_ck_2011_2013": 1.0,
                    "share_ck_2008_2013": 0.25,
                    "share_ck_full": 0.67,
                },
            ],
        )
        _register_view(
            con,
            "ipeds_unit_growth",
            [
                {
                    "k": "A",
                    "t": 2016,
                    "g_kt": 30.0,
                    "g_kt_v1_broad_step": 30.0,
                    "g_kt_v2_broad_cumulative": 28.0,
                    "g_kt_v3_broad_predicted_opt": 15.0,
                    "g_kt_v4_matched_step": 30.0,
                    "g_kt_v5_matched_pulse": 30.0,
                    "g_kt_v6_broad_composition": 0.30,
                    "g_kt_falsification_lead4_broad": 30.0,
                    "g_kt_falsification_lead4_matched": 30.0,
                },
                {
                    "k": "B",
                    "t": 2016,
                    "g_kt": 0.0,
                    "g_kt_v1_broad_step": 0.0,
                    "g_kt_v2_broad_cumulative": 0.0,
                    "g_kt_v3_broad_predicted_opt": 0.0,
                    "g_kt_v4_matched_step": 0.0,
                    "g_kt_v5_matched_pulse": 0.0,
                    "g_kt_v6_broad_composition": 0.0,
                    "g_kt_falsification_lead4_broad": 0.0,
                    "g_kt_falsification_lead4_matched": 0.0,
                },
            ],
        )
        _build_instrument(con)
        row = con.sql("""
            SELECT
                z_ct, z_ct_v2_broad_cumulative, z_ct_v3_broad_predicted_opt,
                z_ct_falsification_lead4_broad, z_ct_falsification_lead4_matched, z_ct_full
            FROM instrument_panel
        """).df().iloc[0]
        self.assertAlmostEqual(float(row["z_ct"]), 22.5)
        self.assertAlmostEqual(float(row["z_ct_v2_broad_cumulative"]), 21.0)
        self.assertAlmostEqual(float(row["z_ct_v3_broad_predicted_opt"]), 11.25)
        self.assertAlmostEqual(float(row["z_ct_falsification_lead4_broad"]), 22.5)
        self.assertAlmostEqual(float(row["z_ct_falsification_lead4_matched"]), 22.5)
        self.assertAlmostEqual(float(row["z_ct_full"]), 9.9, places=6)

    def test_analysis_sample_restrictions_require_active_two_school_years_and_balance(self) -> None:
        con = duckdb.connect()
        _register_view(
            con,
            "analysis_panel",
            [
                {"c": 1, "t": 2008, "z_ct": 0.0, "n_universities": 0, "y_cst_lag0": 10.0},
                {"c": 1, "t": 2009, "z_ct": 5.0, "n_universities": 2, "y_cst_lag0": 11.0},
                {"c": 1, "t": 2010, "z_ct": 4.0, "n_universities": 2, "y_cst_lag0": 12.0},
                {"c": 2, "t": 2008, "z_ct": 0.0, "n_universities": 0, "y_cst_lag0": 20.0},
                {"c": 2, "t": 2009, "z_ct": 3.0, "n_universities": 1, "y_cst_lag0": 21.0},
                {"c": 2, "t": 2010, "z_ct": 2.0, "n_universities": 2, "y_cst_lag0": 22.0},
                {"c": 3, "t": 2008, "z_ct": 0.0, "n_universities": 0, "y_cst_lag0": 30.0},
                {"c": 3, "t": 2009, "z_ct": 5.0, "n_universities": 2, "y_cst_lag0": 31.0},
            ],
        )
        _register_view(
            con,
            "instrument_panel",
            [
                {"c": 1, "t": 2008, "z_ct": 0.0, "n_universities": 0},
                {"c": 1, "t": 2009, "z_ct": 5.0, "n_universities": 2},
                {"c": 1, "t": 2010, "z_ct": 4.0, "n_universities": 2},
                {"c": 2, "t": 2008, "z_ct": 0.0, "n_universities": 0},
                {"c": 2, "t": 2009, "z_ct": 3.0, "n_universities": 1},
                {"c": 2, "t": 2010, "z_ct": 2.0, "n_universities": 2},
                {"c": 3, "t": 2008, "z_ct": 0.0, "n_universities": 0},
                {"c": 3, "t": 2009, "z_ct": 5.0, "n_universities": 2},
            ],
        )
        _register_view(
            con,
            "instrument_components",
            [
                {"c": 1, "t": 2009, "k": "A", "z_ct_component": 2.5},
                {"c": 1, "t": 2009, "k": "B", "z_ct_component": 2.5},
                {"c": 1, "t": 2010, "k": "A", "z_ct_component": 1.0},
                {"c": 1, "t": 2010, "k": "B", "z_ct_component": 3.0},
                {"c": 2, "t": 2009, "k": "A", "z_ct_component": 3.0},
                {"c": 3, "t": 2009, "k": "A", "z_ct_component": 5.0},
            ],
        )
        _register_view(
            con,
            "transition_shares",
            [
                {"c": 1, "k": "A", "share_ck": 0.5},
                {"c": 1, "k": "B", "share_ck": 0.5},
                {"c": 2, "k": "A", "share_ck": 1.0},
                {"c": 3, "k": "A", "share_ck": 1.0},
            ],
        )
        _apply_analysis_sample_restrictions(
            con,
            sample_year_min=2008,
            sample_year_max=2010,
            min_active_shock_schools=2,
            require_balanced_panel=True,
        )
        out = con.sql("SELECT DISTINCT c FROM analysis_panel ORDER BY c").df()["c"].tolist()
        self.assertEqual(out, [1])
        n_rows = con.sql("SELECT COUNT(*) FROM analysis_panel").fetchone()[0]
        self.assertEqual(n_rows, 3)
        kept_components = con.sql("SELECT COUNT(*) FROM instrument_components").fetchone()[0]
        kept_shares = con.sql("SELECT COUNT(*) FROM transition_shares").fetchone()[0]
        self.assertEqual(kept_components, 4)
        self.assertEqual(kept_shares, 2)

    def test_build_interaction_plot_frame_without_omitted_reference(self) -> None:
        coefs = pd.Series({
            "z_ct_x_year_2013": 0.25,
            "z_ct_x_year_2014": 0.10,
            "z_ct_x_year_2015": -0.10,
        })
        ses = pd.Series({
            "z_ct_x_year_2013": 0.05,
            "z_ct_x_year_2014": 0.04,
            "z_ct_x_year_2015": 0.02,
        })
        out = _build_interaction_plot_frame(
            coefs=coefs,
            ses=ses,
            term_specs=[
                (2013, "z_ct_x_year_2013"),
                (2014, "z_ct_x_year_2014"),
                (2015, "z_ct_x_year_2015"),
            ],
            omitted_value=None,
            axis_name="year",
        )
        self.assertEqual(out["year"].tolist(), [2013, 2014, 2015])
        ref = out.loc[out["year"] == 2014].iloc[0]
        self.assertFalse(bool(ref["omitted"]))
        self.assertAlmostEqual(float(ref["coef"]), 0.10)
        self.assertAlmostEqual(float(ref["se"]), 0.04)
        yr_2013 = out.loc[out["year"] == 2013].iloc[0]
        self.assertAlmostEqual(float(yr_2013["se_high"]), 0.30)

    def test_build_analysis_panel_excludes_years_above_panel_max(self) -> None:
        con = duckdb.connect()
        _register_view(
            con,
            "outcomes_wide",
            [
                {"c": 1, "t": 2022, "y_cst_lag0": 10.0, "y_new_hires_lag0": 2.0},
                {"c": 1, "t": 2023, "y_cst_lag0": 12.0, "y_new_hires_lag0": 3.0},
            ],
        )
        _register_view(
            con,
            "opt_new_hires",
            [
                {
                    "c": 1,
                    "t": 2022,
                    "masters_opt_hires": 1.0,
                    "valid_masters_opt_hires": 1.0,
                    "masters_opt_hires_correction_aware": 1.0,
                },
                {
                    "c": 1,
                    "t": 2023,
                    "masters_opt_hires": 2.0,
                    "valid_masters_opt_hires": 2.0,
                    "masters_opt_hires_correction_aware": 2.0,
                },
            ],
        )
        _register_view(
            con,
            "instrument_panel",
            [
                {"c": 1, "t": 2022, "z_ct": 5.0, "n_universities": 2},
                {"c": 1, "t": 2023, "z_ct": 6.0, "n_universities": 2},
            ],
        )
        _register_view(
            con,
            "employer_crosswalk",
            [
                {"preferred_rcid": 1, "f1_state_clean": "CA"},
            ],
        )
        _build_analysis_panel(
            con,
            lag_start=0,
            lag_end=0,
            use_log_y=False,
            panel_year_min=2022,
            panel_year_max=2022,
        )
        out = con.sql("SELECT c, t, z_ct FROM analysis_panel ORDER BY t").df()
        self.assertEqual(out["t"].tolist(), [2022])
        self.assertAlmostEqual(float(out.iloc[0]["z_ct"]), 5.0)

    def test_build_analysis_panel_adds_headcount_state_columns(self) -> None:
        con = duckdb.connect()
        _register_view(
            con,
            "outcomes_wide",
            [
                {"c": 1, "t": 2008, "y_cst_lag0": 10.0, "y_cst_lag1": 8.0, "y_new_hires_lag0": 2.0, "y_new_hires_lag1": 1.0},
                {"c": 1, "t": 2009, "y_cst_lag0": 12.0, "y_cst_lag1": 10.0, "y_new_hires_lag0": 3.0, "y_new_hires_lag1": 2.0},
                {"c": 2, "t": 2008, "y_cst_lag0": 20.0, "y_cst_lag1": 15.0, "y_new_hires_lag0": 1.0, "y_new_hires_lag1": 1.0},
                {"c": 2, "t": 2009, "y_cst_lag0": 21.0, "y_cst_lag1": 20.0, "y_new_hires_lag0": 1.0, "y_new_hires_lag1": 1.0},
            ],
        )
        _register_view(
            con,
            "opt_new_hires",
            [
                {"c": 1, "t": 2008, "masters_opt_hires": 0.0, "valid_masters_opt_hires": 0.0, "masters_opt_hires_correction_aware": 0.0},
                {"c": 1, "t": 2009, "masters_opt_hires": 1.0, "valid_masters_opt_hires": 1.0, "masters_opt_hires_correction_aware": 1.0},
                {"c": 2, "t": 2008, "masters_opt_hires": 0.0, "valid_masters_opt_hires": 0.0, "masters_opt_hires_correction_aware": 0.0},
                {"c": 2, "t": 2009, "masters_opt_hires": 1.0, "valid_masters_opt_hires": 1.0, "masters_opt_hires_correction_aware": 1.0},
            ],
        )
        _register_view(
            con,
            "instrument_panel",
            [
                {"c": 1, "t": 2008, "z_ct": 1.0, "n_universities": 2},
                {"c": 1, "t": 2009, "z_ct": 2.0, "n_universities": 2},
                {"c": 2, "t": 2008, "z_ct": 1.5, "n_universities": 2},
                {"c": 2, "t": 2009, "z_ct": 2.5, "n_universities": 2},
            ],
        )
        _register_view(
            con,
            "employer_crosswalk",
            [
                {"preferred_rcid": 1, "f1_state_clean": "CA"},
                {"preferred_rcid": 2, "f1_state_clean": "NY"},
            ],
        )
        _build_analysis_panel(
            con,
            lag_start=0,
            lag_end=1,
            use_log_y=False,
            panel_year_min=2008,
            panel_year_max=2009,
            conditioning_baseline_window_start=2008,
            conditioning_baseline_window_end=2008,
        )
        out = con.sql(
            """
            SELECT c, t, headcount_lag0_raw, headcount_lag1_raw,
                   headcount_size_baseline, headcount_growth_asinh
            FROM analysis_panel
            WHERE c = 1 AND t = 2009
            """
        ).df()
        row = out.iloc[0]
        self.assertAlmostEqual(float(row["headcount_lag0_raw"]), 12.0)
        self.assertAlmostEqual(float(row["headcount_lag1_raw"]), 10.0)
        self.assertAlmostEqual(float(row["headcount_size_baseline"]), 10.0)
        self.assertAlmostEqual(
            float(row["headcount_growth_asinh"]),
            math.asinh(12.0) - math.asinh(10.0),
            places=6,
        )

    def test_prepare_first_stage_state_panel_uses_pre_window_and_year_bins(self) -> None:
        panel = pd.DataFrame(
            [
                {"c": "A", "t": 2008, "t_num": 2008, "headcount_lag0_raw": 10.0, "headcount_lag1_raw": 9.0},
                {"c": "B", "t": 2008, "t_num": 2008, "headcount_lag0_raw": 20.0, "headcount_lag1_raw": 18.0},
                {"c": "C", "t": 2008, "t_num": 2008, "headcount_lag0_raw": 30.0, "headcount_lag1_raw": 25.0},
                {"c": "D", "t": 2008, "t_num": 2008, "headcount_lag0_raw": 40.0, "headcount_lag1_raw": 38.0},
                {"c": "A", "t": 2014, "t_num": 2014, "headcount_lag0_raw": 100.0, "headcount_lag1_raw": 90.0},
                {"c": "B", "t": 2014, "t_num": 2014, "headcount_lag0_raw": 200.0, "headcount_lag1_raw": 160.0},
                {"c": "C", "t": 2014, "t_num": 2014, "headcount_lag0_raw": 300.0, "headcount_lag1_raw": 220.0},
                {"c": "D", "t": 2014, "t_num": 2014, "headcount_lag0_raw": 400.0, "headcount_lag1_raw": 390.0},
                {"c": "A", "t": 2015, "t_num": 2015, "headcount_lag0_raw": 400.0, "headcount_lag1_raw": 390.0},
                {"c": "B", "t": 2015, "t_num": 2015, "headcount_lag0_raw": 500.0, "headcount_lag1_raw": 420.0},
                {"c": "C", "t": 2015, "t_num": 2015, "headcount_lag0_raw": 600.0, "headcount_lag1_raw": 520.0},
                {"c": "D", "t": 2015, "t_num": 2015, "headcount_lag0_raw": 700.0, "headcount_lag1_raw": 690.0},
            ]
        )
        out = _prepare_first_stage_state_panel(
            panel,
            baseline_window_start=2008,
            baseline_window_end=2008,
            current_size_bins=2,
            current_growth_bins=2,
            joint_size_growth_bins=2,
        )
        baseline_a = out.loc[out["c"] == "A", "baseline_size_decile"].dropna().unique().tolist()
        baseline_d = out.loc[out["c"] == "D", "baseline_size_decile"].dropna().unique().tolist()
        self.assertEqual(baseline_a, [1])
        self.assertEqual(baseline_d, [2])
        row_2014 = out.loc[(out["c"] == "D") & (out["t_num"] == 2014)].iloc[0]
        row_2015 = out.loc[(out["c"] == "A") & (out["t_num"] == 2015)].iloc[0]
        self.assertEqual(int(row_2014["current_size_decile"]), 2)
        self.assertEqual(int(row_2015["current_size_decile"]), 1)
        self.assertTrue(str(row_2014["joint_size_growth_year_fe"]).startswith("2014__"))

    def test_prepare_first_stage_outcomes_creates_asinh_and_rate(self) -> None:
        panel = pd.DataFrame(
            [
                {"masters_opt_hires_correction_aware": 3.0, "headcount_lag0_raw": 10.0},
                {"masters_opt_hires_correction_aware": 0.0, "headcount_lag0_raw": 0.0},
            ]
        )
        out = _prepare_first_stage_outcomes(panel, "masters_opt_hires_correction_aware")
        self.assertAlmostEqual(float(out.loc[0, "x_asinh"]), math.asinh(3.0), places=6)
        self.assertAlmostEqual(float(out.loc[0, "x_rate_headcount"]), 0.3, places=6)
        self.assertAlmostEqual(float(out.loc[1, "x_rate_headcount"]), 0.0, places=6)

    def test_build_first_stage_conditioning_specs_include_expected_fe_blocks(self) -> None:
        specs = _build_first_stage_conditioning_specs(
            lhs="x_bin",
            instrument_col="z_ct",
            continuous_control_cols=["size_x_year_2014", "growth_x_year_2014"],
            headline_first_stage_spec="joint_size_growth_year_fe",
        )
        self.assertEqual([spec["spec_code"] for spec in specs], ["FS0", "FS1", "FS2", "FS3", "FS4", "FS5"])
        self.assertEqual(specs[0]["formula"], "x_bin ~ z_ct | c + t")
        self.assertIn("baseline_size_year_fe", specs[1]["formula"])
        self.assertIn("current_size_year_fe", specs[2]["formula"])
        self.assertIn("current_growth_year_fe", specs[3]["formula"])
        self.assertIn("joint_size_growth_year_fe", specs[4]["formula"])
        self.assertIn("size_x_year_2014", specs[5]["formula"])
        self.assertTrue(bool(specs[4]["is_headline"]))
        self.assertFalse(bool(specs[0]["is_headline"]))

    def test_diagnostic_school_name_lookup_falls_back_to_sample_views(self) -> None:
        con = duckdb.connect()
        _register_view(
            con,
            "school_shift_sample",
            [
                {"k": "A", "school_name": "Alpha University"},
                {"k": "B", "school_name": "Beta Institute"},
            ],
        )
        _register_view(
            con,
            "ipeds_unit_growth",
            [
                {"k": "A", "t": 2022, "g_kt": 5.0},
                {"k": "B", "t": 2022, "g_kt": 0.0},
            ],
        )
        lookup = _diagnostic_school_name_lookup(con).sort_values("k").reset_index(drop=True)
        self.assertEqual(lookup["k"].tolist(), ["A", "B"])
        self.assertEqual(lookup["school_name"].tolist(), ["Alpha University", "Beta Institute"])

    def test_select_regression_instrument_cols_dedupes_and_filters(self) -> None:
        reg_cfg = {
            "instrument_col": "z_ct",
            "instrument_cols": [
                "z_ct",
                "z_ct_v2_broad_cumulative",
                "z_ct",
                "missing_variant",
                "z_ct_v4_matched_step",
            ],
        }
        out = _select_regression_instrument_cols(
            reg_cfg,
            panel_columns=["z_ct", "z_ct_v2_broad_cumulative", "z_ct_v4_matched_step"],
        )
        self.assertEqual(out, ["z_ct", "z_ct_v2_broad_cumulative", "z_ct_v4_matched_step"])

    def test_build_regression_variant_headline_summary_pivots_twfe_specs(self) -> None:
        results_df = pd.DataFrame(
            [
                {
                    "instrument_variant": "z_ct",
                    "label": "first_stage_twfe",
                    "coef_instrument": 0.12,
                    "se_instrument": 0.03,
                    "f_stat": 16.0,
                    "n_obs": 100,
                },
                {
                    "instrument_variant": "z_ct",
                    "label": "reduced_form_twfe",
                    "coef_instrument": 0.05,
                    "se_instrument": 0.02,
                    "f_stat": None,
                    "n_obs": 100,
                },
                {
                    "instrument_variant": "z_ct_v4_matched_step",
                    "label": "first_stage_twfe",
                    "coef_instrument": 0.08,
                    "se_instrument": 0.04,
                    "f_stat": 9.5,
                    "n_obs": 100,
                },
                {
                    "instrument_variant": "z_ct_v4_matched_step",
                    "label": "reduced_form_twfe",
                    "coef_instrument": 0.03,
                    "se_instrument": 0.01,
                    "f_stat": None,
                    "n_obs": 100,
                },
            ]
        )
        out = _build_regression_variant_headline_summary(results_df)
        self.assertEqual(out["instrument_variant"].tolist(), ["z_ct", "z_ct_v4_matched_step"])
        self.assertEqual(out["variant_label"].tolist(), ["Event pre/post level-growth pulse", "V4 matched step"])
        self.assertAlmostEqual(float(out.loc[0, "fs_twfe_coef"]), 0.12)
        self.assertAlmostEqual(float(out.loc[1, "rf_twfe_coef"]), 0.03)

    def test_run_regression_variants_runs_conditional_suite_only_for_main_spec(self) -> None:
        panel = pd.DataFrame(
            {
                "c": [1, 1],
                "t": [2014, 2015],
                "z_alt": [0.1, 0.2],
                "z_ct": [1.0, 2.0],
            }
        )
        reg_cfg = {
            "instrument_col": "z_ct",
            "instrument_cols": ["z_alt", "z_ct"],
            "run_conditional_suite": True,
        }
        calls: list[tuple[str, bool, str]] = []

        def fake_run_regressions(analysis_panel_df, cfg, variant_cfg, out_dir):  # type: ignore[no-untyped-def]
            out_dir.mkdir(parents=True, exist_ok=True)
            calls.append((
                variant_cfg["instrument_col"],
                bool(variant_cfg["run_conditional_suite"]),
                str(out_dir),
            ))
            pd.DataFrame(
                [
                    {
                        "label": "first_stage_twfe",
                        "instrument_col": variant_cfg["instrument_col"],
                        "coef_instrument": 0.1,
                        "se_instrument": 0.01,
                        "f_stat": 100.0,
                        "n_obs": 2,
                    },
                    {
                        "label": "reduced_form_twfe",
                        "instrument_col": variant_cfg["instrument_col"],
                        "coef_instrument": 0.2,
                        "se_instrument": 0.02,
                        "f_stat": 100.0,
                        "n_obs": 2,
                    },
                ]
            ).to_csv(out_dir / "reg_table.csv", index=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with patch(
                "company_shift_share.shift_share_analysis.run_regressions",
                side_effect=fake_run_regressions,
            ):
                run_regression_variants(panel, {}, reg_cfg, out_dir)
            self.assertEqual(
                [(instrument, run_suite) for instrument, run_suite, _ in calls],
                [("z_alt", False), ("z_ct", True)],
            )
            self.assertEqual(Path(calls[0][2]).name, "z_alt")
            self.assertEqual(Path(calls[1][2]), out_dir)
            self.assertTrue((out_dir / "regression_variant_headline_summary.csv").exists())

    def test_run_falsification_tests_disables_conditional_suite(self) -> None:
        con = duckdb.connect()
        con.register(
            "_transition_shares_df",
            pd.DataFrame(columns=["c", "k", "share_ck"]),
        )
        con.sql("CREATE OR REPLACE TEMP VIEW transition_shares AS SELECT * FROM _transition_shares_df")
        con.register(
            "_ipeds_unit_growth_df",
            pd.DataFrame(columns=["k", "g_kt_v1_broad_step", "g_kt_v4_matched_step"]),
        )
        con.sql("CREATE OR REPLACE TEMP VIEW ipeds_unit_growth AS SELECT * FROM _ipeds_unit_growth_df")
        panel = pd.DataFrame(
            {
                "c": [1, 1],
                "t": [2012, 2013],
                "z_ct_falsification_lead4_broad": [0.1, 0.2],
                "z_ct_falsification_lead4_matched": [0.3, 0.4],
            }
        )
        calls: list[tuple[str, bool]] = []

        def fake_run_regressions(analysis_panel_df, cfg, variant_cfg, out_dir):  # type: ignore[no-untyped-def]
            out_dir.mkdir(parents=True, exist_ok=True)
            calls.append((
                variant_cfg["instrument_col"],
                bool(variant_cfg["run_conditional_suite"]),
            ))
            pd.DataFrame(
                [
                    {
                        "label": "first_stage_twfe",
                        "instrument_col": variant_cfg["instrument_col"],
                        "coef_instrument": 0.1,
                        "se_instrument": 0.01,
                        "f_stat": 100.0,
                        "n_obs": 2,
                    }
                ]
            ).to_csv(out_dir / "reg_table.csv", index=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "company_shift_share.shift_share_analysis.run_regressions",
                side_effect=fake_run_regressions,
            ):
                run_falsification_tests(
                    con,
                    panel,
                    {"school_sample_window_start": 2014},
                    {"falsification_pre_period_end": 2013},
                    Path(tmpdir),
                )
        self.assertEqual(
            calls,
            [
                ("z_ct_falsification_lead4_broad", False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
