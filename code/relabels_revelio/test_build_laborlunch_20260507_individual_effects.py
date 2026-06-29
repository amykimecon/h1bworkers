from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "relabels_revelio" / "build_laborlunch_20260507_individual_effects.py"
spec = importlib.util.spec_from_file_location(
    "build_laborlunch_20260507_individual_effects_test_module",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
builder = importlib.util.module_from_spec(spec)
import sys

sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


class IndividualEffectsBuilderTests(unittest.TestCase):
    def test_parse_slide_asset_refs_expands_macros_and_skips_firm_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tex = root / "deck.tex"
            tex.write_text(
                "\n".join(
                    [
                        r"\newcommand{\relabeloutput}{/tmp/relabel}",
                        r"\newcommand{\companyoutput}{/tmp/company}",
                        r"\renewcommand{\outputpath}{/tmp/slides}",
                        r"\providecommand{\figureoutput}{/tmp/figures}",
                        r"\newcommand{\designcomparisonoutput}{/tmp/company_shift_share/design_comparison}",
                        r"\newcommand{\designcomparisonfigures}{\designcomparisonoutput/figures}",
                        r"\includegraphics{\relabeloutput/generalized_relabels_plots/a.png}",
                        r"\input{\relabeloutput/generalized_relabels_plots/b.tex}",
                        r"\includegraphics{\outputpath/slides_20251204/open_doors.png}",
                        r"\includegraphics{\figureoutput/linkedin_match_share.png}",
                        r"\includegraphics{\companyoutput/slides_20260507_shift_share/c.png}",
                        r"\includegraphics{\designcomparisonfigures/nested_macro_firm_asset.png}",
                        r"\input{/tmp/company_shift_share/slides_20260507_shift_share/d.tex}",
                        r"\texttt{relabel\_broad\_bin\_treated\_control\_school\_samples\_part1.tex}",
                    ]
                )
            )

            refs = builder.parse_slide_asset_refs(tex)

        self.assertEqual(
            [str(path) for path in refs],
            [
                "/tmp/relabel/generalized_relabels_plots/a.png",
                "/tmp/relabel/generalized_relabels_plots/b.tex",
                "/tmp/slides/slides_20251204/open_doors.png",
                "/tmp/figures/linkedin_match_share.png",
            ],
        )

    def test_parse_args_preserves_default_slide_output_paths(self) -> None:
        args = builder.parse_args([])

        self.assertEqual(args.foia_plots_dir, builder.DEFAULT_FOIA_PLOTS_DIR)
        self.assertEqual(args.figure_output_dir, builder.DEFAULT_FIGURE_OUTPUT_DIR)
        self.assertEqual(args.revelio_main_output_dir, builder.DEFAULT_REVELIO_MAIN_OUTPUT_DIR)
        self.assertEqual(args.revelio_always_stem_output_dir, builder.DEFAULT_REVELIO_ALWAYS_STEM_OUTPUT_DIR)
        self.assertEqual(args.revelio_econ_output_dir, builder.DEFAULT_REVELIO_ECON_OUTPUT_DIR)
        self.assertEqual(args.revelio_controls_output_dir, builder.DEFAULT_REVELIO_CONTROLS_OUTPUT_DIR)
        self.assertEqual(args.revelio_alt_output_dir, builder.DEFAULT_REVELIO_ALT_OUTPUT_DIR)
        self.assertEqual(args.revelio_alt_tex_macros, builder.DEFAULT_REVELIO_ALT_TEX_MACROS)
        self.assertEqual(args.horizons, [0, 1, 2, 3, 4])
        self.assertEqual(args.revelio_main_did_sample, "full_sample")
        self.assertEqual(args.revelio_relabel_year_min, 2014)
        self.assertEqual(args.revelio_relabel_year_max, 2020)
        self.assertEqual(args.estimation_type, "sun_abraham")
        self.assertTrue(args.revelio_control_comparison)
        self.assertFalse(args.sun_abraham_use_weights)

    def test_parse_args_caps_revelio_horizons_at_four(self) -> None:
        args = builder.parse_args(["--horizons", "0", "3", "4", "5"])

        self.assertEqual(args.horizons, [0, 3, 4])

    def test_filter_revelio_relabel_years_uses_default_cohort_window(self) -> None:
        frame = pd.DataFrame({"relabel_year": [2013, 2014, 2020, 2021], "value": [1, 2, 3, 4]})

        filtered = builder._filter_revelio_relabel_years(
            frame,
            relabel_year_min=builder.DEFAULT_REVELIO_RELABEL_YEAR_MIN,
            relabel_year_max=builder.DEFAULT_REVELIO_RELABEL_YEAR_MAX,
        )

        self.assertEqual(filtered["relabel_year"].tolist(), [2014, 2020])

    def test_parse_args_rejects_unknown_estimator(self) -> None:
        with self.assertRaises(SystemExit):
            builder.parse_args(["--estimation-type", "bad_estimator"])

    def test_estimator_comparison_names_are_only_supported_estimators(self) -> None:
        self.assertEqual(builder._normalize_estimator_name("did"), "twfe")
        self.assertEqual(builder._normalize_estimator_name("sun_abraham_iw"), "sun_abraham")
        self.assertEqual(builder._normalize_estimator_name("callaway_santanna"), "callaway_santanna")
        self.assertIsNone(builder._normalize_estimator_name("stacked_did"))
        self.assertIsNone(builder._normalize_estimator_name("matched_control_did"))
        self.assertEqual(
            builder.ESTIMATOR_COMPARISON_TYPES,
            ("twfe", "sun_abraham", "callaway_santanna"),
        )

    def test_read_yaml_expands_root_and_run_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "\n".join(
                    [
                        'run_tag: "abc"',
                        "paths:",
                        '  output: "{root}/data/out_${run_tag}.parquet"',
                    ]
                )
            )

            data = builder._read_yaml(path)

        self.assertEqual(data["paths"]["output"], f"{builder.PROJECT_ROOT}/data/out_abc.parquet")

    def test_file_ok_and_missing_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.png"
            empty = root / "empty.png"
            missing = root / "missing.png"
            good.write_bytes(b"x")
            empty.write_bytes(b"")

            self.assertTrue(builder._file_ok(good))
            self.assertFalse(builder._file_ok(empty))
            self.assertFalse(builder._file_ok(missing))
            self.assertEqual(builder._missing_assets([good, empty, missing]), [empty, missing])

    def test_run_pdflatex_discards_nul_corrupt_aux_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tex = root / "deck.tex"
            aux = root / "deck.aux"
            nav = root / "deck.nav"
            tex.write_text(r"\documentclass{beamer}\begin{document}\end{document}")
            aux.write_bytes(b"\\newlabel{bad}\x00\x00\n")
            nav.write_text("clean nav\n")

            with patch.object(builder.subprocess, "run") as run_mock:
                builder._run_pdflatex(tex)

            self.assertFalse(aux.exists())
            self.assertEqual(nav.read_text(), "clean nav\n")
            self.assertEqual(run_mock.call_count, 2)
            run_mock.assert_called_with(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "deck.tex"],
                cwd=root,
                check=True,
            )

    def test_run_pdflatex_discards_truncated_aux_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tex = root / "deck.tex"
            aux = root / "deck.aux"
            tex.write_text(r"\documentclass{beamer}\begin{document}\end{document}")
            aux.write_text(r"\newlabel{app_foia_opt_share_pooled_degree_levels}{{26}{84}{Appendix")

            with patch.object(builder.subprocess, "run") as run_mock:
                builder._run_pdflatex(tex)

            self.assertFalse(aux.exists())
            self.assertEqual(run_mock.call_count, 2)

    def test_run_pdflatex_keeps_valid_multiline_aux_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tex = root / "deck.tex"
            aux = root / "deck.aux"
            tex.write_text(r"\documentclass{beamer}\begin{document}\end{document}")
            aux.write_text(
                "\n".join(
                    [
                        r"\HyperFirstAtBeginDocument{\ifx\hyper@anchor\@undefined",
                        r"\global\let\oldnewlabel\newlabel",
                        r"\fi}",
                        r"\gdef \@abspage@last{171}",
                    ]
                )
            )

            with patch.object(builder.subprocess, "run") as run_mock:
                builder._run_pdflatex(tex)

            self.assertTrue(aux.exists())
            self.assertEqual(run_mock.call_count, 2)

    def test_pooled_from_dynamic_uses_treated_post_weights(self) -> None:
        estimates = pd.DataFrame(
            {
                "event_t": [-2, -1, 0, 1, 4],
                "coef": [0.0, 2.0, 4.0, 8.0, 999.0],
                "se": [0.0, 1.0, 2.0, 3.0, 999.0],
            }
        )
        reg_df = pd.DataFrame(
            {
                "event_t": [-1, 0, 1, -1, 0, 1],
                "treated": [1, 1, 1, 0, 0, 0],
                "total_grads": [1.0, 2.0, 1.0, 100.0, 100.0, 100.0],
            }
        )

        coef, se = builder._pooled_from_dynamic(estimates, reg_df)

        self.assertAlmostEqual(coef, 4.5)
        self.assertAlmostEqual(se, float(np.sqrt((0.25**2) * 1.0 + (0.5**2) * 4.0 + (0.25**2) * 9.0)))

    def test_pooled_effect_stats_uses_weighted_treated_pre_mean(self) -> None:
        reg_df = pd.DataFrame(
            {
                "event_t": [-2, -2, 0, 0, -2, 0],
                "treated": [1, 1, 1, 1, 0, 0],
                "outcome": [1.0, 3.0, 5.0, 7.0, 10.0, 20.0],
                "total_grads": [1.0, 3.0, 2.0, 2.0, 100.0, 100.0],
            }
        )

        stats = builder._pooled_effect_stats(reg_df, yvar="outcome", coef=5.0)

        self.assertAlmostEqual(stats["baseline_mean"], 2.5)
        self.assertAlmostEqual(stats["effect_size"], 2.0)
        self.assertAlmostEqual(stats["treated_post_mean"], 6.0)
        self.assertAlmostEqual(stats["control_pre_mean"], 10.0)

    def test_revelio_cached_column_gap_requires_pooled_stats(self) -> None:
        results = pd.DataFrame({"outcome": ["in_us"], "horizon_years": [3], "coef": [0.1]})

        self.assertEqual(
            builder._revelio_cached_column_gap(results, "new_pooled_post"),
            [
                "baseline_mean",
                "effect_size",
                "pooled_post_event_min",
                "pooled_post_event_max",
            ],
        )
        self.assertEqual(builder._revelio_cached_column_gap(results, "old_event_time"), [])

    def test_sun_abraham_param_event_time_is_not_shifted(self) -> None:
        self.assertEqual(
            builder._sun_abraham_param_event_time("C(rel_time)[T.0.0]:cohort_stratum_dummy_1"),
            0,
        )
        self.assertEqual(
            builder._sun_abraham_param_event_time("i(rel_time, cohort_stratum_dummy_1)[T.-3.0]"),
            -3,
        )

    def test_sun_abraham_package_panel_ignores_pair_id_when_requested(self) -> None:
        reg_df = pd.DataFrame(
            {
                "unitid": [1, 1],
                "calendar_year": [2020, 2020],
                "treated": [1, 1],
                "relabel_year": [2020, 2020],
                "event_t": [0, 0],
                "pair_id": ["a", "b"],
                "broad_pair_bin": ["econ_to_quant_econ", "econ_to_quant_econ"],
                "degree_type": ["Master", "Master"],
                "outcome": [0.0, 1.0],
                "total_grads": [1.0, 3.0],
            }
        )

        with_pair = builder._generic_package_panel(reg_df, yvar="outcome", use_weights=True, include_pair_id=True)
        without_pair = builder._generic_package_panel(reg_df, yvar="outcome", use_weights=True, include_pair_id=False)

        self.assertEqual(len(with_pair), 2)
        self.assertEqual(len(without_pair), 1)
        self.assertAlmostEqual(float(without_pair["outcome"].iloc[0]), 0.75)
        self.assertAlmostEqual(float(without_pair["total_grads"].iloc[0]), 4.0)
        entity_parts = str(without_pair["pkg_entity"].iloc[0]).split("||")
        self.assertNotIn("a", entity_parts)
        self.assertNotIn("b", entity_parts)

    def test_sun_abraham_weight_toggle_controls_feols_and_event_aggregation(self) -> None:
        reg_df = pd.DataFrame(
            {
                "unitid": [1, 1, 2, 2, 3, 3],
                "calendar_year": [2018, 2020, 2018, 2020, 2018, 2020],
                "treated": [1, 1, 1, 1, 1, 1],
                "relabel_year": [2020] * 6,
                "event_t": [-2, 0, -2, 0, -2, 0],
                "pair_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
                "broad_pair_bin": ["a", "a", "a", "a", "b", "b"],
                "degree_type": ["Master"] * 6,
                "outcome": [0.0] * 6,
                "total_grads": [1.0, 1.0, 1.0, 1.0, 100.0, 100.0],
            }
        )
        params = pd.Series(
            [10.0, 20.0],
            index=[
                "i(rel_time, cohort_stratum_dummy_0)[T.0.0]",
                "i(rel_time, cohort_stratum_dummy_1)[T.0.0]",
            ],
        )
        cov = pd.DataFrame(np.zeros((2, 2)), index=params.index, columns=params.index)
        feols_kwargs: list[dict[str, object]] = []

        def fake_feols(**kwargs: object) -> object:
            feols_kwargs.append(kwargs)
            return object()

        with (
            patch("pyfixest.estimation.feols", side_effect=fake_feols),
            patch.object(builder.estimator_cmp, "_result_params_and_cov", return_value=(params, cov)),
        ):
            unweighted = builder._estimate_sun_abraham(
                reg_df,
                yvar="outcome",
                reference_event_time=-2,
                use_weights=False,
            )
            weighted = builder._estimate_sun_abraham(
                reg_df,
                yvar="outcome",
                reference_event_time=-2,
                use_weights=True,
            )

        self.assertNotIn("weights", feols_kwargs[0])
        self.assertEqual(feols_kwargs[1]["weights"], "total_grads")
        self.assertEqual(feols_kwargs[0]["vcov"], {"CRV1": "pkg_entity_id"})
        self.assertEqual(feols_kwargs[1]["vcov"], {"CRV1": "pkg_entity_id"})
        self.assertAlmostEqual(
            float(unweighted.loc[unweighted["event_t"].eq(0), "coef"].iloc[0]),
            (2 * 10.0 + 1 * 20.0) / 3.0,
        )
        self.assertAlmostEqual(
            float(weighted.loc[weighted["event_t"].eq(0), "coef"].iloc[0]),
            (2 * 1.0 * 10.0 + 100.0 * 20.0) / 102.0,
        )

    def test_revelio_generic_collapse_keeps_rows_with_missing_optional_bins(self) -> None:
        panel = pd.DataFrame(
            {
                "analysis_variant": ["stage04_all"] * 4,
                "unitid": [1, 1, 2, 2],
                "relabel_year": [2020, 2020, 2020, 2020],
                "treated_ind": [1, 1, 0, 0],
                "cohort_t": [-2, 0, -2, 0],
                "grad_year": [2018, 2020, 2018, 2020],
                "horizon_years": [0, 0, 0, 0],
                "target_year_observed": [1, 1, 1, 1],
                "event_id": ["a", "a", "b", "b"],
                "broad_pair_bin": [pd.NA, pd.NA, pd.NA, pd.NA],
                "degree_type": [pd.NA, pd.NA, pd.NA, pd.NA],
                "user_id": [10, 11, 12, 13],
                "in_us": [1.0, 0.0, 1.0, 0.0],
            }
        )

        collapsed = builder._collapse_revelio_for_generic(
            panel,
            outcome="in_us",
            horizon=0,
            event_window=5,
        )

        self.assertFalse(collapsed.empty)
        self.assertEqual(set(collapsed["event_t"].tolist()), {-2, 0})
        self.assertEqual(set(collapsed["treated"].tolist()), {0, 1})

    def test_revelio_in_us_condition_expected_variants_require_treated_control(self) -> None:
        panel = pd.DataFrame(
            {
                "analysis_variant": ["foia_linked_person_baseline"] * 5,
                "target_year_observed": [1, 1, 1, 1, 1],
                "in_us": [1, 1, 0, 0, 0],
                "treated_ind": [1, 0, 1, 0, 1],
            }
        )

        self.assertEqual(
            builder._in_us_condition_expected_variants(panel),
            ["foia_linked_person_baseline_in_us_1", "foia_linked_person_baseline_in_us_0"],
        )

        no_control = panel[panel["in_us"].eq(1) & panel["treated_ind"].eq(1)].copy()
        no_control = pd.concat([no_control, panel[panel["in_us"].eq(0)]], ignore_index=True)
        self.assertEqual(
            builder._in_us_condition_expected_variants(no_control),
            ["foia_linked_person_baseline_in_us_0"],
        )

    def test_always_stem_button_expected_paths_cover_main_revelio_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(revelio_always_stem_output_dir=Path(tmp))

            paths = builder._always_stem_button_expected_paths(args)

        self.assertEqual(
            [path.name for path in paths],
            [
                "did_att_by_variant_active_us.png",
                "did_att_by_variant_unique_employers.png",
                "did_att_by_variant_employer_tenure.png",
                "did_att_by_variant_compensation.png",
            ],
        )

    def test_full_sample_button_expected_paths_cover_main_revelio_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(revelio_econ_output_dir=Path(tmp))

            paths = builder._revelio_full_sample_button_expected_paths(args)

        self.assertEqual(
            [path.name for path in paths],
            [
                "did_att_by_full_sample_foreign_split_active_us.png",
                "did_att_by_full_sample_foreign_split_unique_employers.png",
                "did_att_by_full_sample_foreign_split_employer_tenure.png",
                "did_att_by_full_sample_foreign_split_compensation.png",
            ],
        )

    def test_revelio_in_us_condition_plot_writes_main_slide_asset(self) -> None:
        results = pd.DataFrame(
            {
                "analysis_variant": [
                    "foia_linked_person_baseline_in_us_1",
                    "foia_linked_person_baseline_in_us_1",
                    "foia_linked_person_baseline_in_us_0",
                    "foia_linked_person_baseline_in_us_0",
                ],
                "outcome": ["n_employers"] * 4,
                "horizon_years": [0, 1, 0, 1],
                "coef": [0.1, 0.2, -0.05, 0.03],
                "se": [0.01, 0.02, 0.015, 0.025],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            out_paths = builder._plot_revelio_in_us_condition_results(
                results,
                output_dir=out_dir,
                output_mode="new_pooled_post",
            )

            self.assertEqual(out_paths, [out_dir / "did_att_by_in_us_condition_unique_employers.png"])
            self.assertTrue(out_paths[0].exists())
            self.assertTrue(out_paths[0].with_suffix(".csv").exists())

    def test_write_revelio_matching_table_creates_slide_tex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "revelio_matching_stats_summary_table.tex"
            panel = pd.DataFrame(
                {
                    "analysis_variant": ["stage04_all", "stage04_all", "foia_linked_person_baseline"],
                    "target_year_observed": [1, 1, 1],
                    "horizon_years": [3, 3, 3],
                    "user_id": [1, 2, 3],
                    "unitid": [10, 20, 10],
                    "treated_ind": [1, 0, 1],
                    "imputed_foreign_ind": [1, 0, 1],
                    "school_match_score": [0.9, 0.8, 0.95],
                    "rsid": [111, 222, 333],
                    "rsid_unitid_match_count": [20, 5, 30],
                    "rsid_unitid_required_count": [10, 10, 10],
                }
            )

            builder.write_revelio_matching_table(panel, out)

            text = out.read_text()
        self.assertIn("Full-sample", text)
        self.assertIn("FOIA-linked", text)
        self.assertIn("RSID support", text)

    def test_generalized_foia_uses_collapsed_program_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                force_rebuild=True,
                foia_output_dir=root / "out",
                foia_plots_dir=root / "plots",
            )
            log = builder.RunLog(cache_hits=[], rebuilt=[], skipped=[], output_paths=[])
            with patch.object(builder.generalized, "run_pipeline", return_value={}) as run_pipeline:
                builder._run_generalized_foia(args=args, slide_assets=[], log=log)

            self.assertEqual(
                run_pipeline.call_args.kwargs["did_spec"],
                builder.generalized.DID_SPEC_COLLAPSED_UNIT_FE,
            )

    def test_foia_status_change_estimation_panel_caps_calendar_year_only_for_status(self) -> None:
        panel = pd.DataFrame(
            {
                "calendar_year": [2019, 2020, 2021, 2022],
                "status_change_share": [0.1, 0.2, 0.3, 0.4],
                "opt_share": [0.5, 0.6, 0.7, 0.8],
            }
        )

        status_panel = builder._foia_estimation_panel_for_outcome(panel, "status_change_share")
        opt_panel = builder._foia_estimation_panel_for_outcome(panel, "opt_share")

        self.assertEqual(status_panel["calendar_year"].tolist(), [2019, 2020])
        self.assertEqual(opt_panel["calendar_year"].tolist(), [2019, 2020, 2021, 2022])

    def test_foia_status_change_cache_key_records_sample_cap(self) -> None:
        args = SimpleNamespace(cache_dir=Path("/tmp/cache"), event_window=5, bootstrap_reps=199, random_seed=42)

        status_path = builder._foia_estimator_cache_path(args, "status_change_share", "twfe")
        opt_path = builder._foia_estimator_cache_path(args, "opt_share", "twfe")

        self.assertIn(builder.FOIA_STATUS_CHANGE_SAMPLE_VERSION, status_path.name)
        self.assertNotIn(builder.FOIA_STATUS_CHANGE_SAMPLE_VERSION, opt_path.name)
        self.assertIn(builder.FOIA_ESTIMATOR_PANEL_VERSION, opt_path.name)

    def test_foia_sun_abraham_weighted_cache_key_is_distinct(self) -> None:
        base = SimpleNamespace(
            cache_dir=Path("/tmp/cache"),
            event_window=5,
            bootstrap_reps=199,
            random_seed=42,
            sun_abraham_use_weights=False,
        )
        weighted = SimpleNamespace(**{**base.__dict__, "sun_abraham_use_weights": True})

        unweighted_path = builder._foia_estimator_cache_path(base, "opt_share", "sun_abraham")
        weighted_path = builder._foia_estimator_cache_path(weighted, "opt_share", "sun_abraham")

        self.assertNotEqual(unweighted_path.name, weighted_path.name)
        self.assertNotIn("sun_abraham_wgrads", unweighted_path.name)
        self.assertIn("sun_abraham_wgrads", weighted_path.name)

    def test_foia_main_plots_use_requested_estimator_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                cache_dir=root / "cache",
                foia_plots_dir=root / "plots",
                event_window=5,
                bootstrap_reps=199,
                random_seed=42,
                force_rebuild=False,
                force_estimator_rebuild=False,
                estimation_type="sun_abraham",
                foia_main_control_comparison=False,
            )
            args.cache_dir.mkdir()
            args.foia_plots_dir.mkdir()
            cache_path = args.cache_dir / "sun_rows.csv"
            cache_path.write_text(
                "\n".join(
                    [
                        "event_t,coef,se,reference_event_t",
                        "-2,0.0,0.0,-2",
                        "0,1.0,0.1,-2",
                    ]
                )
            )
            calls: list[dict[str, object]] = []

            def fake_plot(rows: pd.DataFrame, **kwargs: object) -> Path:
                calls.append({"rows": rows.copy(), **kwargs})
                out_path = Path(kwargs["out_dir"]) / f"{kwargs['file_stem']}.png"
                out_path.write_bytes(b"png")
                out_path.with_suffix(".csv").write_text(rows.to_csv(index=False))
                return out_path

            log = builder.RunLog(cache_hits=[], rebuilt=[], skipped=[], output_paths=[])
            with (
                patch.object(builder, "FOIA_ESTIMATION_APPENDICES", [("slide", "opt_share", "OPT")]),
                patch.object(builder, "_load_or_build_foia_pooled_did_panel", return_value=pd.DataFrame({"x": [1]})),
                patch.object(builder, "_ensure_foia_estimator_cache", return_value=cache_path) as ensure_cache,
                patch.object(builder, "_foia_main_summary_text", return_value="summary box"),
                patch.object(builder.generalized, "plot_did_event_study_generalized", side_effect=fake_plot),
            ):
                builder._ensure_foia_main_estimator_plots(args, log)
                self.assertTrue((args.foia_plots_dir / "pooled_opt_share_did_event_time_never_treated.png").exists())

        ensure_cache.assert_called_once()
        self.assertEqual(ensure_cache.call_args.kwargs["estimation_type"], "sun_abraham")
        self.assertEqual(calls[0]["file_stem"], "pooled_opt_share_did_event_time_never_treated")
        self.assertEqual(calls[0]["summary_text"], "summary box")
        self.assertTrue(calls[0]["rows"]["estimator"].eq("sun_abraham").all())

    def test_foia_internship_outcomes_are_registered_for_labor_lunch(self) -> None:
        yvars = [entry[1] for entry in builder.FOIA_ESTIMATION_APPENDICES]

        self.assertIn("internship_count", builder.DEFAULT_COHORT_YVARS)
        self.assertIn("internship_opt_years", builder.DEFAULT_COHORT_YVARS)
        self.assertIn("internship_count", yvars)
        self.assertIn("internship_opt_years", yvars)

    def test_revelio_new_outcomes_are_registered_for_labor_lunch(self) -> None:
        yvars = [entry[1] for entry in builder.REVELIO_ESTIMATION_APPENDICES]

        self.assertIn("n_employers", builder.indiv.OUTCOMES)
        self.assertIn("avg_employer_tenure_years", builder.indiv.OUTCOMES)
        self.assertIn("in_school", builder.indiv.OUTCOMES)
        self.assertIn("n_internship_positions", builder.indiv.OUTCOMES)
        self.assertIn("n_employers", yvars)
        self.assertIn("avg_employer_tenure_years", yvars)
        self.assertIn("in_school", yvars)
        self.assertIn("n_internship_positions", yvars)
        self.assertEqual(
            builder.indiv.OUTCOME_FILE_LABELS["avg_employer_tenure_years"],
            "employer_tenure",
        )
        self.assertEqual(builder.indiv.OUTCOME_FILE_LABELS["in_school"], "in_school")
        self.assertIn("n_internship_positions", builder.indiv.HORIZON_PROFILE_EXCLUDED_OUTCOMES)

    def test_revelio_full_sample_control_comparison_enables_always_stem(self) -> None:
        args = builder.parse_args(["--revelio-main-did-sample", "full_sample", "--revelio-control-comparison"])

        self.assertEqual(builder._revelio_event_source_mode(args.revelio_main_did_sample), "generalized_final_sample")
        self.assertEqual(
            builder._revelio_primary_control_group(args.revelio_main_did_sample),
            builder.generalized.CONTROL_GROUP_NEVER_TREATED,
        )
        self.assertEqual(
            builder._revelio_control_groups(args),
            (
                builder.generalized.CONTROL_GROUP_NEVER_TREATED,
                builder.generalized.CONTROL_GROUP_ALWAYS_STEM,
            ),
        )

    def test_revelio_econ_only_keeps_never_treated_primary_control(self) -> None:
        args = builder.parse_args(["--revelio-main-did-sample", "econ_only", "--revelio-control-comparison"])

        self.assertEqual(builder._revelio_event_source_mode(args.revelio_main_did_sample), "econ_v2")
        self.assertEqual(
            builder._revelio_primary_control_group(args.revelio_main_did_sample),
            builder.generalized.CONTROL_GROUP_NEVER_TREATED,
        )
        self.assertEqual(builder._revelio_control_groups(args), (builder.generalized.CONTROL_GROUP_NEVER_TREATED,))

    def test_foia_did_panel_cache_missing_registered_outcome_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                cache_dir=root / "cache",
                foia_output_dir=root / "foia",
                force_rebuild=False,
            )
            args.cache_dir.mkdir()
            args.foia_output_dir.mkdir()
            (args.foia_output_dir / "generalized_relabels_panel.parquet").parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"unitid": [1]}).to_parquet(args.foia_output_dir / "generalized_relabels_panel.parquet")
            cache_path = builder._foia_pooled_did_panel_cache_path(args)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            stale_panel = pd.DataFrame(
                {
                    "unitid": [1],
                    "calendar_year": [2020],
                    "treated": [1],
                    "event_t": [0],
                    "opt_share": [0.2],
                }
            )
            stale_panel.to_parquet(cache_path, index=False)

            rebuilt_panel = pd.DataFrame(
                {
                    "unitid": [1, 2],
                    "calendar_year": [2020, 2020],
                    "broad_pair_bin": ["business_52_to_52", "business_52_to_52"],
                    "degree_type": ["Master", "Master"],
                    "treated": [1, 0],
                    "event_t": [0, pd.NA],
                    "total_grads": [10.0, 12.0],
                    "stem_cip_eligible_share": [0.8, 0.7],
                    "opt_share": [0.2, 0.1],
                    "opt_stem_share": [0.1, 0.05],
                    "post_grad_authorization_years_avg": [0.3, 0.2],
                    "opt_duration_years_avg": [0.25, 0.15],
                    "opt_years_avg": [0.3, 0.2],
                    "status_change_share": [0.05, 0.04],
                    "f1_share_of_ctotalt": [0.2, 0.1],
                    "f1_share_of_cnralt": [0.5, 0.4],
                    "ctotalt": [50.0, 60.0],
                    "cnralt": [20.0, 25.0],
                    "avg_tuition": [10000.0, 9000.0],
                    "avg_tuition_ipeds": [11000.0, 9500.0],
                    "avg_fees_ipeds": [1200.0, 1000.0],
                    "avg_students_personal_funds": [22000.0, 21000.0],
                    "avg_total_funds": [44000.0, 42000.0],
                    "unique_employers": [3.0, 2.0],
                    "unique_opt_cities": [2.0, 1.0],
                    "auth_employment_tenure_years": [0.5, 0.4],
                    "employer_opt_intensity_pctile": [75.0, 60.0],
                    "internship_count": [1.0, 0.0],
                    "internship_opt_years": [0.1, 0.0],
                }
            )
            log = builder.RunLog(cache_hits=[], rebuilt=[], skipped=[], output_paths=[])
            with (
                patch.object(builder.generalized, "compute_generalized_did_panel", return_value=pd.DataFrame({"x": [1]})),
                patch.object(builder, "_foia_make_donor_control_panel", return_value=rebuilt_panel.copy()),
            ):
                out = builder._load_or_build_foia_pooled_did_panel(args, log)

            self.assertIn("internship_count", out.columns)
            self.assertIn("internship_opt_years", out.columns)
            self.assertIn("cnralt_share_of_ctotalt", out.columns)
            self.assertIn("avg_fees_ipeds", out.columns)
            self.assertIn("avg_students_personal_funds", out.columns)
            self.assertIn("avg_total_funds", out.columns)
            self.assertAlmostEqual(float(out.loc[out["unitid"].eq(1), "cnralt_share_of_ctotalt"].iloc[0]), 0.4)
            self.assertIn("internship_count", pd.read_parquet(cache_path).columns)
            self.assertIn("cnralt_share_of_ctotalt", pd.read_parquet(cache_path).columns)
            self.assertTrue(any("missing_outcomes" in item for item in log.rebuilt))

    def test_foia_cost_and_fund_outcomes_are_registered_for_labor_lunch(self) -> None:
        yvars = [yvar for _, yvar, _ in builder.FOIA_ESTIMATION_APPENDICES]
        self.assertIn("avg_tuition_ipeds", yvars)
        self.assertIn("avg_fees_ipeds", yvars)
        self.assertIn("avg_students_personal_funds", yvars)
        self.assertIn("avg_total_funds", yvars)

    def test_foia_donor_control_panel_deduplicates_reused_controls(self) -> None:
        stacked = pd.DataFrame(
            {
                "pair_id": [1, 2, 1, 2],
                "unitid": [10, 10, 20, 20],
                "calendar_year": [2020, 2020, 2020, 2020],
                "relabel_year": [2020, 2021, 2020, 2021],
                "relabel_type": ["business_52_to_52"] * 4,
                "broad_pair_bin": ["business_52_to_52"] * 4,
                "degree_type": ["Master"] * 4,
                "treated": [1, 1, 0, 0],
                "event_t": [0, -1, 0, -1],
                "opt_share": [0.5, 0.6, 0.2, 0.2],
                "total_internships": [1.0, 2.0, 3.0, 3.0],
                "total_internship_opt_years": [0.2, 0.3, 0.4, 0.4],
                "internship_count": [0.2, 0.3, 0.4, 0.4],
                "internship_opt_years": [0.02, 0.03, 0.04, 0.04],
                "total_grads": [5.0, 6.0, 7.0, 7.0],
            }
        )

        donor = builder._foia_make_donor_control_panel(stacked)
        controls = donor[donor["treated"].eq(0)]

        self.assertEqual(len(controls), 1)
        self.assertEqual(controls["unitid"].iloc[0], 20)
        self.assertTrue(controls["relabel_year"].isna().all())
        self.assertTrue(controls["event_t"].isna().all())
        self.assertEqual(controls["control_panel_role"].iloc[0], "donor")
        self.assertEqual(float(controls["internship_count"].iloc[0]), 0.4)

    def test_foia_donor_twfe_keeps_controls_without_event_time(self) -> None:
        rows: list[dict[str, object]] = []
        for unitid, treated, relabel_year in [(1, 1, 2020), (2, 0, pd.NA), (3, 0, pd.NA)]:
            for year in range(2015, 2023):
                event_t = year - int(relabel_year) if treated else pd.NA
                rows.append(
                    {
                        "pair_id": str(unitid),
                        "unitid": unitid,
                        "calendar_year": year,
                        "relabel_year": relabel_year,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "degree_type": "Master",
                        "treated": treated,
                        "event_t": event_t,
                        "opt_share": 0.1 + (0.05 if treated and event_t >= 0 else 0.0),
                        "total_grads": 10.0,
                        "control_panel_role": "treated" if treated else "donor",
                    }
                )
        panel = pd.DataFrame(rows)

        reg = builder._prepare_foia_donor_regression_df(panel, yvar="opt_share", event_window=5)
        estimates = builder._compute_foia_donor_twfe_rows(
            panel,
            yvar="opt_share",
            args=SimpleNamespace(event_window=5),
        )

        self.assertEqual(reg["treated"].value_counts().to_dict(), {0: 16, 1: 8})
        self.assertEqual(int(reg.loc[reg["treated"].eq(0), "event_t"].notna().sum()), 0)
        self.assertFalse(estimates.empty)
        self.assertTrue((estimates["control_n_schools"] == 2).all())

    def test_foia_control_comparison_did_uses_laborlunch_event_time_display(self) -> None:
        rows = pd.DataFrame(
            {
                "event_t": [-5, -2, 0, 4],
                "coef": [-0.1, 0.0, 0.2, 0.3],
                "se": [0.02, 0.0, 0.04, 0.05],
                "reference_event_t": [-2, -2, -2, -2],
            }
        )
        captured: dict[str, object] = {}

        def fake_savefig(fig: object, path: Path, *, dpi: int = 220) -> None:
            captured["ax"] = fig.axes[0]
            captured["path"] = Path(path)

        with tempfile.TemporaryDirectory() as tmp, patch.object(builder.llstyle, "savefig", side_effect=fake_savefig):
            args = SimpleNamespace(foia_plots_dir=Path(tmp))
            out_path = builder._plot_foia_main_control_comparison_did(
                {builder.generalized.CONTROL_GROUP_NEVER_TREATED: rows},
                args=args,
                yvar="opt_share",
                summary_text=None,
            )

            ax = captured["ax"]
            self.assertEqual(
                [label.get_text() for label in ax.get_xticklabels()],
                ["-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"],
            )
            vertical_line_x = [
                float(xdata[0])
                for line in ax.lines
                for xdata in [list(line.get_xdata())]
                if len(xdata) == 2 and float(xdata[0]) == float(xdata[1])
            ]
            self.assertIn(builder.generalized.LABORLUNCH_DI_D_EVENT_LINE_X, vertical_line_x)
            self.assertEqual([float(x) for x in ax.lines[0].get_xdata()], [-5.0, -2.0, 0.0, 4.0])
            self.assertEqual(pd.read_csv(out_path.with_suffix(".csv"))["event_t"].tolist(), [-5, -2, 0, 4])

    def test_foia_main_summary_uses_treated_weighted_dynamic_average(self) -> None:
        did_panel = pd.DataFrame(
            [
                {
                    "unitid": unitid,
                    "calendar_year": 2020 + event_t,
                    "event_t": event_t,
                    "treated": treated,
                    "total_grads": 1.0,
                    "opt_stem_share": 0.10 if treated else 0.90,
                }
                for event_t in (-5, -2)
                for unitid, treated in ((10, 1), (20, 0))
            ]
        )
        dynamic_rows = pd.DataFrame(
            {
                "event_t": [-1, 0, 1, 2, 3],
                "coef": [0.01, 0.02, 0.03, 0.04, 0.05],
                "se": [0.001, 0.002, 0.003, 0.004, 0.005],
                "treated_total_grads": [10.0, 20.0, 30.0, 40.0, 50.0],
                "reference_event_t": [-2, -2, -2, -2, -2],
            }
        )

        text = builder._foia_main_summary_text(
            did_panel,
            yvar="opt_stem_share",
            dynamic_rows=dynamic_rows,
            event_window=5,
        )

        self.assertEqual(
            text,
            "Baseline mean (t = -4 to -1): 10.0 pp\n"
            "Dynamic avg (t = 0 to 4): 3.7 pp (0.2 pp)\n"
            "Effect size: 36.7%",
        )

    def test_foia_preperiod_patch_extends_lower_bound_to_cover_event_window(self) -> None:
        panel = pd.DataFrame({"relabel_year": [2014, 2018]})
        old_min = builder.generalized.base.PLOT_YEAR_MIN

        with builder._patched_foia_preperiod_window(panel, 5):
            self.assertEqual(builder.generalized.base.PLOT_YEAR_MIN, min(old_min, 2009))

        self.assertEqual(builder.generalized.base.PLOT_YEAR_MIN, old_min)


if __name__ == "__main__":
    unittest.main()
