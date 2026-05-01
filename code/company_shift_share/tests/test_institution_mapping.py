from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from company_shift_share.institution_mapping import (
    DETERMINISTIC_WITH_FALLBACK_METHOD,
    load_revelio_school_map,
    normalize_school_key_series,
)


class InstitutionMappingTests(unittest.TestCase):
    def test_deterministic_school_map_overrides_legacy_and_backfills_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            deterministic_path = tempdir / "deterministic.parquet"
            ref_path = tempdir / "ref_inst_catalog.parquet"
            legacy_path = tempdir / "legacy_crosswalk.parquet"

            pd.DataFrame(
                [
                    {
                        "inst_key": "ecole polytechnique",
                        "inst_candidates": [
                            {"ref_inst_id": "missing_unitid", "hybrid_score": 0.98},
                            {"ref_inst_id": "polytechnique", "hybrid_score": 0.91},
                        ],
                    }
                ]
            ).to_parquet(deterministic_path, index=False)

            pd.DataFrame(
                [
                    {"ref_inst_id": "missing_unitid", "main_unitid": None},
                    {"ref_inst_id": "polytechnique", "main_unitid": "2001"},
                ]
            ).to_parquet(ref_path, index=False)

            pd.DataFrame(
                [
                    {"university_raw": "Ecole Polytechnique", "UNITID": 9999},
                    {"university_raw": "Stanford University", "UNITID": 1001},
                ]
            ).to_parquet(legacy_path, index=False)

            school_map, metadata = load_revelio_school_map(
                legacy_crosswalk=legacy_path,
                deterministic_triple_map=deterministic_path,
                ref_inst_catalog=ref_path,
            )

            self.assertEqual(metadata["mapping_method"], DETERMINISTIC_WITH_FALLBACK_METHOD)
            out = school_map.set_index("university_raw_key")
            self.assertEqual(out.loc["ecole polytechnique", "unitid"], "2001")
            self.assertEqual(out.loc["ecole polytechnique", "ref_inst_id"], "polytechnique")
            self.assertEqual(out.loc["stanford university", "unitid"], "1001")
            self.assertEqual(
                out.loc["stanford university", "row_mapping_method"],
                "legacy_revelio_ipeds_foia_v1",
            )

    def test_normalize_school_key_series_matches_stage04_style(self) -> None:
        series = pd.Series(["École Polytechnique!!!", " University   of   São Paulo ", None])
        normalized = normalize_school_key_series(series)

        self.assertEqual(normalized.iloc[0], "ecole polytechnique")
        self.assertEqual(normalized.iloc[1], "university of sao paulo")
        self.assertTrue(pd.isna(normalized.iloc[2]))


if __name__ == "__main__":
    unittest.main()
