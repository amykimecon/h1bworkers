from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PIPELINE_ROOT / "01_f1_foia_clean"

spec = importlib.util.spec_from_file_location(
    "f1_indiv_stage01_person_linkage_test_module",
    STAGE_DIR / "person_linkage.py",
)
assert spec is not None and spec.loader is not None
person_linkage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(person_linkage)


class Stage01PersonLinkageTests(unittest.TestCase):
    def test_zip_decimal_suffix_does_not_split_person(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            input_path = tempdir / "foia_raw.parquet"
            crosswalk_path = tempdir / "crosswalk.parquet"
            corrected_path = tempdir / "corrected.parquet"
            full_path = tempdir / "full.parquet"
            tmp_dir = tempdir / "tmp"

            pd.DataFrame(
                [
                    {
                        "filename": "fy2017.xlsx",
                        "student_key": "s2017",
                        "individual_key": "1222551",
                        "year": "2017",
                        "school_name": "Rice University",
                        "country_of_birth": "KUWAIT",
                        "country_of_citizenship": "KUWAIT",
                        "major_1_description": "Econometrics and Quantitative Economics",
                        "employer_name": "Wafra, Inc.",
                        "employer_city": "New York",
                        "employer_state": "NY",
                        "employer_zip_code": "10154.0",
                        "program_start_date": "2017-08-25",
                        "program_end_date": "2018-12-30",
                        "authorization_start_date": "2018-09-03",
                        "authorization_end_date": "2018-11-02",
                    },
                    {
                        "filename": "fy2018.xlsx",
                        "student_key": "s2018",
                        "individual_key": "1055930",
                        "year": "2018",
                        "school_name": "Rice University",
                        "country_of_birth": "KUWAIT",
                        "country_of_citizenship": "KUWAIT",
                        "major_1_description": "Econometrics and Quantitative Economics",
                        "employer_name": "Wafra, Inc.",
                        "employer_city": "New York",
                        "employer_state": "NY",
                        "employer_zip_code": "10154",
                        "program_start_date": "2017-08-25",
                        "program_end_date": "2018-12-30",
                        "authorization_start_date": "2018-09-03",
                        "authorization_end_date": "2018-11-02",
                    },
                    {
                        "filename": "fy2019.xlsx",
                        "student_key": "s2019",
                        "individual_key": "860204",
                        "year": "2019",
                        "school_name": "Rice University",
                        "country_of_birth": "KUWAIT",
                        "country_of_citizenship": "KUWAIT",
                        "major_1_description": "Econometrics and Quantitative Economics",
                        "employer_name": "Wafra, Inc.",
                        "employer_city": "New York",
                        "employer_state": "NY",
                        "employer_zip_code": "10154",
                        "program_start_date": "2017-08-25",
                        "program_end_date": "2018-12-30",
                        "authorization_start_date": "2018-09-03",
                        "authorization_end_date": "2018-11-02",
                    },
                ]
            ).to_parquet(input_path, index=False)

            person_linkage.run_person_id_linkage(
                input_path=input_path,
                crosswalk_out_path=crosswalk_path,
                employment_corrected_out_path=corrected_path,
                full_out_path=full_path,
                tmp_dir=tmp_dir,
                overwrite=True,
            )

            crosswalk = pd.read_parquet(crosswalk_path).sort_values(
                ["year", "individual_key"]
            )

            self.assertEqual(crosswalk["person_id"].nunique(), 1)
            self.assertEqual(crosswalk["year"].tolist(), [2017, 2018, 2019])
            self.assertTrue((crosswalk["flag_ungrouped"] == 0).all())

    def test_zip_leading_zero_drop_does_not_split_person(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            input_path = tempdir / "foia_raw.parquet"
            crosswalk_path = tempdir / "crosswalk.parquet"
            corrected_path = tempdir / "corrected.parquet"
            full_path = tempdir / "full.parquet"
            tmp_dir = tempdir / "tmp"

            pd.DataFrame(
                [
                    {
                        "filename": "fy2017.xlsx",
                        "student_key": "s2017",
                        "individual_key": "1118031",
                        "year": "2017",
                        "school_name": "Boston University",
                        "country_of_birth": "INDIA",
                        "country_of_citizenship": "INDIA",
                        "major_1_description": "Computer Science",
                        "employer_name": "Pegasystems",
                        "employer_city": "CAMBRIDGE",
                        "employer_state": "MA",
                        "employer_zip_code": "2142",
                        "job_title": "Software Engineer",
                        "program_start_date": "2020-09-01",
                        "program_end_date": "2022-05-22",
                        "authorization_start_date": "2021-06-01",
                        "authorization_end_date": "2021-08-31",
                    },
                    {
                        "filename": "fy2018.xlsx",
                        "student_key": "s2018",
                        "individual_key": "381289",
                        "year": "2018",
                        "school_name": "Boston University",
                        "country_of_birth": "INDIA",
                        "country_of_citizenship": "INDIA",
                        "major_1_description": "Computer Science",
                        "employer_name": "Pegasystems",
                        "employer_city": "CAMBRIDGE",
                        "employer_state": "MA",
                        "employer_zip_code": "02142",
                        "job_title": "Software Engineer",
                        "program_start_date": "2020-09-01",
                        "program_end_date": "2022-05-22",
                        "authorization_start_date": "2021-06-01",
                        "authorization_end_date": "2021-08-31",
                    },
                    {
                        "filename": "fy2019.xlsx",
                        "student_key": "s2019",
                        "individual_key": "245529",
                        "year": "2019",
                        "school_name": "Boston University",
                        "country_of_birth": "INDIA",
                        "country_of_citizenship": "INDIA",
                        "major_1_description": "Computer Science",
                        "employer_name": "Pegasystems",
                        "employer_city": "CAMBRIDGE",
                        "employer_state": "MA",
                        "employer_zip_code": "2142",
                        "job_title": "Software Engineer",
                        "program_start_date": "2020-09-01",
                        "program_end_date": "2022-05-22",
                        "authorization_start_date": "2021-06-01",
                        "authorization_end_date": "2021-08-31",
                    },
                ]
            ).to_parquet(input_path, index=False)

            person_linkage.run_person_id_linkage(
                input_path=input_path,
                crosswalk_out_path=crosswalk_path,
                employment_corrected_out_path=corrected_path,
                full_out_path=full_path,
                tmp_dir=tmp_dir,
                overwrite=True,
            )

            crosswalk = pd.read_parquet(crosswalk_path).sort_values(
                ["year", "individual_key"]
            )

            self.assertEqual(crosswalk["person_id"].nunique(), 1)
            self.assertEqual(crosswalk["year"].tolist(), [2017, 2018, 2019])
            self.assertTrue((crosswalk["flag_ungrouped"] == 0).all())


if __name__ == "__main__":
    unittest.main()
