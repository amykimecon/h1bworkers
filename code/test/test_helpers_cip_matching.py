from pathlib import Path
from typing import Dict, Sequence, Set, List, Tuple

import json
import hashlib
import sys 
import os
import duckdb 
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 
from helpers import match_fields_to_cip

def test_match_fields_to_cip_assigns_expected_codes(tmp_path):
    cip_catalog = pd.DataFrame(
        {
            "CIPCode": ["11.0101", "52.0201", "14.0901", "54.0101"],
            "CIPTitle": [
                "Computer Science",
                "Business Administration and Management, General",
                "Computer Engineering",
                "History, General",
            ],
            "CIPDefinition": [
                "Focuses on computation, algorithms, data structures, and software design.",
                "Prepares individuals to plan, organize, direct, and control business operations.",
                "Prepares individuals to apply electrical and computer engineering principles.",
                "Focuses on the general study of historical events and issues.",
            ],
            "Examples": [
                "computer programming; software development",
                "business admin; MBA; management",
                "embedded systems; hardware engineering",
                "history major; general history studies",
            ],
        }
    )
    cip_path = tmp_path / "cip_catalog.csv"
    cip_catalog.to_csv(cip_path, index=False)

    df = pd.DataFrame(
        {
            "field_raw": [
                "Bachelors in Computer Science",
                "Master of Business Administration",
                "Electrical and computer engineering technology",
                "History",
                "Underwater Basket Weaving",
            ]
        }
    )

    result = match_fields_to_cip(df, cip_path, digit_length=6)

    assert list(result["cip_code"]) == ["11.0101", "52.0201", "14.0901", "54.0101", None]
    assert result.loc[0, "cip_title"] == "Computer Science"
    assert result.loc[1, "cip_title"].startswith("Business Administration")
    assert result.loc[0, "cip_match_source"] != "unmatched"
    assert result.loc[1, "cip_match_source"] != "unmatched"
    assert result.loc[0, "cip_match_score"] >= 0.25
    assert result.loc[1, "cip_match_score"] >= 0.25
    assert result.loc[2, "cip_match_score"] is not None
    assert result.loc[2, "cip_match_score"] >= 0.25
    assert result.loc[2, "cip_match_source"] != "unmatched"
    assert result.loc[3, "cip_code"] == "54.0101"
    assert result.loc[3, "cip_match_score"] is not None
    assert result.loc[3, "cip_match_score"] >= 0.25
    assert result.loc[3, "cip_match_source"] != "unmatched"
    assert result.loc[4, "cip_match_source"] == "unmatched"


def test_match_fields_to_cip_debug_details(tmp_path):
    cip_catalog = pd.DataFrame(
        {
            "CIPCode": ["54.0101", "13.1328", "45.0201"],
            "CIPTitle": [
                "History, General",
                "History Teacher Education",
                "Anthropology",
            ],
            "CIPDefinition": [
                "Focuses on the general study of historical events and issues.",
                "Prepares individuals to teach history at various educational levels.",
                "Focuses on human cultures and societies.",
            ],
            "Examples": [
                "history major; general history studies",
                "teacher training in history; social studies education",
                "cultural anthropology; social anthropology",
            ],
        }
    )
    cip_path = tmp_path / "cip_debug_catalog.csv"
    cip_catalog.to_csv(cip_path, index=False)

    df = pd.DataFrame({"field_raw": ["History"]})

    result, debug_info = match_fields_to_cip(df, cip_path, digit_length=6, return_debug=True)

    assert result.loc[0, "cip_code"] == "54.0101"
    debug_record = debug_info[0]
    assert debug_record["field_clean"] == "history"
    stage1 = debug_record.get("stage1_candidates", [])
    assert stage1, "Expected stage1 candidates to be captured"
    assert any(candidate["code"] == "54.0101" for candidate in stage1)
    assert any(candidate["code"] == "13.1328" for candidate in stage1)
    components_keys = {"overlap", "approx", "fuzzy", "penalty", "extra_weight", "approx_component", "fuzzy_component"}
    assert components_keys <= set(stage1[0]["components"].keys())
    stage2 = debug_record.get("stage2_candidates", [])
    stage3 = debug_record.get("stage3_candidates", [])
    assert isinstance(stage2, list)
    assert isinstance(stage3, list)
    assert debug_record.get("selected", {}).get("code") == "54.0101"
    assert debug_record.get("selected", {}).get("stage") in {
        "title",
        "title_precheck",
        "title_fuzzy",
        "title_best_effort",
        "definition",
        "examples",
    }


def test_slash_optional_tokens(tmp_path):
    cip_catalog = pd.DataFrame(
        {
            "CIPCode": ["52.0101", "52.1201"],
            "CIPTitle": [
                "Business/Management Studies",
                "Management Information Systems",
            ],
            "CIPDefinition": [
                "Programs that combine business and management curricula.",
                "Focuses on managing information systems in organizations.",
            ],
            "Examples": [
                "business studies; management studies",
                "information systems management",
            ],
        }
    )
    cip_path = tmp_path / "cip_slash_catalog.csv"
    cip_catalog.to_csv(cip_path, index=False)

    df = pd.DataFrame({"field_raw": ["Business Studies"]})

    result = match_fields_to_cip(df, cip_path, digit_length=6)

    assert result.loc[0, "cip_code"] == "52.0101"
    assert result.loc[0, "cip_match_source"] != "unmatched"


def test_cache_only_mode_uses_existing_matches(tmp_path):
    cip_catalog = pd.DataFrame(
        {
            "CIPCode": ["11.0101", "54.0101"],
            "CIPTitle": [
                "Computer Science",
                "History, General",
            ],
            "CIPDefinition": [
                "Computation and algorithms.",
                "Historical studies.",
            ],
            "Examples": [
                "software development; CS",
                "general history",
            ],
        }
    )
    cip_path = tmp_path / "cip_cache_catalog.csv"
    cip_catalog.to_csv(cip_path, index=False)
    cache_path = tmp_path / "cip_cache.json"

    initial_df = pd.DataFrame({"field_raw": ["Computer Science"]})
    initial_result = match_fields_to_cip(
        initial_df,
        cip_path,
        digit_length=6,
        cache_path=cache_path,
    )
    assert initial_result.loc[0, "cip_code"] == "11.0101"
    assert cache_path.exists()

    cip_path.unlink()  # ensure subsequent runs cannot hit the catalog
    follow_up = pd.DataFrame({"field_raw": ["Computer Science", "History"]})
    follow_result = match_fields_to_cip(
        follow_up,
        cip_path,
        digit_length=6,
        cache_path=cache_path,
        cache_only=True,
    )

    assert follow_result.loc[0, "cip_code"] == "11.0101"
    assert follow_result.loc[0, "cip_match_source"] != "unmatched"
    assert pd.isna(follow_result.loc[1, "cip_code"])
    assert follow_result.loc[1, "cip_match_source"] == "unmatched"

import duckdb as ddb
con = ddb.connect()
x = con.read_parquet(f"{root}/data/int/wrds_users_sep2.parquet")
testdf = con.sql("SELECT * FROM x WHERE field_raw IS NOT NULL").fetch_df_chunk(1)
y, debug_info = help.match_fields_to_cip(testdf, return_debug=True, interactive_hardcode=True, hardcode_path=f"{root}/data/crosswalks/cip/cip_hardcodes.json")
y.sample(100)[['field_raw','cip_title','cip_match_score','cip_match_source']]

# convert list of dicts to df
#pd.json_normalize(debug_info[22]['stage2_candidates'])
# help.add_hardcoded_cip_match("")
# _DEFAULT_HARDCODE_PATH = Path(root) / "data/crosswalks/cip/cip_hardcodes.json"

# dict = help._load_hardcoded_matches(_DEFAULT_HARDCODE_PATH)
