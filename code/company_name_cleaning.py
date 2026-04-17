"""
Compatibility helpers for company-name normalization.

This extracts the small shared surface that older analysis scripts still use
after the broader `revelio_h1b_company_matching` package removal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import re
import unicodedata

import pandas as pd


_US_STATE_ABBR_TO_NAME = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "Washington, D.C.",
}
_US_STATE_NAME_TO_ABBR = {v.casefold(): k for k, v in _US_STATE_ABBR_TO_NAME.items()}

_LEGAL_SUFFIX_RE = re.compile(
    r"""
    (?:\s*,?\s+|\s+)
    (?:
        l\.?\s*l\.?\s*c\.?|
        l\.?\s*p\.?|
        l\.?\s*l\.?\s*p\.?|
        p\.?\s*l\.?\s*c\.?|
        inc(?:orporated)?|
        corp(?:oration)?|
        co(?:mpany)?|
        ltd|
        limited|
        plc|
        llp|
        gmbh|
        s\.?a\.?|
        s\.?r\.?l\.?|
        pvt|
        bv|
        ag|
        oy|
        ab|
        as|
        k\.?k\.?|
        sas|
        spa|
        oyj
    )
    \.?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_TMARK_RE = re.compile(r"[™®©]")
_PAREN_OR_BAR_TAIL_RE = re.compile(r"(\(.*?\)|\|.*$)")
_TOKEN_STREAM_CLEAN_RE = re.compile(r"[^a-z0-9\s]+")


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _squeeze_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_suffixes_iteratively(name: str, max_iter: int = 3) -> str:
    out = name
    for _ in range(max_iter):
        new = _LEGAL_SUFFIX_RE.sub("", out).strip()
        if new == out:
            break
        out = new
    return out


@dataclass
class CleanName:
    raw: str
    rule_clean: str
    clean: str
    stub: str
    base: str
    token_stream: str
    domain: Optional[str]


def normalize_state(x: Any, *, to: str = "name") -> Optional[str]:
    """Normalize a US state abbreviation/full name to a stable representation."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None

    s_clean = re.sub(r"\s+", " ", s).strip()
    if len(s_clean) == 2 and s_clean.upper() in _US_STATE_ABBR_TO_NAME:
        abbr = s_clean.upper()
        return _US_STATE_ABBR_TO_NAME[abbr] if to == "name" else abbr

    abbr = _US_STATE_NAME_TO_ABBR.get(s_clean.casefold())
    if abbr:
        return _US_STATE_ABBR_TO_NAME[abbr] if to == "name" else abbr

    if s_clean.casefold() in {"washington, d.c.", "district of columbia", "district of columbia (dc)"}:
        return "Washington, D.C." if to == "name" else "DC"
    return s_clean


def clean_company_name(name: Any) -> CleanName:
    """
    Standardize a raw company name into the small set of fields that older
    matching scripts expect.
    """
    raw = "" if name is None or (isinstance(name, float) and pd.isna(name)) else str(name)
    rule_clean = _squeeze_spaces(_ascii_fold(raw.replace("&", " And ")))

    clean = _ascii_fold(_TMARK_RE.sub("", rule_clean))
    clean = _squeeze_spaces(_PAREN_OR_BAR_TAIL_RE.sub("", clean))
    clean = re.sub(r"\s?&\s?|\s\+\s", " and ", clean)
    clean = _squeeze_spaces(clean)

    stub = _strip_suffixes_iteratively(clean)
    stub = _squeeze_spaces(re.sub(r"\(.*?\)", "", stub).strip())

    base = re.sub(r"[^A-Za-z0-9]", "", stub).lower()
    token_stream = _squeeze_spaces(_TOKEN_STREAM_CLEAN_RE.sub(" ", stub.lower()))

    return CleanName(
        raw=raw,
        rule_clean=rule_clean,
        clean=clean,
        stub=stub,
        base=base,
        token_stream=token_stream,
        domain=None,
    )
