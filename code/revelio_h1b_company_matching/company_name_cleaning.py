
"""
Utilities for cleaning/normalizing company names, parsing Stata cleaning rules,
and generating token blocks for fuzzy matching.

Designed for:
- FOIA H-1B employer names (HQ/worksite location + NAICS)
- Revelio LinkedIn company names (workforce locations + NAICS/LEI)

Key design goals:
- Deterministic, reproducible IDs
- Conservative default thresholds to avoid over-merging subsidiaries
- Ability to reuse the detailed alias rules in firm_names_preclean.do
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Dict, List, Tuple, Callable, Any
import hashlib
import re
import unicodedata

import pandas as pd
from unidecode import unidecode

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None


# -----------------------------------------------------------------------------
# US state normalization
# -----------------------------------------------------------------------------

_US_STATE_ABBR_TO_NAME: Dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "Washington, D.C.",
}

_US_STATE_NAME_TO_ABBR = {v.casefold(): k for k, v in _US_STATE_ABBR_TO_NAME.items()}


def normalize_state(x: Any, *, to: str = "name") -> Optional[str]:
    """
    Normalize a US state. Supports abbreviations and full names.
    Returns None if missing/unknown.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None

    s_clean = re.sub(r"\s+", " ", s).strip()

    # Abbreviation?
    if len(s_clean) == 2 and s_clean.upper() in _US_STATE_ABBR_TO_NAME:
        ab = s_clean.upper()
        return _US_STATE_ABBR_TO_NAME[ab] if to == "name" else ab

    # Full name?
    ab = _US_STATE_NAME_TO_ABBR.get(s_clean.casefold())
    if ab:
        return _US_STATE_ABBR_TO_NAME[ab] if to == "name" else ab

    # Some FOIA files have "Washington, D.C." already
    if s_clean.casefold() in ("washington, d.c.", "district of columbia", "district of columbia (dc)"):
        return "Washington, D.C." if to == "name" else "DC"

    # If we can't interpret it, return the cleaned string for traceability
    return s_clean


# -----------------------------------------------------------------------------
# NAICS normalization
# -----------------------------------------------------------------------------

def normalize_naics(x: Any) -> Optional[str]:
    """
    Standardize NAICS code to a digit-only string (2 to 6 digits typical).
    Returns None if missing.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    # Some NAICS are longer (e.g., with decimals); keep up to 6 by default.
    return digits[:6]


def naics_prefix_match_level(a: Optional[str], b: Optional[str]) -> int:
    """
    Returns a match level:
      6 = exact 6-digit match
      4 = exact first 4 digits match
      3 = exact first 3 digits match
      2 = exact first 2 digits match
      0 = no match / missing
    """
    if not a or not b:
        return 0
    a = normalize_naics(a)
    b = normalize_naics(b)
    if not a or not b:
        return 0
    if a == b and len(a) >= 6 and len(b) >= 6:
        return 6
    for k in (4, 3, 2):
        if len(a) >= k and len(b) >= k and a[:k] == b[:k]:
            return k
    return 0


# -----------------------------------------------------------------------------
# Company name normalization (general)
# -----------------------------------------------------------------------------

# Broad list of legal suffixes (covers employer_merge.R and common variants)
_LEGAL_SUFFIX_RE = re.compile(
    r"""
    (?:\s*,?\s+|
       \s+)
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

# Things to remove anywhere in the string
_TMARK_RE = re.compile(r"[™®©]")
_PAREN_OR_BAR_TAIL_RE = re.compile(r"(\(.*?\)|\|.*$)")

# Keep only alnum + spaces for token stream
_TOKEN_STREAM_CLEAN_RE = re.compile(r"[^a-z0-9\s]+")

# Extract domain-ish substrings
_DOMAIN_RE = re.compile(r"([A-Za-z0-9\-]+\.[a-z]{2,})(?:[^A-Za-z0-9]|$)")


def _ascii_fold(s: str) -> str:
    # Similar spirit to stringi::stri_trans_general(..., "Latin-ASCII")
    s = unicodedata.normalize("NFKD", s)
    s = unidecode(s)
    return s


def _squeeze_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_suffixes_iteratively(name: str, max_iter: int = 3) -> str:
    out = name
    for _ in range(max_iter):
        new = _LEGAL_SUFFIX_RE.sub("", out).strip()
        if new == out:
            break
        out = new
    return out


def _basic_rule_preclean(s: str) -> str:
    """
    Roughly aligns with the early steps in firm_names_preclean.do:
    - remove punctuation
    - normalize '&' to 'And'
    - drop some extremely common filler words (The, Company, Companies)
    """
    out = _ascii_fold(s)
    out = out.replace("&", " And ")
    out = re.sub(r"[,\.\'\/\-]", " ", out)
    out = _squeeze_spaces(out)

    # common prefixes / fillers
    out = re.sub(r"^\s*The\s+", "", out, flags=re.IGNORECASE)

    # common words the Stata do file removes
    out = re.sub(r"\bCompanies\b", " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\bCompany\b", " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\bCorporation\b", " ", out, flags=re.IGNORECASE)

    # common abbreviations / synonyms from the do file
    out = re.sub(r"\bCntr\b", "Center", out, flags=re.IGNORECASE)
    out = re.sub(r"\bTheatre\b", "Theater", out, flags=re.IGNORECASE)

    out = _squeeze_spaces(out)
    return out


def normalize_rule_string(s: str) -> str:
    """
    Case-insensitive key for rule matching.
    Keep spacing as-is to avoid over-broad matches (e.g., 'Hca ' should not match 'healthcare').
    """
    return str(s).casefold()


@dataclass(frozen=True)
class AtomicRuleCond:
    field: str  # "emp" or "orig"
    kind: str   # "eq" | "startswith" | "contains" | "regex"
    value: str  # string or pattern text
    compiled: Any = None  # compiled regex if kind == "regex"


@dataclass
class StataAliasRule:
    replacement: str
    conds_or: List[AtomicRuleCond] | None = None
    complex_code: Any | None = None  # compiled python expression
    raw_condition: str | None = None

    def apply(self, emp: str, orig: str) -> Optional[str]:
        """
        Apply this rule to (emp, orig). Returns replacement if matched else None.
        """
        emp_k = normalize_rule_string(emp)
        orig_k = normalize_rule_string(orig)

        if self.complex_code is not None:
            env = {
                "emp": emp_k,
                "orig": orig_k,
                "strpos": _stata_strpos,
                "regexm": _stata_regexm,
            }
            try:
                ok = bool(eval(self.complex_code, {"__builtins__": {}}, env))
            except Exception:
                ok = False
            return self.replacement if ok else None

        if not self.conds_or:
            return None

        for c in self.conds_or:
            field_val = emp_k if c.field == "emp" else orig_k
            if c.kind == "eq":
                if field_val == normalize_rule_string(c.value):
                    return self.replacement
            elif c.kind == "startswith":
                if field_val.startswith(normalize_rule_string(c.value)):
                    return self.replacement
            elif c.kind == "contains":
                if normalize_rule_string(c.value) in field_val:
                    return self.replacement
            elif c.kind == "regex":
                pat = c.compiled
                if pat is None:
                    pat = re.compile(c.value, flags=re.IGNORECASE)
                if pat.search(field_val):
                    return self.replacement
        return None


def _stata_strpos(s: str, sub: str) -> int:
    """Stata-like strpos: 1-indexed position, 0 if not found."""
    i = s.find(normalize_rule_string(sub))
    return 0 if i < 0 else i + 1


def _stata_regexm(s: str, pattern: str) -> bool:
    return re.search(pattern, s, flags=re.IGNORECASE) is not None


def _split_outside_quotes(s: str, sep: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    in_q = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '"':
            in_q = not in_q
            buf.append(ch)
        elif (not in_q) and ch == sep:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _translate_stata_condition_to_python(cond: str) -> str:
    """
    Translate a Stata condition (subset used in firm_names_preclean.do) to a Python expression.
    Only used for the relatively small set of complex rules (with '&' and/or parentheses).
    """
    c = cond.strip()
    # normalize function names and variables
    c = re.sub(r"\bustrregexm\b", "regexm", c)
    # allow strpos(employer,...) by mapping "employer" to "orig"
    c = re.sub(r"\bemployer\b", "orig", c)
    # map (rare) "emp" to "emp" (no change)
    # operators outside quotes:
    c = " ".join(c.split())  # squeeze whitespace

    # Replace & and | outside quotes
    c = _replace_outside_quotes(c, "&", " and ")
    c = _replace_outside_quotes(c, "|", " or ")

    return c


def _replace_outside_quotes(s: str, target: str, repl: str) -> str:
    buf: List[str] = []
    in_q = False
    for ch in s:
        if ch == '"':
            in_q = not in_q
            buf.append(ch)
        elif (not in_q) and ch == target:
            buf.append(repl)
        else:
            buf.append(ch)
    return "".join(buf)


def _strip_ws_outside_quotes(s: str) -> str:
    """Remove whitespace outside quoted sections, preserve whitespace inside quotes."""
    buf: List[str] = []
    in_q = False
    for ch in s:
        if ch == '"':
            in_q = not in_q
            buf.append(ch)
        elif ch.isspace() and not in_q:
            continue
        else:
            buf.append(ch)
    return "".join(buf)


_ATOMIC_EQ_RE = re.compile(r'^(emp|employer)==\"([^\"]*)\"$')
_ATOMIC_STRPOS_RE = re.compile(r'^strpos\((emp|employer),\"([^\"]*)\"\)(==1|>0)$')
_ATOMIC_REGEX_RE = re.compile(r'^regexm\((emp|employer),\"([^\"]*)\"\)$')



def load_stata_alias_rules(do_path: str) -> List[StataAliasRule]:
    """
    Parse a Stata .do file in the style of firm_names_preclean.do and extract
    literal alias rules of the form:

        replace emp = "Canonical Name" if <condition>

    We intentionally ignore non-literal RHS commands (subinstr(), substr(), regexr(), ...)
    because the general punctuation/suffix cleanup is implemented directly in Python.

    Returns rules in file order.

    Notes
    -----
    The condition language we support is a practical subset used in firm_names_preclean.do:
    - emp == "X" / employer == "X"
    - strpos(emp, "X") == 1   (starts with)
    - strpos(emp, "X") > 0    (contains)
    - ustrregexm(emp, "pat")  (regex contains)
    - OR with | between atoms
    - A small number of AND rules using & (handled via a safe eval translation)
    """
    rules: List[StataAliasRule] = []
    pat = re.compile(r'^replace\s+emp\s*=\s*"([^"]*)"\s*if\s*(.+)$')

    with open(do_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*") or line.startswith("//"):
                continue

            m = pat.match(line)
            if not m:
                continue

            replacement = m.group(1)
            cond = m.group(2).strip().rstrip(";")

            # # Skip overly broad HCA rule (was matching "healthcare" unintentionally)
            # if replacement.lower() == "hca" and 'strpos(emp,"Hca ")' in cond:
            #     continue

            # Complex rules (AND / parentheses) get translated to a Python expression.
            if _contains_outside_quotes(cond, "&"):
                expr = _translate_stata_condition_to_python(cond)
                try:
                    code_obj = compile(expr, f"<stata_rule:{replacement}>", "eval")
                except SyntaxError:
                    code_obj = None
                rules.append(StataAliasRule(replacement=replacement, complex_code=code_obj, raw_condition=cond))
                continue

            # Otherwise: OR of atomic conditions
            parts = [p.strip() for p in _split_outside_quotes(cond, "|")]
            atomic_conds: List[AtomicRuleCond] = []
            complex_fallback = False

            for p in parts:
                pn = _strip_ws_outside_quotes(p)
                pn = pn.replace("ustrregexm", "regexm")  # normalize function name

                # atomic: equality
                m_eq = _ATOMIC_EQ_RE.match(pn)
                if m_eq:
                    field = "emp" if m_eq.group(1) == "emp" else "orig"
                    atomic_conds.append(AtomicRuleCond(field=field, kind="eq", value=m_eq.group(2)))
                    continue

                # atomic: strpos
                m_sp = _ATOMIC_STRPOS_RE.match(pn)
                if m_sp:
                    field = "emp" if m_sp.group(1) == "emp" else "orig"
                    txt = m_sp.group(2)
                    op = m_sp.group(3)
                    kind = "startswith" if op == "==1" else "contains"
                    atomic_conds.append(AtomicRuleCond(field=field, kind=kind, value=txt))
                    continue

                # atomic: regexm
                m_rx = _ATOMIC_REGEX_RE.match(pn)
                if m_rx:
                    field = "emp" if m_rx.group(1) == "emp" else "orig"
                    pat_txt = m_rx.group(2)
                    atomic_conds.append(
                        AtomicRuleCond(
                            field=field,
                            kind="regex",
                            value=pat_txt,
                            compiled=re.compile(pat_txt, re.IGNORECASE),
                        )
                    )
                    continue

                # Can't parse this OR-atom: fallback to translated eval
                complex_fallback = True
                break

            if complex_fallback:
                expr = _translate_stata_condition_to_python(cond)
                try:
                    code_obj = compile(expr, f"<stata_rule:{replacement}>", "eval")
                except SyntaxError:
                    code_obj = None
                rules.append(StataAliasRule(replacement=replacement, complex_code=code_obj, raw_condition=cond))
            else:
                rules.append(StataAliasRule(replacement=replacement, conds_or=atomic_conds, raw_condition=cond))

    return rules


def _contains_outside_quotes(s: str, ch: str) -> bool:
    in_q = False
    for c in s:
        if c == '"':
            in_q = not in_q
        elif (not in_q) and c == ch:
            return True
    return False


def apply_stata_alias_rules(
    name: Any,
    *,
    rules: Optional[List[StataAliasRule]] = None,
) -> str:
    """
    Apply Stata alias rules (if provided) to a single name, using a Stata-like
    pre-cleaning step first.

    If rules is None, returns the pre-cleaned name.
    """
    raw = "" if name is None or (isinstance(name, float) and pd.isna(name)) else str(name)
    emp = _basic_rule_preclean(raw)
    orig = emp  # we use same normalization for "employer" comparisons
    if not rules:
        return emp

    cur = emp
    for r in rules:
        rep = r.apply(cur, orig)
        if rep is not None:
            cur = rep
    return cur


@dataclass
class CleanName:
    raw: str
    rule_clean: str
    clean: str
    stub: str
    base: str
    token_stream: str
    domain: Optional[str]


def clean_company_name(
    name: Any,
    *,
    stata_rules: Optional[List[StataAliasRule]] = None,
) -> CleanName:
    """
    Produce multiple standardized fields from a raw company name.

    - rule_clean: (optional) application of Stata alias rules + early standardization
    - clean: ascii, no trailing parenthetical/bar section
    - stub: clean with legal suffix stripped
    - base: stub with only alphanumeric (no spaces/punct)
    - token_stream: normalized stream for token frequency / blocking
    - domain: extracted domain (if any)
    """
    raw = "" if name is None or (isinstance(name, float) and pd.isna(name)) else str(name)
    # Apply Stata alias rules (optional) as a first-stage canonicalization
    rule_clean = apply_stata_alias_rules(raw, rules=stata_rules)

    # Remove trademarks and convert to ASCII
    x = _ascii_fold(_TMARK_RE.sub("", rule_clean))
    x = _squeeze_spaces(x)

    # Remove trailing parenthetical or bar-suffix chunks (matches employer_merge.R spirit)
    x = _PAREN_OR_BAR_TAIL_RE.sub("", x).strip()

    # Replace & and + with "and" in a token-friendly way
    x = re.sub(r"\s?&\s?|\s\+\s", " and ", x)
    x = _squeeze_spaces(x)

    # Extract domain (best effort)
    m = _DOMAIN_RE.search(x)
    domain = m.group(1).lower() if m else None

    # Stub: strip legal suffixes
    stub = _strip_suffixes_iteratively(x)

    # Remove any remaining parentheticals
    stub = re.sub(r"\(.*?\)", "", stub).strip()
    stub = _squeeze_spaces(stub)

    # Base: only alphanumeric (case-insensitive)
    base = re.sub(r"[^A-Za-z0-9]", "", stub).lower()

    # Token stream for frequency-based blocking (lower, remove punctuation, squeeze spaces)
    token_stream = _TOKEN_STREAM_CLEAN_RE.sub(" ", stub.lower())
    token_stream = _squeeze_spaces(token_stream)

    clean = x
    return CleanName(
        raw=raw,
        rule_clean=rule_clean,
        clean=clean,
        stub=stub,
        base=base,
        token_stream=token_stream,
        domain=domain,
    )


# -----------------------------------------------------------------------------
# Token frequency + rare-token selection (blocking)
# -----------------------------------------------------------------------------

def compute_token_frequencies(token_streams: Iterable[str]) -> pd.DataFrame:
    """
    Given iterable of token streams (space-separated), return df with columns:
      token, count, freq
    """
    counts: Dict[str, int] = {}
    total = 0
    for s in token_streams:
        if not s:
            continue
        toks = s.split()
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
            total += 1
    if total == 0:
        return pd.DataFrame({"token": [], "count": [], "freq": []})
    df = pd.DataFrame({"token": list(counts.keys()), "count": list(counts.values())})
    df["freq"] = df["count"] / float(total)
    return df.sort_values(["freq", "count", "token"], ascending=[False, False, True]).reset_index(drop=True)


def select_rare_tokens(
    token_stream: str,
    token_freq: Dict[str, float],
    *,
    top_n_stop: int = 100,
    max_tokens: int = 5,
) -> List[str]:
    """
    Select up to `max_tokens` rare tokens from a token stream, excluding the `top_n_stop`
    most frequent tokens (computed over a combined corpus).
    """
    if not token_stream:
        return []
    toks = token_stream.split()
    if not toks:
        return []

    # Build stoplist from top_n_stop tokens
    # Caller should pass token_freq dict and we interpret "top_n_stop" as the number of
    # most frequent tokens to exclude.
    # For performance, create stoplist outside if calling many times.
    # Here we compute on the fly (cheap for moderate dict sizes).
    top_tokens = sorted(token_freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n_stop]
    stop = {t for t, _ in top_tokens}

    # keep tokens not in stop
    toks2 = [t for t in toks if t not in stop]
    if not toks2:
        return []

    # sort by ascending frequency (rarest first), then alphabetical
    toks2_sorted = sorted(toks2, key=lambda t: (token_freq.get(t, 0.0), t))
    # unique, preserve order
    out: List[str] = []
    seen = set()
    for t in toks2_sorted:
        if t in seen:
            continue
        out.append(t)
        seen.add(t)
        if len(out) >= max_tokens:
            break
    return out


# -----------------------------------------------------------------------------
# Similarity helpers
# -----------------------------------------------------------------------------

def name_similarity(a_stub: str, a_base: str, b_stub: str, b_base: str) -> float:
    """
    0-100 similarity score, using RapidFuzz when available.
    """
    if fuzz is None:
        # Fallback: very rough
        return 100.0 if a_base == b_base else 0.0

    # Base ratio catches small typos; token_sort catches word reordering
    s1 = fuzz.ratio(a_base, b_base)
    s2 = fuzz.token_sort_ratio(a_stub.lower(), b_stub.lower())
    return float(max(s1, s2))


def stable_hash_id(prefix: str, parts: Iterable[Any], *, n: int = 12) -> str:
    """
    Deterministic short ID: PREFIX-<sha1(parts)[0:n]>.
    """
    s = "|".join("" if p is None else str(p) for p in parts)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}-{h}"
