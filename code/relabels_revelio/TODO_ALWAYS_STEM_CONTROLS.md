# Implemented: Always-STEM Control Aggregation

The `always_stem` control construction in `relabel_events_generalized.py` now
uses STEM CIP2 aggregates rather than single matched CIP6 programs.

Implemented Python behavior:
- `always_stem` controls are drawn from eligible STEM CIPs in CIP2 families
  `11`, `14`, `26`, and `40`.
- Matching is done on unit/degree/STEM-CIP2 aggregate totals while retaining a
  representative `control_cip6` for audit labels.
- Downstream analysis expands matched always-STEM controls to every eligible
  CIP6 in the matched `control_cip2` family.
- The Texas Stata export carries `role_cip2`, and the wage script uses that
  field for always-STEM control rows when it is present.
