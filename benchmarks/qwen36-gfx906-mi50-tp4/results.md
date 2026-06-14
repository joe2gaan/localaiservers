# Results

This file includes only numbers already present in the public repository.

## Fixed-Token Single-Request Results

```text
c1_2000:  101.47 TPS backend decode
c1_10000:  95.66 TPS backend decode
c1_10000:  95.36 client wall TPS
```

## Interpretation

- `c1_2000` is reported as backend decode TPS.
- `c1_10000` includes both backend decode TPS and client wall TPS.
- Backend decode TPS and client wall TPS are separate metrics.
- The sustained public headline is the documented 90+ TPS backend decode result.
- The 100+ TPS result is a shorter fixed-token backend decode result and should be cited
  with that context.

## Result Categories

- Serving milestone: yes, for the documented fixed-token serving lane.
- Publication gate: yes, for the reproducible Qwen3.6 GFX906 MI50 TP4 public artifact.
- Source milestone: not claimed by this benchmark summary alone.
- Current winner: not claimed here; use the canonical technical README and current
  source records for any future winner language.
