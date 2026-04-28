# Eval Baseline

`report.json` is the committed mini evaluation baseline used by
`python -m vrs.eval.ci`. It is generated from a deterministic synthetic scoring
fixture, so it does not require GPU inference, model downloads, video files, or
large datasets.

Regenerate it after an intentional metric-contract or fixture update:

```bash
uv run python scripts/write_eval_baseline.py --out baselines/eval/report.json
```

Then run the regression gate against a candidate report:

```bash
uv run python -m vrs.eval.ci \
  --baseline baselines/eval/report.json \
  --current runs/eval/report.json \
  --max-f1-drop 0.02
```

Exit codes are:

- `0`: overall and per-class F1 are within tolerance.
- `1`: at least one overall or per-class F1 regression exceeded tolerance.
- `2`: a structural error prevented comparison.
