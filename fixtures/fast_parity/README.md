# Fast parity fixtures

This directory stores Python-generated golden JSON used while migrating Fast
mode to Rust.

Generate a fixture from a fixed photo folder:

```powershell
python scripts\export_fast_golden.py C:\path\to\photos fixtures\fast_parity\sample.json --strength standard --workers 1
```

The exporter records hashes, EXIF summary, quality signals, HSV histogram, ORB
descriptor summaries, ORB pair inliers, and final Python Fast groups. Rust tests
consume the JSON and verify that `pianke-core` produces the same grouping.
