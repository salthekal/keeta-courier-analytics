# Data

The real Keeta operational scorecard is **not committed to this repository**. It contains:

- Personally identifying information (courier names, platform IDs)
- Data subject to platform terms-of-service restrictions
- Proprietary 3PL operational metrics

Place your own scorecard export at `data/keeta_scorecard.xlsx` to run the pipeline.

---

## Expected schema

The Keeta export ships as a single sheet with **27 columns** at courier-day granularity (one row per courier per calendar day). The repo's parser (`src/data_loader.py`) expects exactly these column names:

| # | Column | Type | Notes |
|---|---|---|---|
| 0 | `Date` | int | YYYYMMDD format (e.g., `20260331`) |
| 1 | `Courier ID` | int | Hashed by the loader before any downstream use |
| 2 | `Courier First Name` | str | Dropped by default (`drop_pii=True`) |
| 3 | `Courier Last Name` | str | Dropped by default |
| 4 | `Supervisor` | str | |
| 5 | `Vehicle Type` | str | "Private Car", "Motorcycle", etc. |
| 6 | `Shift_Attendance Summary` | str | Pipe-delimited 6-bucket structured string |
| 7 | `Shift_On-Shift?` | str | "Yes" / "No" |
| 8 | `Shift_Valid Day?` | str | "Yes" / "No" / "-" |
| 9 | `Shift_Courier App Online Time` | str | "11 hr, 54 min" format |
| 10 | `Shift_Valid Online Time` | str | Same format |
| 11 | `Shift_Peak Online Hours` | str | Same format |
| 12 | `Task Volumes_Accepted Tasks` | object | Numeric or `-` for inactive days |
| 13 | `Task Volumes_Tasks with restaurant arrivals` | object | |
| 14 | `Task Volumes_Delivered Tasks` | object | |
| 15 | `Task Volumes_Large Order Tasks Completed` | object | |
| 16 | `Task Volumes_Rejected Tasks` | object | |
| 17 | `Task Volumes_Rejected Tasks (Courier)` | object | |
| 18 | `Task Volumes_Rejected Tasks (Auto)` | object | |
| 19 | `Task Volumes_Cancellation Rate from Delivery Issues` | object | Decimal in `[0, 1]` |
| 20 | `Task Volumes_Order completion rate (non-delivery related)` | object | Decimal in `[0, 1]` |
| 21 | `Delivery Experience_On-time Rate (D)` | object | Decimal in `[0, 1]` |
| 22 | `Delivery Experience_Large order on-time rate` | object | Decimal in `[0, 1]` |
| 23 | `Delivery Experience_Avg Delivery Time of Delivered Orders` | object | Minutes |
| 24 | `Delivery Experience_Delivered Orders Prop. (Over 55min)` | object | Decimal in `[0, 1]` |
| 25 | `Delivery Experience_Overdue Order Tasks` | object | Count |
| 26 | `Delivery Experience_Severely Overdue Order Tasks` | object | Count |

If your export has a slightly different column set, update the `COL_RENAMES` dict in `src/data_loader.py`.

---

## Attendance summary string format

The `Shift_Attendance Summary` field is the most information-dense column and warrants its own note. Format:

```
HH:MM-HH:MM,STATE,DURATION[,qualified] | HH:MM-HH:MM,STATE,DURATION[,qualified] | ...
```

Six time buckets per day:

- `00:00-03:00`
- `03:00-08:00`
- `08:00-12:00`
- `12:00-16:00` (KSA lunch peak)
- `16:00-20:00`
- `20:00-24:00` (KSA dinner peak)

Each segment encodes:
- `STATE` — `On-Shift` or `Off-Shift`
- `DURATION` — variable-length time string (e.g., `0 sec`, `4 hr`, `1 hr, 35 min`, `12 min, 36 sec`). May contain a comma, which is why the parser splits carefully rather than naively on every comma.
- `qualified` — appears only on On-Shift segments that meet the platform's qualified-shift criteria

Example:
```
00:00-03:00,Off-Shift,1 hr, 35 min|03:00-08:00,Off-Shift,0 sec|08:00-12:00,Off-Shift,12 min, 36 sec|12:00-16:00,On-Shift,4 hr,qualified|16:00-20:00,On-Shift,4 hr,qualified|20:00-24:00,On-Shift,2 hr, 6 min,qualified
```

This courier was off the first three buckets, then qualified-on-shift through both peaks and into the late evening.

The parser in `src/attendance_parser.py` produces 8 features per courier-day:

- `total_on_shift_min`
- `total_qualified_min`
- `n_qualified_buckets` (0–6)
- `n_on_shift_buckets` (0–6)
- `attended_lunch_peak` (0/1)
- `attended_dinner_peak` (0/1)
- `peak_buckets_attended` (0/1/2)
- `shift_fragmentation` — count of distinct on-shift "blocks" (Off→On transitions)

---

## Privacy

- **Hash courier IDs first.** The loader does this automatically with a SHA-256-based pseudonymous mapping. Change the `salt` parameter to invalidate any prior mapping.
- **Drop name columns.** `drop_pii=True` by default in `load_keeta_scorecard()`.
- **Never commit raw exports.** The `.gitignore` blocks `data/*.xlsx` and similar. If you must share data with collaborators, share the cleaned + hashed version produced by the loader, not the raw export.

If you operate in a jurisdiction with formal data-protection law (KSA PDPL, EU GDPR, etc.), confirm with your DPO before pushing anything to a public repo.
