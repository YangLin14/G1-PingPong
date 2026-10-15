# The question bank (feeds and target zones)

[繁體中文](question_bank.zh-TW.md) · [Docs index](../README.md)

`make_g1_forehand_bank.py` builds each row in four steps:

1. **Feed.** A bounce point is drawn uniformly in the feed zone, with a pre-bounce velocity
   (vx −5.5…−2.5, vy −2.4…−0.3, vz −3.5…−1.5 m/s) and a random spin up to 5 rev/s. The flight is
   integrated **backward** from the bounce, and the row is kept only if it crossed the far end line
   over the opponent's right (y 0.19–0.72 m, 5–45 cm above the table) and cleared the net by 3 cm.
   That makes it a diagonal feed from the opponent's right. The venue table contact gives the
   post-bounce velocity and spin.
2. **Apex.** RK4 flight (2 ms steps) runs to the apex (vz = 0). The row is kept only if the apex is
   inside the G1 forehand strike box: x 0.42–0.50 m in front of the station, z 0.80–1.10 m. The
   station y follows from the recipe's forehand racket offset (−0.38…−0.32 m).
3. **Return.** One of the four target zones is drawn (recorded in `target_zone_id`), then a
   landing point uniformly inside it. The racket velocity (face normal along it, the V14 contract)
   is solved by damped least squares on (landing x, landing y, net clearance − 0.25 m), weakly
   regularized (λ = 1e-3) toward (1.9, 0.3, 1.0) m/s. The deep and edge targets need fast, flat
   swings, which a stronger pull would forbid.
4. **Accept.** The row is kept if the landing error is ≤ 5 cm, net clearance is within
   0.25 ± 0.10 m (and ≥ 3 cm), racket speed ≤ 3.5 m/s, and `classify_strikes` calls it a success
   for its own zone. Generation continues until every zone has its quota (equal rows per zone).

Statistics of the committed bank (50 000 rows, seed 0):

| Quantity | min | mean | p95 | max |
|---|---|---|---|---|
| racket speed (m/s) | 1.15 | 2.73 | 3.38 | 3.50 |
| apex height above table (m) | 0.13 | 0.24 | 0.32 | 0.34 |
| incoming speed at contact (m/s) | 1.99 | 3.03 | 3.51 | 3.78 |
| net clearance (m) | 0.23 | 0.26 | 0.26 | 0.27 |
| landing residual (m) | 0.000 | 0.002 | 0.003 | 0.047 |

12 500 rows per zone, from 4.6 million candidate feeds. Most are rejected because their apex falls
outside the G1 strike box. The deep/edge zones push the racket speed up to the 3.5 m/s limit.

Regenerate (about 15 min on a laptop CPU) with:
`python3 hope_training/whole_body_tracking/scripts/make_g1_forehand_bank.py --rows 50000`.
Changing the seed or physics changes the rows, so **retrain after regenerating**.

**Faster-feed bank** (`g1_fh_crosscourt_apex_bank_wide`, used by `_FH_XC_RALLY`): same steps with
pre-bounce vx −7.5…−1.8, vy −3.0…−0.2, vz −4.5…−1.0 m/s and spin up to 8 rev/s; 50 000 rows from
7 million candidates. Feed speed before the bounce 4.4–8.5 m/s (mean 6.6), end line → apex
0.38–0.85 s (mean 0.51), incoming speed at contact 2.0–5.1 m/s. Feeds slower than ~4.4 m/s cannot
bounce in the middle third and still peak inside the fixed strike box; they need forward footwork.

```bash
python3 hope_training/whole_body_tracking/scripts/make_g1_forehand_bank.py --rows 50000 \
  --vin-x -7.5 -1.8 --vin-y -3.0 -0.2 --vin-z -4.5 -1.0 --spin-rev-s 8 \
  --out hope_training/whole_body_tracking/assets/venue_tuple/g1_fh_crosscourt_apex_bank_wide
```

