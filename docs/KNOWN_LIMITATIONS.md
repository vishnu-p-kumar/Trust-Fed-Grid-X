# Known limitations

- Local single-process simulator only; no multi-organization deployment or isolation.
- Synthetic fixture is not a real meter dataset. SGCC loader/acquisition pipeline is not included.
- The supplied-data runner requires at least five unique meter IDs and at least two meter groups per class for its customer-group holdout; it errors rather than falling back to row splitting when meter IDs are present but insufficient.
- Trust utility uses coordinator-visible validation data. This is not privacy-preserving evaluation.
- HATTA is a provisional heuristic. Heterogeneity is handled by not penalizing update disagreement, not by a validated heterogeneity estimator.
- Backdoor, collusion-specific, delayed/stale update, and failure-recovery scenarios are incomplete. No attack success rate is claimed.
- No formal DP, accountant, secure aggregation, or privacy guarantee.
- No validation-selected threshold, calibration, multiple-seed confidence intervals, or result figure generator yet.
- Saved checkpoints cannot yet resume training.
- Existing API routes have session checks on operational data/artifact operations, but CSRF protection, rate limiting, deployment security, and end-to-end login/browser behavior have not been comprehensively validated.
- `users.json` was sanitized in the working tree, but credentials or tokens committed in earlier Git history require rotation and history cleanup outside this code change.
