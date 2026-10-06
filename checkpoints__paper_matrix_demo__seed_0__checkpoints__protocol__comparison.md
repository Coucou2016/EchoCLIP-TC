| experiment_id | experiment_title | mae | rmse | auc_ef_lt_50 | ece_ef_lt_50 | load_source | annotation_assisted | demo_is_not_clinical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | EchoCLIP-based zero-shot baseline (official stride under --paper) | 18.75 | 21.32 | 0.5 | 0.3437 | scratch_fallback | False | True |
| R1 | Uniform-16 mean pool (no extra params) | 18.75 | 21.32 | 0.5 | 0.3437 | scratch_fallback | False | True |

> **Honesty:** demo / `scratch_fallback` / `simple_cnn` rows are pipeline wiring only — never report as EchoNet or Nature Medicine EF MAE. State `load_source` and split (TEST vs subset_5000) next to every table number.
