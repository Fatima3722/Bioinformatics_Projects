# AChE workflow revision

Prepared against the public notebook contents at upstream commit `54b7af485aa63f62477b94f3992366ccc6a6b650`.

| Earlier issue | Revised behaviour |
| --- | --- |
| Acquisition and exploratory notebooks use HM74 while later notebooks use AChE | All five notebooks use the same pinned AChE archive. |
| Missing active group produces NaN and an incorrect rejection claim | Group counts and finite test outputs are checked; invalid comparisons are explicitly not interpreted. |
| Static significance claims contradict saved results | Output conclusions are generated from current tests and BH-adjusted p-values. |
| Descriptor order differs from activity order | A validated one-to-one join uses stable record IDs. |
| Supplied features, calculated fingerprints and targets are conflated | The actual feature representation is recorded; the default four-descriptor baseline is explicitly distinguished from optional PaDEL. |
| Training-fit scores or model-selection tables are presented as predictive validation | A preselected forest has a reproducible held-out baseline; model ranking uses training-only group CV. |
| Old warnings, failed model runs and hidden notebook state remain visible | The five notebooks use shared checked functions and are rerun in fresh Python processes. |

