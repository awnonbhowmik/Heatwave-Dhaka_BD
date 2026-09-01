# Reproducibility report

The analysis is configuration-driven and runs with `make analysis`; tests run with `make test`, and output contracts with `make validate`. Source hashes and run metadata are machine-readable. Thin notebooks call package modules and contain no hidden analytical state. A second clean run is compared through the validation script's deterministic manifest.
