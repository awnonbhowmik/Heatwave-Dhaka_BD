# Reproducibility report

The analysis is configuration-driven and runs with `make analysis`; tests run with `make test`, and output contracts with `make validate`. Source hashes and run metadata are machine-readable. Thin notebooks call package modules and contain no hidden analytical state. Two consecutive full runs with seed 20260901 produced byte-identical CSV files across tables, derived data, diagnostics, and forecasts (`REPRODUCIBILITY_NUMERICAL_CSVS_IDENTICAL`). All seven notebooks then executed from restarted kernels.
