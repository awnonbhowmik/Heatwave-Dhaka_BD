.PHONY: analysis two-paper-benchmark test validate validate-two-paper notebooks manuscript-assets manuscripts article clean-results

PYTHON ?= .venv/bin/python

analysis:
	MPLCONFIGDIR=/tmp/heatwave-mpl $(PYTHON) scripts/run_all.py --config config/analysis.yml

two-paper-benchmark:
	MPLCONFIGDIR=/tmp/two-paper-mpl $(PYTHON) scripts/run_two_paper_benchmark.py --config config/two_paper_benchmark.yml

test:
	$(PYTHON) -m pytest -q

validate:
	$(PYTHON) scripts/validate_outputs.py --config config/analysis.yml

validate-two-paper:
	$(PYTHON) scripts/validate_two_paper_outputs.py

notebooks:
	MPLCONFIGDIR=/tmp/heatwave-mpl IPYTHONDIR=/tmp/heatwave-ipython JUPYTER_RUNTIME_DIR=/tmp/heatwave-jupyter $(PYTHON) scripts/execute_notebooks.py

manuscript-assets:
	$(PYTHON) scripts/build_integrated_manuscript_assets.py

manuscripts: manuscript-assets
	$(PYTHON) scripts/build_manuscripts.py

article: analysis two-paper-benchmark test notebooks manuscripts validate validate-two-paper

clean-results:
	@echo "Generated outputs are versioned; remove them only with explicit review."
