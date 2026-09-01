.PHONY: analysis test validate notebooks clean-results

PYTHON ?= .venv/bin/python

analysis:
	MPLCONFIGDIR=/tmp/heatwave-mpl $(PYTHON) scripts/run_all.py --config config/analysis.yml

test:
	$(PYTHON) -m pytest -q

validate:
	$(PYTHON) scripts/validate_outputs.py --config config/analysis.yml

notebooks:
	MPLCONFIGDIR=/tmp/heatwave-mpl $(PYTHON) scripts/execute_notebooks.py

clean-results:
	@echo "Generated outputs are versioned; remove them only with explicit review."
