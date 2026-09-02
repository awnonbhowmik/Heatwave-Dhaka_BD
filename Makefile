.PHONY: analysis test validate notebooks manuscripts article clean-results

PYTHON ?= .venv/bin/python

analysis:
	MPLCONFIGDIR=/tmp/heatwave-mpl $(PYTHON) scripts/run_all.py --config config/analysis.yml

test:
	$(PYTHON) -m pytest -q

validate:
	$(PYTHON) scripts/validate_outputs.py --config config/analysis.yml

notebooks:
	MPLCONFIGDIR=/tmp/heatwave-mpl IPYTHONDIR=/tmp/heatwave-ipython JUPYTER_RUNTIME_DIR=/tmp/heatwave-jupyter $(PYTHON) scripts/execute_notebooks.py

manuscripts:
	$(PYTHON) scripts/build_manuscripts.py

article: analysis test notebooks manuscripts validate

clean-results:
	@echo "Generated outputs are versioned; remove them only with explicit review."
