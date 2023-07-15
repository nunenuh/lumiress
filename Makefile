

run-test:
ifdef dst
	PYTHONPATH=src python -m pytest $(dst) -v
else
	PYTHONPATH=src python -m pytest -v
endif


run-test-cov:
	PYTHONPATH=src pytest --cov=src --cov-report=term-missing --cov-report=xml


precommit:
	pre-commit run --all-files
