lint:
	python -m pycodestyle --max-line-length=140 *.py

format:
	python -m autopep8 --in-place --aggressive --aggressive *.py
.PHONY: lint format