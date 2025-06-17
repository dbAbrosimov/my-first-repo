lint:
	python -m pycodestyle --max-line-length=140 *.py

# Авто-форматирование через black
format:
	black .
.PHONY: lint format

