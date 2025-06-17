# Проверка стиля через pycodestyle
lint:
	python -m pycodestyle app.py parser.py analysis.py

# Авто-форматирование через autopep8
format:
	autopep8 --in-place --aggressive --aggressive app.py parser.py analysis.py

.PHONY: lint format
