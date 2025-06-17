lint:
	pycodestyle app.py
format:
	autopep8 --in-place --aggressive --aggressive app.py parser.py analysis.py
