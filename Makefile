CODE = .
pretty:
	black --target-version py38 --line-length 79 $(CODE)
	isort $(CODE)
	flake8 $(CODE)
