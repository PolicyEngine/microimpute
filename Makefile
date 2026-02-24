install:
	pip install -e ".[dev,docs]"

test:
	pytest tests/ --cov=microimpute --cov-report=xml --maxfail=0

check-format:
	linecheck .
	isort --check-only --profile black microimpute/
	black . -l 79 --check

format:
	linecheck . --fix
	isort --profile black microimpute/
	black . -l 79

documentation:
	cd docs && jupyter book clean . --all
	cd docs && jupyter book build .
	python docs/add_plotly_to_book.py docs/_build

build:
	pip install build
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf docs/_build/

changelog:
	python .github/bump_version.py
	towncrier build --yes --version $$(python -c "import re; print(re.search(r'version = \"(.+?)\"', open('pyproject.toml').read()).group(1))")
# Dashboard commands
dashboard-install:
	cd microimputation-dashboard && npm install

dashboard-dev:
	cd microimputation-dashboard && npm run dev

dashboard-build:
	cd microimputation-dashboard && npm run build

dashboard-start:
	cd microimputation-dashboard && npm run start

dashboard: dashboard-install dashboard-dev

dashboard-clean:
	cd microimputation-dashboard && rm -rf node_modules .next out