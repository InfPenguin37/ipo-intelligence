.PHONY: setup run test

# Run `make setup` once from the repo root, then `conda activate ipo-intel`.
setup:
	conda env create -f environment.yml || conda env update -f environment.yml --prune

run:
	streamlit run src/ipo_intel/app.py

test:
	pytest -q
