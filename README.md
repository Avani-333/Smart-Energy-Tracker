# Smart Energy Tracker for Home Appliances

Python/ML project to estimate energy consumption, predict electricity costs, and analyze appliance usage from CSV datasets. Includes preprocessing, baseline models, clustering, visualizations, and a Streamlit demo UI.

## Quick Start
1) Create a virtual environment and activate it.
2) Install dependencies: `pip install -r requirements.txt`.
3) (Optional) Generate synthetic data: `python -m src.data_gen --rows 240 --output data/raw/sample.csv`.
4) Preprocess: `python -m src.data_prep --input data/raw/sample.csv --output data/processed/clean.csv`.
5) Train/evaluate: `python -m src.models --train data/processed/clean.csv --target usage_kwh --metrics reports/metrics.json` (outputs train/val metrics and saves a model).
6) Explore clustering: `python -m src.clustering --input data/processed/clean.csv --clusters 3`.
7) Launch Streamlit app (PowerShell):
	- `Set-Item Env:PYTHONPATH "$PWD"`
	- `& ".venv/Scripts/python.exe" -m streamlit run src/app.py`

## Results (Latest Run)

| Target | Train Rows | Linear MAE | Linear RMSE | ElasticNet MAE | ElasticNet RMSE | Best |
|--------|-----------|-----------|-----------|-----------|-----------|--------|
| usage_kwh | 192 | 0.001973 | 0.003338 | 0.015625 | 0.020021 | **Linear** |
| cost | 192 | 0.001657 | 0.002939 | 0.011729 | 0.020478 | **Linear** |

Silhouette score (K=3): **0.0333**

## Repository Layout
- data/: raw and processed CSVs.
- notebooks/: EDA and experiments.
- src/: pipeline code (prep, features, models, clustering, viz, app).
- reports/: figures, metrics, experiment logs, saved models.
- tests/: pytest suite.

## Notes
- Expected raw columns are listed in `src/data_prep.py` (`EXPECTED_COLUMNS`).
- Feature engineering adds time sin/cos, weekend flag, peak/off-peak, plus lag and rolling features for usage/power in `src/features.py`.
- Models use a train/validation split and report MAE/RMSE; best of Linear/ElasticNet is saved.
- Streamlit app lets you upload a CSV, view trends, appliance insights, interactive clustering, and receive WhatsApp alerts (click-to-chat or QR).
- See `VIVA_GUIDE.md` for full methodology, Q&A, and demo script.

