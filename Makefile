.PHONY: ew-dash opt-dash create-signal run-backtest

ew-dash:
	uv run marimo edit src/framework/ew_dash.py

opt-dash:
	uv run marimo edit src/framework/opt_dash.py

create-signal:
	uv run python src/signal/create_signal.py

run-backtest:
	uv run python src/framework/run_backtest.py
