import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    import dataframely as dy
    from sf_quant.schema import AlphaSchema, SecurityRetSchema
    import sf_quant.research as sfr
    import sf_quant.performance as sfp
    import polars_ols
    import pandas
    return go, marimo, np, pl, sfp, sfr


@app.cell
def _(marimo):
    marimo.md("""
    # Signal Research

    > **Required columns:** `date` · `barrid` · `signal` · `alpha` · `return`
    """)
    return


@app.cell
def _(marimo):
    signal_file = marimo.ui.text(
        value="data/signal.parquet",
        label="Signal file path:"
    )
    return (signal_file,)


@app.cell
def _(pl, signal_file):
    signal_df = pl.read_parquet(signal_file.value).filter(
        pl.col('alpha').is_not_null()
    )
    return (signal_df,)


@app.cell
def _(marimo, signal_df):
    marimo.stop(signal_df.is_empty(), marimo.md("**⚠️ Please load a valid signal file first**"))
    return


@app.cell
def _(marimo):
    import datetime
    sample_cutoff = marimo.ui.date(value=datetime.date(2018, 1, 1), label="Sample cutoff:")
    sample_mode = marimo.ui.radio(
        options=["Full Sample", "In Sample", "Out of Sample"],
        value="Full Sample",
        label="Sample period:",
    )
    marimo.hstack([sample_mode, sample_cutoff])
    return sample_cutoff, sample_mode


@app.cell
def _(pl, sample_cutoff, sample_mode, signal_df):
    if sample_mode.value == "In Sample":
        signal_df_filtered = signal_df.filter(pl.col("date") <= sample_cutoff.value)
    elif sample_mode.value == "Out of Sample":
        signal_df_filtered = signal_df.filter(pl.col("date") > sample_cutoff.value)
    else:
        signal_df_filtered = signal_df
    return (signal_df_filtered,)


@app.cell
def _(marimo, signal_df_filtered):
    _min = signal_df_filtered.select("date").min().item()
    _max = signal_df_filtered.select("date").max().item()
    marimo.md(f"**Date range:** {_min} → {_max} &nbsp;&nbsp; **({signal_df_filtered['date'].n_unique()} trading days)**")
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Signal Analysis

    Examine signal statistics and characteristics before portfolio construction.
    """)
    return


@app.cell
def _(sfr, signal_df_filtered):
    sfr.get_signal_stats(signal_df_filtered, column='signal')
    return


@app.cell
def _(signal_df_filtered):
    import matplotlib.pyplot as plt
    plt.style.use('default')
    _signal_values = signal_df_filtered.select('signal').to_numpy().flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(_signal_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title("Signal Distribution")
    plt.xlabel("Signal Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(marimo):
    n_quantiles = marimo.ui.slider(
        value=5,
        start=2,
        stop=10,
        step=1,
        label="Number of quantiles:"
    )
    n_quantiles
    return (n_quantiles,)


@app.cell
def _(marimo):
    marimo.md("""
    ## Quantile Portfolio Construction

    Build long-short portfolios based on signal quantiles.
    """)
    return


@app.cell
def _(n_quantiles, sfr, signal_df_filtered):
    # Create quantile portfolios based on alpha signal
    quantile_df = sfr.generate_quantile_ports(
        signal_df_filtered,
        num_bins=n_quantiles.value,
        signal_col='alpha'
    ).drop_nulls()
    return (quantile_df,)


@app.cell
def _(quantile_df):
    # Convert quantile portfolios to long format for performance analysis
    ports_long = (
        quantile_df
        .unpivot(
            index="date",
            variable_name="quantile",
            value_name="return"
        )
    )
    return (ports_long,)


@app.cell
def _(marimo):
    include_ff_regression = marimo.ui.checkbox(
        value=True,
        label="Include Fama-French regression analysis"
    )
    include_ff_regression
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Portfolio Performance Analysis

    Calculate and visualize performance metrics across quantile portfolios.
    """)
    return


@app.cell
def _(pl, ports_long):
    # Calculate cumulative returns for each quantile
    cumul_returns = (
        ports_long
        .with_columns([
            pl.col("return").log1p().cum_sum().over('quantile').alias("cum_return")*100
        ])
        .select(["date", "quantile", "cum_return"])
    )
    return (cumul_returns,)


@app.cell
def _(cumul_returns, go, marimo, n_quantiles, pl):
    # Plot cumulative returns
    fig = go.Figure()

    for i in range(n_quantiles.value):
        quantile = f"p_{i+1}"
        data = cumul_returns.filter(pl.col("quantile") == quantile)
        dates = data.select("date").to_numpy().flatten()
        returns = data.select("cum_return").to_numpy().flatten()

        fig.add_trace(go.Scatter(
            x=dates,
            y=returns,
            mode='lines',
            name=f"Q{i+1}",
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%Y-%m-%d}<br>Cum Return: %{y:.2f}<extra></extra>'
        ))

    _spread_data = cumul_returns.filter(pl.col("quantile") == "spread")
    fig.add_trace(go.Scatter(
        x=_spread_data.select("date").to_numpy().flatten(),
        y=_spread_data.select("cum_return").to_numpy().flatten(),
        mode='lines',
        name="Spread",
        line=dict(color="black", width=2, dash="dash"),
        hovertemplate='<b>Spread</b><br>Date: %{x|%Y-%m-%d}<br>Cum Return: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title="Quantile Portfolio Cumulative Log Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Log Return",
        hovermode='x unified',
        height=500,
        template="plotly_white"
    )

    marimo.ui.plotly(fig)
    return


@app.cell
def _(np, pl, ports_long):
    # Calculate performance metrics
    metrics = (
        ports_long
        .filter(
            pl.col("return").is_not_null() &
            (pl.col("quantile") != "bmk_return")
        )
        .with_columns([
            pl.col("return").log1p().alias("log_return")
        ])
        .group_by("quantile")
        .agg([
            pl.col("return").mean().alias("mean_return"),
            pl.col("return").std().alias("std_return"),
            pl.col("log_return").sum().alias("cum_log_return"),
            pl.col("return").count().alias("n_obs"),
        ])
        .with_columns([
            (pl.col("cum_log_return").exp() - 1).alias("total_return"),
            (pl.col("mean_return") / pl.col("std_return")).alias("sharpe_ratio"),
            (pl.col("mean_return") * 252).alias("annual_return"),  # Assuming daily data
            (pl.col("std_return") * np.sqrt(252)).alias("annual_vol"),
        ])
        .select([
            "quantile",
            "annual_return",
            "annual_vol",
            "sharpe_ratio",
            "total_return",
            "n_obs"
        ])
        .sort("quantile")
    )
    return (metrics,)


@app.cell
def _(marimo, metrics, pl, sfp, signal_df_filtered):
    _spread = metrics.filter(pl.col("quantile") == "spread")
    _spread_sharpe = _spread.select("sharpe_ratio").item()
    _spread_ann_ret = _spread.select("annual_return").item()

    _ics = sfp.generate_alpha_ics(
        signal_df_filtered.select("date", "barrid", "alpha"),
        signal_df_filtered.filter(pl.col("return").is_not_null()).select("date", "barrid", "return"),
    )
    _ic_mean = _ics.select(pl.col("ic").mean()).item()
    _ic_ir = _ics.select(
        (pl.col("ic").mean() / pl.col("ic").std()).alias("icir")
    ).item()

    marimo.md(f"""
    | Metric | Value |
    |--------|-------|
    | Spread Sharpe | **{_spread_sharpe:.3f}** |
    | Spread Ann. Return | **{_spread_ann_ret:.2%}** |
    | IC (mean) | **{_ic_mean:.4f}** |
    | IC IR | **{_ic_ir:.3f}** |
    """)
    return


@app.cell
def _(marimo, metrics):
    marimo.md(f"""
    ## Summary Statistics

    {metrics.to_pandas().to_markdown(index=False)}
    """)
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Fama-French Regression Analysis

    Analyze quantile portfolio factor exposures and risk-adjusted returns.
    """)
    return


@app.cell
def _(quantile_df, sfr):
    sfr.run_quantile_ff_regression(quantile_df)
    return


if __name__ == "__main__":
    app.run()
