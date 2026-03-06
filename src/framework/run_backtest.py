import os
import json
from dotenv import load_dotenv
from sf_backtester import BacktestConfig, BacktestRunner, SlurmConfig
from local_optimizer.local_optimizer import FactorMVO, iter_factor_data
import numpy as np
import datetime as dt
import polars as pl
from scipy import sparse

def run_backtest():
    # Load environment variables from .env file
    load_dotenv()

    project_root = os.getcwd()

    # Helper function to resolve relative paths from project root
    def resolve_path(env_var, default):
        path = os.getenv(env_var, default)
        # If path is relative, make it absolute relative to project root
        if not os.path.isabs(path):
            path = os.path.join(project_root, path)
        return path

    # Get configuration from environment variables with fallback defaults
    signal_path = resolve_path("SIGNAL_PATH", "data/signal.parquet")
    output_dir = resolve_path("WEIGHT_DIR", "data/weights")
    logs_dir = resolve_path("LOG_DIR", "logs")
    signal_name = os.getenv("SIGNAL_NAME", "my_first_signal")
    gamma = int(os.getenv("GAMMA", "50"))
    byu_email = os.getenv("EMAIL", "user@byu.edu")

    # Validate that signal file exists
    if not os.path.exists(signal_path):
        raise FileNotFoundError(
            f"Signal file not found at {signal_path}\n"
            f"Please run 'make create-signal' first to generate the signal."
        )

    # Parse constraints as JSON array
    constraints_str = os.getenv("CONSTRAINTS", "[]")
    try:
        constraints = json.loads(constraints_str)
    except json.JSONDecodeError:
        constraints = []

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Define Slurm Configuration from environment variables
    slurm_config = SlurmConfig(
        n_cpus=int(os.getenv("SLURM_N_CPUS", "8")),
        mem=os.getenv("SLURM_MEM", "32G"),
        time=os.getenv("SLURM_TIME", "03:00:00"),
        mail_type=os.getenv("SLURM_MAIL_TYPE", "BEGIN,END,FAIL"),
        max_concurrent_jobs=int(os.getenv("SLURM_MAX_CONCURRENT_JOBS", "30")),
    )

    # Define Backtest Configuration
    config = BacktestConfig(
        signal_name=signal_name,
        data_path=signal_path,
        gamma=gamma,
        project_root=project_root,
        byu_email=byu_email,
        constraints=constraints,
        slurm=slurm_config,
        output_dir=output_dir,
        logs_dir=logs_dir
    )

    runner = BacktestRunner(config)
    runner.submit(dry_run=False)


def task_run_factor_optimization(signals_df: pl.DataFrame, start: dt.date, end: dt.date, 
                                 unit_beta: bool = False, long_only: bool = False, 
                                 full_investment: bool = False, gamma: float = 1.0):
    print("Indexing data...")
    # Ensure we have all necessary columns for constraints
    required_cols = ["date", "barrid", "alpha"]
    if unit_beta: required_cols.append("predicted_beta")
    
    local_df = signals_df.select(required_cols)
    if isinstance(local_df, pl.LazyFrame):
        local_df = local_df.collect()

    results = []

    for date, data in iter_factor_data(start, end):
        # 1. Align universe and extract vectors
        day_info = local_df.filter(pl.col("date") == date).sort("barrid")
        if day_info.is_empty(): continue
        
        valid_barrids = day_info["barrid"].to_list()
        mask = np.isin(data["barrid"], valid_barrids)
        
        B = data["B"][mask]
        D = data["D"][mask]
        barrids = data["barrid"][mask]
        F = data["F"]
        n_assets = len(barrids)

        alpha_vec = day_info["alpha"].to_numpy()

        # 2. Build Equality Constraints (A w = b)
        A_list = []
        b_list = []

        if full_investment:
            A_list.append(np.ones((1, n_assets)))
            b_list.append(np.array([1.0]))

        if unit_beta:
            beta_vec = day_info["predicted_beta"].to_numpy().reshape(1, -1)
            A_list.append(beta_vec)
            b_list.append(np.array([1.0]))

        A = np.vstack(A_list) if A_list else None
        b = np.concatenate(b_list) if b_list else None

        # 3. Build Inequality Constraints (L w <= d)
        L = None
        d = None
        if long_only:
            # -I * w <= 0  is equivalent to w >= 0
            L = sparse.eye(n_assets, format='csr')
            d = np.zeros(n_assets)

        # 4. Run Optimizer
        optimizer = FactorMVO(alpha_vec, B, F, D, A=A, b=b, L=L, d=d)
        
        try:
            # Note: We pass debug=True to monitor IPM convergence
            weights = optimizer._solve_fixed_gamma(gamma, tol=1e-8, debug=False, max_iter=200)
        except Exception as e:
            print(f"Optimization failed on {date}: {e}")
            continue

        results.append(pl.DataFrame({
            "date": [date] * n_assets,
            "barrid": barrids,
            "weight": weights
        }))

    return pl.concat(results)

if __name__ == "__main__":
    run_backtest()
