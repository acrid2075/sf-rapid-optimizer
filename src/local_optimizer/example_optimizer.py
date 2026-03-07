import polars as pl
import numpy as np
import sf_quant.data as sfd
import datetime as dt
from sf_quant.data._factors import factors
from sf_quant.data.covariance_matrix import _construct_factor_covariance_matrix
from time import perf_counter
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import minres, gmres
from tqdm import tqdm
from local_optimizer import *

def task_run_factor_optimization(signals_df: pl.DataFrame, start: dt.date, end: dt.date, 
                                 unit_beta: bool = False, zero_beta: bool = False, 
                                 long_only: bool = False, full_investment: bool = False, 
                                 gamma: float = 1.0):
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
        elif zero_beta:
            beta_vec = day_info["predicted_beta"].to_numpy().reshape(1, -1)
            A_list.append(beta_vec)
            b_list.append(np.array([0.0]))

        A = np.vstack(A_list) if A_list else None
        b = np.concatenate(b_list) if b_list else None

        # 3. Build Inequality Constraints (L w <= d)
        L = None
        d = None
        if long_only:
            # -I * w <= 0  is equivalent to w >= 0
            L = -sparse.eye(n_assets, format='csr')
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
