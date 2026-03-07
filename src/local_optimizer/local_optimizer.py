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

def iter_factor_data(start, end):
    """
    Stream factor-model inputs between two dates (inclusive).

    Yields (date, data) tuples where ``data`` is a dict containing:
        - ``B``: factor exposure matrix (n_assets x n_factors)
        - ``F``: factor covariance matrix (n_factors x n_factors)
        - ``D``: specific risk variances (length n_assets)
        - ``barrid``: asset identifiers ordered consistently with ``B`` rows
    """
    exposures = (
        sfd.load_exposures(start, end, True, ["date", "barrid"] + factors)
        .fill_nan(0)
        .fill_null(0)
    )
    specific_risk = (
        sfd.load_assets(start, end, ["date", "barrid", "specific_risk"], in_universe=True)
        .fill_nan(0)
        .fill_null(0)
    )

    dates = sorted(exposures.select("date").unique().to_series().to_list())

    for date in tqdm(dates, desc="Loading factor data"):
        exp_date = exposures.filter(pl.col("date").eq(date)).sort("barrid")
        if exp_date.is_empty():
            continue

        # Align specific risk to exposure ordering
        sr_date = (
            exp_date.select("barrid")
            .join(
                specific_risk.filter(pl.col("date").eq(date)).select(
                    "barrid", "specific_risk"
                ),
                on="barrid",
                how="left",
            )
            .fill_null(0)
            .fill_nan(0)
            .sort("barrid")
        )

        barrids = exp_date.select("barrid").to_series().to_list()
        B = exp_date.select(factors).to_numpy()
        F = (
            _construct_factor_covariance_matrix(date)
            .fill_nan(0)
            .fill_null(0)
            .select(factors)
            .to_numpy()
            / 1e4 / 252
        )
        D = np.square(sr_date.select("specific_risk").to_numpy().flatten()) / 1e4 / 252

        yield date, {"B": B, "F": F, "D": D, "barrid": np.array(barrids)}

def load_factor_data(start, end):
    """
    Materialize factor-model inputs between two dates (inclusive) into a dict.

    This wraps :func:`iter_factor_data` for callers that want everything in memory.
    """
    return {date: data for date, data in iter_factor_data(start, end)}

def build_signal_factor_inputs(start, end, signal, df):
    """
    Combine signal alphas with factor inputs for each date.

    Parameters
    ----------
    start, end : datetime-like
        Date range to include (inclusive).
    signal : str
        Name of the signal; the alpha column is expected to be ``{signal}_alpha``.
    df : pl.DataFrame or pl.LazyFrame
        Data containing ``date``, ``barrid`` and the signal alpha column.

    Returns
    -------
    dict
        Keys are dates; values are dicts with ``alpha`` (np.ndarray aligned to
        the factor exposure ordering) and the ``B``, ``F``, ``D`` matrices from
        :func:`load_factor_data`.
    """
    alpha_col = f"{signal}_alpha"

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if alpha_col not in df.columns:
        raise ValueError(f"Column '{alpha_col}' not found in provided dataframe.")

    filtered = (
        df.filter(
            (pl.col("date") >= start)
            & (pl.col("date") <= end)
            & pl.col(alpha_col).is_not_null()
        )
        .select(["date", "barrid", alpha_col])
    )

    alpha_by_date = {}
    for frame in filtered.partition_by("date", maintain_order=True):
        date_value = frame["date"][0]
        alpha_by_date[date_value] = dict(
            zip(frame["barrid"].to_list(), frame[alpha_col].to_list())
        )

    factor_data = load_factor_data(start, end)
    combined = {}

    for date, data in factor_data.items():
        barrids = data.get("barrid")
        if barrids is None:
            raise ValueError("Factor data missing barrid ordering; rerun load_factor_data.")

        alpha_map = alpha_by_date.get(date, {})
        alpha_vec = np.array([alpha_map.get(b, 0.0) for b in barrids])

        combined[date] = {
            "alpha": alpha_vec,
            "B": data["B"],
            "F": data["F"],
            "D": data["D"],
        }

    return combined

def iter_factor_mvos(start, end, signal, df, A=None, b=None, L=None, d=None, d_floor=1e-8):
    """
    Stream per-date FactorMVO instances with alphas aligned to factor exposures.

    Parameters
    ----------
    start, end : datetime-like
        Date range to include (inclusive).
    signal : str
        Signal name; alpha column must be ``{signal}_alpha``.
    df : pl.DataFrame or pl.LazyFrame
        Source data containing alphas.
    A, b, L, d : optional
        Constraint matrices/vectors to pass into :class:`FactorMVO`.
    d_floor : float
        Lower bound for specific risk diagonal entries in the optimizer.

    Yields
    ------
    tuple
        (date, FactorMVO) for each available date.
    """
    alpha_col = f"{signal}_alpha"

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if alpha_col not in df.columns:
        raise ValueError(f"Column '{alpha_col}' not found in provided dataframe.")

    # Keep only relevant slice of alpha data up front
    alpha_slice = (
        df.filter(
            (pl.col("date") >= start)
            & (pl.col("date") <= end)
            & pl.col(alpha_col).is_not_null()
        )
        .select(["date", "barrid", alpha_col])
    )

    alpha_by_date = {}
    for frame in alpha_slice.partition_by("date", maintain_order=True):
        date_value = frame["date"][0]
        alpha_by_date[date_value] = dict(
            zip(frame["barrid"].to_list(), frame[alpha_col].to_list())
        )

    for date, data in iter_factor_data(start, end):
        barrids = data["barrid"]
        alpha_map = alpha_by_date.get(date, {})
        alpha_vec = np.array([alpha_map.get(b, 0.0) for b in barrids])

        yield date, FactorMVO(alpha_vec, data["B"], data["F"], data["D"], A=A, b=b, L=L, d=d, d_floor=d_floor)

class FactorMVO:
    """
    Factor-model mean-variance optimizer that applies the factor covariance as a LinearOperator
    (no explicit nxn covariance computation).

    Objective: maximize alpha^T w - (gamma / 2) * w^T Cov w,  Cov = B F B^T + diag(D)
    Constraints: A w = b, optional L w <= d (not implemented yet), or target active risk.
    """

    def __init__(self, alpha, B, F, D, A=None, b=None, L=None, d=None, d_floor=1e-8):
        self.alpha = alpha          # n
        self.B = B                  # n × k
        self.F = F                  # k × k
        self.D = np.maximum(D, d_floor)  # n, floor to keep H SPD enough for Krylov

        self.A = A                  # m × n  (equality constraints, e.g. UnitBeta, ZeroBeta, FullInvestment)
        self.b = b                  # m

        self.L = L                  # p × n  (inequality constraints, e.g. LongOnly. Must be less than version, hence L)
        self.d = d                  # p
        self.m = A.shape[0] if A is not None else 0
        self.p = L.shape[0] if L is not None else 0

        self.n = len(alpha)
        self.k = F.shape[0]

        self.lastw = None
        self.lasty = None
        self.lasts = None
        self.lastlam = None

        if self.m > 0 and len(b) != self.m:
            raise ValueError(f"b ({len(b)}) must match A ({self.m})")
        if self.p > 0 and len(d) != self.p:
            raise ValueError(f"d ({len(d)}) must match L ({self.p})")
    
    def _make_H_operator(self, gamma):
        """Return LinearOperator for Hessian gamma*Cov without forming Cov."""
        n, _ = self.B.shape # _ really means k

        def matvec(x):
            # x: length n
            Btx = self.B.T @ x            # R^k
            F_Btx = self.F @ Btx         # R^k
            B_F_Btx = self.B @ F_Btx  # R^n
            return gamma * (B_F_Btx + self.D * x)

        return LinearOperator(
            shape=(n, n),
            matvec=matvec,
            dtype=float
        )

    def _make_KKT_operator(self, gamma, s=None, lam=None):
        n, m, p = self.n, self.m, self.p
        H_op = self._make_H_operator(gamma)
        s_lam_ratio = (s / np.maximum(lam, 1e-12)).flatten() if p > 0 else np.array([])

        def matvec(z):
            z = np.asarray(z).flatten()
            dw = z[:n]
            dy = z[n : n + m]
            d_lam = z[n + m :]

            res_dw = H_op.matvec(dw)
            if m > 0: res_dw += (self.A.T @ dy).flatten()
            if p > 0: res_dw -= d_lam

            res_dy = (self.A @ dw).flatten() if m > 0 else np.array([])

            if p > 0:
                res_d_lam = -dw - (s_lam_ratio * d_lam)
            else:
                res_d_lam = np.array([])

            return np.concatenate([res_dw, res_dy, res_d_lam]).astype(np.float64)

        diag_H = gamma * (np.sum(self.B * (self.B @ self.F), axis=1) + self.D)
        diag_blocks = [diag_H]
        if m > 0: diag_blocks.append(np.ones(m))
        if p > 0: diag_blocks.append(s_lam_ratio)
        
        m_inv_diag = 1.0 / np.maximum(np.abs(np.concatenate(diag_blocks)), 1e-10)
        
        K_op = LinearOperator(shape=(n+m+p, n+m+p), matvec=matvec, dtype=np.float64)
        M_op = LinearOperator(shape=(n+m+p, n+m+p), matvec=lambda x: m_inv_diag * x.flatten(), dtype=np.float64)
        return K_op, M_op

    def cov_times(self, w):
        """Compute Cov * w without forming Cov explicitly."""
        Bw = self.B.T @ w
        F_Bw = self.F @ Bw
        return self.B @ F_Bw + self.D * w

    def risk(self, w):
        """Compute sqrt( w^T Cov w )."""
        return np.sqrt(w @ self.cov_times(w))

    def kkt_residuals(self, w, gamma):
        """Return (||A w - b||_2, ||g + A^T lambda||_2) with g = -alpha + gamma*Cov w."""
        g = -self.alpha + gamma * self.cov_times(w)

        if self.A is None:
            primal = 0.0
            dual = np.linalg.norm(g)
            return primal, dual

        primal_vec = self.A @ w - self.b
        ATA = self.A @ self.A.T
        rhs = self.A @ (-g)
        lam, *_ = np.linalg.lstsq(ATA, rhs, rcond=None)
        dual_vec = g + self.A.T @ lam

        return np.linalg.norm(primal_vec), np.linalg.norm(dual_vec)

    def solve(self, gamma, active_risk_target=None, max_iter=50, tol=1e-6, debug=False):
        """
        Solve:
            maximize alpha^T w - (gamma / 2) * w^T Cov w
        with constraints.
        active_risk_target: impose sqrt(w^T Cov w) = active_risk_target via bisection on gamma.
        """

        if active_risk_target is not None:
            return self._solve_for_risk(active_risk_target, gamma_init=gamma, debug=debug)

        return self._solve_fixed_gamma(gamma, max_iter=max_iter, tol=tol, debug=debug)

    def _solve_fixed_gamma(self, gamma, max_iter=500, tol=1e-6, debug=False):
        """
        Newton solve for fixed gamma with equality constraints via KKT MINRES.
        """

        if debug:
            start = perf_counter()
            print(f'[INFO] Started optimizer with gamma={gamma}.')

        n, m, p = self.n, self.m, self.p
        w = np.zeros(n) if self.lastw is None else self.lastw
        s = (np.ones(p) if self.lasts is None else self.lasts) if p > 0 else np.array([])
        lam = (np.ones(p) if self.lastlam is None else self.lastlam) if p > 0 else np.array([])
        y = (np.zeros(m) if self.lasty is None else self.lasty) if m > 0 else np.array([])

        mu = .1

        A = self.A
        b = self.b
        L = self.L
        d = self.d

        lin_maxiter = int(min(1000, 2 * (n + m + p)))

        converged = False
        for _ in range(max_iter): # Might want to track and return this

            # Gradient: g = - alpha + gamma Cov w   (negative because maximizing)
            g = -self.alpha + gamma * self.cov_times(w)
            r_dual = g
            if m > 0: r_dual += self.A.T @ y
            if p > 0: r_dual += self.L.T @ lam

            r_eq = (self.A @ w - b) if m > 0 else np.array([])
            r_ineq = (self.L @ w + s - d) if p > 0 else np.array([])
            r_comp = (s * lam) - mu if p > 0 else np.array([])

            gap = np.dot(s, lam) / p if p > 0 else 0

            rhs_dw = -r_dual
            rhs_dy = -r_eq
            rhs_dlam = -(r_ineq - (r_comp / lam)) if p > 0 else np.array([])
            rhs = np.concatenate([rhs_dw, rhs_dy, rhs_dlam])

            K_op, M_op = self._make_KKT_operator(gamma, s=s, lam=lam)
            sol, info = minres(K_op, rhs, M=M_op, rtol=1e-9, maxiter=lin_maxiter)

            dw = sol[:n]
            dy = sol[n:n + m]
            d_lam = sol[n + m:]

            if p > 0:
                ds = -r_ineq - self.L @ dw
            else:
                ds = np.array([])
            
            alpha_step = 1.0
    
            # Check slacks
            if len(ds) > 0:
                idx = ds < 0
                if np.any(idx):
                    alpha_step = min(alpha_step, .99 * np.min(-s[idx] / ds[idx]))
                    
            # Check multipliers
            if len(d_lam) > 0:
                idx = d_lam < 0
                if np.any(idx):
                    alpha_step = min(alpha_step, .99 * np.min(-lam[idx] / d_lam[idx]))

            w += alpha_step * dw
            if m > 0: y += alpha_step * dy
            if p > 0:
                s += alpha_step * ds
                lam += alpha_step * d_lam

            mu = min(mu * .5, gap * .2)
            res_norm = np.linalg.norm(r_dual)
            if res_norm < tol and (p == 0 or gap < tol):
                converged = True
                break
            
        self.lastw = w
        self.lasty = y
        self.lastlam = lam
        self.lasts = s
        if debug:
            end = perf_counter()
            print(f'[INFO] Optimizer took {(end - start):.4g} seconds to finish.')

        if not converged:
            print(f"[WARN] FactorMVO _solve_fixed_gamma did not reach tol={tol}. "
                  f"Final step norm={res_norm:.3e}, iter={max_iter}, gamma={gamma}.")

        return w

    def _solve_for_risk(self, target, gamma_init, tol=1e-8, debug=False):
        """
        Approximate sqrt(w^T Cov w) = target by bisection on gamma.
        """
        gamma_low = 1e-1
        gamma_high = 1e4
        gamma = gamma_init

        if debug:
            start = perf_counter()
            print(f'[INFO] Started optimizer with target active risk {target:.4g}.')

        w_init = np.zeros(self.n)
        reached = False
        for i in range(40):
            w = self._solve_fixed_gamma(gamma, w0=w_init)
            r = self.risk(w)
            w_init = w  # warm start the next solve

            if np.abs(r - target) < tol:
                reached = True
                break
            elif r < target:
                gamma_high = gamma
            else:
                gamma_low = gamma

            gamma = 0.5 * (gamma_low + gamma_high)

            if debug: print(f'[INFO] Finished iteration {i} with risk {r}.')

        if debug:
            end = perf_counter()
            print(f'[INFO] Optimizer with arget risk tuning took {(end - start):.4g} seconds to finish.')

        if not reached:
            print(f"[WARN] FactorMVO _solve_for_risk did not reach target risk within tol={tol}. "
                  f"Final risk={r:.3e}, target={target}, last gamma={gamma}.")

        return self._solve_fixed_gamma(gamma, w0=w_init)
    