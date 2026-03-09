import os
import json
from dotenv import load_dotenv
from sf_backtester import BacktestConfig, BacktestRunner, SlurmConfig

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

if __name__ == "__main__":
    run_backtest()
