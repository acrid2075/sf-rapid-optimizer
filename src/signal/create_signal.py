import os
import polars as pl
import datetime as dt
from dotenv import load_dotenv
import sf_quant.data as sfd


def load_data() -> pl.DataFrame:
    """
    Load and prepare market data for signal creation.

    Returns:
        pl.DataFrame: Market data with required columns
    """
    # TODO: Load data from source (API, file, database)

    # TODO: Filter data as needed (date range, symbols, quality checks)

    pass

def create_signal():
    """
    Loads data, creates a simple signal, and saves it to parquet.
    """
    # Load environment variables from .env file
    load_dotenv()
    project_root = os.getcwd()
    output_path = os.getenv("SIGNAL_PATH", "data/signal.parquet")
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    # TODO: Load Data
    df = load_data()

    # TODO: Add your signal logic here (remember alpha logic)

    # TODO: Save to data/signal.parquet

    # pl.write_parquet(signal, output_path)

if __name__ == "__main__":
    create_signal()
