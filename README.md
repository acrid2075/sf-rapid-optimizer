# SF-Research

A template project for developing, researching, and backtesting trading signals.

## 🚀 How to Use This Template
#### To create your own repository using this template, follow these quick steps:
1. Click the Button: At the top of this repository page, click the green "Use this template" button and select "Create a new repository."
2. Configure:
   - Choose an Owner (your account).
   - Give your new repository a Name, "sf-research-{signal_name}"
3. Create: Click "Create repository from template."
4. Clone: Once your new repo is ready, clone it to your local machine:
```bash
git clone https://github.com/your-username/your-new-repo.git
```

## 🔄 Automatic Template Updates

This template includes automatic syncing to keep your repository up-to-date with the latest improvements, bug fixes, and security patches.

### How Updates Work
- Every Monday at midnight UTC, GitHub automatically checks for template updates
- If new changes are available, a pull request is created in your repository
- You'll be notified via GitHub, and can review the changes at your convenience
- Updates include version bumps, changelog updates, and framework improvements

### One-Time Setup
When you first receive an update PR, you'll need to configure GitHub Actions permissions (one-time only):

1. Go to **Settings → Actions → General**
2. Under "Workflow permissions," select **"Read and write permissions"**
3. Check **"Allow GitHub Actions to create and approve pull requests"**

After this, updates will flow in automatically. You can always merge, ignore, or customize the updates as needed for your specific project.

## Environment Setup

Before running any commands, you need to configure your environment:

1. **Copy the environment template:**
```bash
cp .env.example .env
```

2. **Edit `.env` with your settings:**
```bash
# Update these values:
SIGNAL_PATH=data/signal.parquet          # Where to save your signal
WEIGHT_DIR=data/weights                  # Where backtest results go
LOG_DIR=logs                              # Where logs go

SIGNAL_NAME="Your Signal Name"            # Name your signal
GAMMA=50                                  # Risk aversion parameter
EMAIL=your-netid@byu.edu                 # Your BYU email
CONSTRAINTS=["ZeroBeta", "ZeroInvestment"]  # Portfolio constraints

# Optional: Customize SLURM settings for cluster jobs
SLURM_N_CPUS=8                           # Number of CPUs
SLURM_MEM=32G                            # Memory allocation
SLURM_TIME=03:00:00                      # Time limit
```

The `.env` file is **not** tracked in git (see `.gitignore`), so each user can have their own settings.

## Project Structure

```
sf-signal/
├── src/
│   ├── framework/
│   │   ├── ew_dash.py            # Equal-weight dashboard (do not edit)
│   │   ├── opt_dash.py           # Optimal portfolio dashboard (do not edit)
│   │   └── run_backtest.py       # Run the backtest (edit config only)
│   └── signal/
│       └── create_signal.py      # Your signal implementation (edit this)
├── data/
│   ├── signal.parquet            # Output: Your signal
│   └── weights/                  # Output: Backtest weights
└── README.md
```

## Workflow

### Required Columns
date: Date column
barrid: Asset identifier
alpha: Alpha signal values
predicted_beta: Predicted beta values
signal: orignal signal values (name can be changed)


### 1. **Implement Signal** (`create_signal.py`)
   - Customize date ranges, data columns, and calculation logic
   - Develop your signal logic
   - Saves signal to `data/signal.parquet`

   ```bash
   make create-signal
   ```

### 2. **View Equal-Weight Performance** (`ew_dash.py`)
   - Compare your signal against an equal-weight baseline
   - Analyze signal characteristics
   - Visualize signal properties and performance

   ```bash
   make ew-dash
   ```

### 3. **Run Backtest** (`run_backtest.py`)
   - Run MVO-based backtest on your signal
   - Generates optimal portfolio weights
   - Saves results to `data/weights.parquet`

   ```bash
   make run-backtest
   ```

### 4. **View Optimized Performance** (`opt_dash.py`)
   - View optimized portfolio performance
   - Analyze backtest returns, drawdowns, and metrics

   ```bash
   make opt-dash
   ```

## Data Files

All data files are stored in the `data/` directory:

- **`data/signal.parquet`**: Output from `create_signal.py`
  - Columns: `date`, `barrid`, `alpha` (your signal), `signal`
  - Format: Parquet (AlphaSchema)

- **`data/weights/*.parquet`**: Output from backtest
  - Contains: Portfolio weights and performance data
  - Format: Parquet

## Quick Start

```bash
# 1. Implement your signal
# Edit src/signal/create_signal.py with your logic
make create-signal

# 2. View equal-weight performance
make ew-dash

# 3. Run backtest
make run-backtest

# 4. View optimized performance
make opt-dash
```

## Template Files (Do Not Need to Edit)

The following files are templates and should not be modified:
- `src/framework/ew_dash.py` - Equal-weight comparison dashboard
- `src/framework/opt_dash.py` - Optimized portfolio dashboard
- `src/framework/run_backtest.py` - Backtest runner

If you want to edit the marimo notebooks use:
```bash
uv run marimo edit src/framework/{}_dash.py
```

**All signal customization happens in `src/signal/create_signal.py`.**

## Configuration

All configuration is managed through the `.env` file (copied from `.env.example`):

- **`SIGNAL_PATH`**: Where to save your generated signal (relative or absolute path)
- **`WEIGHT_DIR`**: Where backtest results will be saved
- **`LOG_DIR`**: Where backtest logs will be saved
- **`SIGNAL_NAME`**: Name for your signal
- **`GAMMA`**: Risk aversion / transaction cost parameter
- **`EMAIL`**: Your BYU email for job notifications
- **`CONSTRAINTS`**: Portfolio constraints as JSON array (e.g., `["ZeroBeta", "ZeroInvestment"]`)
- **`SLURM_N_CPUS`**: Number of CPU cores for cluster jobs
- **`SLURM_MEM`**: Memory allocation for cluster jobs
- **`SLURM_TIME`**: Time limit for cluster jobs
- **`SLURM_MAIL_TYPE`**: Email notifications (BEGIN, END, FAIL)
- **`SLURM_MAX_CONCURRENT_JOBS`**: Maximum parallel jobs

**Note:** Do not edit `src/framework/run_backtest.py` directly. All configuration comes from `.env`.

## Next Steps

1. Implement your signal logic in `src/signal/create_signal.py`
2. Run `make create-signal` to generate your signal
3. Compare against baseline with `make ew-dash`
4. Run backtest with `make run-backtest`
5. Analyze optimized results with `make opt-dash`
6. Iterate and refine your approach

---

## 📝 Maintaining This Template (For Template Maintainers)

If you're maintaining the template repository and need to release updates:

### Release Process
1. **Update the VERSION file** with the new semantic version (e.g., `1.0.1` for bug fixes, `1.1.0` for new features)
   - Follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html): `MAJOR.MINOR.PATCH`

2. **Update CHANGELOG.md** with what changed
   - Use the format: Added, Changed, Deprecated, Removed, Fixed, Security

3. **Commit and push** your changes to the main branch
   ```bash
   git add VERSION CHANGELOG.md [other files]
   git commit -m "chore: release v1.0.1"
   git push
   ```

### Automatic Syncing
- The template sync workflow automatically runs every Monday
- All repositories created from this template will receive a pull request with the updates
- Users can review, merge, or customize the changes as needed

For more details, see [CHANGELOG.md](CHANGELOG.md).

---

**Note**: This is a template project. Customize `src/signal/create_signal.py` with your unique signal logic, then use the workflow above to backtest your ideas.


