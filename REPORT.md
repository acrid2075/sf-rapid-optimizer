# Research Report

**Project Title:**  
**Author(s):**  
**Date:**  
**Version:**  

---

## 1. Summary

Provide a brief overview of:

- Research or development direction  
- Core idea or system built
- Key conclusions, takeaways, resulting actions
- Most important performance or success metrics

### Key Metrics

| Metric | Value | Notes |
|------|------|------|
| Primary Metric |  |  |
| Secondary Metric |  |  |
| Other |  |  |

## 2. Data Requirements

Describe data dependencies.

**Sources**
-  

**Rate of Availability**
-  

**Inputs Required**
-  

**Preprocessing**
-  
 

---

## 3. Approach / System Design

Describe what was built or tested.

Possible topics:

- Conceptual idea or economic intuition  
- Model, system, or architecture  
- Algorithms used  
- Design decisions  
- Tradeoffs considered  

---

## 4. Code Structure

If signal research:

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
   make backtest
   ```

### 4. **View Optimized Performance** (`opt_dash.py`)
   - View optimized portfolio performance
   - Analyze backtest returns, drawdowns, and metrics

   ```bash
   make opt-dash
   ```

If other than signal research, describe organization of implementation.

```
Example:
project/
├── data/
├── src/
├── scripts/
├── results/
└── docs/
```

Explain:

- Main pipeline or workflow
- Important modules
- Execution instructions

---

## 5. Results / Evaluation

Include relevant evidence demonstrating performance.

For signals:

- Cumulative IC table
- Possibly quantile plots
- Active portfolio backtest
- Summary statistic tables
- Other useful tables and plots

Possible items:

- Tables
- Plots
- Benchmarks

Add anything useful for interpreting system behavior.

---

## 6. Performance Discussion

Discuss:

- Strengths
- Weaknesses
- Sensitivity to assumptions or parameters

---

## 7. Limitations

- Known issues:
- Missing features:
- Risks:
- Open questions:

---

## 8. Future Work

Ideas for improvement or continuation:

-  
-  
-  

---

## Appendix (Optional)

### A. Additional Results

### B. Experimental Details

### C. Reproducibility Notes