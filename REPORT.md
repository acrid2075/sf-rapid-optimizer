# Research Report

**Project Title: Rapid MVO**  
**Author(s): Andrew Criddle, Joseph Moore, elaborating on work from Kyle Markham**  
**Date: March 2026**  
**Version: 1**  

---

## 1. Summary

Provide a brief overview of:

This project focuses on improving the speed of optimization through careful matrix multiplication. In doing so, it gains performance improvement of 14x unconstrained and 10x over the current optimizer with both equality and inequality constraints. This can be implemented into the current optimizer as an alternate, faster underlying framework.

### Key Metrics

```
| Metric | Value | Benchmark | Error |
|:---|:---:|:---:|:---:|
| 1 Month Backtest Time | 14.74814 | 213.32540 | 0.0010595 |
| 1 Month Backtest Time, equality constrained | 16.26157 | 268.47322 | 3.20913e-11 |
| 1 Month Backtest Time, equality and inequality constrained | 119.68944 | 1865.92818 | 0.24497 |
```
 

## 2. Data Requirements

In order to operate this optimizer, it requires a signal for every stock each day, as well as the relevant factor exposures, factor covariance, and idiosyncratic variance matrices for each day. Equality and inequality constraints much be linear; the constraints will require the appropriate matrices and vectors Ax = b and Lx ≤ d. Some of this can be dealt with automatically with the available sf-quant data through task_run_factor_optimization. 

---

## 3. Approach / System Design

The core system here is an optimizer perform mean-variance optimization for portfolio construction. The core of the infrastructure here comes from Kyle Markham, who designed the initial optimizer and began work on the inequality constraints. The inspiration for the speed-up has to do with intelligent matrix multiplication: multiplying a vector by a matrix twice is far faster than multiplying a matrix by a matric and then a vetor by a matrix. As such, we can avoid constructing the impractically large covariance matrix in its entirety, instead multiplying the weights directly with the factor exposures, and multiplying the idiosyncratic risk element wise with the idiosyncratic risk rather than constructing the impractically large sparse diagonal. 

The improvements that have been performed from the initial framework include the following:
   - careful alterations on the normalizing constants such that similar gammas will produce very similar results between the portfolio constructed by this rapid optimizer and the existing optimizers
   - implementation of framework for inequality constraints. This involves careful iteration with a slack variable, restricting the movement only where the inequality is close to being violated
   - structured implementation of the four of the most common constraints (Full Investment, Long Only, Unit Beta, Zero Beta) for ease of computation

---

## 4. Code Structure

```
sf-rapid-optimizer/
├── src/
│   └── local_optimizer/
│       ├── example_optimizer.py     # Provides general framework for optimization
│       └── local_optimizer.py       # Provides FactorMVO, the object containing the optimizer
└── README.md
```

### **Perform Backtest** (`task_run_factor_optimization`)
   - Customize date ranges, include a signals_df with columns date, barrid, alpha, and flag relevant constraints, and choose a gamma
   - Calls FactorMVO internally, having constructed the data structure with the relevant constraining matrices and covariance matrix subparts

---

## 5. Performance Discussion

- Strengths
Large speed-ups in performance.
Handles large numbers of constraints well.

- Weaknesses
Restriction to linear constraints. Other varieties of constraints will likely require personalized restructuring within the optimizer.
This includes No Buying on Margin; LongOnly and FullInvestment provide a similar constraint, but it is good to be conscious of this.

---

## 6. Limitations

- Known issues:
There may be slight lapses in inequality holding. This does not occur often, and significant failures do prompt warnings of failures to converge. This happens in particular with constraints that conflict.

- Missing features:
Missing No Buying on Margin, which such nonlinear constraints will take effort to implement into the optimizer.

---

## 7. Future Work

-  Direct implementation into the current optimizer.
-  Address the potential for nonlinear constraints.

---
