# 📈 ML Portfolio Optimization for S&P 500

A comprehensive, modular project for machine learning–driven portfolio optimization using all 503 constituents of the S&P 500. This repository implements advanced data engineering, multiple ML models, and robust backtesting—ideal for academic and research use.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Pipeline](#modeling-pipeline)
- [Evaluation](#evaluation)

---

## Overview

This project analyzes long-term (2010–2025) financial data for the S&P 500 and demonstrates advanced ML for expected-return prediction, cross-sectional feature learning, and dynamic portfolio optimization via the Markowitz, Black-Litterman, and robust optimization frameworks. Suitable for both academic projects and practical quantitative finance prototyping.

---

## Features

- **Full S&P 500 Data:** Covers over 500 U.S. equities, auto-updating tickers from Wikipedia.
- **Rich Feature Engineering:** Returns, volatility, RSI, technicals, PCA, and optional macro/factor integration.
- **Hybrid/Ensemble ML:** Random Forest, XGBoost, stacking, and LSTM/Deep Learning compatibility.
- **Complex Optimization:** Mean-variance (Efficient Frontier), Black-Litterman (market + ML blend), hierarchical and robust allocation.
- **Walk-Forward Backtesting:** Trains and tests using forward-chaining splits across dates (no leakage).
- **Comprehensive Reporting:** Sharpe ratio, max drawdown, annual returns, and allocation heatmaps.
- **Extensible Research Platform:** Modular for rapid expansion—add your own features or models with minimal effort!

---

## Project Structure

```plaintext
portfolio_optimization/
│
├── data_fetch.py              # Script to collect and clean stock data
├── feature_engineering.py     # Generate features for ML models
├── model_training.py          # Train and evaluate ML models
├── portfolio_optimization.py  # Optimize portfolio based on predictions
├── requirements.txt           # Dependencies list
├── README.md                  # Project documentation
│
├── data/                      # Raw and processed data files
│   ├── raw_data.csv
│   └── processed_data.csv
│
├── models/                    # Trained model files
│   └── random_forest_model.pkl
│
├── results/                   # Model outputs and evaluation metrics
│   └── performance_report.csv
│
└── notebooks/                 # Jupyter notebooks for experiments
    └── exploratory_analysis.ipynb
```

---

## Installation

1. **Clone this repo** and enter the directory:
    ```
    git clone https://github.com/yourusername/portfolio_optimization_ml.git
    cd portfolio_optimization_ml
    ```

2. **(Recommended) Create a virtual environment**:
    ```
    python -m venv venv
    source venv/bin/activate       # On Windows: venv\Scripts\activate
    ```

3. **Install requirements**:
    ```
    pip install -r requirements.txt
    ```

---

## Usage

1. Run the full pipeline (data collection, feature creation, modeling, optimization):
    ```
    python main.py
    ```
2. Results, models, and plots will be output to the console or respective subfolders.

3. For EDA and deeper exploration, use the included `notebooks/exploratory_analysis.ipynb`.

---

## Modeling Pipeline

- Scrapes, validates, and conserves minute-by-minute or daily stock data for all S&P 500 members.
- Applies advanced feature engineering (rolling stats, indicators, and factor-based signals).
- Chronological train-test split using the latest 20% of dates as the test set.
- Fits ensemble and/or deep models to predict risk and returns.
- Optimizes the portfolio using Efficient Frontier or Black-Litterman models.
- Backtests, evaluates, and visualizes performance versus S&P 500 and equal-weighted benchmarks.

---

## Evaluation

Metrics reported:
- **Annualized Sharpe Ratio**
- **Cumulative/Annual Return**
- **Maximum Drawdown**
- **Allocation Plots**
- **Benchmark Comparison**
