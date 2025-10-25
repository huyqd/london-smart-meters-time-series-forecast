# London Smart Meters Time Series Forecasting - Pedagogical Learning Repository

This repository documents a comprehensive learning journey through **time series forecasting**, progressing from classical statistical methods to advanced deep learning techniques. The project uses **London smart meter energy consumption data** as the primary dataset for hands-on exploration and experimentation.

## 📚 Learning Path Overview

The notebooks are structured to build progressively from foundational concepts to state-of-the-art techniques:

| Stage | Topic | Notebooks | Key Concepts |
|-------|-------|-----------|--------------|
| **0. Data Foundation** | Data retrieval & exploration | 0 | Data collection and initial understanding |
| **1. Understanding Data** | EDA & Decomposition | 1-2 | Time series patterns, seasonality, trends |
| **2. Classical Methods** | Baseline, ARIMA, Exponential Smoothing | 3-7 | Statistical foundations for forecasting |
| **3. Intermediate Methods** | Regression, Feature Engineering, Forecastability | 6-9 | Machine learning approaches to time series |
| **4. Ensemble & Global Models** | Ensemble methods, local/global forecasting | 10-13 | Combining models and hierarchical forecasting |
| **5. Probabilistic Forecasting** | Uncertainty quantification | 14 | Confidence intervals and distributions |
| **6. Deep Learning** | Neural networks for time series (WIP) | 15+ | LSTMs, Transformers, Attention mechanisms |

---

## 📖 Notebook Guide

### **Stage 0: Data Foundation**

#### **0. Data Retrieval** (`0.data-retrieval.ipynb`)
- Retrieves London smart meter dataset
- Data loading and preprocessing
- Understanding data structure and format

---

### **Stage 1: Understanding Data**

#### **1. Time Series EDA** (`1.time-series-eda.ipynb`)
- Exploratory data analysis of energy consumption patterns
- Visualization of temporal trends
- Identification of patterns and anomalies

#### **2. Time Series Decomposition** (`2.time-series-decomposition.ipynb` / `-fixed.ipynb`)
- Breaking down time series into components (trend, seasonality, residuals)
- Understanding underlying patterns
- Visual interpretation of decomposed components

---

### **Stage 2: Classical Statistical Methods**

#### **3. Baseline Forecasting** (`3.time-series-baseline-forecast.ipynb` / `-fixed.ipynb`)
- Simple forecast methods (naive, seasonal naive, moving average)
- Establishing performance baselines
- Metrics and evaluation frameworks

#### **4. Exponential Smoothing** (`4.time-series-exponential-smoothing.ipynb` / `-fixed.ipynb`)
- Simple, double, and triple exponential smoothing
- Holt-Winters method
- Parameter tuning and cross-validation

#### **5. ARIMA Models** (`5.time-series-arima.ipynb` / `-fixed.ipynb`)
- AutoRegressive Integrated Moving Average (ARIMA) fundamentals
- Stationarity testing (ADF test)
- ACF/PACF analysis and order selection
- Model fitting and diagnostics

#### **6. Statistical Models** (`6.time-series-statistical-models.ipynb`)
- Advanced ARIMA variants (SARIMA)
- Other statistical approaches
- Comparison of classical methods

---

### **Stage 3: Intermediate Machine Learning Methods**

#### **6. Time Series Regression** (`6.time-series-regression.ipynb` / `-fixed.ipynb`)
- Regression-based forecasting approaches
- Feature engineering for time series
- Tree-based models (Random Forest, Gradient Boosting)

#### **8. Feature Engineering** (`8.time-series-feature-engineering.ipynb`)
- Lag features and rolling statistics
- Temporal features (hour, day, week, season)
- Domain-specific feature creation

#### **8. Target Transformation** (`8.time-series-target-transform.ipynb`)
- Normalization and scaling strategies
- Log transformations
- Differencing and its effects

#### **7. Forecastability Analysis** (`7.time-series-forecastability.ipynb`)
- Assessing predictability of time series
- Entropy and autocorrelation measures
- Identifying which periods are easier/harder to forecast

#### **9. ML Strategy** (`9.time-series-ml-strategy.ipynb`)
- Cross-validation strategies for time series
- Walk-forward validation
- Hyperparameter tuning

#### **9. ML Models** (`9.time-series-ml.ipynb`)
- Machine learning models for forecasting
- Performance comparison and ensembling strategies

---

### **Stage 4: Ensemble & Hierarchical Methods**

#### **10. Time Series Ensembles** (`10.time-series-ensemble.ipynb`)
- Combining multiple forecast methods
- Weighted ensemble strategies
- Performance improvement through ensembling

#### **11. Handling Missing Data** (`11.time-series-handle-mising-data.ipynb`)
- Missing data imputation techniques
- Impact on model performance
- Robustness strategies

#### **12. Local Forecasting** (`12.time-series-local-forecast.ipynb`)
- Individual meter-level forecasting
- Meter-specific patterns and models
- Aggregation strategies

#### **13. Glocal Forecasting** (`13.time-series-glocal-forecast.ipynb`)
- Global models with local adaptability
- Transfer learning approaches
- Hierarchical forecasting (bottom-up, top-down, reconciliation)

---

### **Stage 5: Probabilistic Forecasting**

#### **14. Probabilistic Forecasting - Part 1** (`14.time-series-probabilistic-forecast-1.ipynb`)
- Quantile regression
- Prediction intervals
- Uncertainty quantification basics

#### **14. Probabilistic Forecasting - Part 2** (`14.time-series-probabilistic-forecast-2.ipynb`)
- Advanced uncertainty methods
- Distribution-based forecasts

#### **14. Probabilistic Forecasting - Part 3** (`14.time-series-probabilistic-forecast-3.ipynb`)
- Conformal prediction
- Coverage guarantees
- Practical applications

---

### **Stage 6: Deep Learning (In Progress)**

#### **15. Time Series Deep Learning** (`15.time-series-deep-learning.ipynb`)
- Deep learning fundamentals for time series
- Dense neural networks for forecasting
- RNN/LSTM basics

#### **15. Global Deep Learning** (`15.time-series-global-deep-learning.ipynb`)
- Training deep models across multiple time series
- Transfer learning approaches
- Multi-task learning

#### **15. Probabilistic Deep Learning** (`15.time-series-probabilistic-deep-learning.ipynb`)
- Uncertainty in neural networks
- Bayesian deep learning approaches
- Probabilistic outputs from neural models

#### **15.1. Deep Learning Manual (Draft)** (`15.1.time-series-deep-learning-manual-draft.ipynb`)
- Hands-on implementation of deep learning components
- Understanding neural network mechanics

#### **15.1. Deep Learning Manual** (`15.1.time-series-deep-learning-manual.ipynb`)
- Detailed manual implementation
- Building components from scratch

#### **15.2. Deep Learning with Attention** (`15.2.time-series-deep-learning-attention.ipynb`)
- Attention mechanisms in time series
- Transformer-based models
- Interpretability through attention weights

---

## 🎯 Key Learning Objectives

By working through this repository, you will learn:

1. ✅ **Classical methods**: Understanding statistical foundations
2. ✅ **ML approaches**: Applying machine learning to time series
3. ✅ **Ensemble techniques**: Combining predictions effectively
4. ✅ **Hierarchical forecasting**: Working with multiple aggregation levels
5. ✅ **Uncertainty quantification**: Moving beyond point forecasts
6. ✅ **Deep learning**: Modern neural network approaches
7. ✅ **Practical implementation**: Real-world forecasting challenges

---

## 🛠️ Technology Stack

- **Python 3.x**
- **Core Libraries**: pandas, numpy, scikit-learn
- **Time Series**: statsmodels, pmdarima, sktime
- **Deep Learning**: PyTorch, PyMC
- **Visualization**: matplotlib, plotly
- **Optimization**: Optuna, hyperopt

---

## 📊 Dataset

**London Smart Meters Dataset** - Half-hourly energy consumption data for multiple meters across London. The dataset provides a rich environment for exploring:
- Daily patterns
- Weekly seasonality
- Annual trends
- Weather effects
- Multi-level hierarchies (meter → area → region)

---

## 💡 How to Use This Repository

1. **Sequential Learning**: Start from notebook 0 and progress through the stages
2. **Selective Focus**: Jump to sections matching your learning goals
3. **Experimentation**: Modify cells and try variations
4. **Reference**: Use as a reference guide for future time series projects

---

## 📝 Status

- ✅ Stages 0-5: Complete
- 🚧 Stage 6 (Deep Learning): In Progress
  - LSTM and manual implementations underway
  - Attention mechanisms being explored
  - Probabilistic deep learning integration in progress

---

## 📚 Additional Resources

- Tutorial notebooks in `hef_tutorial/` for hierarchical forecasting
- PyMC tutorials for Bayesian approaches
- Optuna integration examples for hyperparameter optimization
- Utility modules: `utils.py`, `metrics_utils.py`, `plotting_utils.py`, `summary_utils.py`

---

## 🚀 Next Steps

- Complete deep learning explorations
- Implement hybrid models combining classical and deep learning
- Explore real-time forecasting scenarios
- Build production-ready pipelines

---

**Note**: This is a living learning repository. Content evolves as new concepts are explored and implemented.
