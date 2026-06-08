# Federal Reserve Rate Predictor — AWS Bedrock + Claude

A prompt engineering research project that uses Anthropic's Claude (via AWS Bedrock) to predict U.S. Federal Reserve interest rate decisions from historical macroeconomic data. No model training — pure LLM inference.

---

## Overview

The project systematically tests 5 different prompt engineering strategies to see how accurately Claude can forecast FOMC rate decisions when given recent economic indicators. Predictions are validated against 112 actual Fed decisions (December 2015 – February 2025) and evaluated using standard regression error metrics.

---

## AWS Infrastructure

| Component | Detail |
|---|---|
| **Service** | AWS Bedrock (inference only — no SageMaker training jobs) |
| **Model** | `anthropic.claude-3-5-sonnet-20240620-v1:0` (Claude 3.5 Sonnet) |
| **Region** | `us-east-1` |
| **Auth** | AWS credentials configured in your environment (IAM role or access keys with Bedrock permissions) |

---

## Repository Structure

```
aws/
├── Main_Analysis.ipynb                        # Main notebook — full pipeline
├── Notebook_with_Sensitivity.ipynb            # Shock & sensitivity analysis
├── AWS_LLM_Fed.ipynb                          # Initial exploratory notebook
├── Data_Exploration.ipynb                     # Data loading utilities
│
├── Data_cleaned.csv                           # Input: 32 economic indicators (masked with noise, monthly)
├── Monthly_Fed_Funds_Target.csv               # Input: Actual Fed Funds Target rates (monthly)
├── Fed_Funds_Target.csv                       # Input: Actual Fed Funds Target rates (daily)
│
├── Claude_Prompt_1_Predictions.csv            # Output: Prompt variant 1 predictions
├── Claude_Prompt_2_Predictions.csv            # Output: Prompt variant 2 predictions
├── Claude_Prompt_3_Predictions.csv            # Output: Prompt variant 3 predictions
├── Claude_Prompt_4_Predictions.csv            # Output: Prompt variant 4 predictions
├── Claude_Prompt_5_Predictions.csv            # Output: Prompt variant 5 predictions
├── Claude_Prompt_2_Average_Predictions.csv    # Output: Ensemble average across 100 simulations
│
├── Claude_Fed_Funds_Validation.csv            # Validation results with explanations
├── FedFundsRate_Error.csv                     # MAE/MSE/RMSE per simulation run
├── FedFundsRate_ErrorSummary.csv              # Statistical summary across 100 runs
│
├── figLLMvFed.png                             # Chart: Actual vs average predicted rate
├── FedFundsRate_Simulation_vs_Actual.png      # Chart: 100 simulation bands vs actual
│
└── simulations/                               # All 100 individual Monte Carlo run CSVs
    ├── Claude_Prompt_2_Simulation_1.csv
    ├── Claude_Prompt_2_Simulation_2.csv
    ├── ...
    └── Claude_Prompt_2_Simulation_100.csv
```

---

## Input Data

**`Data_cleaned.csv`** — Masked Monthly economic indicators, Dec 2015 – Feb 2025:

| Category | Variables |
|---|---|
| Unemployment | U3 (official rate), U6 (broad/discouraged workers) |
| Inflation | CPI (headline & core), PCE (headline & core), 1-yr & 5-yr inflation expectations |
| Employment | Nonfarm payroll change, average weekly earnings growth, average hourly earnings growth |
| Money Supply | M1, M2, M2 growth rate |
| Real Estate | House price growth, commercial real estate price growth |
| Output | GDP growth rate |

**`Monthly_Fed_Funds_Target.csv`** — Actual FOMC target rate decisions (ground truth for validation).

---

## How It Works

### Pipeline (run cells in this order in the main notebook)

**1. Setup**
- Install: `boto3`, `pandas`, `tqdm`, `matplotlib`, `seaborn`, `scikit-learn`
- Load both CSV datasets

**2. Prompt Engineering — 5 Variants Tested**

Each prompt receives 6 months of lookback economic data formatted as a markdown table and asks Claude to respond in JSON:
```json
{ "predicted_rate": 5.25, "confidence": 0.80, "explanation": "..." }
```

| Prompt | Framing |
|---|---|
| 1 | Financial analyst perspective |
| 2 | Central bank advisor (FOMC-focused) |
| 3 | Quantitative model (data-driven, inflation/labor focus) |
| 4 | Simulated FOMC policy discussion |
| 5 | Direct instructional command |

**3. Validation**
- Test each prompt on 112 monthly dates (Dec 2015 – Feb 2025)
- Compare predicted vs actual Fed rate
- Calculate MAE, MSE, RMSE per prompt

**4. Monte Carlo Simulation**
- Run 100 simulations using Prompt 2 (Central Bank Advisor — selected as final model)
- Aggregate ensemble average predictions
- Measure error stability across runs
- Individual run files saved to `simulations/`

**5. Sensitivity Analysis** (`Notebook_with_Sensitivity.ipynb`)
- Apply economic shocks (e.g. unemployment +2%, inflation +5%)
- Compare base-case vs shocked predictions
- Assess how sensitive the model is to economic changes

---

## Results

### Prompt Comparison (MAE — lower is better)

| Rank | Prompt | Framing | Dates Evaluated | MAE |
|---|---|---|---|---|
| 1 | Prompt 1 | Financial Analyst | 112/112 | **0.1492** |
| 2 | Prompt 2 | Central Bank Advisor | 106/112 | 0.1518 |
| 3 | Prompt 4 | Policy Simulation | 103/112 | 0.1575 |
| 4 | Prompt 5 | Instructional Command | 104/112 | 0.1622 |
| 5 | Prompt 3 | Quantitative Model | 112/112 | 0.1689 |

> Average error of ~0.15% — less than one quarter-point (25 bps) Fed move.
>
> Prompt 2 was selected for the Monte Carlo simulation based on prior analysis and is hardcoded in the notebook (`prompt_index = 1`). This can be updated to any prompt variant as needed.

### Monte Carlo Simulation (Prompt 2, 100 runs)

| Metric | Mean | Std Dev | Range |
|---|---|---|---|
| MAE | 0.1483 | ±0.0045 | 0.1398 – 0.1606 |
| RMSE | 0.1762 | ±0.0097 | — |

Low variance across 100 runs confirms consistent model behavior.

---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `n` | `6` | Months of lookback history fed to each prompt |
| `temperature` | `0.5` | Moderate randomness in Claude responses |
| `max_tokens` | `500` | Max Claude output length |
| `model_id` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | Claude model via Bedrock |
| `bedrock_region` | `us-east-1` | AWS region |
| Simulations | `100` | Monte Carlo runs for ensemble analysis |
| Test dates | `112` | Monthly FOMC dates evaluated |

---

## Prerequisites

### Python Dependencies
```bash
pip install boto3 pandas tqdm matplotlib seaborn scikit-learn
```

### AWS Setup
1. An AWS account with **Amazon Bedrock** access enabled in `us-east-1`
2. Model access granted for **Anthropic Claude 3.5 Sonnet** in the Bedrock console
3. AWS credentials configured in your environment:
   ```bash
   aws configure
   # or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
   ```
   If running on SageMaker, the instance execution role needs `bedrock:InvokeModel` permission.

---

## Quickstart

1. Open `Main_Analysis.ipynb` in JupyterLab or SageMaker Studio
2. Run all cells in order
3. Prediction CSVs and charts will be saved to the working directory
4. For sensitivity analysis, open `Notebook_with_Sensitivity.ipynb` separately
