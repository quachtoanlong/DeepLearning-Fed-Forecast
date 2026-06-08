# DeepLearning-Fed-Forecast

Two complementary approaches to forecasting U.S. Federal Reserve interest rate decisions using deep learning and large language models. Each sub-project lives in its own folder with its own notebook and data.

---

## Repository Structure

```
DeepLearning-Fed-Forecast/
├── RNN/    # LSTM and GRU Fed Model
└── LLM/    # LLM Fed Model
```

---

## Sub-projects

### [`RNN/`](./RNN) — LSTM and GRU Forecaster

A one-step-ahead forecaster for the U.S. effective federal funds rate built with a **LSTM** and **GRU**. Hyperparameters are tuned via Bayesian optimisation (`scikit-optimize`) over a 6-dimensional search space (lag window, learning rate, epochs, hidden size, layers, dropout).

**Highlights:**
- Rolling-origin evaluation from 2016 onward
- Two forecasting experiments across different training cutoffs (end-2015, end-2019)

> **Note:** The shipped `Input/Data_LQ.csv` is synthetic (schema-matched random noise). Real data comes from Haver Analytics. FRED public substitutes are recommended if you don't have a Haver licence.

**Requirements:** `torch`, `scikit-optimize`, `pandas`, `numpy`, `matplotlib`, `jupyter`

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r RNN/requirements.txt
jupyter lab RNN/LSTM_model_effective_FFrate_v3.ipynb
```

---

### [`LLM/`](./LLM) — Prompt Engineering with Claude (AWS Bedrock)

A prompt engineering research project that uses **Anthropic's Claude 3.5 Sonnet** (via AWS Bedrock) to predict FOMC rate decisions from historical macroeconomic data — no model training, pure LLM inference.

Five prompt framing strategies are systematically tested and validated against **112 actual Fed decisions** (December 2015 – February 2025).

**Highlights:**
- 5 prompt variants: Financial Analyst, Central Bank Advisor, Quantitative Model, Policy Simulation, Instructional Command
- Best MAE: **0.1492%** (Prompt 1 — Financial Analyst) — less than one 25 bps Fed move
- Monte Carlo: 100 simulations of the best prompt; MAE stable at 0.1483 ± 0.0045
- Sensitivity analysis: economic shock scenarios (e.g. unemployment +2%, inflation +5%)

**Prompt comparison (MAE — lower is better):**

| Rank | Prompt | Framing | Dates Evaluated | MAE |
|---|---|---|---|---|
| 1 | Prompt 1 | Financial Analyst | 112/112 | **0.1492** |
| 2 | Prompt 2 | Central Bank Advisor | 106/112 | 0.1518 |
| 3 | Prompt 4 | Policy Simulation | 103/112 | 0.1575 |
| 4 | Prompt 5 | Instructional Command | 104/112 | 0.1622 |
| 5 | Prompt 3 | Quantitative Model | 112/112 | 0.1689 |

**Requirements:** `boto3`, `pandas`, `tqdm`, `matplotlib`, `seaborn`, `scikit-learn`  
**AWS:** Bedrock access in `us-east-1` with Claude 3.5 Sonnet model access enabled

```bash
pip install boto3 pandas tqdm matplotlib seaborn scikit-learn
# Then open LLM/Main_Analysis.ipynb in JupyterLab or SageMaker Studio
```

---

## Licence

Code: MIT. See [`RNN/LICENSE`](./RNN/LICENSE). Synthetic sample data in `RNN/Input/` is also MIT-licensed. Any real data you supply locally is governed by your data provider's terms.
