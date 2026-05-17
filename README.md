# Forecasting the Effective Fed Funds Rate with an LSTM

A one-step-ahead forecaster for the U.S. effective federal funds rate, built
with a PyTorch LSTM and benchmarked against a VAR baseline. Hyperparameters
are tuned via Bayesian optimisation (`scikit-optimize`) over a 6-dimensional
search space.

## What's in here

```
.
├── LSTM_model_effective_FFrate_v3.ipynb   # the model
├── Input/
│   └── Data_LQ.csv                        # SYNTHETIC sample (see below)
├── requirements.txt
├── LICENSE                                # MIT
└── .gitignore
```

## Important: the shipped data is synthetic

`Input/Data_LQ.csv` in this repository is **not real macroeconomic data**.
It is a schema-matched file: same 22 columns, same 329 monthly observations
from Jan-1998 to May-2025, but every value is **uniform random noise** drawn
within the original min/max range of each column. The notebook will run end
to end against it, but the model will not learn anything meaningful — the
output is for verifying the pipeline only.

The real input series come from Haver Analytics, which prohibits
redistribution of subscriber data. To reproduce the actual results you need
to supply your own `Input/Data_LQ.csv` with the same schema (see below).

## Column schema

The CSV must have a `Date` column in `d/m/Y` format plus the following 21
numeric columns (order isn't strict; the notebook indexes by name):

| Column | Meaning |
|---|---|
| `FF` | Effective federal funds rate (target) |
| `CINF1`, `CINF5` | 1y and 5y inflation expectations |
| `MLU6` | Civilian employment level |
| `FM2` | M2 money stock |
| `LH` | Not in labor force |
| `PCU` | PCE price index |
| `PCUSLFE` | Core PCE price index |
| `JCBM`, `JCXFEBM` | Additional PCE-type deflators |
| `USPHPIM` | House price index |
| `GSACPPIC` | Commodity / PPI series |
| `PZRAW` | Raw industrials price index |
| `LANAGRD` | Non-farm payrolls (dropped before training) |
| `LKPRIVA` | Private sector employment |
| `LEPRIVA` | Average hourly earnings (dropped before training) |
| `Yield6m`, `Yield1Y`, `Yield2Y`, `Yield10Y` | Treasury yields |
| `MGDPN` | Nominal GDP |

Public substitutes from [FRED](https://fred.stlouisfed.org/) exist for almost
all of these (`FEDFUNDS`, `M2SL`, `PCEPI`, `PCEPILFE`, `DGS6MO`/`DGS1`/
`DGS2`/`DGS10`, `CPIAUCSL`, `PAYEMS`, `GDP`, etc.). If you don't have a Haver
licence, swapping these in is the recommended path.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # or your env of choice
pip install -r requirements.txt
jupyter lab LSTM_model_effective_FFrate_v3.ipynb
```

A CUDA GPU is used automatically if available, otherwise CPU.

## What the notebook does

1. **Loads + transforms data.** Reads `Input/Data_LQ.csv`, converts most level
   series to year-over-year percent changes, constructs two yield-curve
   spreads (`YC_2y_10y = Yield10Y − Yield2Y`, `YC_6m_1y = Yield1Y − Yield6m`),
   shifts `FF` by –1 so the target is next month's rate, drops the first 13
   rows of NaNs introduced by the YoY transforms, and MinMax-scales.
2. **Defines the model.** A configurable PyTorch `LSTM(input_size, hidden,
   output, num_layers, dropout)` with two fully-connected heads, plus a `GRU`
   variant for comparison.
3. **Hyperparameter search (optional).** `bayesian_optimization(...)` wraps
   `gp_minimize` over `lag_num ∈ {4,6,12,24}`, `learn_rate ∈ [1e-4, 0.1]`,
   `num_epochs ∈ [300, 800]`, `hidden_size ∈ [38, 190]`, `num_layers ∈ [1,7]`,
   `dropout ∈ [0, 0.7]`. Toggled via the `CV` flag — currently `False`, which
   reuses the best config from a previous search.
4. **Three forecasting experiments.**
   - Rolling-origin one-step forecast from 2016 onward, with a VAR(lag_num)
     baseline computed at each step.
   - Fixed-sample LSTM trained through 2015-12, predicted forward.
   - Same with a 2020-01 cutoff.
5. **Saves outputs.** Timestamped CSVs and PNGs (see `.gitignore` —
   these are excluded from version control).

## Known issues

- **Cell 11 metric bug.** `pred` and `actual` are swapped in the variable
  assignment, and the MAE line computes `mean(actual − pred)` (mean error /
  bias) rather than `mean(|actual − pred|)`. RMSE is unaffected by the swap.
- **Saved best `hidden_size = 224` is outside the current search space**
  `[38, 190]`. If you re-run the Bayesian search you'll need to widen the
  upper bound (e.g. `Integer(38, 256, name='hidden_size')`) to be able to
  recover the previous best.

## Licence

Code: MIT (see `LICENSE`). The synthetic sample data in `Input/` is also
MIT-licensed. Any real data you supply locally is your responsibility and
governed by your own data provider's terms.
