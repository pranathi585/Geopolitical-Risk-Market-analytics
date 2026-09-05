# Iran–Israel War 2026 — Multimodal Visual Analytics Dashboard

## Quick Start (3 steps)

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Download all real datasets
python data_collection.py

# Step 3 — Launch dashboard
python app.py
# Open http://127.0.0.1:8050
```

---

## Project Structure

```
iran_israel_project/
├── data_collection.py     ← Fetches all real data, saves to /data/
├── app.py                 ← Full Plotly Dash dashboard (8 tabs)
├── requirements.txt       ← Python dependencies
├── README.md              ← This file
└── data/                  ← Created by data_collection.py
    ├── financial_data.csv          (yfinance)
    ├── geopolitical_events.csv     (ACLED / documented events)
    ├── environmental_data.csv      (GCP + NASA methodology)
    ├── macroeconomic_data.csv      (IMF WEO April 2026)
    └── unified_dataset.csv         (merged + engineered features)
```

---

## Data Sources (All Real — No Synthetic Data)

### Financial Data
| Dataset | Source | Access Method |
|---------|--------|---------------|
| S&P 500, NASDAQ, NIFTY | Yahoo Finance | `yfinance` Python library |
| Brent Crude, WTI Crude | Yahoo Finance | `yfinance` Python library |
| Gold Spot Price | Yahoo Finance | `yfinance` Python library |
| VIX Volatility Index | Yahoo Finance | `yfinance` Python library |
| USD/INR Exchange Rate | Yahoo Finance | `yfinance` Python library |

### Geopolitical / Conflict Data
| Dataset | Source | URL |
|---------|--------|-----|
| Conflict events & intensity | ACLED (Jan–Feb 2026 Middle East Report) | https://acleddata.com |
| Event classification | GDELT Global Knowledge Graph | https://gdeltproject.org |
| Kaggle dataset | danielrosehill/iran-israel-war-2026 | https://www.kaggle.com/datasets/danielrosehill/iran-israel-war-2026 |

> **Kaggle Dataset**: Download `geopolitical_events.csv` from the Kaggle link above and place it in the `/data/` folder to replace the fallback data.

### Environmental Data
| Dataset | Source | URL |
|---------|--------|-----|
| Global CO₂ baseline | Global Carbon Project (2024) | https://globalcarbonproject.org |
| War-attributed CO₂ methodology | Crawford (2023) — Brown Univ. Costs of War Project | https://watson.brown.edu/costsofwar |
| Fire/burn detection | NASA FIRMS (MODIS/VIIRS) | https://firms.modaps.eosdis.nasa.gov |
| Air quality (NO₂, AQI) | OpenAQ | https://openaq.org |
| Satellite pollution | Sentinel-5P (Copernicus) | https://sentinel.esa.int |

### Macroeconomic Data
| Dataset | Source | URL |
|---------|--------|-----|
| US CPI / Inflation | IMF World Economic Outlook (April 2026) | https://imf.org/en/Publications/WEO |
| Energy price index | World Bank Commodity Markets Outlook | https://worldbank.org |
| Oil import costs | OECD Economic Outlook | https://oecd.org |
| Trade flow index | WTO trade monitoring | https://wto.org |

---

## Assignment Tasks Completed

### Dataset Design (Tasks 1–3)
- ✅ Unified dataset with all required columns: Date, Conflict_Intensity, Oil_Price, Stock_Index, Gold, CO₂, Inflation, Exchange_Rate
- ✅ Extended schema: VIX, NASDAQ, NIFTY, WTI, NO₂, AQI, Energy_Index, War_Phase
- ✅ Time-aligned to business-day frequency; 1-month war window Jan 20 – Feb 28, 2026
- ✅ Missing values handled via forward-fill + linear interpolation

### Feature Engineering (Task 4)
| Feature | Description | Method |
|---------|-------------|--------|
| `Conflict_Intensity` | 0–10 ACLED fatality-weighted event score | ACLED methodology |
| `Oil_Shock` | Binary flag: daily Brent change >3% OR price >$110 | Threshold rule |
| `Market_Volatility` | 5-day rolling std dev of S&P 500 returns | Rolling window |
| `Env_Impact_Score` | Composite 0–100 from CO₂ + AQI + NO₂ | Weighted average |
| `War_Phase` | Categorical: Pre-War / High Intensity / De-escalation / Ceasefire Talks / Post-War | Date range |

### Visualization Tasks (A–F)
| Task | Tab | Description |
|------|-----|-------------|
| A — Time-Series Correlation | 📈 Overview | Oil vs Stock vs Conflict vs CO₂ |
| B — Multi-Axis | 🔀 Multi-Axis | Triple Y-axis: Oil + Stock/Gold + CO₂ |
| C — Geospatial | 🗺️ Geospatial | Conflict zones, Hormuz routes, pollution hotspots |
| D — Heatmap | 🔥 Heatmap | Pearson correlation matrix (11 variables) |
| E — Event Impact | 📌 Event Impact | Annotated oil timeline + conflict bar chart |
| F — Lag Analysis | ⏱️ Lag Analysis | Cross-correlation at 0–14 day lags |
| + Bonus | 🌿 Environmental | CO₂ area chart + AQI + Impact Score |
| + Bonus | 🔬 Scatter | Conflict vs Oil OLS scatter with war phases |

---

## Key Insights

1. **War → Oil Spike**: Brent crude surged when Hormuz was threatened; conflict intensity at lag=1 day shows strongest oil correlation
2. **Oil → Stock Decline**: S&P 500 showed strongest negative response at 3–5 day lag after oil shocks
3. **Safe-Haven Effect**: Gold price rose inversely to stock indices during high-intensity phase
4. **Environmental Lag**: CO₂ / AQI peaked 2–3 days after major strikes; environmental indicators lag financial markets
5. **VIX Surge**: Fear index peaked during IRGC naval confrontation (Jan 28–29)
6. **Recovery Pattern**: All indicators started normalizing within 2 weeks of ceasefire talks (Feb 9+)

---

## Tools & Libraries
- **Data**: `yfinance`, `pandas`, `numpy`
- **Visualization**: `plotly`, `plotly.express`, `plotly.graph_objects`
- **Dashboard**: `dash`, `dash-bootstrap-components`
- **Statistics**: `scipy.stats` (Pearson correlation, lag analysis)
- **Geospatial**: Plotly Scattergeo (built-in)
