import os
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = "data"
WAR_START = "2026-01-01"
FETCH_END = "2026-04-10"
WAR_ONSET = "2026-01-20"  # Start of 1-month high intensity phase
WAR_END   = "2026-02-28"  # End of 1-month high intensity phase

print(f"============================================================")
print(f"  IRAN–ISRAEL WAR 2026 — Data Processing (Tasks 3 & 4)")
print(f"  Time Alignment : {WAR_START} → {FETCH_END}")
print(f"  War Period     : {WAR_ONSET} → {WAR_END}")
print(f"============================================================\n")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3: TIME ALIGNMENT & DATA INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
print("[1/2] Loading raw datasets and aligning time scales …")

# Load Financial
fin_path = os.path.join(DATA_DIR, "financial_data.csv")
fin_df = pd.read_csv(fin_path, index_col="Date", parse_dates=True) if os.path.exists(fin_path) else pd.DataFrame()

# Load Geopolitical
geo_path = os.path.join(DATA_DIR, "geopolitical_events.csv")
geo_df = pd.read_csv(geo_path, index_col="Date", parse_dates=True) if os.path.exists(geo_path) else pd.DataFrame()

# Load Environmental
env_path = os.path.join(DATA_DIR, "environmental_data.csv")
env_df = pd.read_csv(env_path, index_col="Date", parse_dates=True) if os.path.exists(env_path) else pd.DataFrame()

# Load Macroeconomic
macro_path = os.path.join(DATA_DIR, "macroeconomic_data.csv")
macro_df = pd.read_csv(macro_path, index_col="Date", parse_dates=True) if os.path.exists(macro_path) else pd.DataFrame()

# Define the common daily timescale
bdays = pd.date_range(WAR_START, FETCH_END, freq="D")
unified = pd.DataFrame(index=bdays)
unified.index.name = "Date"

# Handle Missing Values (Task 3 requirement)
# Financial: Forward fill missing weekends, then backward fill leading holidays (like Jan 1)
if not fin_df.empty:
    fin_aligned = fin_df.reindex(bdays).ffill().bfill()
    unified = unified.join(fin_aligned, how="left")

# Geopolitical: Keep zeros (0 intensity means peaceful day), forward fill missing intermediate days
if not geo_df.empty:
    geo_daily = geo_df[["Conflict_Intensity", "Fatalities"]].reindex(bdays).fillna(0)
    unified = unified.join(geo_daily, how="left")

# Environmental
if not env_df.empty:
    env_aligned = env_df.reindex(bdays).ffill().bfill()
    unified = unified.join(env_aligned, how="left")

# Macroeconomic
if not macro_df.empty:
    macro_aligned = macro_df.reindex(bdays).ffill().bfill()
    unified = unified.join(macro_aligned, how="left")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("[2/2] Engineering features …")

# 1. Oil Shock Indicator — 1 if Brent daily change > 3% OR price > $110
if "BRENT" in unified.columns:
    unified["Oil_Daily_Return"] = unified["BRENT"].pct_change().fillna(0) * 100
    unified["Oil_Shock"] = (
        (unified["Oil_Daily_Return"].abs() > 3) |
        (unified["BRENT"] > 110)
    ).astype(int)
else:
    unified["Oil_Daily_Return"] = 0.0
    unified["Oil_Shock"] = 0

# 2. Market Volatility Measure — 5-day rolling std of SP500 returns
if "SP500" in unified.columns:
    sp_ret = unified["SP500"].pct_change().fillna(0) * 100
    unified["Market_Volatility"] = sp_ret.rolling(5, min_periods=1).std().fillna(0)
else:
    unified["Market_Volatility"] = 0.0

# 3. Environmental Impact Score — composite (0–100)
if "War_CO2_kt" in unified.columns:
    co2_norm  = (unified["War_CO2_kt"]  / unified["War_CO2_kt"].max()).fillna(0)
    aqi_norm  = (unified["AQI"]         / unified["AQI"].max()).fillna(0)
    no2_norm  = (unified["NO2_ugm3"]    / unified["NO2_ugm3"].max()).fillna(0)
    unified["Env_Impact_Score"] = ((co2_norm * 0.5 + aqi_norm * 0.3 + no2_norm * 0.2) * 100).round(2)
else:
    unified["Env_Impact_Score"] = 0

# 4. Conflict Intensity Index
# Already integrated but ensuring it's on a 0-10 scale explicitly for the dashboard
unified["Conflict_Intensity"] = unified["Conflict_Intensity"].clip(upper=10)

# 5. War Phase label (Highlights the 1-month war period)
def war_phase(d):
    ts = pd.Timestamp(d)
    if ts < pd.Timestamp(WAR_ONSET):                return "Pre-War"
    elif ts <= pd.Timestamp("2026-02-07"):           return "High Intensity" # Start of 1-Month period
    elif ts <= pd.Timestamp("2026-02-20"):           return "De-escalation"
    elif ts <= pd.Timestamp(WAR_END):                return "Ceasefire Talks" # End of 1-month period
    else:                                             return "Post-War"

unified["War_Phase"] = unified.index.map(war_phase)

# Save Final Unified Dataset
unified.to_csv(os.path.join(DATA_DIR, "unified_dataset.csv"))

print(f"\n  ✓ unified_dataset.csv saved successfully!")
print(f"    Shape      : {unified.shape}")
print(f"    Date range : {unified.index[0].date()} → {unified.index[-1].date()}")
print(f"    Columns    : {len(unified.columns)} total features")

print("\n" + "=" * 60)
print("  DATA PROCESSING COMPLETE (Tasks 3 & 4 achieved)")
print("  → Now run:  python app.py to launch the dashboard")
print("=" * 60 + "\n")
