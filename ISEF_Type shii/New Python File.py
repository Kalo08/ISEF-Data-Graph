"""
ISEF Project: Automated Detection of Tritium Using Time Series Analysis

Aligns with Research Plan (C.1.b):
- Baseline from data prior to leak (2005–2009)
- Time series at regular intervals; ML anomaly detection vs baseline
- Compare ML-detected anomalies with descriptive statistics (e.g. Z-score)
- Hypothesis: anomalies will be identified before official detection (Jan 7, 2010)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Study period and baseline settings
YEAR_START = 2005
YEAR_END = 2024
BASELINE_END_YEAR = 2009  # pre-leak baseline ends before 2010
OFFICIAL_DETECTION_DATE = "2010-01-07"  # Vermont Yankee tritium leak first reported

# ----- STEP 2: Load dataset (C.1.a: NRC reports → standardized spreadsheets) -----
df = pd.read_csv("tritium_data.csv")
print("Raw data preview:")
print(df.head())

# ======================== Column mapping for NRC-style reports ========================
# Supports: 'Distance (km)' or 'Distance (km) & Direction'; 'Min'/'Max' or 'Minimum'/'Maximum'

column_rename_dict = {
    'Year': 'Year',
    'Medium': 'Medium',
    'Radionuclide': 'Radionuclide',
    'Location Name': 'Location_Name',
    'Indicator/ Control': 'Indicator_Type',
    'Distance (km) & Direction': 'Distance_km_and_Dir',
    'Distance (km)': 'Distance_km',
    'Mean Measurement (pCi/L)': 'Mean_pCi_L',
    'Fraction Detectable': 'Fraction_Detectable',
    'Min (pCi/L)': 'Min',
    'Max (pCi/L)': 'Max',
    'Minimum (pCi/L)': 'Min',
    'Maximum (pCi/L)': 'Max',
    'LLD (pCi/L)': 'LLD'
}
df = df.rename(columns=column_rename_dict)

# Add a fake Quarter column as all data is yearly, set to Q1 (or estimate Q4 or as needed)
if "Quarter" not in df.columns:
    df["Quarter"] = 1  # Default to Q1 for all rows, since source does not break out quarters

# If Distance_km numeric field not yet extracted, do so:
if "Distance_km" not in df.columns:
    # Extract numeric value from 'Distance_km_and_Dir' (e.g. "0.3 km")
    df["Distance_km"] = df["Distance_km_and_Dir"].str.extract(r'([\d.]+)').astype(float)

# Clean up Indicator_Type for mapping
df["Indicator_Type"] = df["Indicator_Type"].str.strip()

# Clean up and convert numeric fields
# Convert 'Mean_pCi_L', 'Min', 'Max', 'LLD' to numeric (LLD might have "500 +4" form)
df["Mean_pCi_L"] = pd.to_numeric(df["Mean_pCi_L"], errors="coerce")
df["Min"] = pd.to_numeric(df["Min"], errors="coerce")
df["Max"] = pd.to_numeric(df["Max"], errors="coerce")
df["LLD"] = pd.to_numeric(df["LLD"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")  # extract first numeric part only

# Convert Fraction_Detectable to a fraction: form is like "0/4", "1/4", etc.
def frac_detect(x):
    try:
        a, b = x.split("/")
        return int(a) / int(b) if int(b) != 0 else 0
    except:
        return 0
df["Fraction_Detectable"] = df["Fraction_Detectable"].astype(str).apply(frac_detect)

# ----- STEP 3: Create time column (data at regular intervals) -----
df = df.dropna(subset=["Year", "Quarter"])
df["Year"] = df["Year"].astype(int)
df["Quarter"] = df["Quarter"].astype(int)
df["Date"] = pd.PeriodIndex(df["Year"].astype(str) + "Q" + df["Quarter"].astype(str), freq="Q").to_timestamp()
df = df.sort_values("Date")

# ----- STEP 4: Filter to tritium + groundwater (environmental monitoring focus) -----
df = df[df["Medium"].str.lower() == "groundwater"]
# Limit analysis to chosen study period (e.g., 2005–2024)
df = df[(df["Year"] >= YEAR_START) & (df["Year"] <= YEAR_END)]

# ----- STEP 5: Extra features for model (C.3: concentration, detection status, etc.) -----
df["Range"] = df["Max"] - df["Min"]
df["Indicator_Binary"] = df["Indicator_Type"].map({"Indicator": 1, "Control": 0})
df = df.fillna(0)

# ----- STEP 6–7: Features and scaling -----
features = ["Mean_pCi_L", "Range", "Fraction_Detectable", "LLD", "Indicator_Binary", "Distance_km"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----- STEP 8: Baseline = pre-leak only (2005–2009); train ML on baseline -----
baseline = df[df["Year"] <= BASELINE_END_YEAR]
if len(baseline) == 0:
    print("Warning: No baseline data (Year <= %d). Training on full dataset instead." % BASELINE_END_YEAR)
    train_df = df
else:
    train_df = baseline
X_base = scaler.fit_transform(train_df[features])

model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
model.fit(X_base)

# ----- STEP 9: Apply model to full dataset -----
X_full = scaler.transform(df[features])
df["Anomaly"] = model.predict(X_full)  # -1 = anomaly, 1 = normal

# ----- STEP 10: Traditional method — Z-score (descriptive statistics) -----
df["Z"] = (df["Mean_pCi_L"] - df["Mean_pCi_L"].mean()) / df["Mean_pCi_L"].std()
df["Z_Anomaly"] = df["Z"].abs() > 3

# ----- HYPOTHESIS: First anomaly date vs official detection -----
anomaly_rows = df[df["Anomaly"] == -1]
first_ml_anomaly_date = anomaly_rows["Date"].min() if len(anomaly_rows) > 0 else None
first_z_anomaly_date = df[df["Z_Anomaly"]]["Date"].min() if df["Z_Anomaly"].any() else None

# ----- Time series trend (simple rolling mean for visualization) -----
df["Mean_pCi_L_rolling"] = df["Mean_pCi_L"].rolling(window=3, min_periods=1).mean()

# Ensure Location_Name exists for filters
if "Location_Name" not in df.columns and "Location" in df.columns:
    df["Location_Name"] = df["Location"]
elif "Location_Name" not in df.columns:
    df["Location_Name"] = "N/A"

# ==================== INTERACTIVE: Toggle Graph/Table + Filters ====================
from matplotlib.widgets import RadioButtons, Slider, CheckButtons, Button
from matplotlib.gridspec import GridSpec

def get_filtered_df(year_min, year_max, anomalies_only, normal_only, selected_locations):
    """Apply current filter state to full df."""
    out = df[(df["Year"] >= year_min) & (df["Year"] <= year_max)]
    if selected_locations and "All" not in selected_locations:
        out = out[out["Location_Name"].astype(str).isin(selected_locations)]
    if anomalies_only:
        out = out[out["Anomaly"] == -1]
    if normal_only:
        out = out[out["Anomaly"] == 1]
    return out.sort_values("Date").copy()

def draw_graph(ax, data):
    ax.clear()
    if len(data) == 0:
        ax.text(0.5, 0.5, "No data after filters", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return
    data = data.sort_values("Date").reset_index(drop=True)
    ax.plot(data["Date"], data["Mean_pCi_L"], "o-", label="Mean Tritium (pCi/L)", markersize=4)
    data_rolling = data["Mean_pCi_L"].rolling(window=3, min_periods=1).mean()
    ax.plot(data["Date"], data_rolling, "--", color="gray", alpha=0.8, label="Trend (3-period rolling mean)")
    anom = data[data["Anomaly"] == -1]
    if len(anom) > 0:
        ax.scatter(anom["Date"], anom["Mean_pCi_L"], color="red", s=60, zorder=5, label="ML-detected anomaly")
    ax.axvline(pd.Timestamp(OFFICIAL_DETECTION_DATE), color="green", linestyle=":", label="Official detection (Jan 7, 2010)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Tritium (pCi/L)")
    ax.set_title("Tritium Time Series and ML Anomaly Detection (ISEF)")
    ax.legend()
    ax.grid(True, alpha=0.3)

def draw_table(ax, data):
    ax.clear()
    ax.axis("off")
    if len(data) == 0:
        ax.text(0.5, 0.5, "No data after filters", ha="center", va="center", transform=ax.transAxes)
        return
    tbl = data[["Date", "Mean_pCi_L", "Anomaly", "Z_Anomaly"]].copy()
    tbl["Date_str"] = tbl["Date"].dt.year.astype(str) + "-Q" + tbl["Date"].dt.quarter.astype(str)
    tbl["Anomaly_label"] = tbl["Anomaly"].map({-1: "Anomaly", 1: "Normal"})
    tbl["Z_Anomaly_label"] = tbl["Z_Anomaly"].map({True: "Yes", False: "No"})
    loc_col = data["Location_Name"].astype(str) if "Location_Name" in data.columns else pd.Series(["N/A"] * len(data), index=data.index)
    display = tbl[["Date_str", "Mean_pCi_L", "Anomaly_label", "Z_Anomaly_label"]].copy()
    display.insert(1, "Location", loc_col.values)
    display = display.tail(50)
    display.columns = ["Date", "Mean (pCi/L)", "Location", "ML Anomaly", "Z-Score Anomaly"]
    t = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
        colColours=["#4472C4"] * 5,
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1.0, 1.6)
    ax.set_title("Results: ML vs Z-Score (filtered, last 50 rows)", fontsize=11, pad=10)

# State for filters (updated by widgets)
state = {
    "view": "Graph",
    "year_min": int(df["Year"].min()) if len(df) else YEAR_START,
    "year_max": int(df["Year"].max()) if len(df) else YEAR_END,
    "anomalies_only": False,
    "normal_only": False,
    "locations": list(df["Location_Name"].astype(str).unique()) if "Location_Name" in df.columns else ["N/A"],
    "selected_locations": list(df["Location_Name"].astype(str).unique()) if "Location_Name" in df.columns else ["N/A"],
}

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 4, figure=fig, height_ratios=[0.08, 0.12, 1], hspace=0.35, wspace=0.3)
ax_main = fig.add_subplot(gs[2, :])
ax_radio = fig.add_subplot(gs[0, :2])
ax_slider_min = fig.add_subplot(gs[1, 0])
ax_slider_max = fig.add_subplot(gs[1, 1])
ax_cb = fig.add_subplot(gs[1, 2])
ax_loc = fig.add_subplot(gs[1, 3])

radio = RadioButtons(ax_radio, ("Graph", "Table"), active=0)
year_min_slider = Slider(ax_slider_min, "Year from", YEAR_START, YEAR_END, valinit=state["year_min"], valstep=1)
year_max_slider = Slider(ax_slider_max, "Year to", YEAR_START, YEAR_END, valinit=state["year_max"], valstep=1)
cb_labels = ["Anomalies only", "Normal only"]
cb = CheckButtons(ax_cb, cb_labels, [False, False])
loc_options = state["locations"] if state["locations"] else ["All"]
loc_states = [True] * len(loc_options)
loc_cb = CheckButtons(ax_loc, loc_options, loc_states)

def update(_):
    state["view"] = radio.value_selected
    state["year_min"] = int(year_min_slider.val)
    state["year_max"] = int(year_max_slider.val)
    state["anomalies_only"] = cb.get_status()[0]
    state["normal_only"] = cb.get_status()[1]
    state["selected_locations"] = [loc_options[i] for i, s in enumerate(loc_cb.get_status()) if s]
    data = get_filtered_df(
        state["year_min"], state["year_max"],
        state["anomalies_only"], state["normal_only"],
        state["selected_locations"],
    )
    if state["view"] == "Graph":
        draw_graph(ax_main, data)
    else:
        draw_table(ax_main, data)
    fig.canvas.draw_idle()

radio.on_clicked(update)
year_min_slider.on_changed(update)
year_max_slider.on_changed(update)
cb.on_clicked(update)
loc_cb.on_clicked(update)

# Initial draw
update(None)
plt.savefig("tritium_anomalies_isef.png", dpi=150)
plt.show()

# ==================== RESULTS SUMMARY FOR ISEF ====================
summary_tbl = df[["Date", "Mean_pCi_L", "Anomaly", "Z_Anomaly"]].copy()
summary_tbl["Date_str"] = summary_tbl["Date"].dt.year.astype(str) + "-Q" + summary_tbl["Date"].dt.quarter.astype(str)
summary_tbl["Anomaly_label"] = summary_tbl["Anomaly"].map({-1: "Anomaly", 1: "Normal"})
summary_tbl["Z_Anomaly_label"] = summary_tbl["Z_Anomaly"].map({True: "Yes", False: "No"})
table_display = summary_tbl[["Date_str", "Mean_pCi_L", "Anomaly_label", "Z_Anomaly_label"]].tail(20)
table_display.columns = ["Date", "Mean (pCi/L)", "ML Anomaly", "Z-Score Anomaly"]

print("\n" + "=" * 60)
print("ISEF RESULTS SUMMARY — Automated Tritium Anomaly Detection")
print("=" * 60)
print(f"Official detection date (Vermont Yankee): {OFFICIAL_DETECTION_DATE}")
print(f"First ML-detected anomaly date:           {first_ml_anomaly_date}")
print(f"First Z-score anomaly date (|Z|>3):      {first_z_anomaly_date}")
print(f"ML anomalies count: { (df['Anomaly'] == -1).sum() }  |  Z-score anomalies: { df['Z_Anomaly'].sum() }")
if first_ml_anomaly_date is not None and pd.Timestamp(first_ml_anomaly_date) < pd.Timestamp(OFFICIAL_DETECTION_DATE):
    print("Hypothesis support: ML flagged an anomaly BEFORE official detection date.")
else:
    print("Hypothesis: Compare first anomaly date above to official detection; add more data for stronger test.")
print("=" * 60)
print("\nFull table (last 20 rows):")
print(table_display.to_string(index=False))
