# ============================================================
#  House Price Prediction — Linear Regression
#  Input : house_price_dataset_large.csv  (train/test)
#          house_price_dataset_small.csv  (held-out validation)
#  Libs  : pandas, numpy, scikit-learn, matplotlib, seaborn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

LARGE_CSV = "house_price_dataset_large.csv"
SMALL_CSV = "house_price_dataset_small.csv"

# ─── 1. LOAD DATA ────────────────────────────────────────────
print("=" * 60)
print("  HOUSE PRICE PREDICTION — LINEAR REGRESSION")
print("=" * 60)

df_large = pd.read_csv(LARGE_CSV)
df_small = pd.read_csv(SMALL_CSV)
df_large.columns = df_large.columns.str.strip()
df_small.columns = df_small.columns.str.strip()

FEATURE_COL = "Size (sq ft)"
TARGET_COL  = "Price ($)"

print(f"\n Large dataset : {len(df_large):,} rows  -> train / test split")
print(f"  Small dataset : {len(df_small):,} rows  -> held-out validation")
print(f"  Feature       : '{FEATURE_COL}'")
print(f"  Target        : '{TARGET_COL}'")

# ─── 2. EDA ──────────────────────────────────────────────────
print("\n" + "─" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("─" * 60)
for label, df in [("Large", df_large), ("Small", df_small)]:
    print(f"\n  [{label}]")
    print(f"    Missing values : {df.isnull().sum().sum()}")
    print(f"    Size  : {df[FEATURE_COL].min():,.0f} – {df[FEATURE_COL].max():,.0f} sq ft  "
          f"(mean {df[FEATURE_COL].mean():,.0f})")
    print(f"    Price : ${df[TARGET_COL].min():,.0f} – ${df[TARGET_COL].max():,.0f}  "
          f"(mean ${df[TARGET_COL].mean():,.0f})")

# ─── 3. PREPARE DATA ─────────────────────────────────────────
X_large = df_large[[FEATURE_COL]].values
y_large = df_large[TARGET_COL].values
X_small = df_small[[FEATURE_COL]].values
y_small = df_small[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X_large, y_large, test_size=0.2, random_state=42
)
print(f"\n  Train : {len(X_train):,} | Test : {len(X_test):,} | Validation : {len(X_small):,}")

# ─── 4. SCALE ────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
X_small_sc = scaler.transform(X_small)

# ─── 5. TRAIN ────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)

slope     = model.coef_[0] / scaler.scale_[0]
intercept = model.intercept_ - slope * scaler.mean_[0]
print(f"\n  Regression line : Price = {slope:,.2f} x Size + {intercept:,.2f}")

# ─── 6. PREDICT ──────────────────────────────────────────────
y_pred_train = model.predict(X_train_sc)
y_pred_test  = model.predict(X_test_sc)
y_pred_val   = model.predict(X_small_sc)

# ─── 7. EVALUATE ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("MODEL PERFORMANCE")
print("─" * 60)

def report(label, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{label}]")
    print(f"    MAE  : ${mae:>12,.2f}")
    print(f"    MSE  : ${mse:>12,.2f}")
    print(f"    RMSE : ${rmse:>12,.2f}")
    print(f"    R2   :  {r2:.4f}  ({r2*100:.2f}% variance explained)")
    return mae, mse, rmse, r2

train_m = report("TRAIN",      y_train, y_pred_train)
test_m  = report("TEST",       y_test,  y_pred_test)
val_m   = report("VALIDATION", y_small, y_pred_val)

cv = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="r2")
print(f"\n  [5-Fold CV R2]  {[round(s,4) for s in cv]}")
print(f"    Mean : {cv.mean():.4f} +/- {cv.std():.4f}")

# ─── 8. SAMPLE PREDICTIONS ───────────────────────────────────
print("\n" + "─" * 60)
print("SAMPLE PREDICTIONS — Validation Set (first 8 rows)")
print("─" * 60)
sample = pd.DataFrame({
    FEATURE_COL:             X_small[:8, 0],
    "Actual Price ($)":      y_small[:8],
    "Predicted Price ($)":   y_pred_val[:8].round(2),
    "Error ($)":             (y_pred_val[:8] - y_small[:8]).round(2),
})
print(sample.to_string(index=False))

# ─── 9. VISUALISE ────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
fig = plt.figure(figsize=(18, 13), facecolor="#0d1117")
fig.suptitle("House Price Prediction — Linear Regression",
             fontsize=22, fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)
ACCENT = "#00d4ff"; GOLD = "#ffd700"; PINK = "#ff6b9d"
GREEN = "#39d353"; BG = "#161b22"; TEXT = "#c9d1d9"

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.set_title(title, color=ACCENT, fontsize=11, fontweight="bold", pad=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

fmt_dollar = plt.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k")

# Plot 1: Regression line
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(X_large[:, 0], y_large, alpha=0.35, s=12, color=ACCENT, label="Large dataset", edgecolors="none")
ax1.scatter(X_small[:, 0], y_small, alpha=0.75, s=22, color=PINK,   label="Validation dataset", edgecolors="none", zorder=5)
x_line = np.linspace(X_large[:, 0].min(), X_large[:, 0].max(), 200)
y_line = slope * x_line + intercept
ax1.plot(x_line, y_line, color=GOLD, lw=2.5, label=f"y = {slope:,.0f}x + {intercept:,.0f}")
ax1.set_xlabel("Size (sq ft)"); ax1.set_ylabel("Price ($)")
ax1.yaxis.set_major_formatter(fmt_dollar)
ax1.legend(fontsize=9, facecolor=BG, labelcolor=TEXT, framealpha=0.8)
style_ax(ax1, "Regression Line — All Data")

# Plot 2: Metrics summary card
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(BG); ax2.axis("off")
ax2.set_title("Performance Summary", color=ACCENT, fontsize=11, fontweight="bold", pad=9)
for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")
rows = [
    ("",       "R²",               "RMSE",               "MAE"),
    ("Train",  f"{train_m[3]:.4f}", f"${train_m[2]:,.0f}", f"${train_m[0]:,.0f}"),
    ("Test",   f"{test_m[3]:.4f}",  f"${test_m[2]:,.0f}",  f"${test_m[0]:,.0f}"),
    ("Val",    f"{val_m[3]:.4f}",   f"${val_m[2]:,.0f}",   f"${val_m[0]:,.0f}"),
]
col_colors = [GOLD, ACCENT, PINK, GREEN]
for r, row in enumerate(rows):
    for c, cell in enumerate(row):
        color = col_colors[c] if r == 0 else (TEXT if c == 0 else "white")
        fs = 9 if r == 0 else 10
        fw = "bold" if r == 0 or c == 0 else "normal"
        ax2.text(0.05 + c*0.24, 0.88 - r*0.20, cell,
                 transform=ax2.transAxes, color=color, fontsize=fs, fontweight=fw)
ax2.text(0.5, 0.08, f"CV R²: {cv.mean():.4f} ± {cv.std():.4f}",
         transform=ax2.transAxes, color=GOLD, fontsize=9, ha="center")

# Plot 3: Actual vs Predicted (test)
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y_test, y_pred_test, alpha=0.45, s=14, color=ACCENT, edgecolors="none")
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
ax3.plot(lims, lims, color=GOLD, lw=2, linestyle="--")
ax3.set_xlabel("Actual Price ($)"); ax3.set_ylabel("Predicted Price ($)")
ax3.xaxis.set_major_formatter(fmt_dollar); ax3.yaxis.set_major_formatter(fmt_dollar)
style_ax(ax3, "Actual vs Predicted (Test)")

# Plot 4: Actual vs Predicted (validation)
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(y_small, y_pred_val, alpha=0.75, s=24, color=PINK, edgecolors="none")
lims2 = [min(y_small.min(), y_pred_val.min()), max(y_small.max(), y_pred_val.max())]
ax4.plot(lims2, lims2, color=GOLD, lw=2, linestyle="--")
ax4.set_xlabel("Actual Price ($)"); ax4.set_ylabel("Predicted Price ($)")
ax4.xaxis.set_major_formatter(fmt_dollar); ax4.yaxis.set_major_formatter(fmt_dollar)
style_ax(ax4, "Actual vs Predicted (Validation)")

# Plot 5: Residuals
ax5 = fig.add_subplot(gs[1, 2])
res_test = y_test - y_pred_test
res_val  = y_small - y_pred_val
ax5.hist(res_test, bins=30, color=ACCENT, alpha=0.6, label="Test",       edgecolor="none")
ax5.hist(res_val,  bins=15, color=PINK,   alpha=0.85, label="Validation", edgecolor="none")
ax5.axvline(0, color=GOLD, lw=2, linestyle="--")
ax5.set_xlabel("Residual ($)"); ax5.set_ylabel("Count")
ax5.xaxis.set_major_formatter(fmt_dollar)
ax5.legend(fontsize=8, facecolor=BG, labelcolor=TEXT)
style_ax(ax5, "Residuals Distribution")

# Plot 6: Price distribution
ax6 = fig.add_subplot(gs[2, 0])
ax6.hist(y_large, bins=35, color=GOLD, edgecolor="none", alpha=0.85)
ax6.axvline(y_large.mean(), color=PINK, lw=2, linestyle="--",
            label=f"Mean: ${y_large.mean():,.0f}")
ax6.set_xlabel("Price ($)"); ax6.set_ylabel("Count")
ax6.xaxis.set_major_formatter(fmt_dollar)
ax6.legend(fontsize=8, facecolor=BG, labelcolor=TEXT)
style_ax(ax6, "Price Distribution (Large Dataset)")

# Plot 7: Size distribution
ax7 = fig.add_subplot(gs[2, 1])
ax7.hist(X_large[:, 0], bins=35, color="#a78bfa", edgecolor="none", alpha=0.85)
ax7.axvline(X_large[:, 0].mean(), color=GOLD, lw=2, linestyle="--",
            label=f"Mean: {X_large[:,0].mean():,.0f} sq ft")
ax7.set_xlabel("Size (sq ft)"); ax7.set_ylabel("Count")
ax7.legend(fontsize=8, facecolor=BG, labelcolor=TEXT)
style_ax(ax7, "Size Distribution (Large Dataset)")

# Plot 8: Cross-validation
ax8 = fig.add_subplot(gs[2, 2])
bars = ax8.bar(range(1, 6), cv, color=ACCENT, edgecolor="none", alpha=0.9, width=0.6)
ax8.axhline(cv.mean(), color=GOLD, lw=2, linestyle="--",
            label=f"Mean R² = {cv.mean():.4f}")
for bar, val in zip(bars, cv):
    ax8.text(bar.get_x() + bar.get_width()/2, val + 0.002,
             f"{val:.3f}", ha="center", va="bottom", color=TEXT, fontsize=8)
ax8.set_xlabel("Fold"); ax8.set_ylabel("R² Score")
ax8.set_xticks(range(1, 6))
ax8.legend(fontsize=8, facecolor=BG, labelcolor=TEXT)
style_ax(ax8, "5-Fold Cross Validation")

plt.savefig("house_price_prediction.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()

# ─── 10. SAVE PREDICTIONS CSV ────────────────────────────────
out = df_small.copy()
out["Predicted Price ($)"] = y_pred_val.round(2)
out["Error ($)"]           = (y_pred_val - y_small).round(2)
out["Error (%)"]           = ((y_pred_val - y_small) / y_small * 100).round(2)
out.to_csv("validation_predictions.csv", index=False)

print("\nDone! Files saved:")
print("  house_price_prediction.png")
print("  validation_predictions.csv")
print("=" * 60)