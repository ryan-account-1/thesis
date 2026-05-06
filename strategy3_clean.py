import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import OLS as OLS_reg
from statsmodels.tools import add_constant
from matplotlib.lines import Line2D

OPTION_CSV = r"C:\Users\ryans\Downloads\option prices 2012-2025 full.csv"
SPX_CSV    = r"C:\Users\ryans\Downloads\spx_close 2000-2025.csv"
VIX_CSV    = r"C:\Users\ryans\Downloads\vix close 2000-2025.csv"

LOSS_THRESHOLD_Q    = 0.10
SKEW_THRESHOLD_Z    = -0.75
TARGET_DTE          = 30
DTE_MIN             = 20
DTE_MAX             = 40
TARGET_DELTA        = 0.25
DELTA_BAND          = 0.15
NOTIONAL_MULTIPLIER = 100
APPLY_TC            = True
MAX_LOSS_DOLLARS    = -5_000
MAX_HOLD_DAYS       = 5
STARTING_CAPITAL    = 1_000_000
MARGIN_PER_CONTRACT = 20_000

# loading + cleaning
print("Loading data")

df_raw = pd.read_csv(OPTION_CSV, parse_dates=["date", "exdate"])
df_raw["exercise_style"] = df_raw["exercise_style"].astype(str).str.strip().str[0]
df_raw["issuer"]         = df_raw["issuer"].astype(str).str.strip()
df_raw = df_raw[
    (df_raw["exercise_style"] == "E") &
    (df_raw["issuer"].str.contains("S&P 500", case=False, na=False))
]
df_raw["strike_price"] = df_raw["strike_price"] / 1000.0
df_raw["DTE"]          = (df_raw["exdate"] - df_raw["date"]).dt.days
df_raw["Year"]         = df_raw["date"].dt.year
df_raw = df_raw[(df_raw["best_bid"] > 0) & (df_raw["best_offer"] > 0)].copy()
df_raw = df_raw[(df_raw["open_interest"] >= 50) & (df_raw["volume"] >= 3)].copy()

df_raw["mid"]         = (df_raw["best_bid"] + df_raw["best_offer"]) / 2.0
df_raw["spread_pct"]  = (df_raw["best_offer"] - df_raw["best_bid"]) / df_raw["mid"]
df_raw["half_spread"] = (df_raw["best_offer"] - df_raw["best_bid"]) / 2.0

df_broad = df_raw.copy()

cond_dte = (df_raw["DTE"] >= DTE_MIN) & (df_raw["DTE"] <= DTE_MAX)
cond_131415 = (
    (df_raw["Year"] >= 2013) & (df_raw["Year"] <= 2015) &
    (df_raw["open_interest"] >= 50) & (df_raw["volume"] >= 3) &
    (df_raw["spread_pct"] <= 0.08)
)
cond_16up = (
    (df_raw["Year"] >= 2016) &
    (df_raw["open_interest"] >= 100) & (df_raw["volume"] >= 10) &
    (df_raw["spread_pct"] <= 0.05)
)
df_entry = df_raw[cond_dte & (cond_131415 | cond_16up)].copy()

keep_cols = ["date","exdate","cp_flag","strike_price","impl_volatility",
             "delta","DTE","mid","vega","spread_pct","half_spread"]
df_broad = df_broad[keep_cols].copy()
df_entry = df_entry[keep_cols].copy()
del df_raw

# risk reversal
print("Building RR series")

def iv_at_delta_flexible(sub, target, band=DELTA_BAND):
    sub = sub.dropna(subset=["delta","impl_volatility"]).sort_values("delta")
    below = sub[sub["delta"] <= target]
    above = sub[sub["delta"] >= target]
    if (not below.empty) and (not above.empty):
        lower, upper = below.iloc[-1], above.iloc[0]
        d1,iv1 = float(lower.delta), float(lower.impl_volatility)
        d2,iv2 = float(upper.delta), float(upper.impl_volatility)
        if d1 == d2: return iv1
        return iv1 + (target - d1) * (iv2 - iv1) / (d2 - d1)
    if sub.empty: return np.nan
    nearest = sub.iloc[(sub["delta"] - target).abs().argmin()]
    return float(nearest["impl_volatility"]) if abs(nearest["delta"] - target) <= band else np.nan

def iv25_time_interp(day):
    exs = (day.assign(diff=(day.DTE - TARGET_DTE).abs())
             .sort_values("diff")["exdate"].drop_duplicates().tolist())
    pts = []
    for ex in exs:
        sub = day[day["exdate"] == ex]
        c = iv_at_delta_flexible(sub[sub["cp_flag"]=="C"], +TARGET_DELTA)
        p = iv_at_delta_flexible(sub[sub["cp_flag"]=="P"], -TARGET_DELTA)
        dte_val = int(sub["DTE"].iloc[0])
        if np.isfinite(c) and np.isfinite(p):
            pts.append((dte_val, c, p, ex))
        if len(pts) >= 2: break
    if not pts: return None
    if len(pts) == 1:
        d,c,p,ex = pts[0]
        return ex, d, c, p, c-p
    (d1,c1,p1,ex1),(d2,c2,p2,ex2) = sorted(pts, key=lambda x: x[0])
    if d1 == d2: return ex1, d1, c1, p1, c1-p1
    w = (TARGET_DTE - d1) / (d2 - d1)
    c30 = c1 + w*(c2-c1)
    p30 = p1 + w*(p2-p1)
    best_ex = ex1 if abs(d1-TARGET_DTE) <= abs(d2-TARGET_DTE) else ex2
    return best_ex, TARGET_DTE, c30, p30, c30-p30

rows = [
    ([d] + list(iv25_time_interp(day) or [None, np.nan, np.nan, np.nan, np.nan]))
    for d, day in df_entry.groupby("date")
]
rr_df = pd.DataFrame(rows, columns=["date","exdate","DTE","IV25C","IV25P","RR"])
rr_df["RR"] = rr_df["RR"].interpolate(limit=2)

# macro + rolling OLS -> AbRR
print("Computing AbRR")

spx = pd.read_csv(SPX_CSV, parse_dates=["date"])[["date","close"]].drop_duplicates("date")
vix = pd.read_csv(VIX_CSV, parse_dates=["date"])[["date","vix"]].drop_duplicates("date")

rr_df = (rr_df.merge(spx, on="date", how="left")
               .merge(vix, on="date", how="left")
               .sort_values("date").reset_index(drop=True))
rr_df["SPX_ret"] = np.log(rr_df["close"] / rr_df["close"].shift(1))
rr_df["Month"]   = rr_df["date"].dt.month

ROLL_WINDOW = 252
rr_df["RR_hat"] = np.nan
for i in range(ROLL_WINDOW, len(rr_df)):
    win = rr_df.iloc[i-ROLL_WINDOW:i].dropna(subset=["RR","vix","Month"])
    if len(win) < 40: continue
    X = pd.concat([
        pd.DataFrame({"const":1.0, "vix": win["vix"].astype(float)}),
        pd.get_dummies(win["Month"].astype(int), prefix="m", drop_first=True).astype(float)
    ], axis=1)
    model = sm.OLS(win["RR"].astype(float), X).fit()
    row   = rr_df.iloc[[i]]
    x_new = pd.concat([
        pd.DataFrame({"const":1.0, "vix": row["vix"].astype(float)}),
        pd.get_dummies(row["Month"].astype(int), prefix="m", drop_first=True)
          .astype(float).reindex(columns=X.columns[2:], fill_value=0.0)
    ], axis=1)
    rr_df.loc[i,"RR_hat"] = float(model.predict(x_new).iloc[0])

rr_df["AbRR"] = rr_df["RR"] - rr_df["RR_hat"]

# events + tests
print("Events & tests")

analysis_df = rr_df[rr_df["date"] >= pd.Timestamp("2014-01-01")].copy()
artifact_dates = [pd.Timestamp("2015-08-24"), pd.Timestamp("2018-02-05")]
analysis_df = analysis_df[~analysis_df["date"].isin(artifact_dates)].reset_index(drop=True)

analysis_df["ret_pct_rank"] = analysis_df["SPX_ret"].rolling(126).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

mu_abrr  = analysis_df["AbRR"].mean()
sig_abrr = analysis_df["AbRR"].std()
analysis_df["AbRR_z"] = (analysis_df["AbRR"] - mu_abrr) / sig_abrr

loss_cond = analysis_df["ret_pct_rank"] <= LOSS_THRESHOLD_Q
skew_cond = analysis_df["AbRR_z"]       <= SKEW_THRESHOLD_Z
analysis_df["Event_t"]    = loss_cond & skew_cond
analysis_df["Event_lag1"] = analysis_df["Event_t"].shift(1).fillna(False)
analysis_df["AbRR_median"] = analysis_df["AbRR"].rolling(252, min_periods=63).median()
analysis_df["RR_median"] = analysis_df["RR"].rolling(252, min_periods=63).median()
analysis_df["fwd_ret_10d"] = analysis_df["close"].pct_change(10).shift(-10)
analysis_df["fwd_RR_10d"]  = analysis_df["RR"].shift(-10) - analysis_df["RR"]

sig_spx   = analysis_df[analysis_df["Event_lag1"]==True]["fwd_ret_10d"].dropna()
nosig_spx = analysis_df[analysis_df["Event_lag1"]==False]["fwd_ret_10d"].dropna()
t1, p1 = stats.ttest_ind(sig_spx, nosig_spx, equal_var=False)

sig_rr   = analysis_df[analysis_df["Event_lag1"]==True]["fwd_RR_10d"].dropna()
nosig_rr = analysis_df[analysis_df["Event_lag1"]==False]["fwd_RR_10d"].dropna()
t2, p2 = stats.ttest_ind(sig_rr, nosig_rr, equal_var=False)

n_events = analysis_df["Event_lag1"].sum()
print(f"\n  Events: {n_events}")
print(f"  Test 1 (SPX fwd): t={t1:.3f} p={p1:.4f}")
print(f"  Test 2 (RR reversion): t={t2:.3f} p={p2:.4f}")

# corrections for stats
signal_dates = analysis_df[analysis_df["Event_lag1"] == True]["date"].sort_values().reset_index(drop=True)

gaps = []
overlap_pairs = []
for i in range(1, len(signal_dates)):
    d1 = signal_dates.iloc[i-1]
    d2 = signal_dates.iloc[i]
    idx1 = analysis_df[analysis_df["date"] == d1].index[0]
    idx2 = analysis_df[analysis_df["date"] == d2].index[0]
    gap = idx2 - idx1
    gaps.append(gap)
    if gap < 10:
        overlap_pairs.append((d1.date(), d2.date(), gap))

overlapping_dates = set()
for d1, d2, gap in overlap_pairs:
    overlapping_dates.add(d1)
    overlapping_dates.add(d2)

print(f"\n  Overlap: {len(overlap_pairs)}/{len(gaps)} pairs < 10d")
print(f"  Dates in overlap: {len(overlapping_dates)}/{len(signal_dates)}")

test_df = analysis_df[["date", "Event_lag1", "fwd_RR_10d", "fwd_ret_10d"]].dropna().copy()
test_df["signal"] = test_df["Event_lag1"].astype(float)

X2 = add_constant(test_df["signal"])
y2 = test_df["fwd_RR_10d"]
model2_ols = OLS_reg(y2, X2).fit()
model2_nw  = OLS_reg(y2, X2).fit(cov_type='HAC', cov_kwds={'maxlags': 9})

print(f"\n  Newey-West Test 2: t={model2_nw.tvalues['signal']:.3f} p={model2_nw.pvalues['signal']:.6f}")

X1 = add_constant(test_df["signal"])
y1 = test_df["fwd_ret_10d"]
model1_nw = OLS_reg(y1, X1).fit(cov_type='HAC', cov_kwds={'maxlags': 9})
print(f"  Newey-West Test 1: t={model1_nw.tvalues['signal']:.3f} p={model1_nw.pvalues['signal']:.6f}")

non_overlap_dates = []
last_idx = -999
for d in signal_dates:
    idx = analysis_df[analysis_df["date"] == d].index[0]
    if idx - last_idx >= 10:
        non_overlap_dates.append(d)
        last_idx = idx

nonoverlap_set = set(non_overlap_dates)
sig_rr_no = analysis_df[analysis_df["date"].isin(nonoverlap_set)]["fwd_RR_10d"].dropna()
nosig_rr_no = analysis_df[~analysis_df["date"].isin(nonoverlap_set)]["fwd_RR_10d"].dropna()
t_no, p_no = stats.ttest_ind(sig_rr_no, nosig_rr_no, equal_var=False)

print(f"  Non-overlap Test 2: n={len(sig_rr_no)} t={t_no:.3f} p={p_no:.6f}")

# sensitivity

print("sensitivity analyis")
print(f"  {'Percentile':<12} {'Z-Score':<10} {'Events':<8}")
print(f"  {'-'*12} {'-'*10} {'-'*8}")
for q in [0.05, 0.075, 0.10, 0.125, 0.15]:
    for z in [-1.0, -0.75, -0.5]:
        loss_c = analysis_df["ret_pct_rank"] <= q
        skew_c = analysis_df["AbRR_z"] <= z
        n_ev = (loss_c & skew_c).sum()
        marker = " <" if q == 0.10 and z == -0.75 else ""
        print(f"  {q:<12.3f} {z:<10.2f} {n_ev:<8}{marker}")

# index data
print("Indexing")
broad_by_date = {d: g for d,g in df_broad.groupby("date")}
entry_by_date = {d: g for d,g in df_entry.groupby("date")}

# backtesting
print("Simulation")

trade_log        = []
skipped_no_data  = 0
open_trade_until = pd.Timestamp("1900-01-01")
entry_dates      = analysis_df[analysis_df["Event_lag1"]==True]["date"].tolist()

for entry_date in entry_dates:
    if entry_date <= open_trade_until:
        continue

    entry_row = analysis_df[analysis_df["date"] == entry_date]
    if entry_row.empty: continue

    entry_idx  = entry_row.index[0]
    spx_entry  = entry_row["close"].values[0]
    abrr_entry = entry_row["AbRR"].values[0]
    rr_entry   = entry_row["RR"].values[0]
    vix_entry  = entry_row["vix"].values[0]

    day_opts = entry_by_date.get(entry_date)
    if day_opts is None or day_opts.empty:
        skipped_no_data += 1; continue
    day_opts = day_opts.copy()

    day_opts["dte_diff"] = (day_opts["DTE"] - TARGET_DTE).abs()
    best_exdate = day_opts.sort_values("dte_diff")["exdate"].iloc[0]
    chain = day_opts[day_opts["exdate"] == best_exdate].copy()

    calls = chain[chain["cp_flag"] == "C"].copy()
    puts  = chain[chain["cp_flag"] == "P"].copy()
    if calls.empty or puts.empty:
        skipped_no_data += 1; continue

    calls["delta_diff"] = (calls["delta"] - TARGET_DELTA).abs()
    best_call_row = calls.sort_values("delta_diff").iloc[0]
    if abs(best_call_row["delta"] - TARGET_DELTA) > DELTA_BAND:
        skipped_no_data += 1; continue

    puts["delta_diff"] = (puts["delta"] - (-TARGET_DELTA)).abs()
    best_put_row = puts.sort_values("delta_diff").iloc[0]
    if abs(best_put_row["delta"] - (-TARGET_DELTA)) > DELTA_BAND:
        skipped_no_data += 1; continue

    vega_call = best_call_row.get("vega", np.nan)
    vega_put  = best_put_row.get("vega",  np.nan)
    if pd.isna(vega_call) or pd.isna(vega_put) or vega_put == 0:
        skipped_no_data += 1; continue

    qty_call = 1.0
    qty_put  = qty_call * (vega_call / vega_put)

    strike_call = best_call_row["strike_price"]
    strike_put  = best_put_row["strike_price"]
    expiry_dte  = int(best_call_row["DTE"])

    iv_put_entry  = best_put_row["impl_volatility"]
    iv_call_entry = best_call_row["impl_volatility"]
    iv_spread_entry = iv_put_entry - iv_call_entry

    gross_premium = (qty_put * best_put_row["mid"]) - (qty_call * best_call_row["mid"])
    initial_net_prem = gross_premium

    tc_entry = 0.0
    if APPLY_TC:
        tc_entry = (qty_call * best_call_row["half_spread"] +
                    qty_put  * best_put_row["half_spread"]) * NOTIONAL_MULTIPLIER

    cash = gross_premium * NOTIONAL_MULTIPLIER - tc_entry

    net_delta   = (qty_call * best_call_row["delta"]) - (qty_put * best_put_row["delta"])
    shares_held = -net_delta
    cash -= shares_held * spx_entry * NOTIONAL_MULTIPLIER

    days_held   = 0
    exit_reason = f"Time Stop ({MAX_HOLD_DAYS}D)"
    curr_idx    = entry_idx

    for i in range(1, MAX_HOLD_DAYS + 1):
        curr_idx = entry_idx + i
        if curr_idx >= len(analysis_df): break

        curr_row   = analysis_df.iloc[curr_idx]
        curr_date  = curr_row["date"]
        curr_spx   = curr_row["close"]
        curr_abrr  = curr_row["AbRR"]
        curr_rr    = curr_row["RR"]
        days_held  = i

        rr_median   = curr_row["RR_median"]   if pd.notna(curr_row["RR_median"])   else 0
        abrr_median = curr_row["AbRR_median"] if pd.notna(curr_row["AbRR_median"]) else 0

        if curr_rr >= rr_median:
            exit_reason = "RR Reverted to Median"; break

        if curr_abrr >= abrr_median:
            exit_reason = "AbRR Reverted"; break

        today_broad = broad_by_date.get(curr_date, pd.DataFrame())
        if not today_broad.empty:
            today_chain = today_broad[(today_broad["exdate"] == best_exdate) & (today_broad["spread_pct"] <= 0.15)]
            _c = today_chain[(today_chain["cp_flag"]=="C") & (today_chain["strike_price"]==strike_call)]
            _p = today_chain[(today_chain["cp_flag"]=="P") & (today_chain["strike_price"]==strike_put)]
            if not (_c.empty or _p.empty):
                mtm = ((qty_call * _c["mid"].values[0]) - (qty_put * _p["mid"].values[0])) * NOTIONAL_MULTIPLIER
                running_pnl = cash + mtm + shares_held * curr_spx * NOTIONAL_MULTIPLIER
                if running_pnl <= MAX_LOSS_DOLLARS:
                    exit_reason = f"Stop Loss (${abs(MAX_LOSS_DOLLARS):,.0f})"; break

        today_broad = broad_by_date.get(curr_date, pd.DataFrame())
        if today_broad.empty: continue
        today_chain = today_broad[(today_broad["exdate"] == best_exdate) & (today_broad["spread_pct"] <= 0.15)]
        _c = today_chain[(today_chain["cp_flag"]=="C") & (today_chain["strike_price"]==strike_call)]
        _p = today_chain[(today_chain["cp_flag"]=="P") & (today_chain["strike_price"]==strike_put)]
        if _c.empty or _p.empty: continue

        new_delta   = (qty_call * _c["delta"].values[0]) - (qty_put * _p["delta"].values[0])
        tgt_shares  = -new_delta
        cash       -= (tgt_shares - shares_held) * curr_spx * NOTIONAL_MULTIPLIER
        shares_held = tgt_shares

    exit_row  = analysis_df.iloc[curr_idx] if curr_idx < len(analysis_df) else analysis_df.iloc[-1]
    exit_date = exit_row["date"]
    exit_spx  = exit_row["close"]
    rr_exit   = exit_row["RR"]
    iv_spread_exit = np.nan

    exit_broad = broad_by_date.get(exit_date, pd.DataFrame())
    if exit_broad.empty: continue
    exit_chain = exit_broad[exit_broad["exdate"] == best_exdate]

    ex_c = exit_chain[(exit_chain["cp_flag"]=="C") & (exit_chain["strike_price"]==strike_call)]
    ex_p = exit_chain[(exit_chain["cp_flag"]=="P") & (exit_chain["strike_price"]==strike_put)]
    if ex_c.empty or ex_p.empty: continue

    price_call = ex_c["mid"].values[0]
    price_put  = ex_p["mid"].values[0]

    cash += ((qty_call * price_call) - (qty_put * price_put)) * NOTIONAL_MULTIPLIER

    tc_exit = 0.0
    if APPLY_TC:
        tc_exit = (qty_call * ex_c["half_spread"].values[0] +
                   qty_put  * ex_p["half_spread"].values[0]) * NOTIONAL_MULTIPLIER
        cash -= tc_exit

    cash += shares_held * exit_spx * NOTIONAL_MULTIPLIER

    if "impl_volatility" in ex_c.columns and "impl_volatility" in ex_p.columns:
        iv_spread_exit = ex_p["impl_volatility"].values[0] - ex_c["impl_volatility"].values[0]

    trade_log.append({
        "Entry_Date":       entry_date.date(),
        "Exit_Date":        exit_date.date(),
        "Days_Held":        days_held,
        "Exit_Reason":      exit_reason,
        "VIX_Entry":        round(vix_entry, 2),
        "RR_Entry":         round(rr_entry, 4),
        "RR_Exit":          round(rr_exit, 4) if pd.notna(rr_exit) else np.nan,
        "RR_Change":        round(rr_exit - rr_entry, 4) if pd.notna(rr_exit) else np.nan,
        "IV_Spread_Entry":  round(iv_spread_entry, 4),
        "TC_Total_$":       round(tc_entry + tc_exit, 2),
        "Net_PnL_$":        round(cash, 2),
    })
    open_trade_until = exit_date

print(f"\n  Trades: {len(trade_log)}/{n_events}")
print(f"  Skipped (data): {skipped_no_data}")
print(f"  Skipped (overlap): {n_events - len(trade_log) - skipped_no_data}")

# results
results_df = pd.DataFrame(trade_log)

if not results_df.empty:
    results_df["Entry_Date"] = pd.to_datetime(results_df["Entry_Date"])
    results_df["Exit_Date"]  = pd.to_datetime(results_df["Exit_Date"])

    n          = len(results_df)
    wins       = results_df[results_df["Net_PnL_$"] > 0]
    losses     = results_df[results_df["Net_PnL_$"] <= 0]
    win_rate   = len(wins) / n
    avg_win    = wins["Net_PnL_$"].mean()   if not wins.empty   else 0
    avg_loss   = losses["Net_PnL_$"].mean() if not losses.empty else 0
    rr_ratio   = abs(avg_win / avg_loss)    if avg_loss != 0    else np.nan
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    total_pnl  = results_df["Net_PnL_$"].sum()

    timeline = analysis_df[["date","close"]].copy()
    timeline["date"] = pd.to_datetime(timeline["date"])
    timeline.set_index("date", inplace=True)
    timeline["Daily_PnL"] = 0.0
    for _, row in results_df.iterrows():
        if row["Exit_Date"] in timeline.index:
            timeline.loc[row["Exit_Date"], "Daily_PnL"] += row["Net_PnL_$"]

    timeline["Strat_Equity"] = STARTING_CAPITAL + timeline["Daily_PnL"].cumsum()
    timeline["Strat_Ret"]    = timeline["Strat_Equity"].pct_change().fillna(0)
    spx_start = timeline["close"].iloc[0]
    timeline["SPX_Equity"]   = (timeline["close"] / spx_start) * STARTING_CAPITAL
    timeline["SPX_Ret"]      = timeline["SPX_Equity"].pct_change().fillna(0)

    cov      = timeline["Strat_Ret"].cov(timeline["SPX_Ret"])
    spx_var  = timeline["SPX_Ret"].var()
    beta     = cov / spx_var if spx_var != 0 else 0
    timeline["Peak"]     = timeline["Strat_Equity"].cummax()
    timeline["Drawdown"] = (timeline["Strat_Equity"] - timeline["Peak"]) / timeline["Peak"]
    max_dd   = timeline["Drawdown"].min() * 100

    rr_winners = results_df[results_df["Net_PnL_$"] > 0]["RR_Change"].dropna()
    rr_losers  = results_df[results_df["Net_PnL_$"] <= 0]["RR_Change"].dropna()
    iv_winners = results_df[results_df["Net_PnL_$"] > 0]["IV_Spread_Entry"].dropna()
    iv_losers  = results_df[results_df["Net_PnL_$"] <= 0]["IV_Spread_Entry"].dropna()

    backtest_years  = (results_df["Exit_Date"].max() - results_df["Entry_Date"].min()).days / 365.25
    trades_per_year = n / backtest_years
    ann_pnl_1c      = total_pnl / backtest_years
    ann_ret_margin  = ann_pnl_1c / MARGIN_PER_CONTRACT
    cagr_margin     = (1 + total_pnl / MARGIN_PER_CONTRACT) ** (1 / backtest_years) - 1

    trade_ret_pct  = results_df["Net_PnL_$"] / MARGIN_PER_CONTRACT
    trade_sharpe   = (trade_ret_pct.mean() / trade_ret_pct.std()) * np.sqrt(trades_per_year) if trade_ret_pct.std() != 0 else 0

    gross_profit  = wins["Net_PnL_$"].sum()        if not wins.empty   else 0
    gross_loss    = abs(losses["Net_PnL_$"].sum())  if not losses.empty else 1
    profit_factor = gross_profit / gross_loss

    avg_hold    = results_df["Days_Held"].mean()
    median_hold = results_df["Days_Held"].median()

    pnl_signs = (results_df["Net_PnL_$"] > 0).astype(int).tolist()
    max_consec_loss = 0
    curr_consec     = 0
    for s in pnl_signs:
        if s == 0: curr_consec += 1; max_consec_loss = max(max_consec_loss, curr_consec)
        else: curr_consec = 0

    best_trade  = results_df.loc[results_df["Net_PnL_$"].idxmax()]
    worst_trade = results_df.loc[results_df["Net_PnL_$"].idxmin()]
    rr_compressed = (results_df["RR_Change"] > 0).mean() * 100

    vix_bins   = [0, 15, 20, 25, 100]
    vix_labels = ["<15 (Low)", "15-20 (Normal)", "20-25 (Elevated)", ">25 (High Fear)"]
    results_df["VIX_Regime"] = pd.cut(results_df["VIX_Entry"], bins=vix_bins, labels=vix_labels)
    vix_breakdown = results_df.groupby("VIX_Regime", observed=True).agg(
        Trades=("Net_PnL_$", "count"), Win_Rate=("Net_PnL_$", lambda x: (x > 0).mean() * 100),
        Avg_PnL=("Net_PnL_$", "mean"), Total_PnL=("Net_PnL_$", "sum"),
        Avg_IV_Sprd=("IV_Spread_Entry", "mean")).round(2)

    # output
    print(f"\n  Win Rate: {win_rate*100:.1f}%")
    print(f"  Avg Win: ${avg_win:,.0f} | Avg Loss: ${avg_loss:,.0f}")
    print(f"  Risk-Reward: {rr_ratio:.2f}x | Profit Factor: {profit_factor:.2f}x")
    print(f"  Expectancy: ${expectancy:,.0f} | Total P&L: ${total_pnl:,.0f}")
    print(f"  Avg Hold: {avg_hold:.1f}d | Max Consec Loss: {max_consec_loss}")
    print(f"  Best: ${best_trade['Net_PnL_$']:,.2f} ({best_trade['Entry_Date'].date()}) | Worst: ${worst_trade['Net_PnL_$']:,.2f} ({worst_trade['Entry_Date'].date()})")
    print(f"  Beta: {beta:.4f} | Max DD: {max_dd:.2f}%")
    print(f"  CAGR: {cagr_margin*100:.1f}% | Ann Return: {ann_ret_margin*100:.1f}% | Trade Sharpe: {trade_sharpe:.2f}")
    print(f"  Trade Skewness: {trade_ret_pct.skew():.2f} | Trade Kurtosis: {trade_ret_pct.kurtosis():.2f}")
    print(f"  IV Spread Entry: {results_df['IV_Spread_Entry'].mean():+.4f} | RR Compressed: {rr_compressed:.1f}%")
    print(f"  RR Change Winners: {rr_winners.mean():+.5f} | Losers: {rr_losers.mean():+.5f}")
    print(f"  IV Spread Winners: {iv_winners.mean():+.4f} | Losers: {iv_losers.mean():+.4f}")

    print(f"\n  VIX Regime Breakdown:")
    for regime, row in vix_breakdown.iterrows():
        print(f"    {str(regime):<20} n={int(row['Trades'])} WR={row['Win_Rate']:.1f}% Avg=${row['Avg_PnL']:,.0f} Tot=${row['Total_PnL']:,.0f} Sprd={row['Avg_IV_Sprd']:.4f}")

    print(f"\n  Trade Log:")
    pd.set_option("display.max_rows", 200); pd.set_option("display.width", 200)
    print(results_df[["Entry_Date","Exit_Date","Days_Held","Exit_Reason","VIX_Entry","RR_Entry","RR_Exit","RR_Change","IV_Spread_Entry","TC_Total_$","Net_PnL_$"]].to_string(index=False))
    print(f"\n  Exit Reasons:")
    print(results_df["Exit_Reason"].value_counts().to_string())

# figures

# Event Study
PRE_DAYS = 5; POST_DAYS = 10
signal_idx = analysis_df[analysis_df["Event_t"] == True].index.tolist()
rr_paths = []
for idx in signal_idx:
    start = idx - PRE_DAYS; end = idx + POST_DAYS
    if start < 0 or end >= len(analysis_df): continue
    window = analysis_df.loc[start:end, "RR"].values
    if len(window) == PRE_DAYS + POST_DAYS + 1 and not np.any(np.isnan(window)):
        rr_paths.append(window)
rr_paths = np.array(rr_paths)
mean_rr = rr_paths.mean(axis=0); se_rr = rr_paths.std(axis=0) / np.sqrt(rr_paths.shape[0])
days = np.arange(-PRE_DAYS, POST_DAYS + 1)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.fill_between(days, mean_rr - 1.96*se_rr, mean_rr + 1.96*se_rr, alpha=0.2, color="#4A90D9", label="95% CI")
ax.plot(days, mean_rr, color="#2C5F8A", linewidth=2.0, marker="o", markersize=4, label="Mean RR")
ax.axvline(x=0, color="#D04040", linewidth=1.0, linestyle="--", alpha=0.7, label="Signal (t=0)")
ax.axvline(x=1, color="#27AE60", linewidth=1.0, linestyle=":", alpha=0.7, label="Entry (t+1)")
ax.set_xlabel("Days relative to signal"); ax.set_ylabel("RR"); ax.set_xticks(days)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9, loc="lower right")
plt.tight_layout(); plt.savefig("figure2_event_study_rr.png", dpi=300, bbox_inches="tight"); plt.show()

# Signal Timeline
fig, ax = plt.subplots(figsize=(12, 4.5))
sig_dates = analysis_df[analysis_df["Event_lag1"] == True]["date"].sort_values()
spx_ts = analysis_df[["date", "close"]].dropna().sort_values("date")
ax.plot(spx_ts["date"], spx_ts["close"], color="#333333", linewidth=0.8)
for d in sig_dates:
    row = spx_ts[spx_ts["date"] == d]
    if not row.empty: ax.scatter(d, row["close"].iloc[0], color="#D04040", s=35, edgecolors="#8B0000", linewidths=0.5, marker="v", zorder=3)
ymin, ymax = ax.get_ylim(); ax.set_ylim(ymin, ymax * 1.10)
for s, e, l in [("2018-10-01","2018-12-31","Q4 2018"), ("2019-05-01","2019-10-15","Trade War"), ("2020-02-19","2020-04-30","COVID-19"), ("2023-08-01","2023-11-15","Rate Hikes")]:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.12, color="#4A90D9")
    ax.text(pd.Timestamp(s)+(pd.Timestamp(e)-pd.Timestamp(s))/2, ymax*1.04, l, ha="center", va="top", fontsize=7, color="#2C5F8A", fontweight="bold")
ax.set_ylabel("SPX"); ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(axis="y", alpha=0.3)
ax.scatter([], [], color="#D04040", s=35, marker="v", edgecolors="#8B0000", linewidths=0.5, label=f"Signals (n={len(sig_dates)})")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout(); plt.savefig("figure1_signal_timeline.png", dpi=300, bbox_inches="tight"); plt.show()

# P&L Distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(results_df["Net_PnL_$"], bins=20, color="#000000", edgecolor="black", alpha=0.75)
ax.axvline(0, color="orange", linestyle="--", linewidth=1.2)
ax.axvline(expectancy, color="#27AE60", linestyle="--", linewidth=1.5, label=f"Expectancy (${expectancy:,.0f})")
ax.set_xlabel("Net P&L per trade ($)"); ax.set_ylabel("Frequency")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig("figure3_pnl_distribution.png", dpi=300, bbox_inches="tight"); plt.show()

# Trade Bars
fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#27AE60" if p > 0 else "#E74C3C" for p in results_df["Net_PnL_$"]]
ax.bar(range(n), results_df["Net_PnL_$"].values, color=colors, edgecolor="black", linewidth=0.4, width=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(n)); ax.set_xticklabels([d.strftime("%Y-%m") for d in results_df["Entry_Date"]], rotation=75, fontsize=7, ha="right")
ax.set_ylim(-1200, 1500)
max_idx = results_df["Net_PnL_$"].idxmax()
bar_idx = list(results_df.index).index(max_idx)
ax.annotate(f"${results_df.loc[max_idx, 'Net_PnL_$']:,.0f}", xy=(bar_idx, 1450), fontsize=7, ha="center", color="#000000", fontweight="bold")
ax.set_ylabel("Net P&L ($)"); ax.set_xlabel("Trade dates")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("figure4_trade_bars.png", dpi=300, bbox_inches="tight"); plt.show()

# Equity Curve
MARGIN = 20_000
margin_equity = pd.Series(index=timeline.index, data=0.0)
cumulative = MARGIN
for _, r in results_df.iterrows():
    exit_d = r["Exit_Date"]
    if exit_d in margin_equity.index: cumulative += r["Net_PnL_$"]
    margin_equity.loc[exit_d:] = cumulative
margin_equity = margin_equity.replace(0, np.nan); margin_equity.iloc[0] = MARGIN; margin_equity = margin_equity.ffill()
margin_norm = (margin_equity / MARGIN) * 100

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(margin_norm.index, margin_norm.values, color="#000000", linewidth=1.8)
for _, r in results_df.iterrows():
    ax.axvspan(r["Entry_Date"], r["Exit_Date"], color="#27AE60" if r["Net_PnL_$"] > 0 else "#E74C3C", alpha=0.10)
ax.set_ylabel("Equity (base=100)"); ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("figure5_equity_curve.png", dpi=300, bbox_inches="tight"); plt.show()

# IV Spread vs P&L by VIX Regime
fig, ax = plt.subplots(figsize=(7, 4.5))
valid = results_df.dropna(subset=["IV_Spread_Entry"])
clrs = []
for _, r in valid.iterrows():
    v = r["VIX_Entry"]
    if v < 15: clrs.append("#3498DB")
    elif v < 20: clrs.append("#F1C40F")
    elif v < 25: clrs.append("#E67E22")
    else: clrs.append("#E74C3C")
ax.scatter(valid["IV_Spread_Entry"], valid["Net_PnL_$"], c=clrs, edgecolors="black", linewidths=0.5, s=70, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
legend_el = [Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=8,markeredgecolor='black',markeredgewidth=0.5,label=l)
             for c,l in [("#3498DB","VIX<15"),("#F1C40F","VIX 15-20"),("#E67E22","VIX 20-25"),("#E74C3C","VIX>25")]]
ax.legend(handles=legend_el, fontsize=9, loc="upper left")
ax.set_xlabel("Put IV - Call IV at Entry"); ax.set_ylabel("Net P&L ($)")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figure6_ivspread_vs_pnl_vix.png", dpi=300, bbox_inches="tight"); plt.show()

# RR Change
fig, ax = plt.subplots(figsize=(7, 4))
if not rr_winners.empty: ax.hist(rr_winners, bins=12, color="#27AE60", alpha=0.6, edgecolor="black", linewidth=0.5, label=f"Winners (n={len(rr_winners)})")
if not rr_losers.empty: ax.hist(rr_losers, bins=12, color="#E74C3C", alpha=0.6, edgecolor="black", linewidth=0.5, label=f"Losers (n={len(rr_losers)})")
ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_xlabel("\u0394RR During Trade"); ax.set_ylabel("Frequency")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig("figure7_rr_change_winners_losers.png", dpi=300, bbox_inches="tight"); plt.show()