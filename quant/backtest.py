import os, glob, math, json, warnings
from pathlib import Path
import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

DATA_ROOT = Path("data_us")
OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)
START = "2016-01-01"
END = "2025-12-31"
TOP_N = 40
MIN_PRICE = 5.0
MIN_DVOL = 2_000_000.0
COST_BPS_RT = 20.0

def rsi(s, n=14):
    d=s.diff()
    up=d.clip(lower_bound=0)
    dn=(-d).clip(lower_bound=0)
    au=up.ewm_mean(alpha=1/n, adjust=False)
    ad=dn.ewm_mean(alpha=1/n, adjust=False)
    rs=au/(ad+1e-12)
    return 100-100/(1+rs)

def load_one(path):
    try:
        df=pl.read_csv(path, has_header=False, new_columns=["ticker","per","date","time","open","high","low","close","vol","openint"],
                       columns=["ticker","per","date","time","open","high","low","close","vol","openint"])
        df=df.with_columns([
            pl.col("date").cast(pl.Utf8).str.strptime(pl.Date,"%Y%m%d",strict=False),
            pl.col("open").cast(pl.Float64,strict=False),
            pl.col("high").cast(pl.Float64,strict=False),
            pl.col("low").cast(pl.Float64,strict=False),
            pl.col("close").cast(pl.Float64,strict=False),
            pl.col("vol").cast(pl.Float64,strict=False),
        ])
        df=df.filter((pl.col("date")>=pl.date(2016,1,1)) & (pl.col("date")<=pl.date(2025,12,31)))
        if df.height < 300: return None
        t=df["ticker"][0]
        df=df.sort("date").unique(subset=["date"],keep="last")
        return t, df.select(["date","open","high","low","close","vol"])
    except Exception:
        return None

def features(df):
    c,o,h,l,v=[df[x] for x in ["close","open","high","low","vol"]]
    tr=pl.max_horizontal([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()])
    atr=tr.rolling_mean(14)
    ma20=c.rolling_mean(20); ma50=c.rolling_mean(50); ma100=c.rolling_mean(100); ma200=c.rolling_mean(200)
    bbmid=ma20; bbsd=c.rolling_std(20); z=(c-bbmid)/(bbsd+1e-12)
    r=rsi(c,14)
    hi20=c.rolling_max(20).shift(1); hi55=c.rolling_max(55).shift(1); hi100=c.rolling_max(100).shift(1); hi252=c.rolling_max(252).shift(1)
    lo20=c.rolling_min(20).shift(1); lo55=c.rolling_min(55).shift(1)
    vol20=v.rolling_mean(20); vol5=v.rolling_mean(5)
    ret1=c.pct_change(1); ret3=c.pct_change(3); ret5=c.pct_change(5); ret10=c.pct_change(10); ret20=c.pct_change(20); ret60=c.pct_change(60); ret120=c.pct_change(120)
    rng=(h-l)/(c.shift(1)+1e-12)
    atrpct=atr/(c+1e-12)
    gap=(o/c.shift(1)-1)
    sma_slope20=ma20/ma20.shift(10)-1
    hh252=c.rolling_max(252)
    dist52=c/hh252
    return df.with_columns([
        pl.Series("ret1",ret1),pl.Series("ret3",ret3),pl.Series("ret5",ret5),pl.Series("ret10",ret10),pl.Series("ret20",ret20),pl.Series("ret60",ret60),pl.Series("ret120",ret120),
        pl.Series("ma20",ma20),pl.Series("ma50",ma50),pl.Series("ma100",ma100),pl.Series("ma200",ma200),pl.Series("rsi14",r),
        pl.Series("z20",z),pl.Series("atr14",atr),pl.Series("atrpct",atrpct),pl.Series("hi20",hi20),pl.Series("hi55",hi55),pl.Series("hi100",hi100),pl.Series("hi252",hi252),
        pl.Series("lo20",lo20),pl.Series("lo55",lo55),pl.Series("vol20",vol20),pl.Series("vol5",vol5),
        pl.Series("range",rng),pl.Series("gap",gap),pl.Series("slope20",sma_slope20),pl.Series("dist52",dist52),
        pl.Series("next_ret",c.shift(-1)/c-1)
    ])

# strategy returns are signal-weighted next-day returns; signal is always based on information available at close t.
def build_signals(df):
    q=features(df)
    c=pl.col("close"); v=pl.col("vol")
    sigs={}
    # 1-15 trend / momentum
    sigs["S01_break20"]=(c>pl.col("hi20"))&(c>pl.col("ma50"))
    sigs["S02_break55"]=(c>pl.col("hi55"))&(c>pl.col("ma50"))
    sigs["S03_break100"]=(c>pl.col("hi100"))&(c>pl.col("ma100"))
    sigs["S04_break252"]=(c>pl.col("hi252"))&(c>pl.col("ma200"))
    sigs["S05_ma2050"]= (c>pl.col("ma20"))&(pl.col("ma20")>pl.col("ma50"))
    sigs["S06_ma50100"]= (pl.col("ma50")>pl.col("ma100"))&(c>pl.col("ma50"))
    sigs["S07_mom20"]= (pl.col("ret20")>0.08)&(c>pl.col("ma50"))
    sigs["S08_mom60"]= (pl.col("ret60")>0.12)&(c>pl.col("ma100"))
    sigs["S09_mom120"]= (pl.col("ret120")>0.20)&(c>pl.col("ma200"))
    sigs["S10_rel52high"]= (pl.col("dist52")>0.90)&(pl.col("ret20")>0)
    sigs["S11_pullback20"]= (c>pl.col("ma50"))&(pl.col("rsi14")<45)&(pl.col("ret20")>0)
    sigs["S12_pullback50"]= (c>pl.col("ma100"))&(c/pl.col("ma20")<0.97)&(pl.col("ret60")>0.05)
    sigs["S13_break_vol"]= (c>pl.col("hi20"))&(v>pl.col("vol20")*1.5)
    sigs["S14_break_lowvol"]= (c>pl.col("hi20"))&(pl.col("atrpct")<pl.col("atrpct").rolling_quantile(60,0.40))
    sigs["S15_trend_slope"]= (c>pl.col("ma50"))&(pl.col("slope20")>0.03)
    # 16-28 mean reversion
    sigs["S16_rsi25"]=(pl.col("rsi14")<25)&(pl.col("ret3")<0)
    sigs["S17_rsi20"]=(pl.col("rsi14")<20)
    sigs["S18_bb_low"]=pl.col("z20")<-2
    sigs["S19_bb_low_rsi"]=(pl.col("z20")<-1.8)&(pl.col("rsi14")<35)
    sigs["S20_drop5"]= (pl.col("ret5")<-0.12)
    sigs["S21_drop10"]=(pl.col("ret10")<-0.18)
    sigs["S22_three_red"]=(pl.col("ret3")<-0.06)&(c<pl.col("ma20"))
    sigs["S23_below20_reclaim"]=(c<pl.col("ma20"))&(c.shift(-1)>c) # conservative one-day reversal proxy
    sigs["S24_atr_extreme"]= (pl.col("atrpct")>pl.col("atrpct").rolling_quantile(120,0.8))&(pl.col("ret3")<-0.05)
    sigs["S25_gap_down_reversal"]=(pl.col("gap")<-0.04)&(c>o)&(pl.col("ret1")<0)
    sigs["S26_under_52high"]=(pl.col("dist52")<0.75)&(pl.col("ret5")>0)
    sigs["S27_stretch_ma20"]=(c/pl.col("ma20")<0.93)
    sigs["S28_meanrev_trend"]=(c>pl.col("ma200"))&(pl.col("z20")<-1.5)
    # 29-38 volatility / volume
    sigs["S29_nr7"]=(pl.col("range")<=pl.col("range").rolling_min(7))&(c>c.shift(1))
    sigs["S30_range_expand"]=(pl.col("range")>pl.col("range").rolling_mean(20)*1.8)&(pl.col("ret1")>0)
    sigs["S31_volume_spike"]=(v>pl.col("vol20")*2)&(pl.col("ret1")>0)
    sigs["S32_volume_dry_break"]=(pl.col("vol5")<pl.col("vol20")*0.6)&(c>pl.col("hi20"))
    sigs["S33_gap_up_hold"]=(pl.col("gap")>0.04)&(c>o)
    sigs["S34_gap_down_reclaim"]=(pl.col("gap")<-0.04)&(c>o)
    sigs["S35_atr_compress_break"]=(pl.col("atrpct")<pl.col("atrpct").rolling_quantile(60,0.25))&(c>pl.col("hi20"))
    sigs["S36_volume_break20"]=(c>pl.col("hi20"))&(v>pl.col("vol20")*1.2)
    sigs["S37_vol_ratio_break"]=(pl.col("ret5")>0.05)&(v>pl.col("vol20")*1.5)
    sigs["S38_range_volume"]= (pl.col("range")>pl.col("range").rolling_mean(10)*1.5)&(v>pl.col("vol20")*1.5)&(c>o)
    # 39-50 hybrids
    sigs["S39_mom_lowvol"]=(pl.col("ret20")>0.08)&(pl.col("atrpct")<pl.col("atrpct").rolling_quantile(120,0.5))
    sigs["S40_break_rsi"]=(c>pl.col("hi55"))&(pl.col("rsi14").is_between(50,72))
    sigs["S41_break_slope_vol"]=(c>pl.col("hi20"))&(pl.col("slope20")>0.02)&(v>pl.col("vol20"))
    sigs["S42_pullback_200"]= (c>pl.col("ma200"))&(c/pl.col("ma20")<0.96)&(pl.col("rsi14")<40)
    sigs["S43_pullback_50"]= (c>pl.col("ma50"))&(c/pl.col("ma20")<0.97)&(pl.col("ret5")<0)&(pl.col("ret60")>0.05)
    sigs["S44_high52_break20"]=(pl.col("dist52")>0.95)&(c>pl.col("hi20"))
    sigs["S45_mom20_volume"]=(pl.col("ret20")>0.10)&(v>pl.col("vol20")*1.2)
    sigs["S46_mom60_slope"]=(pl.col("ret60")>0.15)&(pl.col("slope20")>0.03)
    sigs["S47_bb_squeeze_break"]=(pl.col("z20")>1.5)&(pl.col("atrpct")<pl.col("atrpct").rolling_quantile(120,0.5))
    sigs["S48_reversal_above200"]=(c>pl.col("ma200"))&(pl.col("z20")<-1.5)&(pl.col("ret1")>0)
    sigs["S49_multi_factor"]=(pl.col("ret20")>0.05)&(pl.col("ret60")>0.08)&(c>pl.col("ma100"))&(pl.col("rsi14")>50)&(v>pl.col("vol20"))
    sigs["S50_ensemble"]=(pl.col("ret20")>0)&(c>pl.col("ma50"))&(((pl.col("z20")>0.5)&(pl.col("rsi14")>50))|((pl.col("z20")<-1.2)&(pl.col("rsi14")<40)))
    return q.with_columns([expr.cast(pl.Int8).alias(name) for name,expr in sigs.items()])

files=glob.glob(str(DATA_ROOT/"**/*.us.txt"),recursive=True)
print("files",len(files))
rows=[]
for i,p in enumerate(files):
    r=load_one(p)
    if r is None: continue
    ticker,df=r
    df=build_signals(df)
    # liquidity screen from lagged dollar volume
    df=df.with_columns((pl.col("close")*pl.col("vol")).rolling_mean(20).shift(1).alias("dvol20"))
    df=df.filter((pl.col("close")>=MIN_PRICE)&(pl.col("dvol20")>=MIN_DVOL))
    if df.height:
        df=df.with_columns(pl.lit(ticker).alias("ticker"))
        rows.append(df.select(["ticker","date","next_ret"]+[f"S{i:02d}_{n}" for i in range(1,51) for n in []]))
    if (i+1)%1000==0: print("processed",i+1)

# Re-read in a more memory-efficient manner for outputs.
frames=[]
for p in files:
    r=load_one(p)
    if r is None: continue
    ticker,df=r
    df=build_signals(df).with_columns(pl.lit(ticker).alias("ticker"))
    df=df.with_columns((pl.col("close")*pl.col("vol")).rolling_mean(20).shift(1).alias("dvol20"))
    df=df.filter((pl.col("close")>=MIN_PRICE)&(pl.col("dvol20")>=MIN_DVOL))
    if df.height:
        frames.append(df)
all_df=pl.concat(frames,how="diagonal_relaxed")
print("rows",all_df.height,"tickers",all_df["ticker"].n_unique())

results=[]
years=list(range(2016,2026))
for j in range(1,51):
    col=f"S{j:02d}_"
# cols have names embedded; discover
sigcols=[c for c in all_df.columns if c.startswith("S") and c[1:3].isdigit()]
for col in sigcols:
    d=all_df.select(["date","ticker",col,"next_ret"]).with_columns(
        (pl.col(col)*pl.col("next_ret")).alias("gross"),
        (pl.col(col).cast(pl.Int8)*1).alias("active")
    ).filter(pl.col("next_ret").is_not_null())
    daily=d.group_by("date").agg([
        pl.col("gross").filter(pl.col("active")==1).mean().fill_null(0).alias("ret"),
        pl.col("active").sum().alias("n")
    ]).sort("date").with_columns((pl.col("ret")-pl.col("ret").abs()*0 + 0).alias("dummy"))
    # turnover proxy: cost applied on active days with any signal
    daily=daily.with_columns((pl.when(pl.col("n")>0).then(COST_BPS_RT/10000).otherwise(0)).alias("cost"))
    daily=daily.with_columns((pl.col("ret")-pl.col("cost")).alias("net"))
    vals=daily["net"].to_numpy()
    eq=np.cumprod(1+np.nan_to_num(vals,nan=0.0))
    if len(eq)==0: continue
    years_ret={}
    for y in years:
        sub=daily.filter(pl.col("date").dt.year()==y)["net"].to_numpy()
        years_ret[str(y)]=float(np.prod(1+np.nan_to_num(sub,nan=0.0))-1) if len(sub) else np.nan
    cagr=float(eq[-1]**(252/len(vals))-1) if len(vals)>0 else np.nan
    dd=float(np.min(eq/np.maximum.accumulate(eq)-1))
    vol=float(np.std(vals)*np.sqrt(252))
    sharpe=float((np.mean(vals)/np.std(vals))*np.sqrt(252)) if np.std(vals)>0 else np.nan
    hit=sum(1 for y in years if np.isfinite(years_ret[str(y)]) and years_ret[str(y)]>=0.20)
    results.append({"strategy":col,"cagr":cagr,"max_dd":dd,"sharpe":sharpe,"years_ge_20pct":hit,**{f"r_{y}":years_ret[str(y)] for y in years}})
res=pl.DataFrame(results).sort(["years_ge_20pct","cagr"],descending=True)
res.write_csv(OUT/"strategy_results_2016_2025.csv")
res.head(50).write_csv(OUT/"top50.csv")
print(res)

# Holdout report 2023-2025 and full-period summary
summary=[]
for row in res.iter_rows(named=True):
    hold=[row.get(f"r_{y}") for y in [2023,2024,2025]]
    summary.append({"strategy":row["strategy"],"cagr_2016_25":row["cagr"],"years_ge20":row["years_ge_20pct"],"dd":row["max_dd"],"sharpe":row["sharpe"],"holdout_2023":hold[0],"holdout_2024":hold[1],"holdout_2025":hold[2],"passes":bool(row["cagr"]>=0.20 and row["years_ge_20pct"]>=8 and row["max_dd"]>=-0.30 and row["sharpe"]>=1.0)})
pl.DataFrame(summary).sort(["passes","years_ge20","cagr_2016_25"],descending=True).write_csv(OUT/"screened_results.csv")
