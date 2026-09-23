from pathlib import Path
import glob, warnings
import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

DATA_ROOT = Path("data_us")
OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)

MIN_PRICE = 5.0
MIN_DVOL = 2_000_000.0
COST_BPS_RT = 20.0
YEARS = list(range(2016, 2026))

STRATEGIES = [
"S01_break20","S02_break55","S03_break100","S04_break252","S05_ma2050","S06_ma50100",
"S07_mom20","S08_mom60","S09_mom120","S10_rel52high","S11_pullback20","S12_pullback50",
"S13_break_vol","S14_break_lowvol","S15_trend_slope","S16_rsi25","S17_rsi20","S18_bb_low",
"S19_bb_low_rsi","S20_drop5","S21_drop10","S22_three_red","S23_reversal","S24_atr_extreme",
"S25_gap_down_reversal","S26_under_52high","S27_stretch_ma20","S28_meanrev_trend","S29_nr7",
"S30_range_expand","S31_volume_spike","S32_volume_dry_break","S33_gap_up_hold","S34_gap_down_reclaim",
"S35_atr_compress_break","S36_volume_break20","S37_vol_ratio_break","S38_range_volume",
"S39_mom_lowvol","S40_break_rsi","S41_break_slope_vol","S42_pullback_200","S43_pullback_50",
"S44_high52_break20","S45_mom20_volume","S46_mom60_slope","S47_bb_squeeze_break",
"S48_reversal_above200","S49_multi_factor","S50_ensemble"
]

def load_one(path):
    try:
        df = pl.read_csv(path, has_header=True, ignore_errors=True)
        df = df.rename({
            "<TICKER>":"ticker","<PER>":"per","<DATE>":"date","<TIME>":"time",
            "<OPEN>":"open","<HIGH>":"high","<LOW>":"low","<CLOSE>":"close","<VOL>":"vol","<OPENINT>":"openint"
        })
        df = df.with_columns([
            pl.col("date").cast(pl.Utf8).str.strptime(pl.Date,"%Y%m%d",strict=False),
            pl.col("open").cast(pl.Float64,strict=False),
            pl.col("high").cast(pl.Float64,strict=False),
            pl.col("low").cast(pl.Float64,strict=False),
            pl.col("close").cast(pl.Float64,strict=False),
            pl.col("vol").cast(pl.Float64,strict=False),
        ])
        df = df.filter((pl.col("date")>=pl.date(2016,1,1)) & (pl.col("date")<=pl.date(2025,12,31)))
        if df.height < 300:
            return None
        df = df.sort("date").unique(subset=["date"],keep="last")
        return df.select(["date","open","high","low","close","vol"])
    except Exception:
        return None

def rsi(c,n=14):
    d=c.diff()
    up=d.clip(lower_bound=0)
    dn=(-d).clip(lower_bound=0)
    au=up.ewm_mean(alpha=1/n,adjust=False)
    ad=dn.ewm_mean(alpha=1/n,adjust=False)
    return 100-100/(1+(au/(ad+1e-12)))

def build(df):
    c,o,h,l,v=[df[x] for x in ["close","open","high","low","vol"]]
    tr=pl.max_horizontal([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()])
    atr=tr.rolling_mean(14)
    ma20=c.rolling_mean(20); ma50=c.rolling_mean(50); ma100=c.rolling_mean(100); ma200=c.rolling_mean(200)
    sd20=c.rolling_std(20); z20=(c-ma20)/(sd20+1e-12)
    r=rsi(c)
    hi20=c.rolling_max(20).shift(1); hi55=c.rolling_max(55).shift(1); hi100=c.rolling_max(100).shift(1); hi252=c.rolling_max(252).shift(1)
    vol20=v.rolling_mean(20); vol5=v.rolling_mean(5)
    ret1=c.pct_change(1); ret3=c.pct_change(3); ret5=c.pct_change(5); ret10=c.pct_change(10); ret20=c.pct_change(20); ret60=c.pct_change(60); ret120=c.pct_change(120)
    rng=(h-l)/(c.shift(1)+1e-12); atrpct=atr/(c+1e-12); gap=o/c.shift(1)-1; slope20=ma20/ma20.shift(10)-1
    dist52=c/c.rolling_max(252)
    # No future information: next-day return is used only as the realized payoff after signal.
    next_ret=c.shift(-1)/c-1
    x=df.with_columns([
        ret1.alias("ret1"),ret3.alias("ret3"),ret5.alias("ret5"),ret10.alias("ret10"),ret20.alias("ret20"),ret60.alias("ret60"),ret120.alias("ret120"),
        ma20.alias("ma20"),ma50.alias("ma50"),ma100.alias("ma100"),ma200.alias("ma200"),r.alias("rsi14"),z20.alias("z20"),
        atr.alias("atr14"),atrpct.alias("atrpct"),hi20.alias("hi20"),hi55.alias("hi55"),hi100.alias("hi100"),hi252.alias("hi252"),
        vol20.alias("vol20"),vol5.alias("vol5"),rng.alias("range"),gap.alias("gap"),slope20.alias("slope20"),dist52.alias("dist52"),
        next_ret.alias("next_ret")
    ])
    c=pl.col
    sig={
      "S01_break20":(c("close")>c("hi20"))&(c("close")>c("ma50")),
      "S02_break55":(c("close")>c("hi55"))&(c("close")>c("ma50")),
      "S03_break100":(c("close")>c("hi100"))&(c("close")>c("ma100")),
      "S04_break252":(c("close")>c("hi252"))&(c("close")>c("ma200")),
      "S05_ma2050":(c("close")>c("ma20"))&(c("ma20")>c("ma50")),
      "S06_ma50100":(c("ma50")>c("ma100"))&(c("close")>c("ma50")),
      "S07_mom20":(c("ret20")>0.08)&(c("close")>c("ma50")),
      "S08_mom60":(c("ret60")>0.12)&(c("close")>c("ma100")),
      "S09_mom120":(c("ret120")>0.20)&(c("close")>c("ma200")),
      "S10_rel52high":(c("dist52")>0.90)&(c("ret20")>0),
      "S11_pullback20":(c("close")>c("ma50"))&(c("rsi14")<45)&(c("ret20")>0),
      "S12_pullback50":(c("close")>c("ma100"))&(c("close")/c("ma20")<0.97)&(c("ret60")>0.05),
      "S13_break_vol":(c("close")>c("hi20"))&(c("vol")>c("vol20")*1.5),
      "S14_break_lowvol":(c("close")>c("hi20"))&(c("atrpct")<c("atrpct").rolling_quantile(quantile=0.40, window_size=60)),
      "S15_trend_slope":(c("close")>c("ma50"))&(c("slope20")>0.03),
      "S16_rsi25":(c("rsi14")<25)&(c("ret3")<0),
      "S17_rsi20":(c("rsi14")<20),
      "S18_bb_low":c("z20")<-2,
      "S19_bb_low_rsi":(c("z20")<-1.8)&(c("rsi14")<35),
      "S20_drop5":c("ret5")<-0.12,
      "S21_drop10":c("ret10")<-0.18,
      "S22_three_red":(c("ret3")<-0.06)&(c("close")<c("ma20")),
      "S23_reversal":(c("close")<c("ma20"))&(c("close")>c("open"))&(c("ret1")<0),
      "S24_atr_extreme":(c("atrpct")>c("atrpct").rolling_quantile(quantile=0.80, window_size=120))&(c("ret3")<-0.05),
      "S25_gap_down_reversal":(c("gap")<-0.04)&(c("close")>c("open")),
      "S26_under_52high":(c("dist52")<0.75)&(c("ret5")>0),
      "S27_stretch_ma20":c("close")/c("ma20")<0.93,
      "S28_meanrev_trend":(c("close")>c("ma200"))&(c("z20")<-1.5),
      "S29_nr7":(c("range")<=c("range").rolling_min(7))&(c("close")>c("close").shift(1)),
      "S30_range_expand":(c("range")>c("range").rolling_mean(20)*1.8)&(c("ret1")>0),
      "S31_volume_spike":(c("vol")>c("vol20")*2)&(c("ret1")>0),
      "S32_volume_dry_break":(c("vol5")<c("vol20")*0.6)&(c("close")>c("hi20")),
      "S33_gap_up_hold":(c("gap")>0.04)&(c("close")>c("open")),
      "S34_gap_down_reclaim":(c("gap")<-0.04)&(c("close")>c("open")),
      "S35_atr_compress_break":(c("atrpct")<c("atrpct").rolling_quantile(quantile=0.25, window_size=60))&(c("close")>c("hi20")),
      "S36_volume_break20":(c("close")>c("hi20"))&(c("vol")>c("vol20")*1.2),
      "S37_vol_ratio_break":(c("ret5")>0.05)&(c("vol")>c("vol20")*1.5),
      "S38_range_volume":(c("range")>c("range").rolling_mean(10)*1.5)&(c("vol")>c("vol20")*1.5)&(c("close")>c("open")),
      "S39_mom_lowvol":(c("ret20")>0.08)&(c("atrpct")<c("atrpct").rolling_quantile(quantile=0.50, window_size=120)),
      "S40_break_rsi":(c("close")>c("hi55"))&(c("rsi14").is_between(50,72)),
      "S41_break_slope_vol":(c("close")>c("hi20"))&(c("slope20")>0.02)&(c("vol")>c("vol20")),
      "S42_pullback_200":(c("close")>c("ma200"))&(c("close")/c("ma20")<0.96)&(c("rsi14")<40),
      "S43_pullback_50":(c("close")>c("ma50"))&(c("close")/c("ma20")<0.97)&(c("ret5")<0)&(c("ret60")>0.05),
      "S44_high52_break20":(c("dist52")>0.95)&(c("close")>c("hi20")),
      "S45_mom20_volume":(c("ret20")>0.10)&(c("vol")>c("vol20")*1.2),
      "S46_mom60_slope":(c("ret60")>0.15)&(c("slope20")>0.03),
      "S47_bb_squeeze_break":(c("z20")>1.5)&(c("atrpct")<c("atrpct").rolling_quantile(quantile=0.50, window_size=120)),
      "S48_reversal_above200":(c("close")>c("ma200"))&(c("z20")<-1.5)&(c("ret1")>0),
      "S49_multi_factor":(c("ret20")>0.05)&(c("ret60")>0.08)&(c("close")>c("ma100"))&(c("rsi14")>50)&(c("vol")>c("vol20")),
      "S50_ensemble":(c("ret20")>0)&(c("close")>c("ma50"))&(((c("z20")>0.5)&(c("rsi14")>50))|((c("z20")<-1.2)&(c("rsi14")<40)))
    }
    return x.with_columns([e.cast(pl.Int8).alias(k) for k,e in sig.items()])

files=[p for p in glob.glob(str(DATA_ROOT/"**/*.us.txt"),recursive=True) if "/etfs/" not in p.lower()]
print("stock files",len(files))

# Daily aggregate: equal-weight every qualifying stock signal, then subtract a 20bp round-trip proxy only on days with trades.
agg={s:{} for s in STRATEGIES}
counts={s:{} for s in STRATEGIES}
all_dates=set()

for i,p in enumerate(files,1):
    df=load_one(p)
    if df is None: continue
    df=df.with_columns((pl.col("close")*pl.col("vol")).rolling_mean(20).shift(1).alias("dvol20"))
    df=build(df).filter((pl.col("close")>=MIN_PRICE)&(pl.col("dvol20")>=MIN_DVOL)&pl.col("next_ret").is_not_null())
    if df.height==0: continue
    dates=df["date"].to_list()
    all_dates.update(dates)
    nr=df["next_ret"].to_numpy()
    for s in STRATEGIES:
        a=df[s].to_numpy().astype(bool)
        if not a.any(): continue
        vals=np.where(a,nr,0.0)
        # accumulate sum/count by date
        for dt,val,active in zip(dates,vals,a):
            if active:
                agg[s][dt]=agg[s].get(dt,0.0)+float(val)
                counts[s][dt]=counts[s].get(dt,0)+1
    if i%1000==0: print("processed",i)

out=[]
calendar=pl.DataFrame({"date":sorted(all_dates)}).sort("date")
for s in STRATEGIES:
    ds=sorted(agg[s])
    rows=[]
    for dt in ds:
        n=counts[s][dt]
        rows.append((dt,agg[s][dt]/n if n else 0.0,n))
    active=pl.DataFrame(rows,schema=["date","gross","n"],orient="row").sort("date") if rows else pl.DataFrame({"date":[],"gross":[],"n":[]},schema={"date":pl.Date,"gross":pl.Float64,"n":pl.Int64})
    daily=calendar.join(active,on="date",how="left").with_columns([
        pl.col("gross").fill_null(0.0),
        pl.col("n").fill_null(0)
    ]).with_columns([
        pl.when(pl.col("n")>0).then(pl.lit(COST_BPS_RT/10000)).otherwise(pl.lit(0.0)).alias("cost"),
        pl.when(pl.col("n")>0).then(pl.col("gross")-pl.lit(COST_BPS_RT/10000)).otherwise(pl.col("gross")).alias("net")
    ])
    vals=daily["net"].to_numpy()
    eq=np.cumprod(1+np.nan_to_num(vals,nan=0.0))
    if len(eq)<252: continue
    cagr=float(eq[-1]**(252/len(eq))-1)
    dd=float(np.min(eq/np.maximum.accumulate(eq)-1))
    sd=float(np.std(vals))
    sharpe=float(np.mean(vals)/sd*np.sqrt(252)) if sd>0 else np.nan
    yr={}
    for y in YEARS:
        yy=daily.filter(pl.col("date").dt.year()==y)["net"].to_numpy()
        yr[y]=float(np.prod(1+np.nan_to_num(yy,nan=0.0))-1) if len(yy) else np.nan
    hit=sum(1 for y in YEARS if np.isfinite(yr[y]) and yr[y]>=0.20)
    active_days=int((daily["n"]>0).sum())
    avg_positions=float(daily.filter(pl.col("n")>0)["n"].mean()) if active_days else 0.0
    out.append({"strategy":s,"cagr":cagr,"max_dd":dd,"sharpe":sharpe,"years_ge_20":hit,"active_days":active_days,"avg_positions":avg_positions,
                **{f"r_{y}":yr[y] for y in YEARS}})
res=pl.DataFrame(out).sort(["years_ge_20","cagr"],descending=True)
res.write_csv(OUT/"strategy_results_2016_2025.csv")
screen=[]
for r in res.iter_rows(named=True):
    screen.append({"strategy":r["strategy"],"cagr":r["cagr"],"years_ge20":r["years_ge_20"],"max_dd":r["max_dd"],"sharpe":r["sharpe"],"active_days":r["active_days"],"avg_positions":r["avg_positions"],
                   "2023":r["r_2023"],"2024":r["r_2024"],"2025":r["r_2025"],
                   "passes":bool(r["cagr"]>=0.20 and r["years_ge_20"]>=8 and r["max_dd"]>=-0.30 and r["sharpe"]>=1.0)})
pl.DataFrame(screen).write_csv(OUT/"screened_results.csv")
print(res)
