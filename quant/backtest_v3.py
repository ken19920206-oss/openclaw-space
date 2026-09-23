from pathlib import Path
import glob, warnings, math
import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

DATA_ROOT = Path("data_us")
OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)

MIN_PRICE = 10.0
MIN_DVOL = 20_000_000.0
ROUND_TRIP_COST = 0.002
YEARS = list(range(2016, 2026))
HOLD_DAYS = [1, 3, 5, 10, 20]

BASE_STRATEGIES = [
    "break20", "break55", "ma2050", "mom20", "mom60",
    "pullback20", "rsi25", "bb_low", "gap_down_reversal", "multi_factor"
]
STRATEGIES = [f"{b}_H{h}" for b in BASE_STRATEGIES for h in HOLD_DAYS]
SPLIT_RATIOS = np.array([0.10, 0.125, 0.20, 0.25, 1/3, 0.40, 0.50,
                         2.0, 2.5, 3.0, 4.0, 5.0, 10.0], dtype=float)

def repair_splits(df):
    """Infer obvious split/reverse-split events and back-adjust prior OHLC."""
    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    n = len(c)
    if n < 2:
        return df, 0
    ratio = c[1:] / np.where(c[:-1] == 0, np.nan, c[:-1])
    open_ratio = o[1:] / np.where(c[:-1] == 0, np.nan, c[:-1])
    event = np.ones(n, dtype=float)
    split_count = 0
    for i, r in enumerate(ratio, start=1):
        if not np.isfinite(r):
            continue
        j = int(np.argmin(np.abs(SPLIT_RATIOS - r) / SPLIT_RATIOS))
        target = SPLIT_RATIOS[j]
        if abs(r / target - 1.0) <= 0.015 and np.isfinite(open_ratio[i-1]) and abs(open_ratio[i-1] / target - 1.0) <= 0.03:
            event[i] = target
            split_count += 1
    if split_count == 0:
        return df, 0
    shifted = np.ones(n, dtype=float)
    shifted[:-1] = event[1:]
    scale = np.cumprod(shifted[::-1])[::-1]
    vol_scale = np.where(scale == 0, 1.0, scale)
    df = df.with_columns([
        (pl.col("open") * pl.Series(scale)).alias("open"),
        (pl.col("high") * pl.Series(scale)).alias("high"),
        (pl.col("low") * pl.Series(scale)).alias("low"),
        (pl.col("close") * pl.Series(scale)).alias("close"),
        (pl.col("vol") / pl.Series(vol_scale)).alias("vol"),
    ])
    return df, split_count

def load_one(path):
    try:
        df = pl.read_csv(path, has_header=True, ignore_errors=True)
        df = df.rename({
            "<TICKER>":"ticker","<PER>":"per","<DATE>":"date","<TIME>":"time",
            "<OPEN>":"open","<HIGH>":"high","<LOW>":"low","<CLOSE>":"close",
            "<VOL>":"vol","<OPENINT>":"openint"
        })
        df = df.with_columns([
            pl.col("date").cast(pl.Utf8).str.strptime(pl.Date,"%Y%m%d",strict=False),
            pl.col("open").cast(pl.Float64,strict=False),
            pl.col("high").cast(pl.Float64,strict=False),
            pl.col("low").cast(pl.Float64,strict=False),
            pl.col("close").cast(pl.Float64,strict=False),
            pl.col("vol").cast(pl.Float64,strict=False),
        ])
        df = df.filter(
            (pl.col("date")>=pl.date(2016,1,1)) &
            (pl.col("date")<=pl.date(2025,12,31))
        ).sort("date").unique(subset=["date"],keep="last")
        if df.height < 300:
            return None, {"path":str(path),"rows":df.height,"splits":0}
        df, splits = repair_splits(df)
        return df.select(["date","open","high","low","close","vol"]), {
            "path":str(path), "rows":df.height, "splits":splits
        }
    except Exception:
        return None, {"path":str(path),"rows":0,"splits":0}

def rsi(c,n=14):
    d=c.diff()
    up=d.clip(lower_bound=0)
    dn=(-d).clip(lower_bound=0)
    au=up.ewm_mean(alpha=1/n,adjust=False)
    ad=dn.ewm_mean(alpha=1/n,adjust=False)
    return 100-100/(1+(au/(ad+1e-12)))

def build_features(df):
    c,o,h,l,v=[df[x] for x in ["close","open","high","low","vol"]]
    tr=pl.max_horizontal([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()])
    atr=tr.rolling_mean(14)
    ma20=c.rolling_mean(20); ma50=c.rolling_mean(50); ma100=c.rolling_mean(100); ma200=c.rolling_mean(200)
    sd20=c.rolling_std(20)
    z20=(c-ma20)/(sd20+1e-12)
    hi20=c.rolling_max(20).shift(1)
    hi55=c.rolling_max(55).shift(1)
    vol20=v.rolling_mean(20)
    ret1=c.pct_change(1); ret5=c.pct_change(5); ret20=c.pct_change(20); ret60=c.pct_change(60)
    atrpct=atr/(c+1e-12)
    gap=o/c.shift(1)-1
    return df.with_columns([
        ret1.alias("ret1"),ret5.alias("ret5"),ret20.alias("ret20"),ret60.alias("ret60"),
        ma20.alias("ma20"),ma50.alias("ma50"),ma100.alias("ma100"),ma200.alias("ma200"),
        rsi(c).alias("rsi14"),z20.alias("z20"),atrpct.alias("atrpct"),
        hi20.alias("hi20"),hi55.alias("hi55"),vol20.alias("vol20"),gap.alias("gap"),
    ])

def build_signals(df):
    c=pl.col
    return df.with_columns([
        ((c("close")>c("hi20"))&(c("close")>c("ma50"))).cast(pl.Int8).alias("break20"),
        ((c("close")>c("hi55"))&(c("close")>c("ma50"))).cast(pl.Int8).alias("break55"),
        ((c("close")>c("ma20"))&(c("ma20")>c("ma50"))).cast(pl.Int8).alias("ma2050"),
        ((c("ret20")>0.08)&(c("close")>c("ma50"))).cast(pl.Int8).alias("mom20"),
        ((c("ret60")>0.12)&(c("close")>c("ma100"))).cast(pl.Int8).alias("mom60"),
        ((c("close")>c("ma50"))&(c("rsi14")<45)&(c("ret20")>0)).cast(pl.Int8).alias("pullback20"),
        ((c("rsi14")<25)&(c("ret5")<0)).cast(pl.Int8).alias("rsi25"),
        (c("z20")<-2).cast(pl.Int8).alias("bb_low"),
        ((c("gap")<-0.04)&(c("close")>c("open"))).cast(pl.Int8).alias("gap_down_reversal"),
        ((c("ret20")>0.05)&(c("ret60")>0.08)&(c("close")>c("ma100"))&
         (c("rsi14")>50)&(c("vol")>c("vol20"))).cast(pl.Int8).alias("multi_factor"),
    ])

def add_trade_to_agg(agg_sum, agg_n, strategy, dates, opens, closes, entry_idx, hold):
    exit_idx = entry_idx + hold - 1
    if entry_idx >= len(closes) or exit_idx >= len(closes):
        return False
    day_rets = np.empty(hold, dtype=float)
    day_rets[0] = closes[entry_idx] / opens[entry_idx] - 1.0
    if hold > 1:
        day_rets[1:] = closes[entry_idx+1:exit_idx+1] / closes[entry_idx:exit_idx] - 1.0
    if not np.isfinite(day_rets).all() or (np.abs(day_rets)>0.90).any():
        return False
    day_rets -= ROUND_TRIP_COST / hold
    for k, ret in enumerate(day_rets):
        dt = dates[entry_idx+k]
        agg_sum[strategy][dt] = agg_sum[strategy].get(dt, 0.0) + float(ret)
        agg_n[strategy][dt] = agg_n[strategy].get(dt, 0) + 1
    return True

files=[p for p in glob.glob(str(DATA_ROOT/"**/*.us.txt"),recursive=True)
       if "/etfs/" not in p.lower()]
print("stock files",len(files))

agg_sum={s:{} for s in STRATEGIES}
agg_n={s:{} for s in STRATEGIES}
quality=[]
all_dates=set()

for i,p in enumerate(files,1):
    df, stat = load_one(p)
    quality.append(stat)
    if df is None:
        continue
    df=df.with_columns(
        (pl.col("close")*pl.col("vol")).rolling_mean(20).shift(1).alias("dvol20")
    )
    df=build_signals(build_features(df)).fill_null(0).filter(
        (pl.col("close")>=MIN_PRICE)&
        (pl.col("dvol20")>=MIN_DVOL)
    )
    if df.height < 250:
        continue

    dates=df["date"].to_list()
    all_dates.update(dates)
    opens=df["open"].to_numpy()
    closes=df["close"].to_numpy()

    for base in BASE_STRATEGIES:
        sig=df[base].to_numpy().astype(bool)
        idxs=np.flatnonzero(sig)
        for hold in HOLD_DAYS:
            strategy=f"{base}_H{hold}"
            next_free=0
            for idx in idxs:
                entry=idx+1
                if entry < next_free:
                    continue
                if not add_trade_to_agg(agg_sum,agg_n,strategy,dates,opens,closes,entry,hold):
                    break
                next_free=entry+hold

    if i%1000==0:
        print("processed",i)

calendar=pl.DataFrame({"date":sorted(all_dates)}).sort("date")
out=[]

for s in STRATEGIES:
    rows=[(dt,agg_sum[s][dt],agg_n[s][dt]) for dt in sorted(agg_sum[s])]
    active=pl.DataFrame(
        rows,schema=["date","sum_ret","n"],orient="row"
    ) if rows else pl.DataFrame(
        {"date":[],"sum_ret":[],"n":[]},
        schema={"date":pl.Date,"sum_ret":pl.Float64,"n":pl.Int64}
    )
    daily=calendar.join(active,on="date",how="left").with_columns([
        pl.col("sum_ret").fill_null(0.0),
        pl.col("n").fill_null(0),
    ]).with_columns(
        pl.when(pl.col("n")>0)
          .then(pl.col("sum_ret")/pl.col("n"))
          .otherwise(pl.lit(0.0))
          .alias("net")
    )

    vals=daily["net"].to_numpy()
    eq=np.cumprod(1+np.nan_to_num(vals,nan=0.0))
    if len(eq)<252:
        continue

    cagr=float(eq[-1]**(252/len(eq))-1)
    dd=float(np.min(eq/np.maximum.accumulate(eq)-1))
    vals=np.nan_to_num(vals,nan=0.0,posinf=0.0,neginf=0.0)
    sd=float(np.std(vals))
    sharpe=float(np.mean(vals)/sd*np.sqrt(252)) if sd>0 else 0.0

    yr={}
    for y in YEARS:
        yy=daily.filter(pl.col("date").dt.year()==y)["net"].to_numpy()
        yr[y]=float(np.prod(1+np.nan_to_num(yy,nan=0.0))-1) if len(yy) else np.nan

    hit=sum(1 for y in YEARS if np.isfinite(yr[y]) and yr[y]>=0.20)
    active_days=int((daily["n"]>0).sum())
    avg_positions=float(daily.filter(pl.col("n")>0)["n"].mean()) if active_days else 0.0

    out.append({
        "strategy":s,"cagr":cagr,"max_dd":dd,"sharpe":sharpe,
        "years_ge20":hit,"active_days":active_days,
        "avg_positions":avg_positions,
        **{f"r_{y}":yr[y] for y in YEARS}
    })

res=pl.DataFrame(out).sort(["years_ge20","cagr"],descending=True)
res.write_csv(OUT/"strategy_results_2016_2025.csv")

screen=res.with_columns(
    (
        (pl.col("cagr")>=0.20)&
        (pl.col("years_ge20")>=8)&
        (pl.col("max_dd")>=-0.30)&
        (pl.col("sharpe")>=1.0)
    ).alias("passes")
)
screen.write_csv(OUT/"screened_results.csv")

pl.DataFrame(quality).write_csv(OUT/"data_quality.csv")

summary={
    "stock_files":len(files),
    "files_with_inferred_splits":sum(1 for q in quality if q["splits"]>0),
    "inferred_split_events":sum(q["splits"] for q in quality),
    "files_with_300plus_rows":sum(1 for q in quality if q["rows"]>=300),
    "strategies":len(STRATEGIES),
    "passes":int(screen["passes"].sum()),
}
print("QUALITY",summary)
print(res)
