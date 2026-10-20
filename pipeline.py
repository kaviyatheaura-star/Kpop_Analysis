"""Comeback momentum pipeline for the South Korea Top 50 playlist.

Usage:  python pipeline.py Atlantic_South_Korea.csv [min_gap_days]
Outputs: clean.csv, episodes.csv, songs.csv, validation.txt (in ./output)
"""
import re
import sys
import numpy as np
import pandas as pd

W_SPIKE = (0.4, 0.3, 0.3)     # rank jump, popularity rate, peak rank
W_FANDOM = (0.4, 0.3, 0.3)    # re-entries, spike sharpness, recovery speed


def norm_key(song: str, artist: str) -> str:
    s = song.lower()
    s = re.sub(r"\((feat|with)[^)]*\)|\[(feat|with)[^\]]*\]", "", s)  # strip features
    s = re.sub(r"\s+", " ", s).strip()
    a = artist.lower().split(",")[0].split("&")[0].strip()          # primary artist
    return f"{s}|{a}"


def clean(path):
    df = pd.read_csv(path)
    log = []
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    n_raw = len(df)
    # a date holding two full snapshots: keep the first block of 50 rows
    counts = df.groupby("date").size()
    for d in counts[counts > 50].index:
        idx = df.index[df.date == d]
        df = df.drop(idx[50:])
        log.append(f"{d.date()}: had {counts[d]} rows (two snapshots); kept first 50")
    df = df.sort_values(["date", "position"]).reset_index(drop=True)
    df["duration_min"] = df.duration_ms / 60000
    df["key"] = [norm_key(s, a) for s, a in zip(df.song, df.artist)]
    # same key twice on one day (e.g. two versions) -> keep best rank
    dup = df.duplicated(["date", "key"], keep="first").sum()
    if dup:
        log.append(f"{dup} same-day duplicate song/artist rows dropped (best rank kept)")
        df = df.drop_duplicates(["date", "key"], keep="first")
    # popularity == 0 while a song sits in the Top 50 is an API gap, not a real score:
    # treat as missing and interpolate within the song
    zero = df.popularity == 0
    df["popularity"] = df.popularity.where(~zero)
    df["popularity"] = (df.groupby("key").popularity
                        .transform(lambda x: x.interpolate(limit_direction="both")))
    df["popularity"] = (df.popularity.fillna(df.groupby("key").popularity.transform("median"))
                        .fillna(df.popularity.median()).round().astype(int))
    log.append(f"{int(zero.sum())} zero-popularity rows treated as missing and interpolated within each song")
    # album attributes can differ across days for one song: use the mode
    for c in ["album_type", "total_tracks"]:
        df[c] = df.groupby("key")[c].transform(lambda x: x.mode().iloc[0])
    all_days = pd.date_range(df.date.min(), df.date.max())
    missing = all_days.difference(df.date.unique())
    log.append(f"rows raw={n_raw}, clean={len(df)}; date range {df.date.min().date()} to {df.date.max().date()}")
    log.append(f"missing snapshot dates (treated as unobserved, not as exits): {[str(d.date()) for d in missing]}")
    per_day = df.groupby("date").size()
    log.append(f"days with !=50 rows after cleaning: {int((per_day != 50).sum())}")
    log.append(f"popularity range after repair: {df.popularity.min()}-{df.popularity.max()}")
    return df, log


def build_episodes(df, min_gap=3):
    days = np.sort(df.date.unique())
    didx = {d: i for i, d in enumerate(days)}           # index over OBSERVED days
    df = df.assign(di=df.date.map(didx))
    last_di = len(days) - 1
    rows = []
    for key, g in df.groupby("key"):
        g = g.sort_values("di")
        di = g.di.values
        brk = np.where(np.diff(di) - 1 >= min_gap)[0] + 1   # >= min_gap absent days
        for n, part in enumerate(np.split(np.arange(len(g)), brk), start=1):
            e = g.iloc[part]
            pos, pop = e.position.values, e.popularity.values
            k = min(4, len(e)) - 1                           # day 4 of episode
            pk = int(np.argmin(pos))
            tail = pos[pk:]
            slope = np.polyfit(np.arange(len(tail)), tail, 1)[0] if len(tail) >= 3 else np.nan
            prior = np.nan
            gap_days = np.nan
            if n > 1:
                prev = g.iloc[:part[0]]
                prior = prev.popularity.values[-1]
                gap_days = di[part[0]] - prev.di.values[-1] - 1
            rows.append(dict(
                key=key, song=e.song.iloc[0], artist=e.artist.iloc[0],
                episode=n, is_reentry=n > 1,
                left_censored=(n == 1 and e.di.iloc[0] == 0),
                right_censored=bool(e.di.iloc[-1] == last_di),
                start=e.date.iloc[0], end=e.date.iloc[-1],
                entry_rank=int(pos[0]), entry_pop=int(pop[0]), prior_pop=prior,
                gap_days=gap_days,
                rank_jump=51 - int(pos[0]),
                pop_rate=(pop[k] - pop[0]) / max(k, 1),
                peak_rank=int(pos.min()), peak_pop=int(pop.max()),
                days_to_peak=int(pk), recovery_speed=(int(pos[0]) - int(pos.min())) / max(pk, 1) if pk > 0 else 0.0,
                retention_days=len(e), decay_slope=slope,
                volatility=float(np.std(np.diff(pos))) if len(e) > 2 else np.nan,
                album_type=e.album_type.iloc[0], total_tracks=int(e.total_tracks.iloc[0]),
                duration_min=e.duration_min.iloc[0], is_explicit=bool(e.is_explicit.iloc[0]),
            ))
    ep = pd.DataFrame(rows)

    def z(s):
        return (s - s.mean()) / (s.std() or 1)
    raw = W_SPIKE[0] * z(ep.rank_jump) + W_SPIKE[1] * z(ep.pop_rate) + W_SPIKE[2] * z(51 - ep.peak_rank)
    ep["spike_score"] = (raw.rank(pct=True) * 100).round(1)   # 0-100 so ratios are meaningful
    return ep


def build_songs(ep):
    pct = lambda s: s.rank(pct=True)
    g = ep.groupby(["key"]).agg(
        song=("song", "first"), artist=("artist", "first"),
        episodes=("episode", "max"), album_type=("album_type", "first"),
        total_tracks=("total_tracks", "first"), is_explicit=("is_explicit", "first"),
        duration_min=("duration_min", "first"),
        total_days=("retention_days", "sum"), best_rank=("peak_rank", "min"),
        mean_spike=("spike_score", "mean"), mean_pop_rate=("pop_rate", "mean"),
        mean_recovery=("recovery_speed", "mean"),
    ).reset_index()
    g["reentries"] = g.episodes - 1
    sharp = g.mean_pop_rate.clip(lower=0)
    g["fandom_score"] = (100 * (W_FANDOM[0] * pct(g.reentries) + W_FANDOM[1] * pct(sharp)
                                + W_FANDOM[2] * pct(g.mean_recovery))).round(1)
    return g


def album_advantage(ep):
    re_ = ep[ep.is_reentry]
    a = re_[re_.album_type == "album"].spike_score.mean()
    s = re_[re_.album_type == "single"].spike_score.mean()
    return a / s if s else np.nan


if __name__ == "__main__":
    import os
    path = sys.argv[1]
    gap = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.makedirs("output", exist_ok=True)
    df, log = clean(path)
    ep = build_episodes(df, gap)
    songs = build_songs(ep)
    df.drop(columns=["album_cover_url"]).to_csv("output/clean.csv", index=False)
    ep.to_csv("output/episodes.csv", index=False)
    songs.to_csv("output/songs.csv", index=False)
    log.append(f"min_gap={gap}: {len(ep)} episodes, {int(ep.is_reentry.sum())} re-entries, "
               f"{int((songs.reentries > 0).sum())} songs with >=1 re-entry")
    open("output/validation.txt", "w").write("\n".join(log))
    print("\n".join(log))
