"""Machine-learning layer for the comeback momentum project.

1. Re-entry predictor     (classification) - will a song that just left the chart return within H days?
2. Comeback strength      (regression)     - how many days will a re-entry stay on the chart?
3. Song archetypes        (clustering)     - K-Means on song-level momentum features
4. Spike detector         (anomaly)        - Isolation Forest on episode features
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (GradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             r2_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _last_day_stats(df):
    """Rank and popularity on every (song, date) so we can look up an episode's last day."""
    return df.set_index(["key", "date"])[["position", "popularity"]]


def _base_features(ep):
    f = pd.DataFrame(index=ep.index)
    f["is_single"] = (ep.album_type == "single").astype(int)
    f["total_tracks"] = ep.total_tracks
    f["duration_min"] = ep.duration_min
    f["is_explicit"] = ep.is_explicit.astype(int)
    return f


# --------------------------------------------------------------------------
# 1. re-entry predictor
# --------------------------------------------------------------------------
def build_exit_table(df, ep, horizon=30):
    """One row per chart exit that has `horizon` days of follow-up (no right-censoring)."""
    ep = ep.sort_values(["key", "episode"]).copy()
    last = _last_day_stats(df)
    ep["next_start"] = ep.groupby("key").start.shift(-1)
    ep["days_to_next"] = (ep.next_start - ep.end).dt.days
    max_date = df.date.max()
    elig = ep[(~ep.right_censored) & (ep.end + pd.Timedelta(days=horizon) <= max_date)].copy()
    elig["label"] = (elig.days_to_next <= horizon).astype(int)

    idx = list(zip(elig.key, elig.end))
    ld = last.reindex(idx)
    first_seen = df.groupby("key").date.min()
    cum_days = ep.groupby("key").retention_days.cumsum()

    X = _base_features(elig)
    X["last_rank"] = ld.position.values
    X["last_pop"] = ld.popularity.values
    X["peak_rank"] = elig.peak_rank
    X["entry_rank"] = elig.entry_rank
    X["retention_days"] = elig.retention_days
    X["volatility"] = elig.volatility.fillna(0)
    X["decay_slope"] = elig.decay_slope.fillna(0)
    X["episodes_so_far"] = elig.episode
    X["prev_gap_days"] = elig.gap_days.fillna(-1)
    X["days_since_first_seen"] = (elig.end - elig.key.map(first_seen)).dt.days
    X["total_days_so_far"] = cum_days.loc[elig.index]
    return X, elig.label, elig


def train_reentry_model(df, ep, horizon=30):
    X, y, meta = build_exit_table(df, ep, horizon)
    order = meta.end.argsort().values            # time-ordered split: train on past, test on future
    cut = int(len(X) * 0.7)
    tr, te = order[:cut], order[cut:]
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    models = {
        "Baseline (majority class)": DummyClassifier(strategy="prior"),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                random_state=SEED, n_jobs=-1),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        rows.append(dict(model=name, roc_auc=roc_auc_score(yte, p),
                         avg_precision=average_precision_score(yte, p)))
        fitted[name] = m
    res = pd.DataFrame(rows).round(3)
    rf = fitted["Random Forest"]
    pi = permutation_importance(rf, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=SEED)
    imp = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    info = dict(n_train=len(tr), n_test=len(te), positive_rate_train=float(ytr.mean()),
                positive_rate_test=float(yte.mean()), horizon=horizon)
    return dict(results=res, importance=imp, model=rf, features=list(X.columns), info=info,
                table=X.assign(label=y, key=meta.key, song=meta.song, artist=meta.artist, end=meta.end))


def score_current_exits(df, ep, bundle, horizon=30):
    """Probability of re-entry for songs that most recently left the chart (still inside the horizon window)."""
    last = _last_day_stats(df)
    max_date = df.date.max()
    cur = ep.sort_values(["key", "episode"]).groupby("key").tail(1)
    cur = cur[(~cur.right_censored) & (cur.end + pd.Timedelta(days=horizon) > max_date)].copy()
    if cur.empty:
        return pd.DataFrame()
    ld = last.reindex(list(zip(cur.key, cur.end)))
    first_seen = df.groupby("key").date.min()
    cum = ep.groupby("key").retention_days.sum()
    X = _base_features(cur)
    X["last_rank"] = ld.position.values
    X["last_pop"] = ld.popularity.values
    X["peak_rank"] = cur.peak_rank
    X["entry_rank"] = cur.entry_rank
    X["retention_days"] = cur.retention_days
    X["volatility"] = cur.volatility.fillna(0)
    X["decay_slope"] = cur.decay_slope.fillna(0)
    X["episodes_so_far"] = cur.episode
    X["prev_gap_days"] = cur.gap_days.fillna(-1)
    X["days_since_first_seen"] = (cur.end - cur.key.map(first_seen)).dt.days
    X["total_days_so_far"] = cur.key.map(cum)
    p = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    return (cur[["song", "artist", "end", "peak_rank"]].assign(reentry_probability=p.round(3))
            .sort_values("reentry_probability", ascending=False))


# --------------------------------------------------------------------------
# 2. comeback strength (regression on re-entries)
# --------------------------------------------------------------------------
def train_comeback_model(ep):
    ep = ep.sort_values(["key", "episode"]).copy()
    prev = ep.groupby("key")[["peak_rank", "retention_days", "spike_score"]].shift(1)
    ep["prev_peak_rank"], ep["prev_retention"], ep["prev_spike"] = prev.peak_rank, prev.retention_days, prev.spike_score
    re_ = ep[ep.is_reentry & ~ep.right_censored & ep.prev_peak_rank.notna()].copy()   # features known at re-entry time

    X = _base_features(re_)
    X["entry_rank"] = re_.entry_rank
    X["entry_pop"] = re_.entry_pop
    X["gap_days"] = re_.gap_days
    X["prior_pop"] = re_.prior_pop
    X["pop_vs_prior"] = re_.entry_pop - re_.prior_pop
    X["episode_number"] = re_.episode
    X["prev_peak_rank"] = re_.prev_peak_rank
    X["prev_retention"] = re_.prev_retention
    y = np.log1p(re_.retention_days)

    gkf = GroupKFold(n_splits=5)
    models = {
        "Baseline (mean)": None,
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                       subsample=0.8, random_state=SEED),
    }
    rows, oof_gb = [], np.zeros(len(X))
    for name, m in models.items():
        oof = np.zeros(len(X))
        for tr, te in gkf.split(X, y, groups=re_.key):
            if m is None:
                oof[te] = y.iloc[tr].mean()
            else:
                m.fit(X.iloc[tr], y.iloc[tr])
                oof[te] = m.predict(X.iloc[te])
        rows.append(dict(model=name, r2=r2_score(y, oof),
                         mae_days=mean_absolute_error(re_.retention_days, np.expm1(oof))))
        if name == "Gradient Boosting":
            oof_gb = oof
    res = pd.DataFrame(rows).round(3)
    final = models["Gradient Boosting"].fit(X, y)
    pi = permutation_importance(final, X, y, n_repeats=10, random_state=SEED)
    imp = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    pred = pd.DataFrame({"actual_days": re_.retention_days.values, "predicted_days": np.expm1(oof_gb).round(1),
                         "song": re_.song.values})
    return dict(results=res, importance=imp, model=final, features=list(X.columns), predictions=pred, n=len(X))


# --------------------------------------------------------------------------
# 3. song archetypes (K-Means)
# --------------------------------------------------------------------------
ARCH_FEATURES = ["reentries", "total_days", "best_rank", "mean_spike", "mean_pop_rate", "mean_recovery"]


def cluster_songs(songs, k_range=range(2, 7)):
    X = songs[ARCH_FEATURES].fillna(0).copy()
    X["total_days"] = np.log1p(X.total_days)             # tame heavy tails
    X["reentries"] = np.log1p(X.reentries)
    Z = StandardScaler().fit_transform(X)
    scores = {}
    for k in k_range:
        scores[k] = silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z))
    best_k = max(scores, key=scores.get)
    km = KMeans(best_k, n_init=10, random_state=SEED).fit(Z)
    out = songs.copy()
    out["cluster"] = km.labels_
    prof = out.groupby("cluster")[ARCH_FEATURES].mean().round(2)
    prof["size"] = out.groupby("cluster").size()
    # name clusters from their profiles: each rule claims the best-matching unassigned cluster
    names, free = {}, set(prof.index)
    rules = [("Fandom-driven revivers", "reentries", True), ("Long-run staples", "total_days", True),
             ("Fast risers", "mean_recovery", True), ("Chart-edge flickers", "best_rank", True)]
    for label, col, hi in rules:
        if not free:
            break
        c = prof.loc[sorted(free), col].idxmax() if hi else prof.loc[sorted(free), col].idxmin()
        names[c] = label
        free.discard(c)
    for c in free:
        names[c] = f"Mixed profile {c}"
    out["archetype"] = out.cluster.map(names)
    prof["archetype"] = prof.index.map(names)
    xy = PCA(2, random_state=SEED).fit_transform(Z)
    out["pc1"], out["pc2"] = xy[:, 0], xy[:, 1]
    return dict(songs=out, profile=prof, silhouette=pd.Series(scores).round(3), k=best_k)


# --------------------------------------------------------------------------
# 4. spike detector (Isolation Forest)
# --------------------------------------------------------------------------
SPIKE_FEATURES = ["rank_jump", "pop_rate", "peak_rank", "retention_days", "days_to_peak", "recovery_speed"]


def detect_spikes(ep, contamination=0.05):
    X = ep[SPIKE_FEATURES].fillna(0).copy()
    X["retention_days"] = np.log1p(X.retention_days)
    Z = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=SEED).fit(Z)
    out = ep.copy()
    out["anomaly_score"] = (-iso.score_samples(Z)).round(3)       # higher = more unusual
    out["is_ml_spike"] = iso.predict(Z) == -1
    return out
