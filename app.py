import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import ml
from pipeline import clean, build_episodes, build_songs, album_advantage, W_FANDOM

st.set_page_config(page_title="K-Pop Comeback Momentum", layout="wide")
CSV = "Atlantic_South_Korea.csv"


@st.cache_data
def load(min_gap):
    df, log = clean(CSV)
    return df, build_episodes(df, min_gap), log


@st.cache_data(show_spinner="Training machine-learning models...")
def run_ml(min_gap):
    df_, ep_, _ = load(min_gap)
    sg = build_songs(ep_)
    reent = ml.train_reentry_model(df_, ep_)
    return dict(
        reentry=reent,
        current=ml.score_current_exits(df_, ep_, reent),
        comeback=ml.train_comeback_model(ep_),
        clusters=ml.cluster_songs(sg),
        spikes=ml.detect_spikes(ep_),
    )


st.title("South Korea Top 50: Comeback Momentum & Fandom Intensity")

# ---------------- Sidebar ----------------
sb = st.sidebar
min_gap = sb.slider("Re-entry gap (days absent)", 1, 14, 3,
                    help="A song must be off the chart this many days before a return counts as a re-entry.")
df, ep_all, log = load(min_gap)
dmin, dmax = df.date.min().date(), df.date.max().date()
dr = sb.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if len(dr) != 2:
    st.stop()
types = sb.multiselect("Album type", ["single", "album"], ["single", "album"])
artists = sb.multiselect("Artist", sorted(ep_all.artist.unique()))
songs_sel = sb.multiselect("Song", sorted(ep_all.song.unique()))
min_re = sb.slider("Min re-entries per song", 0, 10, 0)
meaningful = sb.slider("Ignore episodes that never peak above rank", 1, 50, 50,
                       help="50 = keep everything. Lower values drop edge-of-chart flicker around #50.")
exclude_lc = sb.checkbox("Exclude first episodes already on chart at the start of the data", True)
with sb.expander("Fandom score weights"):
    w1 = st.slider("Re-entry frequency", 0.0, 1.0, W_FANDOM[0])
    w2 = st.slider("Spike sharpness", 0.0, 1.0, W_FANDOM[1])
    w3 = st.slider("Recovery speed", 0.0, 1.0, W_FANDOM[2])

ep = ep_all[(ep_all.start >= pd.Timestamp(dr[0])) & (ep_all.start <= pd.Timestamp(dr[1]))]
ep = ep[ep.album_type.isin(types) & (ep.peak_rank <= meaningful)]
if artists:
    ep = ep[ep.artist.isin(artists)]
if songs_sel:
    ep = ep[ep.song.isin(songs_sel)]
if exclude_lc:
    ep = ep[~ep.left_censored]
ep = ep[ep.groupby("key").episode.transform("max") - 1 >= min_re]
if ep.empty:
    st.warning("No episodes match these filters.")
    st.stop()

songs = build_songs(ep)
tot = w1 + w2 + w3 or 1
pct = lambda s: s.rank(pct=True)
songs["fandom_score"] = (100 * (w1 * pct(songs.reentries) + w2 * pct(songs.mean_pop_rate.clip(lower=0))
                                + w3 * pct(songs.mean_recovery)) / tot).round(1)
re_ = ep[ep.is_reentry]

# ---------------- KPIs ----------------
c = st.columns(6)
c[0].metric("Songs with re-entries", int((songs.reentries > 0).sum()))
c[1].metric("Avg re-entries / song", f"{songs.reentries.mean():.2f}")
c[2].metric("Momentum spike (re-entry)", f"{re_.spike_score.mean():.1f}" if len(re_) else "n/a")
c[3].metric("Retention days (re-entry)", f"{re_.retention_days.mean():.1f}" if len(re_) else "n/a")
c[4].metric("Rank recovery (ranks/day)", f"{re_.recovery_speed.mean():.1f}" if len(re_) else "n/a")
c[5].metric("Album comeback index", f"{album_advantage(ep):.2f}" if len(re_) else "n/a",
            help=">1: album tracks' comebacks are stronger than singles'")

t1, t2, t3, t4, t5, t7, t6 = st.tabs(["Re-entry timeline", "Momentum spikes", "Comeback vs first entry",
                                      "Content attributes", "Fandom leaderboard", "Machine learning", "Data notes"])

# ---------------- Timeline ----------------
with t1:
    cand = songs.sort_values(["reentries", "total_days"], ascending=False)
    pick = st.selectbox("Song", cand.key, format_func=lambda k: f"{cand.set_index('key').loc[k, 'song']} - "
                        f"{cand.set_index('key').loc[k, 'artist']} ({int(cand.set_index('key').loc[k, 'reentries'])} re-entries)")
    h = df[(df.key == pick)].sort_values("date")
    e = ep_all[ep_all.key == pick]
    fig = go.Figure(go.Scatter(x=h.date, y=h.position, mode="markers+lines", line=dict(width=1), name="Rank"))
    for _, r in e.iterrows():
        fig.add_vrect(x0=r.start, x1=r.end, opacity=0.12, line_width=0,
                      fillcolor="green" if r.is_reentry else "blue")
    fig.update_yaxes(autorange="reversed", range=[51, 1], title="Chart position (1 = top)")
    fig.update_layout(title="Rank history (blue = first entry, green = re-entries)", height=380)
    st.plotly_chart(fig, width='stretch')
    st.dataframe(e[["episode", "start", "end", "retention_days", "entry_rank", "peak_rank",
                    "gap_days", "spike_score"]], hide_index=True, width='stretch')
    st.subheader("Re-entry counts")
    st.plotly_chart(px.histogram(songs, x="reentries", nbins=20, title="Songs by number of re-entries"),
                    width='stretch')
    if len(re_):
        st.plotly_chart(px.histogram(re_[re_.gap_days <= 60], x="gap_days", nbins=30,
                                     title="Days off-chart before re-entry (<= 60 days shown)"),
                        width='stretch')

# ---------------- Spikes ----------------
with t2:
    a, b = st.columns(2)
    a.plotly_chart(px.scatter(ep, x="rank_jump", y="pop_rate", color="is_reentry", hover_data=["song", "artist"],
                              title="Rank jump vs popularity change rate"), width='stretch')
    b.plotly_chart(px.scatter(ep, x="retention_days", y="spike_score", color="is_reentry", log_x=True,
                              hover_data=["song", "artist"], title="Spike strength vs days retained"),
                   width='stretch')
    st.subheader("Strongest episodes")
    st.dataframe(ep.sort_values("spike_score", ascending=False).head(25)[
        ["song", "artist", "episode", "start", "entry_rank", "peak_rank", "retention_days",
         "decay_slope", "volatility", "spike_score"]], hide_index=True, width='stretch')

# ---------------- Comeback vs first ----------------
with t3:
    ep["type"] = np.where(ep.is_reentry, "Re-entry", "First entry")
    m = ep.groupby("type")[["spike_score", "entry_rank", "peak_rank", "retention_days",
                            "days_to_peak", "volatility"]].agg(["mean", "median"]).round(2)
    st.dataframe(m, width='stretch')
    a, b = st.columns(2)
    a.plotly_chart(px.box(ep, x="type", y="spike_score", title="Spike score"), width='stretch')
    b.plotly_chart(px.box(ep, x="type", y="retention_days", log_y=True, title="Days retained (log)"),
                   width='stretch')
    st.caption("Most re-entries are short returns near the bottom of the chart. Use the "
               "'never peak above rank' filter to isolate meaningful comebacks.")

# ---------------- Content ----------------
with t4:
    base = re_ if len(re_) else ep
    st.caption(f"Based on {len(base)} {'re-entry' if len(re_) else 'all'} episodes.")
    a, b = st.columns(2)
    a.plotly_chart(px.box(base, x="album_type", y="spike_score", title="Single vs album track"),
                   width='stretch')
    b.plotly_chart(px.box(base, x="is_explicit", y="spike_score", title="Explicit vs clean"),
                   width='stretch')
    a, b = st.columns(2)
    a.plotly_chart(px.scatter(base, x="total_tracks", y="spike_score", trendline="lowess",
                              title="Album size vs spike"), width='stretch')
    b.plotly_chart(px.scatter(base, x="duration_min", y="spike_score", trendline="lowess",
                              title="Duration (min) vs spike"), width='stretch')
    rows = []
    s_, a_ = base[base.album_type == "single"].spike_score, base[base.album_type == "album"].spike_score
    if len(s_) > 1 and len(a_) > 1:
        rows.append(("Single vs album (Mann-Whitney)", stats.mannwhitneyu(s_, a_).pvalue))
    x_, c_ = base[base.is_explicit].spike_score, base[~base.is_explicit].spike_score
    if len(x_) > 1 and len(c_) > 1:
        rows.append(("Explicit vs clean (Mann-Whitney)", stats.mannwhitneyu(x_, c_).pvalue))
    if len(base) > 3:
        rows.append(("Duration vs spike (Spearman)", stats.spearmanr(base.duration_min, base.spike_score).pvalue))
        rows.append(("Album size vs spike (Spearman)", stats.spearmanr(base.total_tracks, base.spike_score).pvalue))
    st.dataframe(pd.DataFrame(rows, columns=["Test", "p-value"]).round(4), hide_index=True)

# ---------------- Leaderboard ----------------
with t5:
    top = songs.sort_values("fandom_score", ascending=False).head(25)
    top["label"] = top.song.str.slice(0, 32) + " - " + top.artist.str.slice(0, 20)
    st.plotly_chart(px.bar(top.iloc[::-1], x="fandom_score", y="label", orientation="h", color="reentries",
                           height=700, title="Fandom Intensity Proxy Score (0-100)"), width='stretch')
    st.dataframe(songs.sort_values("fandom_score", ascending=False)[
        ["song", "artist", "reentries", "total_days", "best_rank", "mean_spike", "mean_recovery", "fandom_score"]
    ].round(2), hide_index=True, width='stretch')
    st.download_button("Download songs table", songs.to_csv(index=False), "songs.csv")


# ---------------- Machine learning ----------------
with t7:
    R = run_ml(min_gap)
    st.caption("Models are trained on the full dataset for the chosen re-entry gap; sidebar filters do not change them.")
    m1, m2, m3, m4 = st.tabs(["Re-entry predictor", "Comeback strength", "Song archetypes", "ML spike detector"])

    with m1:
        rr = R["reentry"]
        i = rr["info"]
        st.markdown(f"**Question:** when a song leaves the Top 50, will it come back within {i['horizon']} days?")
        st.write(f"Trained on the earliest {i['n_train']} exits and tested on the latest {i['n_test']} "
                 f"(time-ordered split, so the test set is genuinely the future). "
                 f"About {i['positive_rate_test']:.0%} of test exits did return.")
        st.dataframe(rr["results"], hide_index=True, width="stretch")
        st.caption("ROC-AUC 0.5 = guessing, 1.0 = perfect. Random Forest should clearly beat the baseline.")
        imp = rr["importance"].head(10).iloc[::-1]
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                               title="What drives re-entry? (permutation importance, drop in ROC-AUC)"),
                        width="stretch")
        st.subheader("Songs that just left the chart: likelihood of returning")
        if R["current"].empty:
            st.info("No songs are inside the prediction window at the end of the data.")
        else:
            st.dataframe(R["current"].head(15), hide_index=True, width="stretch")

    with m2:
        cb = R["comeback"]
        st.markdown(f"**Question:** can we predict how long a re-entry stays on the chart? ({cb['n']} re-entries, "
                    "5-fold cross-validation grouped by song so a song never appears in both train and test.)")
        st.dataframe(cb["results"], hide_index=True, width="stretch")
        best_r2 = cb["results"].r2.max()
        if best_r2 < 0.1:
            st.warning("Honest result: none of the models beat simply predicting the average (R2 near or below 0). "
                       "Comeback length is not predictable from the information available at re-entry time. "
                       "This suggests it depends on outside events (comebacks, performances, awards) that are not in the data.")
        imp = cb["importance"].head(8).iloc[::-1]
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                               title="Feature importance (for reference only if the model has no skill)"),
                        width="stretch")

    with m3:
        cl = R["clusters"]
        st.markdown(f"**K-Means** grouped songs into **{cl['k']} archetypes** (k chosen by silhouette score).")
        st.plotly_chart(px.scatter(cl["songs"], x="pc1", y="pc2", color="archetype",
                                   hover_data=["song", "artist", "reentries", "total_days"],
                                   title="Songs in 2D (PCA of momentum features)"), width="stretch")
        st.dataframe(cl["profile"][["archetype", "size", "reentries", "total_days", "best_rank",
                                    "mean_spike", "mean_recovery"]], width="stretch")
        pick = st.selectbox("Show songs in archetype", sorted(cl["songs"].archetype.unique()))
        st.dataframe(cl["songs"][cl["songs"].archetype == pick].sort_values("fandom_score", ascending=False)[
            ["song", "artist", "reentries", "total_days", "best_rank", "fandom_score"]].head(30),
            hide_index=True, width="stretch")
        st.caption("Silhouette scores by k: " + ", ".join(f"k={k}: {v}" for k, v in cl["silhouette"].items()))

    with m4:
        sp = R["spikes"]
        st.markdown("**Isolation Forest** flags the ~5% most unusual episodes across rank jump, popularity rate, "
                    "peak rank, duration and recovery speed. No hand-set thresholds.")
        st.plotly_chart(px.scatter(sp, x="rank_jump", y="retention_days", color="is_ml_spike", log_y=True,
                                   hover_data=["song", "artist", "anomaly_score"],
                                   title="Flagged episodes (True) vs normal"), width="stretch")
        st.dataframe(sp[sp.is_ml_spike].sort_values("anomaly_score", ascending=False)[
            ["song", "artist", "episode", "entry_rank", "peak_rank", "retention_days", "anomaly_score"]].head(25),
            hide_index=True, width="stretch")

# ---------------- Notes ----------------
with t6:
    st.write("**Validation log**")
    st.code("\n".join(log))
    st.markdown("""
- Rank is censored at 50: absent songs are treated as rank 51 for rank-jump.
- Missing snapshot dates are skipped, not counted as chart exits.
- Episodes already on the chart on day 1 are left-censored; episodes still running on the last day are right-censored.
- Popularity is a smoothed score, not stream counts, so spike sharpness is muted.
""")
