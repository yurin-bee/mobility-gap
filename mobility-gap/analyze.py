"""
정읍시 고령운전자 이동공백 분석 - 2단계: 지수 산출, AI 군집 분석, 시각화

입력 (collect_data.py 결과, data 폴더)
  travel_times.csv, hubs.csv, hospitals.csv
  (선택) elderly.csv : 읍면동, 고령비율(0~1)  -> 있으면 지수에 반영

출력 (figures 폴더)
  fig1_map.png        마을회관별 이동공백지수 지도
  fig2_scatter.png    자동차 vs 대중교통 소요시간 + 군집
  fig3_emd.png        읍면동별 평균 이동공백비
  map_interactive.html 확대해서 볼 수 있는 지도
  ranking.csv         이동공백지수 순위표
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
FIG = sys.argv[2] if len(sys.argv) > 2 else "figures"
WATERMARK = os.environ.get("WATERMARK", "")
os.makedirs(FIG, exist_ok=True)

for f in ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
CLUSTER_COLORS = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626"]


def mark(fig):
    if WATERMARK:
        fig.text(0.5, 0.5, WATERMARK, ha="center", va="center", fontsize=40,
                 color="#dc2626", alpha=0.18, rotation=25, weight="bold")


t = pd.read_csv(f"{DATA}/travel_times.csv")
hubs = pd.read_csv(f"{DATA}/hubs.csv")
hosp = pd.read_csv(f"{DATA}/hospitals.csv")

# ---------- 1. 지표 계산 ----------
memo = t["대중교통_메모"].fillna("").astype(str)
near = memo.str.contains("700m")                                   # 거점 바로 옆: 걸어서 감
missing = t["대중교통_분"].isna() & (memo.str.contains("quota") | (memo == ""))   # 한도 때문에 못 받음
t["대중교통_경로없음"] = t["대중교통_분"].isna() & ~near & ~missing     # ODsay가 경로 없다고 답함
t["대중교통_실측"] = t["대중교통_분"].notna()
walk = t["직선거리_km"] * 1.3 * 1000 / 67     # 도로 우회계수 1.3, 노인 보행 4km/h(약 67m/분)
t.loc[near, "대중교통_분"] = t.loc[near, "직선거리_km"] * 1000 / 67

# ---------- 1-1. 못 받은 마을의 대중교통 시간 예측 (지도학습 회귀) ----------
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

t["대중교통_추정"] = False
MIN_TRAIN = 40
train = t[t["대중교통_실측"]].copy()
model_report = {"학습_마을수": len(train), "학습_읍면동수": train["읍면동"].nunique()}


def features(d):
    f = pd.DataFrame({"자동차_분": d["자동차_분"], "도로거리_km": d["도로거리_km"],
                      "직선거리_km": d["직선거리_km"],
                      "도로_직선비": d["도로거리_km"] / d["직선거리_km"].clip(lower=0.1),
                      "동지역": d["읍면동"].astype(str).str.endswith("동").astype(int)})
    for h in sorted(t["거점"].unique()):
        f[f"거점_{h}"] = (d["거점"] == h).astype(int)
    return f


if missing.any() and len(train) >= MIN_TRAIN:
    Xtr, ytr = features(train), train["대중교통_분"].astype(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cands = {"선형회귀": LinearRegression(),
             "랜덤포레스트": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42),
             "그래디언트부스팅": GradientBoostingRegressor(random_state=42)}
    scores = {}
    for name, mdl in cands.items():
        pred = cross_val_predict(mdl, Xtr, ytr, cv=cv)
        scores[name] = (np.mean(np.abs(pred - ytr)), pred)
    best = min(scores, key=lambda k: scores[k][0])
    mae, cvpred = scores[best]
    model = cands[best].fit(Xtr, ytr)
    t.loc[missing, "대중교통_분"] = model.predict(features(t[missing]))
    t.loc[missing, "대중교통_추정"] = True
    # 학습 범위를 벗어난 예측(외삽)은 표시
    t["외삽"] = t["대중교통_추정"] & (t["자동차_분"] > train["자동차_분"].max() * 1.1)
    model_report.update({"선택_모델": best, "교차검증_MAE_분": round(float(mae), 1),
                         "모델별_MAE": {k: round(float(v[0]), 1) for k, v in scores.items()},
                         "추정_마을수": int(missing.sum()), "외삽_마을수": int(t["외삽"].sum())})
    # 그림: 실측 vs 교차검증 예측
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    lim = max(ytr.max(), cvpred.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=GRID, lw=1)
    ax.scatter(ytr, cvpred, s=22, color="#2563eb", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("실제 대중교통 시간 (분, ODsay)", color=MUTED)
    ax.set_ylabel("모델 예측 (분, 5겹 교차검증)", color=MUTED)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f"대중교통 시간 예측 정확도 ({best}, 평균오차 {mae:.1f}분)",
                 loc="left", fontsize=12, color=INK, weight="bold")
    mark(fig)
    fig.savefig(f"{FIG}/fig5_model.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
else:
    t["외삽"] = False
    if missing.any():
        model_report["경고"] = (f"실측 {len(train)}곳으로는 예측하지 않음(최소 {MIN_TRAIN}곳). "
                              "못 받은 마을은 분석에서 제외")
        t = t[~missing].copy()

import json as _json
with open(f"{FIG}/model_report.json", "w", encoding="utf-8") as f:
    _json.dump(model_report, f, ensure_ascii=False, indent=2)
print("예측 모델:", model_report)

# 대중교통 경로가 아예 없으면 '걸어서 가는 시간'으로 대체하고,
# 버스보다 걷는 게 빠른 경우(거점 근처)도 걷는 시간을 씀. 대중교통이 자동차보다 빠를 수는 없게 아래로 제한
w = walk.reindex(t.index)
t["대중교통_분_보정"] = np.minimum(t["대중교통_분"].fillna(w), w).clip(lower=t["자동차_분"])
t["이동공백비"] = t["대중교통_분_보정"] / t["자동차_분"].clip(lower=5)
t["추가시간_분"] = t["대중교통_분_보정"] - t["자동차_분"]

el_path = f"{DATA}/elderly.csv"
if os.path.exists(el_path):
    el = pd.read_csv(el_path)
    t = t.merge(el[["읍면동", "고령비율"]], on="읍면동", how="left")
else:
    t["고령비율"] = np.nan


def mm(s):
    s = s.astype(float)
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0


# 이동공백지수 (0~100): 비율 50% + 추가시간 30% + 고령비율 20%
if t["고령비율"].notna().any():
    t["이동공백지수"] = 100 * (0.5 * mm(t["이동공백비"]) + 0.3 * mm(t["추가시간_분"])
                           + 0.2 * mm(t["고령비율"].fillna(t["고령비율"].mean())))
else:
    t["이동공백지수"] = 100 * (0.6 * mm(t["이동공백비"]) + 0.4 * mm(t["추가시간_분"]))

# ---------- 2. AI 군집 분석 (K-means) ----------
feat = ["자동차_분", "대중교통_분_보정", "이동공백비", "직선거리_km"]
X = StandardScaler().fit_transform(t[feat])
km = KMeans(n_clusters=4, n_init=20, random_state=42).fit(X)
t["군집"] = km.labels_
# 군집을 평균 이동공백지수 순으로 이름 붙이기 (0=양호 ... 3=심각)
# 군집은 '심각도'가 아니라 '유형'으로 이름 붙임 (특성으로 결정적으로 배정)
prof = t.groupby("군집").agg(car=("자동차_분", "mean"), ratio=("이동공백비", "mean"))
far = prof["car"].idxmax()                                   # 거점에서 가장 먼 무리
rest = prof.drop(far)
poor = rest["ratio"].idxmax()                                # 가깝지만 배율이 큰 무리
rest = rest.drop(poor)
near_c = rest["car"].idxmin()                                # 시내 인접
mid = rest.drop(near_c).index[0]
rename = {near_c: 0, mid: 1, poor: 2, far: 3}
t["군집"] = t["군집"].map(rename)
NAMES = ["A 시내 인접형", "B 중거리형", "C 근거리 연결불량형", "D 원거리 산간형"]
t["군집명"] = t["군집"].map(dict(enumerate(NAMES)))

summary = (t.groupby("군집명")
             .agg(마을수=("경로당명", "count"),
                  자동차_평균분=("자동차_분", "mean"),
                  대중교통_평균분=("대중교통_분_보정", "mean"),
                  평균_이동공백비=("이동공백비", "mean"),
                  경로없음=("대중교통_경로없음", "sum"))
             .round(1))
summary.to_csv(f"{FIG}/cluster_summary.csv", encoding="utf-8-sig")

rank = t.sort_values("이동공백지수", ascending=False)
rank[["읍면동", "경로당명", "자동차_분", "대중교통_분_보정", "대중교통_추정", "이동공백비",
      "추가시간_분", "이동공백지수", "군집명", "대중교통_메모"]].round(1) \
    .to_csv(f"{FIG}/ranking.csv", index=False, encoding="utf-8-sig")

# ---------- 3. 그림 1: 지도 ----------
fig, ax = plt.subplots(figsize=(8, 8.5))
sc = ax.scatter(t["x"], t["y"], c=t["이동공백지수"], cmap="YlOrRd", s=28,
                edgecolor="white", linewidth=0.4, vmin=0, vmax=100, zorder=2)
ax.scatter(hosp["XPos"], hosp["YPos"], s=6, color="#94a3b8", zorder=1, label="병원")
ax.scatter(hubs["x"], hubs["y"], marker="*", s=380, color="#2563eb",
           edgecolor="white", linewidth=1, zorder=3, label="의료 거점(시내)")
for _, r in t.groupby("읍면동")[["x", "y"]].mean().reset_index().iterrows():
    ax.text(r["x"], r["y"], r["읍면동"], fontsize=8, color=MUTED, ha="center", zorder=4)
placed = []
for _, r in rank.head(8).iterrows():   # 상위 마을 이름, 가까이 붙은 마을은 하나만
    if any(abs(r["x"] - px) < 0.02 and abs(r["y"] - py) < 0.015 for px, py in placed):
        continue
    placed.append((r["x"], r["y"]))
    right = r["x"] > t["x"].quantile(0.85)
    ax.annotate(r["경로당명"], (r["x"], r["y"]), xytext=(-6 if right else 6, 6), textcoords="offset points",
                ha="right" if right else "left", fontsize=8, color=INK, weight="bold", zorder=5)
    if len(placed) == 5:
        break
cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
cb.set_label("이동공백지수 (높을수록 취약)", color=MUTED)
ax.set_aspect(1 / np.cos(np.radians(t["y"].mean())))
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.legend(loc="lower left", frameon=False, fontsize=9)
ax.set_title("정읍시 경로당(마을회관)별 이동공백지수", loc="left", fontsize=14, color=INK,
             weight="bold", pad=24)
ax.text(0, 1.01, "의료 거점까지 대중교통 vs 자동차 소요시간 비교", transform=ax.transAxes,
        fontsize=9, color=MUTED, va="bottom")
mark(fig)
fig.savefig(f"{FIG}/fig1_map.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- 4. 그림 2: 산점도 ----------
fig, ax = plt.subplots(figsize=(8, 6))
lim = max(t["자동차_분"].max(), 1) * 1.1
for k, (ratio, lab) in enumerate([(1, "같음"), (2, "2배"), (3, "3배"), (5, "5배")]):
    xs = np.array([0, lim])
    ax.plot(xs, xs * ratio, color=GRID if ratio != 3 else "#fca5a5", lw=1, zorder=0)
    ax.text(lim, lim * ratio, f" {lab}", fontsize=8, color=MUTED, va="center", clip_on=False) \
        if lim * ratio < t["대중교통_분_보정"].max() * 1.1 else None
for c in range(4):
    d = t[t["군집"] == c]
    m = d[~d["대중교통_추정"]]; e = d[d["대중교통_추정"]]
    ax.scatter(m["자동차_분"], m["대중교통_분_보정"], s=22, color=CLUSTER_COLORS[c],
               alpha=0.85, edgecolor="white", linewidth=0.4, label=f"{NAMES[c]} ({len(d)})")
    ax.scatter(e["자동차_분"], e["대중교통_분_보정"], s=20, facecolor="none",
               edgecolor=CLUSTER_COLORS[c], linewidth=0.9, alpha=0.8)
ax.set_xlim(0, lim)
ax.set_ylim(0, t["대중교통_분_보정"].max() * 1.1)
ax.set_xlabel("자동차 소요시간 (분)", color=MUTED)
ax.set_ylabel("대중교통 소요시간 (분)", color=MUTED)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(title="K-means 군집", frameon=False, fontsize=9, title_fontsize=9, loc="upper left")
if t["대중교통_추정"].any():
    ax.text(0.99, 0.01, "채운 점: ODsay 실측 · 빈 점: 모델 추정", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED)
ax.set_title("면허를 반납하면, 병원 가는 시간은 몇 배가 되는가", loc="left",
             fontsize=14, color=INK, weight="bold")
mark(fig)
fig.savefig(f"{FIG}/fig2_scatter.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- 5. 그림 3: 읍면동별 막대 ----------
emd = t.groupby("읍면동")["이동공백비"].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, max(4, len(emd) * 0.3)))
colors = ["#dc2626" if v >= 3 else "#f59e0b" if v >= 2 else "#94a3b8" for v in emd]
ax.barh(emd.index, emd.values, color=colors, height=0.65)
for i, v in enumerate(emd.values):
    ax.text(v + 0.05, i, f"{v:.1f}배", va="center", fontsize=8, color=INK)
ax.axvline(3, color="#fca5a5", lw=1, ls="--")
ax.set_xlabel("평균 이동공백비 (대중교통 ÷ 자동차)", color=MUTED)
ax.tick_params(axis="y", length=0)
ax.set_title("읍·면·동별 평균 이동공백비", loc="left", fontsize=14, color=INK, weight="bold")
mark(fig)
fig.savefig(f"{FIG}/fig3_emd.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- 6. 인터랙티브 지도 ----------
try:
    import folium
    import branca.colormap as cm
    m = folium.Map(location=[t["y"].mean(), t["x"].mean()], zoom_start=11, tiles="OpenStreetMap")
    cmap = cm.LinearColormap(["#fff7bc", "#fd8d3c", "#bd0026"], vmin=0, vmax=100,
                             caption="이동공백지수")
    for _, r in t.iterrows():
        folium.CircleMarker(
            [r["y"], r["x"]], radius=5, color=cmap(r["이동공백지수"]), fill=True,
            fill_opacity=0.9, weight=0.5,
            tooltip=(f"<b>{r['경로당명']}</b> ({r['읍면동']})<br>"
                     f"자동차 {r['자동차_분']:.0f}분 / 대중교통 "
                     f"{'경로 없음' if r['대중교통_경로없음'] else f'{r['대중교통_분_보정']:.0f}분'}<br>"
                     f"이동공백비 {r['이동공백비']:.1f}배 · 지수 {r['이동공백지수']:.0f}")).add_to(m)
    for _, h in hubs.iterrows():
        folium.Marker([h["y"], h["x"]], tooltip=f"의료 거점 (병원 {h['병원수']}곳)",
                      icon=folium.Icon(color="blue", icon="plus")).add_to(m)
    cmap.add_to(m)
    m.save(f"{FIG}/map_interactive.html")
except ImportError:
    pass

print(summary.to_string())
print("\n이동공백지수 상위 10개 마을")
print(rank[["읍면동", "경로당명", "자동차_분", "대중교통_분_보정", "이동공백비", "이동공백지수"]]
      .head(10).round(1).to_string(index=False))
