"""
보고서용 그림 (analyze.py 실행 후)

    python report_figures.py            # figures/report/ 에 그림 저장

입력
  figures/villages_scored.csv   analyze.py가 만든 마을별 결과
  figures/cv_predictions.csv    회귀모델 교차검증 예측
  figures/model_report.json
  data/hubs.csv, data/hospitals.csv, data/elderly.csv
  data/jeongeup_admdong.geojson 정읍시 행정동 경계
      (통계청 SGIS 행정동 경계, vuski/admdongkor ver20261001 가공본, CC BY 4.0)

전국 통계와 배출계수는 아래 NATIONAL, EF 에 출처와 함께 적어 두었다.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

FIG = "figures"
OUT = "figures/report"
os.makedirs(OUT, exist_ok=True)

# ---------------- 스타일 ----------------
for f in ["Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "xtick.color": "#898781", "ytick.color": "#898781", "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5, "xtick.major.size": 0, "ytick.major.size": 0,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAR, BUS = "#2a78d6", "#eb6834"          # 자동차 = 파랑, 대중교통 = 주황 (모든 그림 공통)
EV = "#1baf7a"                            # 전기 전환 시나리오
GRAY = "#b9b8b1"
ORANGES = LinearSegmentedColormap.from_list(
    "gap", ["#fde6db", "#f7b393", "#eb6834", "#c2481c", "#7d2a0c"])
DPI = 220


def title(fig, main, sub=None, x=0.035, y=0.985):
    fig.text(x, y, main, fontsize=15.5, weight="bold", color=INK, va="top")
    if sub:
        fig.text(x, y - 0.058 * (6 / fig.get_figheight()), sub, fontsize=10.2, color=INK2, va="top")


def source(fig, text, y=0.012):
    fig.text(0.035, y, text, fontsize=7.8, color=MUTED, va="bottom")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", name)


def hgrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


# ---------------- 데이터 ----------------
t = pd.read_csv(f"{FIG}/villages_scored.csv")
for c in ["이동공백비", "추가시간_분", "자동차_분", "대중교통_분_보정", "이동공백지수"]:
    t[c + "_r"] = t[c].round(1)            # 보고서·순위표와 같은 반올림 기준
hubs = pd.read_csv("data/hubs.csv")
hosp = pd.read_csv("data/hospitals.csv")
eld = pd.read_csv("data/elderly.csv")
geo = json.load(open("data/jeongeup_admdong.geojson", encoding="utf-8"))
cvp = pd.read_csv(f"{FIG}/cv_predictions.csv")
mrep = json.load(open(f"{FIG}/model_report.json", encoding="utf-8"))
TYPES = ["A 시내 인접형", "B 중거리형", "C 근거리 연결불량형", "D 원거리 산간형"]

N = len(t)
n3 = int((t["이동공백비_r"] >= 3).sum())
n30 = int((t["추가시간_분_r"] >= 30).sum())
n60 = int((t["대중교통_분_보정_r"] >= 60).sum())
car_m, bus_m = t["자동차_분"].mean(), t["대중교통_분_보정"].mean()

# 전국 통계 (출처는 그림 하단에도 표기)
NATIONAL = {
    # 도로교통공단, 고령운전자(65세 이상) 교통사고 현황(가해자 기준)
    "acc_year": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "acc": [30012, 33239, 31072, 31841, 34652, 39614, 42369],
    # 경찰청·국회입법조사처 자료(복기왕 의원실 분석, 2026.7 보도)
    "drivers_2021": 401, "drivers_2025": 563, "old_driver_growth": 40.2, "all_driver_growth": 3.6,
    "share_2021": 15.7, "share_2025": 23.7, "return_rate": 2.51,
    # 2024 국가 온실가스 잠정배출량 (만 톤, 2006 IPCC 기준)
    "ghg_total": 69158, "ghg_transport": 9746,
}

# 배출계수
#  휘발유·LPG: 에너지공단 EG-TIPS 순발열량 × 탄소배출계수(tC/TJ) × 44/12
#  전력: 2023년도 국가 전력배출계수 0.4173 tCO2eq/MWh (기후에너지환경부)
#  연비·전비는 대표값을 '가정'으로 둠 (그림에 표기)
GAS_KG_L = 30.1e-6 * 19.731 * 44 / 12 * 1000     # TJ/L × tC/TJ × 44/12 → kgCO2/L = 2.18
LPG_KG_L = 45.5e-6 * 18.094 * 44 / 12 * 1000 * 0.58   # 부탄 밀도 0.58 kg/L 가정 → 1.75 kgCO2/L
GRID_KG_KWH = 0.4173
EF = {  # kgCO2/km (차량 1대 기준)
    "gas_car": GAS_KG_L / 12.0,      # 휘발유 승용 12 km/L
    "lpg_taxi": LPG_KG_L / 9.5,      # LPG 택시 9.5 km/L
    "ev_taxi": GRID_KG_KWH / 5.5,    # 전기 승용 5.5 km/kWh
    "ev_van": GRID_KG_KWH / 4.0,     # 소형 전기 승합 4.0 km/kWh
}


# =====================================================================
# 그림 1. 핵심 수치
# =====================================================================
def fig_key_numbers():
    fig = plt.figure(figsize=(11, 2.7))
    tiles = [
        (f"{N}곳", "분석한 정읍시 마을(경로당)", "경로당 707곳 중 좌표 확보"),
        (f"{n3/N*100:.0f}%", f"대중교통이 자동차의 3배 이상", f"{n3}곳"),
        (f"+{bus_m - car_m:.0f}분", "의료 거점까지 평균 추가시간", f"자동차 {car_m:.1f}분 → 대중교통 {bus_m:.1f}분"),
        (f"{n60}곳", "대중교통으로 편도 1시간 이상", f"왕복이면 2시간 이상"),
    ]
    w = 0.225
    for i, (big, lab, small) in enumerate(tiles):
        x0 = 0.03 + i * (w + 0.018)
        fig.patches.append(FancyBboxPatch((x0, 0.04), w, 0.72, boxstyle="round,pad=0,rounding_size=0.025",
                                          transform=fig.transFigure, fc="#f7f6f3", ec="none"))
        col = BUS if i else INK
        fig.text(x0 + 0.02, 0.56, big, fontsize=30, weight="bold", color=col, va="center")
        fig.text(x0 + 0.02, 0.29, lab, fontsize=10.5, color=INK, va="center", weight="medium")
        fig.text(x0 + 0.02, 0.15, small, fontsize=9, color=MUTED, va="center")
    fig.text(0.03, 0.95, "면허를 반납하면, 병원은 세 배 멀어진다", fontsize=16, weight="bold", color=INK, va="top")
    save(fig, "R01_핵심수치")


# =====================================================================
# 그림 2. 전국 맥락: 고령운전자 사고 증가 vs 낮은 반납률
# =====================================================================
def fig_national():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.28})
    yrs, acc = NATIONAL["acc_year"], np.array(NATIONAL["acc"])
    bars = a.bar([str(y) for y in yrs], acc / 1e4, width=0.62, color=[GRAY] * 6 + [BUS], zorder=2)
    for r in bars:
        r.set_capstyle("round")
    for x, v in zip(range(len(yrs)), acc):
        a.text(x, v / 1e4 + 0.08, f"{v/1e4:.1f}", ha="center", va="bottom", fontsize=9,
               color=INK if x == len(yrs) - 1 else INK2, weight="bold" if x == len(yrs) - 1 else None)
    a.set_ylim(0, 5.0); hgrid(a)
    a.set_ylabel("사고 건수 (만 건)", color=MUTED, fontsize=9.5)
    inc = (acc[-1] / acc[0] - 1) * 100
    a.set_title(f"65세 이상 운전자 가해 교통사고  {inc:+.0f}% (2018→2024)", loc="left",
                fontsize=11.5, color=INK, weight="bold", pad=10)
    a.spines["bottom"].set_visible(True)

    # 오른쪽: 고령운전자 증가율 vs 전체 운전자 증가율, 반납률
    g_old = NATIONAL["old_driver_growth"]   # 보도 원문 수치(만 명 단위 반올림 전 기준)
    vals = [g_old, NATIONAL["all_driver_growth"]]
    labs = ["고령 운전자", "전체 운전자"]
    b.barh(labs[::-1], vals[::-1], height=0.36, color=[GRAY, BUS], zorder=2)
    b.set_ylim(-1.6, 1.5); b.spines["bottom"].set_visible(False)
    for i, v in enumerate(vals[::-1]):
        b.text(v + 1, i, f"+{v:.1f}%", va="center", fontsize=10.5, color=INK, weight="bold")
    b.set_xlim(0, 52); b.set_xticks([])
    b.tick_params(axis="y", labelsize=10.5, colors=INK)
    b.set_title("운전자 수 증가율 (2021→2025)", loc="left", fontsize=11.5, color=INK, weight="bold", pad=10)
    b.text(0, 1.32, f"고령 운전자 {NATIONAL['drivers_2021']}만 명 → {NATIONAL['drivers_2025']}만 명", fontsize=9,
           color=MUTED)
    b.axhline(-0.75, color=GRID, lw=0.8)
    b.text(0, -1.1, f"면허 자진반납률은 {NATIONAL['return_rate']}%", fontsize=12.5, color=INK, weight="bold")
    b.text(0, -1.38, "운전을 멈추면 대안이 없는 농촌일수록\n반납이 더 어렵다", fontsize=9.2, color=INK2, va="center")
    # 반납률 콜아웃
    fig.suptitle("")
    fig.text(0.035, 1.08, "고령 운전자는 빠르게 늘고, 사고도 늘지만, 면허는 반납되지 않는다",
             fontsize=15.5, weight="bold", color=INK, va="top")
    source(fig, "출처: 도로교통공단 고령운전자 교통사고 현황(가해자 기준)  ·  경찰청·국회입법조사처 자료(2026.7 보도)",
           y=-0.04)
    save(fig, "R02_전국맥락")


# =====================================================================
# 그림 4. 지도 (행정동 경계 + 마을별 이동공백지수)
# =====================================================================
def polys(feature):
    g = feature["geometry"]
    rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
    return [np.array(r) for r in rings]


def fig_map():
    fig, ax = plt.subplots(figsize=(9.6, 8.7))
    patches, names = [], []
    for f in geo["features"]:
        for r in polys(f):
            patches.append(MplPolygon(r, closed=True))
    ax.add_collection(PatchCollection(patches, fc="#f4f3ef", ec="white", lw=1.6, zorder=1))
    # 시내(동 지역) 살짝 진하게
    dpat = [MplPolygon(r, closed=True) for f in geo["features"]
            if f["properties"]["읍면동"].endswith("동") for r in polys(f)]
    ax.add_collection(PatchCollection(dpat, fc="#e9e8e2", ec="white", lw=1.6, zorder=1))
    # 읍면동 이름
    for f in geo["features"]:
        rs = polys(f); big = max(rs, key=len)
        nm = f["properties"]["읍면동"]
        if nm.endswith("동"):
            continue
        cx, cy = big[:, 0].mean(), big[:, 1].mean()
        ax.text(cx, cy, nm, ha="center", va="center", fontsize=8.6, color="#9a9890", zorder=2)

    s = t.sort_values("이동공백지수")
    sc = ax.scatter(s["x"], s["y"], c=s["이동공백지수"], cmap=ORANGES, vmin=20, vmax=90, s=26,
                    edgecolor="white", linewidth=0.45, zorder=4)
    # 의료 거점
    ax.scatter(hubs["x"], hubs["y"], s=230, marker="o", fc="white", ec=CAR, lw=2.4, zorder=7)
    ax.scatter(hubs["x"], hubs["y"], s=45, marker="+", color=CAR, lw=2, zorder=8)
    # 상위 마을 라벨
    top = t.sort_values("이동공백지수", ascending=False).head(6)
    offs = [(-0.075, 0.03), (-0.085, -0.018), (0.03, 0.035), (0.035, -0.03), (-0.08, 0.05), (0.04, 0.02)]
    for (_, r), (dx, dy) in zip(top.iterrows(), offs):
        ax.annotate(f"{r['경로당명'].replace('경로당','')} {r['이동공백지수']:.0f}점\n"
                    f"차 {r['자동차_분']:.0f}분 → 대중교통 {r['대중교통_분_보정']:.0f}분",
                    (r["x"], r["y"]), (r["x"] + dx, r["y"] + dy), fontsize=8.1, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7), zorder=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    ax.set_aspect(1 / np.cos(np.radians(35.6)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.autoscale_view()
    # 색 막대
    cax = fig.add_axes([0.62, 0.135, 0.26, 0.016])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.5, colors=MUTED)
    cb.set_label("이동공백지수 (높을수록 취약)", fontsize=9, color=INK2, labelpad=4)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=CAR, mew=2.2, ms=11,
                              label="의료 거점 (DBSCAN, 3곳)"),
                       Line2D([], [], marker="o", ls="", mfc=BUS, mec="white", ms=7, label="경로당(마을회관)")],
              loc="lower left", frameon=False, fontsize=9, bbox_to_anchor=(0.0, 0.0))
    fig.text(0.07, 0.975, "시내에서 멀어질수록, 그리고 서쪽 평야의 연결 나쁜 마을일수록 공백이 크다",
             fontsize=14.5, weight="bold", color=INK, va="top")
    fig.text(0.07, 0.94, f"정읍시 경로당 {N}곳의 이동공백지수 (대중교통 ÷ 자동차 50% + 추가시간 30% + 고령비율 20%)",
             fontsize=9.8, color=INK2, va="top")
    source(fig, "경계: 통계청 SGIS 행정동 경계(vuski/admdongkor 가공, CC BY 4.0)  ·  소요시간: 카카오모빌리티, ODsay  ·  고령비율: 행정안전부 2026.9",
           y=0.06)
    save(fig, "R04_지도")


# =====================================================================
# 그림 5. 유형별 자동차 vs 대중교통 (작은 다중 그림)
# =====================================================================
def fig_scatter_types():
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 4.0), sharex=True, sharey=True,
                            gridspec_kw={"wspace": 0.08})
    lim_x, lim_y = 50, 120
    for ax, ty in zip(axs, TYPES):
        xs = np.array([0, lim_x])
        for k, lab in [(1, "같음"), (2, "2배"), (3, "3배")]:
            ax.plot(xs, xs * k, color=GRID if k != 3 else "#f2c4ae", lw=1 if k != 3 else 1.4, zorder=1)
            xe = min(lim_x, lim_y / k)
            ax.text(xe * 0.97, xe * k * 0.97 + (2 if k == 3 else 0), lab, fontsize=8, color=MUTED if k != 3 else BUS,
                    ha="right", va="bottom")
        o = t[t["군집명"] != ty]
        ax.scatter(o["자동차_분"], o["대중교통_분_보정"], s=9, color="#e4e3dd", lw=0, zorder=2)
        d = t[t["군집명"] == ty]
        m = d["대중교통_실측"]
        ax.scatter(d.loc[~m, "자동차_분"], d.loc[~m, "대중교통_분_보정"], s=15, fc="none", ec=BUS, lw=0.8, zorder=3)
        ax.scatter(d.loc[m, "자동차_분"], d.loc[m, "대중교통_분_보정"], s=18, fc=BUS, ec="white", lw=0.5, zorder=4)
        ax.set_xlim(0, lim_x); ax.set_ylim(0, lim_y)
        ax.grid(color="#f0efea", lw=0.6); ax.set_axisbelow(True)
        ax.spines["bottom"].set_visible(True)
        ax.set_title(f"{ty}", loc="left", fontsize=11, weight="bold", color=INK, pad=24)
        ax.text(0, 1.035, f"{len(d)}곳 · 평균 {d['자동차_분'].mean():.0f}분 → {d['대중교통_분_보정'].mean():.0f}분 "
                f"({d['이동공백비'].mean():.1f}배)", transform=ax.transAxes, fontsize=8.6, color=INK2)
        ax.set_xlabel("자동차 (분)", color=MUTED, fontsize=9)
    axs[0].set_ylabel("대중교통 (분)", color=MUTED, fontsize=9)
    fig.text(0.035, 1.2, "K-means가 찾은 네 가지 마을 유형", fontsize=15.5, weight="bold", color=INK, va="top")
    fig.text(0.035, 1.115, "‘근거리 연결불량형’은 시내에서 차로 9분인데도 대중교통은 3.6배 걸린다. 정류장 거리만 보는 기존 기준으로는 보이지 않는 사각지대",
             fontsize=9.8, color=INK2, va="top")
    fig.text(0.035, -0.07, "● 채운 점: ODsay 실측   ○ 빈 점: 회귀모델 추정   회색: 다른 유형", fontsize=8.6, color=MUTED)
    save(fig, "R05_유형별산점도")


# =====================================================================
# 그림 6. 유형별 평균 소요시간 덤벨
# =====================================================================
def fig_type_dumbbell():
    g = t.groupby("군집명").agg(car=("자동차_분", "mean"), bus=("대중교통_분_보정", "mean"),
                              n=("경로당명", "size"), r=("이동공백비", "mean")).reindex(TYPES)
    fig, ax = plt.subplots(figsize=(10, 3.9))
    y = np.arange(len(g))[::-1]
    for yi, (ty, r) in zip(y, g.iterrows()):
        ax.plot([r.car, r.bus], [yi, yi], color="#f2c4ae", lw=6, solid_capstyle="round", zorder=1)
        ax.scatter(r.car, yi, s=110, color=CAR, ec="white", lw=1.6, zorder=3)
        ax.scatter(r.bus, yi, s=110, color=BUS, ec="white", lw=1.6, zorder=3)
        ax.text(r.car - 1.8, yi, f"{r.car:.0f}분", ha="right", va="center", fontsize=9.5, color=INK2)
        ax.text(r.bus + 1.8, yi, f"{r.bus:.0f}분", ha="left", va="center", fontsize=9.5, color=INK2)
        ax.text((r.car + r.bus) / 2, yi + 0.28, f"+{r.bus - r.car:.0f}분 · {r.r:.1f}배", ha="center",
                fontsize=9.2, color=INK, weight="bold")
    ax.set_yticks(y); ax.set_yticklabels([f"{ty}  ({int(r.n)}곳)" for ty, r in g.iterrows()],
                                         fontsize=10.2, color=INK)
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(g) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.set_xlabel("의료 거점까지 편도 소요시간 (분)", color=MUTED, fontsize=9.5)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=CAR, ms=9, label="자동차"),
                       Line2D([], [], marker="o", ls="", color=BUS, ms=9, label="대중교통(도보·대기·환승 포함)")],
              loc="lower right", frameon=False, fontsize=9.2, ncol=2, bbox_to_anchor=(1, 1.0))
    fig.text(0.035, 1.07, "원거리 산간형은 배율이 낮아 보여도, 차보다 53분을 더 쓴다", fontsize=15.5,
             weight="bold", color=INK, va="top")
    save(fig, "R06_유형별소요시간")


# =====================================================================
# 그림 7. 읍면동별 이동공백지수 + 고령비율 (두 패널, 같은 순서)
# =====================================================================
def fig_district():
    g = t.groupby("읍면동").agg(idx=("이동공백지수", "mean"), n=("경로당명", "size")).join(
        eld.set_index("읍면동")[["고령비율", "고령인구"]])
    g = g.sort_values("idx")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 7.6), sharey=True,
                               gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.06})
    y = np.arange(len(g))
    top = set(g["idx"].nlargest(5).index)
    colors = [BUS if n in top else "#f2c4ae" for n in g.index]
    a.hlines(y, 0, g["idx"], color=colors, lw=2.2, zorder=2)
    a.scatter(g["idx"], y, s=60, color=colors, ec="white", lw=1.2, zorder=3)
    for yi, (n, r) in zip(y, g.iterrows()):
        a.text(r.idx + 1.2, yi, f"{r.idx:.0f}", va="center", fontsize=8.8,
               color=INK if n in top else INK2, weight="bold" if n in top else None)
    a.set_xlim(0, 75); a.set_yticks(y); a.set_yticklabels(g.index, fontsize=9.6, color=INK)
    a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
    a.set_title("평균 이동공백지수", loc="left", fontsize=11, weight="bold", color=INK)
    b.barh(y, g["고령비율"] * 100, height=0.62, color="#d9d8d1", zorder=2)
    for yi, v in zip(y, g["고령비율"] * 100):
        b.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=8.6, color=INK2)
    b.set_xlim(0, 68); b.grid(axis="x", color=GRID, lw=0.6); b.set_axisbelow(True)
    b.set_title("65세 이상 인구 비율", loc="left", fontsize=11, weight="bold", color=INK)
    b.tick_params(axis="y", length=0)
    fig.text(0.035, 0.995, "공백이 큰 면 지역은 노인 비율도 절반을 넘는다", fontsize=15.5, weight="bold",
             color=INK, va="top")
    fig.text(0.035, 0.957, "읍·면·동 23곳, 경로당 평균  ·  진한 색: 지수 상위 5곳", fontsize=9.8, color=INK2, va="top")
    source(fig, "고령비율: 행정안전부 주민등록 인구통계, 행정동별 연령별 인구현황(2026년 9월)", y=0.0)
    save(fig, "R07_읍면동")


# =====================================================================
# 그림 8. 대표 사례 (실측 상위)
# =====================================================================
def fig_cases():
    d = t[t["대중교통_실측"]].sort_values("이동공백지수", ascending=False).head(6)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.barh(yi, r["대중교통_분_보정"], height=0.56, color=BUS, zorder=2)
        ax.barh(yi, r["자동차_분"], height=0.56, color=CAR, zorder=3)
        ax.text(r["자동차_분"] / 2, yi, f"{r['자동차_분']:.0f}", ha="center", va="center", color="white",
                fontsize=9, weight="bold", zorder=4)
        ax.text(r["대중교통_분_보정"] + 1.5, yi, f"{r['대중교통_분_보정']:.0f}분  ({r['이동공백비']:.1f}배)",
                va="center", fontsize=9.6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['경로당명'].replace('경로당','')}  ·  {r['읍면동']}" for _, r in d.iterrows()],
                       fontsize=10, color=INK)
    ax.set_xlim(0, 118); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("의료 거점까지 편도 (분)", color=MUTED, fontsize=9.5)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CAR, label="자동차"),
                       plt.Rectangle((0, 0), 1, 1, color=BUS, label="대중교통")],
              loc="lower right", frameon=False, ncol=2, fontsize=9.2, bbox_to_anchor=(1, 1.0))
    fig.text(0.035, 1.07, "차로 20분이면 가는 병원이, 버스로는 1시간 반", fontsize=15.5, weight="bold",
             color=INK, va="top")
    fig.text(0.035, 0.995, "ODsay로 실측한 마을 중 이동공백지수 상위 6곳", fontsize=9.8, color=INK2, va="top")
    save(fig, "R08_대표사례")


# =====================================================================
# 그림 9. 회귀모델 정확도
# =====================================================================
def fig_model():
    best = mrep["선택_모델"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.32})
    y, p = cvp["실측_분"], cvp[f"예측_{best}"]
    lim = 110
    mae = mrep["교차검증_MAE_분"]
    xs = np.array([0, lim])
    a.fill_between(xs, xs - mae, xs + mae, color="#dbe8f8", zorder=1, lw=0)
    a.plot(xs, xs, color=CAR, lw=1, zorder=2)
    a.scatter(y, p, s=30, color=CAR, ec="white", lw=0.7, zorder=3)
    a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect("equal")
    a.grid(color="#f0efea", lw=0.6); a.set_axisbelow(True); a.spines["bottom"].set_visible(True)
    a.set_xlabel("실측 대중교통 시간 (분)", color=MUTED, fontsize=9.3)
    a.set_ylabel("5겹 교차검증 예측 (분)", color=MUTED, fontsize=9.3)
    a.text(8, 98, f"띠: ±{mae}분 (평균절대오차)", fontsize=8.8, color=CAR)
    a.set_title(f"{best} · 실측 {len(cvp)}곳", loc="left", fontsize=11, weight="bold", color=INK)
    sc = mrep["모델별_MAE"]
    names = list(sc.keys()); vals = [sc[k] for k in names]
    cols = [CAR if k == best else "#cfdcee" for k in names]
    b.barh(names[::-1], vals[::-1], color=cols[::-1], height=0.36, zorder=2)
    for i, v in enumerate(vals[::-1]):
        b.text(v + 0.2, i, f"{v}분", va="center", fontsize=10, color=INK, weight="bold")
    b.set_xlim(0, 11.5); b.set_xticks([]); b.tick_params(axis="y", labelsize=10, colors=INK)
    b.set_title("모델별 평균절대오차 (작을수록 정확)", loc="left", fontsize=11, weight="bold", color=INK)
    fig.text(0.035, 1.08, f"ODsay 호출 한도로 못 잰 {mrep['추정_마을수']}곳은 회귀모델로 추정했다",
             fontsize=15.5, weight="bold", color=INK, va="top")
    save(fig, "R09_예측모델")


# =====================================================================
# 그림 10. 탄소: 국가 수송부문 + 병원 왕복 1회당 CO2
# =====================================================================
def carbon_numbers():
    d = t[t["이동공백비_r"] >= 3]["도로거리_km"].median()   # 공백 마을 → 거점 편도 도로거리 중앙값
    modes = [
        ("면허 유지\n본인 자가용(휘발유)", 2 * d * EF["gas_car"], GRAY),
        ("현행 복지택시\nLPG 단독 탑승", 4 * d * EF["lpg_taxi"], GRAY),
        ("전기택시\n단독 탑승", 4 * d * EF["ev_taxi"], EV),
        ("전기 DRT\n3명 합승", 4 * d * EF["ev_van"] / 3, EV),
    ]
    return d, modes


def fig_carbon():
    d, modes = carbon_numbers()
    fig = plt.figure(figsize=(11.5, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.45], height_ratios=[1, 1], wspace=0.3, hspace=0.75)
    # (a) 국가 배출량 중 수송 비중
    a = fig.add_subplot(gs[0, 0])
    tot, tr = NATIONAL["ghg_total"], NATIONAL["ghg_transport"]
    a.barh([0], [tot / 100], color="#e4e3dd", height=0.5)
    a.barh([0], [tr / 100], color=BUS, height=0.5)
    a.text(0, -0.48, f"수송 {tr/100:,.1f}백만 t", ha="left", va="center", color=BUS,
           fontsize=9.6, weight="bold")
    a.text(tot / 100, 0.42, f"국가 총배출 {tot/100:,.1f}백만 t", ha="right", fontsize=9, color=INK2)
    a.set_xlim(0, tot / 100); a.set_ylim(-0.7, 0.75); a.axis("off")
    a.set_title(f"2024년 국가 온실가스 중 수송 {tr/tot*100:.1f}%", loc="left", fontsize=11, weight="bold", color=INK)
    # (b) 연료별
    c = fig.add_subplot(gs[1, 0])
    efv = [("휘발유 승용", EF["gas_car"]), ("LPG 택시", EF["lpg_taxi"]), ("전기 승용", EF["ev_taxi"]),
           ("소형 전기승합", EF["ev_van"])]
    c.barh([n for n, _ in efv][::-1], [v * 1000 for _, v in efv][::-1], height=0.55,
           color=[EV, EV, GRAY, GRAY], zorder=2)
    for i, (_, v) in enumerate(efv[::-1]):
        c.text(v * 1000 + 4, i, f"{v*1000:.0f}", va="center", fontsize=9, color=INK)
    c.set_xlim(0, 260); c.set_xticks([]); c.tick_params(axis="y", labelsize=9.2, colors=INK)
    c.set_title("차량 1km당 CO2 (g)", loc="left", fontsize=11, weight="bold", color=INK)
    # (c) 병원 왕복 1회당 1인 CO2
    b = fig.add_subplot(gs[:, 1])
    labs = [m[0] for m in modes]; vals = [m[1] for m in modes]; cols = [m[2] for m in modes]
    yy = np.arange(len(modes))[::-1]
    b.barh(yy, vals, color=cols, height=0.58, zorder=2)
    base = modes[1][1]
    for yi, (lab, v, col) in zip(yy, modes):
        txt = f"{v:.1f} kg"
        if col == EV:
            txt += f"   복지택시 대비 −{(1 - v/base)*100:.0f}%"
        b.text(v + 0.08, yi, txt, va="center", fontsize=10, color=INK, weight="bold" if col == EV else None)
    b.set_yticks(yy); b.set_yticklabels(labs, fontsize=9.8, color=INK)
    b.set_xlim(0, max(vals) * 1.75); b.grid(axis="x", color=GRID, lw=0.6); b.set_axisbelow(True)
    b.set_xlabel("병원 왕복 1회, 1인당 CO2 (kg)", color=MUTED, fontsize=9.3)
    b.set_title(f"공백 마을 → 의료 거점 왕복 1회 (편도 {d:.1f}km, 중앙값)", loc="left", fontsize=11,
                weight="bold", color=INK)
    fig.text(0.035, 1.07, "이동공백을 전기·합승 교통으로 메우면, 같은 병원길의 탄소는 절반 이하로",
             fontsize=15.5, weight="bold", color=INK, va="top")
    source(fig, "배출: 2024 국가 온실가스 잠정배출량, 에너지공단 EG-TIPS 연료 발열량·배출계수, 2023 국가 전력배출계수 0.4173 tCO2/MWh\n"
           "가정: 연비 휘발유 12km/L · LPG 9.5km/L, 전비 승용 5.5km/kWh · 승합 4.0km/kWh, 택시·DRT는 시내에서 출발해 빈 차로 돌아오는 거리 포함(편도의 4배)",
           y=-0.1)
    save(fig, "R10_탄소")
    return d, modes


def fig_carbon_scale():
    d, modes = carbon_numbers()
    base = modes[1][1]
    trips = np.linspace(0, 100000, 101)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for (lab, v, col), ls in zip(modes[2:], ["-", "-"]):
        red = (base - v) * trips / 1000
        ax.plot(trips / 1e4, red, color=EV if "DRT" in lab else "#7fd2b2", lw=2.4)
        ax.text(10.1, red[-1], lab.replace("\n", " "), va="center", fontsize=9.6, color=INK)
    # 기준 시나리오: 공백 마을 n3곳 × 주 3회 왕복
    sc_trips = n3 * 3 * 52
    v_drt = modes[3][1]
    red_sc = (base - v_drt) * sc_trips / 1000
    ax.scatter([sc_trips / 1e4], [red_sc], s=70, color=EV, ec="white", lw=1.5, zorder=5)
    ax.annotate(f"공백 마을 {n3}곳에서 주 3회씩\n연 {sc_trips:,}회 → 연 {red_sc:.0f}t 감축",
                (sc_trips / 1e4, red_sc), (sc_trips / 1e4 - 2.6, red_sc + 95), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, None)
    hgrid(ax); ax.spines["bottom"].set_visible(True)
    ax.set_xlabel("연간 병원 왕복 이용 횟수 (만 회)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("연간 CO2 감축 (t)", color=MUTED, fontsize=9.5)
    fig.text(0.035, 1.07, "이용이 늘수록 감축도 비례해서 커진다", fontsize=15.5, weight="bold", color=INK, va="top")
    fig.text(0.035, 0.995, "현행 LPG 복지택시 대비  ·  감축량 = 왕복 1회당 감축량 × 이용 횟수 (이용 횟수는 가정)", fontsize=9.8,
             color=INK2, va="top")
    save(fig, "R11_탄소확대")
    return sc_trips, red_sc


if __name__ == "__main__":
    fig_key_numbers()
    fig_national()
    fig_map()
    fig_scatter_types()
    fig_type_dumbbell()
    fig_district()
    fig_cases()
    fig_model()
    d, modes = fig_carbon()
    sc_trips, red_sc = fig_carbon_scale()
    summary = {"공백마을_3배이상": n3, "추가30분이상": n30, "편도60분이상": n60,
               "평균_자동차": round(car_m, 1), "평균_대중교통": round(bus_m, 1),
               "편도_도로거리_중앙값_km": round(d, 2),
               "왕복1회_1인_CO2_kg": {m[0].replace("\n", " "): round(m[1], 2) for m in modes},
               "배출계수_kg_km": {k: round(v, 4) for k, v in EF.items()},
               "시나리오_연간왕복": sc_trips, "시나리오_DRT_감축_t": round(red_sc, 1)}
    json.dump(summary, open(f"{OUT}/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
