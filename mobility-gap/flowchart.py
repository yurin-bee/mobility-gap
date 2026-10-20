"""보고서용 분석 알고리즘 흐름도"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

for f in ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break

INK, MUTED, LINE = "#1f2937", "#6b7280", "#cbd5e1"
BLUE, BLUE_BG = "#2563eb", "#eff6ff"
RED, RED_BG = "#dc2626", "#fef2f2"
GRAY_BG = "#f8fafc"

steps = [
    ("1", "의료 거점(시내) 찾기",
     "심평원 전국 병의원 현황\n내과·정형외과·안과\n재활의학과·치과·한의원",
     "DBSCAN 밀도 기반 군집화\n반경 700m 내 병원 5곳 이상 → 거점", True),
    ("2", "마을회관 위치 확보",
     "정읍시 경로당 현황 (707곳)",
     "카카오 주소검색 API로\n도로명주소 → 위경도 변환", False),
    ("3", "소요시간 측정",
     "카카오모빌리티 길찾기 API\nODsay 대중교통 길찾기 API\n+ 회귀모델(선형회귀 등 3종 비교)",
     "마을회관 → 가장 가까운 거점\n자동차 680곳 · 대중교통 실측 74곳\n나머지 606곳은 회귀모델로 추정", True),
    ("4", "이동공백지수 산출",
     "3단계 결과\n(버스보다 걷는 게 빠르면 도보시간)",
     "이동공백비 = 대중교통 ÷ 자동차\n지수 = 이동공백비 60%\n+ 추가시간 40%  (0~100점)", False),
    ("5", "마을 유형 분류",
     "4개 지표 (자동차·대중교통 시간,\n이동공백비, 거점까지 거리)",
     "K-means 군집 분석 (k=4)\n시내 인접 · 중거리\n근거리 연결불량 · 원거리 산간", True),
]

fig, ax = plt.subplots(figsize=(10, 12.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 12.2); ax.axis("off")

ax.text(0.3, 11.85, "고령운전자 이동공백 분석 알고리즘", fontsize=17, weight="bold", color=INK)
ax.text(0.3, 11.5, "\"면허를 반납하면 병원 가는 시간이 몇 배가 되는가\"를 마을 단위로 측정",
        fontsize=10, color=MUTED)
ax.text(1.35, 11.0, "입력 데이터", fontsize=9, color=MUTED, ha="center", weight="bold")
ax.text(5.1, 11.0, "처리 단계", fontsize=9, color=MUTED, ha="center", weight="bold")
ax.text(8.45, 11.0, "방법", fontsize=9, color=MUTED, ha="center", weight="bold")

y, h, gap = 9.55, 1.25, 0.5
for num, title, data, method, ai in steps:
    # 입력 데이터
    ax.add_patch(FancyBboxPatch((0.1, y), 2.5, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=GRAY_BG, ec=LINE, lw=1))
    ax.text(1.35, y + h / 2, data, ha="center", va="center", fontsize=8.3, color=INK, linespacing=1.5)
    # 처리 단계
    ax.add_patch(FancyBboxPatch((3.3, y), 3.6, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=BLUE_BG, ec=BLUE, lw=1.6))
    ax.add_patch(plt.Circle((3.75, y + h / 2), 0.25, color=BLUE))
    ax.text(3.75, y + h / 2, num, ha="center", va="center", color="white", fontsize=11, weight="bold")
    ax.text(4.15, y + h / 2, title, va="center", fontsize=12, weight="bold", color=INK)
    if ai:
        ax.text(6.75, y + h - 0.2, "AI", ha="right", va="center", fontsize=8, weight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.25", fc=RED, ec="none"))
    # 방법
    ax.add_patch(FancyBboxPatch((7.15, y), 2.65, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=LINE, lw=1))
    ax.text(8.47, y + h / 2, method, ha="center", va="center", fontsize=8.3, color=INK, linespacing=1.5)
    # 화살표
    ax.annotate("", xy=(3.28, y + h / 2), xytext=(2.62, y + h / 2),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    ax.plot([6.92, 7.13], [y + h / 2] * 2, color=LINE, lw=1, ls=":")
    if num != "5":
        ax.annotate("", xy=(5.1, y - gap + 0.03), xytext=(5.1, y - 0.03),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
    y -= h + gap

# 결과 → 정책
yb = y + h + gap - 0.05 - 1.55
ax.annotate("", xy=(5.1, yb + 1.38), xytext=(5.1, yb + 1.78 - 0.05),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
ax.add_patch(FancyBboxPatch((0.1, yb+0.25), 9.7, 1.1, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc=RED_BG, ec=RED, lw=1.6))
ax.text(0.45, yb + 1.0, "결과 → 정책 제안", fontsize=12, weight="bold", color=RED)
ax.text(0.45, yb + 0.5,
        "원거리 산간형은 전기택시 전환, 근거리 연결불량형은 전기 DRT 연계  ·  이동공백지수 순 예산 재배분\n"
        "면허 반납자 정기 이용권 연계  ·  전화·마을회관 호출 방식  ·  자가용 대체에 따른 CO2 감축량 추정",
        fontsize=9, color=INK, linespacing=1.7)

ax.set_ylim(yb + 0.15, 12.2)
fig.set_size_inches(10, 12.2 - yb - 0.15)
fig.savefig("figures/fig0_algorithm.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
