"""
정읍시 고령운전자 이동공백 분석 - 1단계: 데이터 수집 (노트북에서 실행)

하는 일
  1) 심평원 병원정보 API  -> 정읍시 어르신 주요 진료과 병원 좌표  (hospitals.csv)
  2) 카카오 주소검색 API   -> 경로당(마을회관) 주소를 좌표로 변환  (villages.csv)
  3) DBSCAN 군집화        -> 병원 밀집지 = '의료 거점(시내)' 찾기   (hubs.csv)
  4) 카카오모빌리티 길찾기 -> 마을회관 -> 가장 가까운 의료 거점, 자동차 소요시간
  5) ODsay 대중교통 길찾기 -> 같은 구간, 대중교통 소요시간           (travel_times.csv)

준비
  pip install requests pandas scikit-learn
  아래 KEY 3개를 채우고, 경로당 CSV를 이 파일과 같은 폴더에 둔 뒤
  python collect_data.py

중간에 끊겨도 다시 실행하면 이미 받은 행은 건너뜁니다.
"""
import os, time, json, math
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# ==== API 키는 코드에 쓰지 않고 같은 폴더의 keys.json에서 읽습니다 ====
# keys.json 예시 (이 파일은 .gitignore에 있어 GitHub에 올라가지 않습니다)
# {"KAKAO_REST_KEY": "...", "ODSAY_KEY": "...", "DATA_GO_KR_KEY": "..."}
with open("keys.json", encoding="utf-8") as f:
    _k = json.load(f)
DATA_GO_KR_KEY = _k["DATA_GO_KR_KEY"]
KAKAO_REST_KEY = _k["KAKAO_REST_KEY"]
ODSAY_KEY      = _k["ODSAY_KEY"]


def find_village_csv():
    """같은 폴더에서 이름에 '경로당'이 들어간 CSV를 찾음 (Mac 한글 파일명 정규화 대응)"""
    import glob, unicodedata
    hits = [f for f in (glob.glob("*.csv") + glob.glob("data/*.csv"))
            if "경로당" in unicodedata.normalize("NFC", f)]
    if not hits:
        raise SystemExit("[파일 없음] 이 폴더에 이름에 '경로당'이 들어간 CSV가 없어요. "
                         "작업 폴더나 data 폴더에 경로당 CSV를 넣어 주세요.")
    print(f"  경로당 파일: {unicodedata.normalize('NFC', hits[0])}")
    return hits[0]
# ======================================================================

OUT = "data"
os.makedirs(OUT, exist_ok=True)
KAKAO_H = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}

# 어르신이 자주 가는 진료과 (심평원 코드)
DEPTS = {"01": "내과", "05": "정형외과", "12": "안과", "21": "재활의학과"}
KINDS = {"51": "치과의원", "41": "치과병원", "93": "한의원"}
SIDO_JEONBUK = "350000"   # 심평원 시도코드: 전북


# 심평원 병원정보서비스 주소 후보 (서비스 개편으로 주소가 바뀐 적이 있어 순서대로 시도)
HOSP_URLS = [
    "https://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList",
    "https://apis.data.go.kr/B551182/hospInfoService1/getHospBasisList1",
    "http://apis.data.go.kr/B551182/hospInfoService1/getHospBasisList1",
]
_hosp_url = None
ERR_HINT = {
    "12": "서비스 주소가 없거나 폐기됨",
    "20": "이 API에 활용신청이 안 되어 있음 (공공데이터포털에서 '병원정보서비스' 활용신청 확인)",
    "22": "오늘 호출 한도 초과",
    "30": "등록되지 않은 키 (승인 후 30분~1시간 기다리거나, Decoding 키인지 확인)",
    "31": "키 사용기간 만료",
}


def _parse(text):
    """JSON 또는 XML 응답을 (오류코드, totalCount, items) 로 변환"""
    try:
        j = json.loads(text)
        if "response" in j:
            head, body = j["response"].get("header", {}), j["response"].get("body", {})
            items = body.get("items") or {}
            items = items.get("item", []) if isinstance(items, dict) else []
            if isinstance(items, dict):
                items = [items]
            return head.get("resultCode", "00"), int(body.get("totalCount", 0) or 0), items
        code = j.get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader", {}).get("returnReasonCode", "99")
        return str(code), 0, []
    except ValueError:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(text)
        code = root.findtext(".//returnReasonCode") or root.findtext(".//resultCode") or "00"
        items = [{c.tag: c.text for c in it} for it in root.iter("item")]
        return code, int(root.findtext(".//totalCount") or 0), items


def fetch_hosp(params):
    global _hosp_url
    urls = [_hosp_url] if _hosp_url else HOSP_URLS
    last = ""
    for url in urls:
        rows, page = [], 1
        while True:
            p = {"serviceKey": DATA_GO_KR_KEY, "pageNo": page, "numOfRows": 500,
                 "sidoCd": SIDO_JEONBUK, "_type": "json", **params}
            r = requests.get(url, params=p, timeout=30)
            code, total, items = _parse(r.text)
            if code not in ("00", "0", "INFO-000"):
                last = f"{url}\n  -> 오류코드 {code}: {ERR_HINT.get(code, r.text[:200])}"
                break
            if _hosp_url is None:
                _hosp_url = url
                print(f"  병원 API 주소 확인: {url}")
            rows += items
            if page * 500 >= total:
                return rows
            page += 1
        if code != "12":   # 주소 문제가 아니면 다른 주소를 시도해도 소용없음
            break
    raise SystemExit(f"[병원 API 오류]\n  {last}")


def _find_files(*words):
    import glob, unicodedata
    out = []
    for f in glob.glob("*.xlsx") + glob.glob("*.csv") + glob.glob("data/*.xlsx"):
        n = unicodedata.normalize("NFC", f)
        if all(w in n for w in words):
            out.append(f)
    return out


def _read_any(f):
    if f.lower().endswith(".xlsx"):
        return pd.read_excel(f)
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(f, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"[파일 오류] {f} 를 읽을 수 없어요.")


def _col(df, *keys):
    """열 이름에 keys가 모두 들어간 첫 열 (파일 버전마다 열 이름이 조금씩 달라서)"""
    for c in df.columns:
        if all(k in str(c).replace(" ", "") for k in keys):
            return c
    return None


KEYWORDS = {"내과": "내과", "정형외과": "정형외과", "안과": "안과", "재활의학": "재활의학과",
            "치과": "치과", "한의원": "한의원", "한방": "한의원"}


def hospitals_from_file():
    """심평원 '전국 병의원 및 약국 현황' 파일에서 정읍 병원 추출 (API 대체 경로)"""
    basic = _find_files("병원정보")
    if not basic:
        return None
    print(f"  병원 파일 사용: {basic[0]}")
    h = _read_any(basic[0])
    c_name, c_kind = _col(h, "요양기관명"), _col(h, "종별코드명")
    c_sggu, c_emd, c_addr = _col(h, "시군구코드명"), _col(h, "읍면동"), _col(h, "주소")
    c_x, c_y, c_id = _col(h, "좌표", "x") or _col(h, "X"), _col(h, "좌표", "y") or _col(h, "Y"), _col(h, "암호화요양기호")
    if not all([c_name, c_kind, c_sggu, c_x, c_y]):
        raise SystemExit(f"[병원 파일] 필요한 열을 못 찾았어요. 열 목록: {list(h.columns)}")
    h = h[h[c_sggu].astype(str).str.contains("정읍")].copy()

    # 진료과목 파일이 있으면 진료과목 코드로, 없으면 기관명·종별명으로 분류
    dept = _find_files("진료과목")
    h["분류"] = ""
    if dept and c_id:
        d = _read_any(dept[0])
        d_id, d_nm = _col(d, "암호화요양기호"), _col(d, "진료과목코드명")
        d = d[d[d_id].isin(h[c_id])]
        for kw, lab in KEYWORDS.items():
            ids = set(d.loc[d[d_nm].astype(str).str.contains(kw), d_id])
            h.loc[h[c_id].isin(ids), "분류"] += "," + lab
        print(f"  진료과목 파일 사용: {dept[0]}")
    for kw, lab in KEYWORDS.items():   # 이름·종별에 과목이 드러난 경우 보완
        m = (h[c_name].astype(str) + h[c_kind].astype(str)).str.contains(kw)
        h.loc[m & ~h["분류"].str.contains(lab), "분류"] += "," + lab
    h["분류"] = h["분류"].str.strip(",")
    h = h[h["분류"] != ""].dropna(subset=[c_x, c_y])
    out = pd.DataFrame({"yadmNm": h[c_name], "XPos": h[c_x].astype(float), "YPos": h[c_y].astype(float),
                        "분류": h["분류"], "종별": h[c_kind],
                        "읍면동": h[c_emd] if c_emd else "", "주소": h[c_addr] if c_addr else ""})
    return out


def step1_hospitals():
    path = f"{OUT}/hospitals.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    df = hospitals_from_file()
    if df is not None:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[1] 정읍 병원 {len(df)}곳 저장 (파일)")
        return df
    rows = []
    for code, name in DEPTS.items():
        for it in fetch_hosp({"dgsbjtCd": code}):
            it["분류"] = name; rows.append(it)
        print(f"  {name} 조회 완료")
    for code, name in KINDS.items():
        for it in fetch_hosp({"clCd": code}):
            it["분류"] = name; rows.append(it)
        print(f"  {name} 조회 완료")
    df = pd.DataFrame(rows)
    df = df[df["sgguCdNm"].astype(str).str.contains("정읍")]
    df = df.dropna(subset=["XPos", "YPos"])
    # 같은 병원이 여러 진료과로 중복될 수 있어 병원명+좌표 기준으로 정리
    df = (df.groupby(["yadmNm", "XPos", "YPos"], as_index=False)
            .agg(분류=("분류", lambda s: ",".join(sorted(set(s)))),
                 종별=("clCdNm", "first"), 읍면동=("emdongNm", "first"),
                 주소=("addr", "first")))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[1] 정읍 병원 {len(df)}곳 저장")
    return df


def step2_geocode():
    path = f"{OUT}/villages.csv"
    if os.path.exists(path):
        v = pd.read_csv(path)
    else:
        f = find_village_csv()
        try:
            v = pd.read_csv(f, encoding="cp949")
        except UnicodeDecodeError:
            v = pd.read_csv(f, encoding="utf-8-sig")
        v["x"], v["y"] = np.nan, np.nan
    todo = v[v["x"].isna()].index
    for n, i in enumerate(todo):
        addr = v.at[i, "소재지도로명주소"]
        r = requests.get("https://dapi.kakao.com/v2/local/search/address.json",
                         headers=KAKAO_H, params={"query": addr}, timeout=10)
        if r.status_code != 200:
            raise SystemExit(f"[카카오 오류 {r.status_code}] {r.text[:200]}\n"
                             "REST API 키와 '카카오맵 사용 설정 ON'을 확인하세요.")
        docs = r.json().get("documents", [])
        if docs:
            v.at[i, "x"], v.at[i, "y"] = float(docs[0]["x"]), float(docs[0]["y"])
        if n % 50 == 0:
            v.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  좌표 변환 {n}/{len(todo)}")
        time.sleep(0.05)
    v.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[2] 경로당 {v['x'].notna().sum()}/{len(v)}곳 좌표 변환 완료")
    return v


def step3_hubs(h, eps_m=700, min_samples=5):
    """병원이 반경 eps_m 안에 min_samples개 이상 모인 곳을 '의료 거점'으로 정의"""
    xy = np.radians(h[["YPos", "XPos"]].to_numpy(dtype=float))
    lab = DBSCAN(eps=eps_m / 6371000, min_samples=min_samples,
                 metric="haversine").fit_predict(xy)
    h["군집"] = lab
    h.to_csv(f"{OUT}/hospitals.csv", index=False, encoding="utf-8-sig")
    hubs = (h[h["군집"] >= 0].groupby("군집")
              .agg(x=("XPos", "mean"), y=("YPos", "mean"),
                   병원수=("yadmNm", "count"),
                   진료과=("분류", lambda s: ",".join(sorted(set(",".join(s).split(",")))))))
    hubs = hubs.reset_index()
    hubs.to_csv(f"{OUT}/hubs.csv", index=False, encoding="utf-8-sig")
    print(f"[3] 의료 거점 {len(hubs)}곳 발견 (병원 수: {hubs['병원수'].tolist()})")
    return hubs


def haversine_km(x1, y1, x2, y2):
    x1, y1, x2, y2 = map(np.radians, [x1, y1, x2, y2])
    a = np.sin((y2-y1)/2)**2 + np.cos(y1)*np.cos(y2)*np.sin((x2-x1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def car_min(sx, sy, ex, ey):
    r = requests.get("https://apis-navi.kakaomobility.com/v1/directions", headers=KAKAO_H,
                     params={"origin": f"{sx},{sy}", "destination": f"{ex},{ey}"}, timeout=10)
    try:
        s = r.json()["routes"][0]["summary"]
        return s["duration"] / 60, s["distance"] / 1000
    except Exception:
        return np.nan, np.nan


class QuotaExceeded(Exception):
    pass


def transit(sx, sy, ex, ey):
    r = requests.get("https://api.odsay.com/v1/api/searchPubTransPathT",
                     params={"SX": sx, "SY": sy, "EX": ex, "EY": ey,
                             "apiKey": ODSAY_KEY}, timeout=15)
    j = r.json()
    if "error" in j:
        err = j["error"]
        err = err[0] if isinstance(err, list) else err
        msg = str(err.get("message", err) if isinstance(err, dict) else err)
        if "quota" in msg.lower() or "limit" in msg.lower():
            raise QuotaExceeded(msg)
        return {"대중교통_분": np.nan, "도보_분": np.nan, "환승": np.nan,
                "대중교통_메모": msg[:60]}
    p = j["result"]["path"][0]["info"]
    return {"대중교통_분": p["totalTime"], "도보_분": p.get("totalWalk", 0) / 67,  # 노인 보행 약 67m/분
            "환승": p.get("busTransitCount", 0) + p.get("subwayTransitCount", 0) - 1,
            "대중교통_메모": ""}


def _need_transit(row):
    memo = str(row.get("대중교통_메모", "") or "")
    return pd.isna(row.get("대중교통_분")) and (memo in ("", "nan") or "quota" in memo.lower())


def step4_travel(v, hubs):
    """자동차 시간은 한 번만, 대중교통 시간은 빠진 마을만 이어서 받음.
    ODsay 하루 한도에 걸리면 저장하고 멈춤 -> 다음 날 다시 실행하면 이어서 진행.
    읍면동별로 돌아가며 받아, 중간에 멈춰도 받은 마을이 정읍 전역에 고르게 퍼진 표본이 되게 함."""
    path = f"{OUT}/travel_times.csv"
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    key = lambda d: d["경로당명"].astype(str) + "|" + d["소재지도로명주소"].astype(str)
    prev = {k: r for k, r in zip(key(old), old.to_dict("records"))} if len(old) else {}
    # 23개 읍면동을 돌아가며 한 곳씩(각 읍면동 안에서는 무작위) -> 하루 한도가 작아도 정읍 전역이 고르게 표본에 들어감
    v = v.dropna(subset=["x", "y"]).sample(frac=1, random_state=42)
    v["_순번"] = v.groupby("읍면동").cumcount()
    v = v.sort_values(["_순번", "읍면동"]).drop(columns="_순번")
    rows, quota_hit, called = [], False, 0
    for n, (k, r) in enumerate(zip(key(v), v.to_dict("records"))):
        row = prev.get(k)
        if row is None:
            d = haversine_km(r["x"], r["y"], hubs["x"].values, hubs["y"].values)
            hub = hubs.iloc[int(np.argmin(d))]
            cm, ckm = car_min(r["x"], r["y"], hub["x"], hub["y"])
            row = {"읍면동": r["읍면동"], "경로당명": r["경로당명"],
                   "소재지도로명주소": r["소재지도로명주소"], "x": r["x"], "y": r["y"],
                   "거점": int(hub["군집"]), "직선거리_km": round(float(d.min()), 2),
                   "자동차_분": cm, "도로거리_km": ckm,
                   "대중교통_분": np.nan, "도보_분": np.nan, "환승": np.nan, "대중교통_메모": ""}
        if not quota_hit and _need_transit(row):
            hub = hubs[hubs["군집"] == row["거점"]].iloc[0]
            try:
                row.update(transit(row["x"], row["y"], hub["x"], hub["y"]))
                called += 1
                time.sleep(0.3)
            except QuotaExceeded as e:
                quota_hit = True
                row["대중교통_메모"] = "quota"
                print(f"  ODsay 하루 한도 도달 ({called}건 받음): {e}")
        elif quota_hit and _need_transit(row):
            row["대중교통_메모"] = "quota"
        rows.append(row)
        if n % 20 == 0:
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    left = int(out.apply(_need_transit, axis=1).sum())
    print(f"[4] 마을 {len(out)}곳 / 대중교통 남은 마을 {left}곳 -> {path}")
    if left:
        print("    한도가 풀리면(보통 다음 날) 'python collect_data.py'를 다시 실행하세요. 받은 건 건너뜁니다.")
    else:
        print("    완료. python analyze.py 로 분석을 실행하세요.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "geo":
        # 병원 API 키가 활성화되기 전에 마을회관 좌표 변환만 먼저 실행
        step2_geocode()
        print("좌표 변환만 완료. 병원 키가 활성화되면 'python collect_data.py'로 전체 실행하세요.")
    else:
        h = step1_hospitals()
        v = step2_geocode()
        hubs = step3_hubs(h)
        step4_travel(v, hubs)
