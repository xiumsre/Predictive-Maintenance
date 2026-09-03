"""
설비 센서 시뮬레이터 (물리 기반)
CNC 밀링 설비 3대를 1분 단위로 시뮬레이션합니다.
- truth : 오염 없는 참값 (정답지)
- observed : 현장에서 실제로 받는 더러운 데이터

*CNC : Computerized Numerical Control
컴퓨터로 프로그래밍된 명령에 따라 자동으로 금속이나 소재를 깎아 원하는 모양을 만드는 공작기계
- 공구가 계속 깎다 보면 마모되고, 부하가 커지면 전류,온도가 오르는 등 물리적 인과관계가 뚜렷


"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm



names = {f.name for f in fm.fontManager.ttflist}
for cand in ["D2Coding"]:
	if cand in names:
		plt.rcParams["font.family"] = cand
		break
	else:
		print("[WARN] 한글 폰트를 찾지 못했습니다.")
	plt.rcParams["axes.unicode_minus"] = False






# CNC 설비 3대의 스펙을 딕셔너리로 정의
# machine_id : (품질등급, 과부하 한계, 공구 교체주기(분))
# 품질등급 L,M,H
# osf_limit : 과부하 한계(rpm)
# 공구 교체주기 : tool_life(분)
MACHINES = {
"CNC-01": {"type": "L", "osf_limit": 11000, "tool_life": 210},
"CNC-02": {"type": "M", "osf_limit": 12000, "tool_life": 225},
"CNC-03": {"type": "H", "osf_limit": 13000, "tool_life": 240},
}


# 기본 오염 강도
# 관측 데이터 오염 강도 (기본값 = "현장급")
# 오염규칙을 하드코딩하지 않기 위함
# 처음부터 어떤 오염이 존재하는지 명시
POLLUTION = {
"dropout_rate": 0.015, # 통신 끊김
"dropout_len": (3, 40), # 끊김 길이(분)
"nan_rate": 0.008, # 개별 센서값만 NaN
"spike_rate": 0.004, # 센서 튐(전기 노이즈)
"dup_rate": 0.006, # 같은 레코드 중복 전송
"ts_jitter_rate": 0.05, # 타임스탬프 흔들림
"unit_mix_rate": 0.10, # 단위 혼재(K 대신 섭씨)
"drift_per_day": 0.35, # 온도 센서 드리프트 (K/day)
}


#####################################################################
# 1. 물리기반 센서 시뮬레이터
#####################################################################



#물리 법칙으로 센서 데이터 생성
#시간 흐름에 따라 센서값들 순서대로 계산

#그냥 랜덤 숫자는 변수들 사이에 관계가 없어서 모델이 뭘 배울 게 없음
#실제처럼 부하->토크->온도->전류로 이어지는 인과관계가 있으면,
#전류는 안 변했는데 토크만 튀었다 = 센서 오작동과 같은 이상 탐지 로직 가능


#물리모델
def _simulate_one(machine_id: str, n_minutes: int, start: pd.Timestamp,
                  rng: np.random.Generator) -> pd.DataFrame:
    spec = MACHINES[machine_id]
    ts = pd.date_range(start, periods=n_minutes, freq="min")

    # --- 공정 부하: 근무 시간대에 높고 야간에 낮음 (일주기) ---
    hour = ts.hour + ts.minute / 60.0
    duty = 0.55 + 0.45 * np.sin((hour - 6) / 24 * 2 * np.pi)     # 0.1 ~ 1.0
    duty = np.clip(duty + rng.normal(0, 0.05, n_minutes), 0.05, 1.0)

    # --- 공기 온도: 계절/일교차 + 랜덤워크 ---
    air = 298.0 + 2.0 * np.sin((hour - 14) / 24 * 2 * np.pi)
    air = air + np.cumsum(rng.normal(0, 0.02, n_minutes))         # 완만한 표류
    air = air + rng.normal(0, 0.15, n_minutes)

    # --- 공구 마모: 누적되다가 교체하면 0으로 ---
    tool_life = spec["tool_life"]
    wear_rate = 1.0 + 0.6 * duty          # 부하 클수록 빨리 닳음
    wear = np.zeros(n_minutes)
    acc = rng.uniform(0, 60)              # 시작 시점 마모도는 랜덤
    limit = tool_life * rng.uniform(0.90, 1.15)
    for i in range(n_minutes):
        acc += wear_rate[i]
        if acc > limit:                   # 계획 교체 (정비반 재량으로 조금씩 다름)
            acc = 0.0
            limit = tool_life * rng.uniform(0.90, 1.15)
        wear[i] = acc

    # --- 회전수: 부하에 반비례(무거운 절삭일수록 저속) ---
    rpm = 2860 - 1500 * duty + rng.normal(0, 45, n_minutes)
    rpm = np.clip(rpm, 1150, 2900)

    # --- 토크: 부하에 비례, 마모되면 저항 증가 ---
    torque = 10 + 40 * duty + 0.02 * wear + rng.normal(0, 2.0, n_minutes)
    torque = np.clip(torque, 3.0, 80.0)

    # --- 냉각(HVAC) 이상: 가끔 공장 공조가 죽어 실내가 더워짐 ---
    hvac_fail = np.zeros(n_minutes, dtype=bool)
    for _ in range(max(1, n_minutes // 2000)):
        s = rng.integers(0, max(1, n_minutes - 120))
        hvac_fail[s:s + rng.integers(40, 120)] = True
    air = air + 5.5 * hvac_fail                 # 실내 온도 상승

    # --- 공정 온도: 공기온도 + 절삭열. 쿨런트가 process 쪽은 어느 정도 잡아줌 ---
    power_w = torque * rpm * 2 * np.pi / 60.0                     # [W]
    proc = air + 8.5 + power_w / 1400.0 + 0.004 * wear
    proc = proc - 6.0 * hvac_fail                # 온도차(방열 여력)가 줄어듦
    proc = proc + rng.normal(0, 0.12, n_minutes)

    # --- 진동: 마모·회전수에 비례. 마모 후반에 급격히 커짐 ---
    vib = (0.8 + 0.0009 * rpm + 0.9 * (wear / tool_life) ** 3
           + rng.normal(0, 0.06, n_minutes))
    vib = np.clip(vib, 0.1, None)

    # --- 전류: 전력/전압(380V, 역률 0.85, 3상) ---
    current = power_w / (380 * 1.732 * 0.85) + rng.normal(0, 0.15, n_minutes)
    current = np.clip(current, 0.2, None)

    # --- 습도: 온도와 약한 음의 관계 ---
    humid = 55 - 1.8 * (air - 298) + rng.normal(0, 2.5, n_minutes)
    humid = np.clip(humid, 15, 95)


	#계산된 센서값들을 하나의 데이터프레임으로
    df = pd.DataFrame({
        "ts": ts,
        "machine_id": machine_id,
        "type": spec["type"],
        "air_temp_k": air,
        "process_temp_k": proc,
        "rot_speed_rpm": rpm,
        "torque_nm": torque,
        "tool_wear_min": wear,
        "vibration_mms": vib,
        "current_a": current,
        "humidity_pct": humid,
    })

    # ------------------------------------------------------------------
    # 고장 라벨 (AI4I 2020 정의 그대로)
    # 
	# UCI AI4I 2020 데이터셋 정의 ------------------------------------------------------------------
    
	#공구마모고장
    twf = (wear >= 200) & (wear <= 240) & (rng.random(n_minutes) < 0.004)
	#방열실패
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
	#전력이상
    pwf = (power_w < 3500) | (power_w > 9000)
	#과부하 고장
    osf = (wear * torque) > spec["osf_limit"]
	#원인 불명
    rnf = rng.random(n_minutes) < 0.0002                 

    df["twf"] = twf.astype(int)
    df["hdf"] = hdf.astype(int)
    df["pwf"] = pwf.astype(int)
    df["osf"] = osf.astype(int)
    df["rnf"] = rnf.astype(int)
    
    #다섯 조건 중 하나라도 True면 True
    #.astype(int) : True/False를 1/0 숫자로 변환
    df["machine_failure"] = (twf | hdf | pwf | osf | rnf).astype(int)
    df["power_w"] = power_w
    return df




# (오염 없는) 참값 만들기
# -> 설비 3대를 합쳐서 최종 참값 데이터셋 완성	
def simulate_truth(n_minutes: int = 1440, start: str | pd.Timestamp = "2024-01-01",
                   seed: int = 42) -> pd.DataFrame:
    """오염 없는 참값을 생성합니다."""
    #난수생성기
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)
    #딕셔너리에 있는 3개 설비에 대해 각각 물리모델 함수 호출
    parts = [_simulate_one(m, n_minutes, start, rng) for m in MACHINES]
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["ts", "machine_id"]).reset_index(drop=True)




#14일치 만들어보기
truth = simulate_truth(n_minutes=1440 * 14, start="2024-01-01", seed=42)

#기본 검산
print("설비 수 :", truth["machine_id"].nunique())
print("기간 :", truth["ts"].min(), "~", truth["ts"].max())
print("행 수 :", f"{len(truth):,}")


#고장모드 분포
#각 고장 유형이 몇 번 발생했는지, 전체에서 몇%인지 계산
modes = truth[["twf", "hdf", "pwf", "osf", "rnf", "machine_failure"]].sum()
print(pd.DataFrame({"건수": modes, "비율(%)": (modes / len(truth) * 100).round(3)}))



#센서 요약
#물리적으로 말이 되는 값인지 확인
'''
-공기온도 293.97~307.90 K = 약 21~35 ℃ → 공장 실내로 타당합니다
-공정온도가 공기온도보다 항상 높습니다 → 절삭열이 나니까 당연합니다
-회전수 1,220~2,900 rpm → CNC 밀링 스펙 범위입니다
진동 1.75~4.70 mm/s → ISO 10816 기준 "양호~주의" 구간입니다
-전력 1,958~9,564 W → 고장 판정선(3,500 / 9,000 W)이 이 범위 안에 있습니다. 그래야 PWF가 "가끔"
발생합니다. 범위 밖이면 고장이 0건이거나 전부 고장입니다
'''
cols = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm",
"tool_wear_min", "vibration_mms", "current_a", "power_w"]
print(truth[cols].describe().loc[["mean", "std", "min", "50%", "max"]].round(2))





# 시각화 코드
'''
<CNC-01의 이틀치>
1) 공구 마모가 톱니모양으로 누적되다 교체 시점에 0으로 떨어짐
2) 토크가 일주기로 오르내림 (주간 부하 ↑, 야간 ↓)
3) 진동이 마모 후반에 치솟음 (3제곱 관계)
4) 빨간 세로선 - 고장 시점
'''
sub = truth[(truth["machine_id"] == "CNC-01")].iloc[:2880]  # 이틀치 = 1440*2분

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

axes[0].plot(sub["ts"], sub["tool_wear_min"], color="steelblue")
axes[0].set_ylabel("tool_wear_min")

axes[1].plot(sub["ts"], sub["torque_nm"], color="seagreen")
axes[1].set_ylabel("torque_nm")

axes[2].plot(sub["ts"], sub["vibration_mms"], color="darkorange")
axes[2].set_ylabel("vibration_mms")

# 고장 시점에 빨간 세로선
fail_times = sub.loc[sub["machine_failure"] == 1, "ts"]
for ax in axes:
    for t in fail_times:
        ax.axvline(t, color="red", alpha=0.3)

fig.suptitle("CNC-01 이틀치 참값 — 마모 누적과 교체, 그리고 고장 시점")
plt.tight_layout()
plt.show()






#####################################################################
# 2. 현장급 오염 주입
#####################################################################


# 깨끗한 센서 데이터는 존재하지 않는다!
# 설계 의도 : 현실에서 일어나는 순서대로 오염을 넣는다
'''
설비 단계 드리프트 (센서가 늙는다)
↓
수집기 단계 단위 혼재 (태그 설정이 잘못됐다)
↓
전기 단계 센서 튐 (노이즈가 탄다)
↓
센서 단계 개별 결측 (값을 못 읽었다)
↓
통신 단계 구간 끊김 (네트워크가 끊겼다)
↓
전송 단계 중복·타임스탬프 흔들림

return_masks=True면 "어디에 무엇을 주입했는지" 정답지를 함께 돌려줍니다.
전처리 성능을 정밀도·재현율로 채점하기 위한 것입니다.
'''

SENSOR_COLS = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min", "vibration_mms", "current_a", "humidity_pct"]

# 6가지 오염

def pollute(truth: pd.DataFrame, seed: int = 7, cfg: dict | None = None, return_masks: bool = False):
	#기본 오염 강도
	c = dict(POLLUTION)

	#사용자가 cfg를 넘기면 그 값으로 덮어쓴다
	if cfg: c.update(cfg)

	# 오염 주입 전용 난수 생성기 
	rng = np.random.default_rng(seed)
 
	df = truth.copy()
	masks = pd.DataFrame(index=df.index) # ★ 주입 정답지 (오염을 어디에 넣었는지)
     
	# 현장에서도 세부 고장코드는 정비 후에야 붙습니다
    # 실시간으로 받을 수 없는 정보는 관측 데이터에서 뺀다
    # 관측 데이터는 참값 라벨 중 machine_failure만 남긴다 => 기계 이상 자체는 현장에서도 실시간으로 알 수 있는 정보
	df = df.drop(columns=["twf", "hdf", "pwf", "osf", "rnf", "power_w"])

	# 데이터 시작 시점부터 며칠이 지났는지 계산 
	t0 = df["ts"].min()
	days = (df["ts"] - t0).dt.total_seconds() / 86400.0

	# --- (a) 센서 드리프트: CNC-02 온도 센서만 서서히 위로 밀림 ---
	# 한 대만 밀리면 센서 문제. 세 대가 다 밀리면 공정 변화
	m2 = df["machine_id"] == "CNC-02"
	df.loc[m2, "process_temp_k"] += c["drift_per_day"] * days[m2]


 	# --- (b) 단위 혼재: 특정 구간에서 온도가 섭씨로 들어옴 ---
	n = len(df)
	unit_block = np.zeros(n, dtype=bool)
	n_blocks = max(1, int(n * c["unit_mix_rate"] / 200))
	for _ in range(n_blocks):
		s = rng.integers(0, n - 200)
		unit_block[s:s + 200] = True
	df.loc[unit_block, "air_temp_k"] -= 273.15
	df.loc[unit_block, "process_temp_k"] -= 273.15

	# 어디를 오염시켰는지 정답 기록 (mask)
	masks["unit_temp"] = unit_block
	# 진동 단위도 일부는 m/s^2 로 (×9.81)
	vib_block = rng.random(n) < 0.04
	df.loc[vib_block, "vibration_mms"] *= 9.81
	masks["unit_vib"] = vib_block

	# --- (c) 센서 튐: 값이 순간적으로 10~50배 또는 0 ---
	for col in SENSOR_COLS:
		hit = rng.random(n) < c["spike_rate"]
		mode = rng.random(n)
		df.loc[hit & (mode < 0.5), col] = df.loc[hit & (mode < 0.5), col] * rng.uniform(8, 40)
		df.loc[hit & (mode >= 0.5), col] = 0.0
		masks[f"spike_{col}"] = hit

	# --- (d) 개별 결측 ---
	# 컬럼별로 nan_rate 확률로 해당 셀만 NaN 처리.
	for col in SENSOR_COLS:
		hit = rng.random(n) < c["nan_rate"]
		df.loc[hit, col] = np.nan
		masks[f"nan_{col}"] = hit

	# --- (e) 통신 끊김: 행 자체가 사라짐 ---
	# 랜덤 시작점에서 일정 길이 (dropout_len)만큼 행 전체를 삭제
	# x3은 설비 3대분이 동시에 빠지는 걸 표현
	drop_mask = np.zeros(n, dtype=bool)
	n_drop = int(n * c["dropout_rate"] / 10)
	for _ in range(max(1, n_drop)):
		s = rng.integers(0, n)
		ln = rng.integers(*c["dropout_len"]) * 3      # 설비 3대 × 분
		drop_mask[s:s + ln] = True
	masks["dropped"] = drop_mask
	keep = ~drop_mask
	df = df[keep].copy()
	kept_masks = masks[keep].copy()

	# --- (f) 중복 전송 ---
	# 남은 행 중 일부를 골라 그대로 복사해서 뒤에 이어붙임 (pd.concat)
	# is_dup 컬럼으로 "이게 중복본이다"를 마킹
	n2 = len(df)
	dup_idx = rng.random(n2) < c["dup_rate"]
	dups = df[dup_idx].copy()
	kept_masks["is_dup"] = False
	dup_masks = kept_masks[dup_idx].copy()
	dup_masks["is_dup"] = True
	df = pd.concat([df, dups], ignore_index=True)
	kept_masks = pd.concat([kept_masks, dup_masks], ignore_index=True)

	# --- (g) 타임스탬프 흔들림 + 순서 뒤섞임 ---
	# 무작위로 흔들어서 실제 네트워크로 데이터가 들어올 때 순서 보장이 안되는 상황 재현
	n3 = len(df)
	jitter = np.where(rng.random(n3) < c["ts_jitter_rate"],
					  rng.integers(-90, 90, n3), 0)
	df["ts"] = df["ts"] + pd.to_timedelta(jitter, unit="s")
	kept_masks["ts_jittered"] = jitter != 0
	order = rng.permutation(n3)
	df = df.iloc[order].reset_index(drop=True)
	kept_masks = kept_masks.iloc[order].reset_index(drop=True)

	# --- (h) 실제 수집기가 붙이는 메타 컬럼 ---
	# 실제 DB에 적재된 시각 붙이기 (원래 발생 시각과 수집된 시각 구분하기 위함)
	df["collected_at"] = pd.Timestamp("2024-01-01")
	df["ts"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")   # 문자열로 들어옴(현실)
	if return_masks:
		return df, kept_masks
	return df




# 실행 -> 오염된 데이터 확인
	 

	