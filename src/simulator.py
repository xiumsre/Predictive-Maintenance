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



#물리 법칙으로 센서 데이터 생성
#시간 흐름에 따라 센서값들 순서대로 계산

#그냥 랜덤 숫자는 변수들 사이에 관계가 없어서 모델이 뭘 배울 게 없음
#실제처럼 부하->토크->온도->전류로 이어지는 인과관계가 있으면,
#전류는 안 변했는데 토크만 튀었다 = 센서 오작동과 같은 이상 탐지 로직 가능

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



	