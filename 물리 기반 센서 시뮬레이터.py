import pandas as pd
import numpy as np
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

