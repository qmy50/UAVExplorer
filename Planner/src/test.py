import numpy as np
hfov_deg = 79
w, h = 640, 480
fx = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)  # ≈ 388
fy = fx
cx, cy = w / 2.0, h / 2.0  # 320, 240

print(fx)
print(cx)
print(cy)



