#--cong:utf-8--
import numpy as np
import cv2


def invert(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# =========================
# 1. Body <- SVC
# =========================
rvec = [-0.035000639417800267, 2.683251120117661, -1.6100801604567245]
t = [2.181947469711111,-1.0814440250399866, 1.0172706842424688]

R, _ = cv2.Rodrigues(np.array(rvec))
T_body_svc = np.eye(4)
T_body_svc[:3,:3] = R
T_body_svc[:3,3] = t


# =========================
# 2. SVC <- D455
# =========================
T_d455_svc = np.eye(4)
T_d455_svc[:3,:3] = np.array([
    [ 0.9988213,   0.01744383,  0.04529602],
    [-0.01314425,  0.99552857, -0.09354193],
    [-0.04672521,  0.09283629,  0.99458443]
])
T_d455_svc[:3,3] = np.array([0.330893, 0.059162, -0.173514]) 

T_svc_d455 = invert(T_d455_svc)


# =========================
# 3. D455 <- IMU
# =========================
T_cam0_imu = np.eye(4)
T_cam0_imu[:3,:3] = np.array([
    [0.999994, 0.000078, 0.003535],
    [-0.000097, 0.999987, 0.005146],
    [-0.003535, -0.005147, 0.999981]
])
T_cam0_imu[:3,3] = np.array([-0.028933, -0.007453, -0.015558])

T_d455_imu = invert(T_cam0_imu)


# =========================
# 4. BODY <- IMU
# =========================
T_body_imu = T_body_svc @ T_svc_d455 @ T_d455_imu

# =========================
# 5. IMU <- BODY（你要的）
# =========================
T_imu_body = invert(T_body_imu)

print("T_imu_body =\n", T_imu_body)
print("T_body_imu =\n", T_body_imu)

# =========================
# 6. SVC RGB -> D455 IMU (你需要的)
# =========================
# T_svc_imu = T_svc_d455 @ T_d455_imu (SVC 坐标系到 IMU 坐标系)
T_svc_imu = T_svc_d455 @ T_d455_imu
T_imu_svc = invert(T_svc_imu)

print("\n========================================")
print("SVC RGB -> D455 IMU 外参 (T_imu_svc)")
print("========================================")
print("R =\n", T_imu_svc[:3, :3])
print("t =", T_imu_svc[:3, 3])
rvec_svc_imu, _ = cv2.Rodrigues(T_imu_svc[:3, :3])
euler = np.degrees(rvec_svc_imu.flatten())
print(f"欧拉角 (deg, xyz/RPY) = {euler}")
print(f"rvec (Rodrigues) = {rvec_svc_imu.flatten()}")
print("\nT_imu_svc =\n", T_imu_svc)
print()
print("T_svc_imu =\n", T_svc_imu)