import numpy as np

def check_timestamps():
    wheel_file = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/6.24/speed/recordSpeed_20260624_153736.txt"
    imu_file = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/6.24/time/scene_1/imu0/data.csv"

    wheel_data = np.loadtxt(wheel_file)
    imu_data = np.loadtxt(imu_file, delimiter=',', skiprows=1)

    print(f"轮速文件: {len(wheel_data)} 条, 时间范围: {wheel_data[0,0]} ~ {wheel_data[-1,0]} (纳秒)")
    print(f"轮速秒范围: {wheel_data[0,0]/1e9:.3f} ~ {wheel_data[-1,0]/1e9:.3f} 秒")
    print(f"IMU 文件: {len(imu_data)} 条, 时间范围: {imu_data[0,0]} ~ {imu_data[-1,0]} (纳秒)")
    print(f"IMU 秒范围: {imu_data[0,0]/1e9:.3f} ~ {imu_data[-1,0]/1e9:.3f} 秒")
    overlap_start = max(wheel_data[0,0], imu_data[0,0])
    overlap_end = min(wheel_data[-1,0], imu_data[-1,0])
    if overlap_end > overlap_start:
        overlap_sec = (overlap_end - overlap_start) / 1e9
        print(f"重叠时长: {overlap_sec:.2f} 秒")
    else:
        print("⚠️ 两个数据没有时间重叠！")

if __name__ == "__main__":
    check_timestamps()