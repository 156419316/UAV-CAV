state = {
    # 无人机特征图（经过编码压缩）
    'uav_feature': tensor(16, 32, 32),  # [通道, 高, 宽]
    # 地面车辆状态（多车）
    'ground_vehicles': [
        {
            'id': 0,
            'position': [x, y, z],          # 全局坐标系
            'velocity': [vx, vy, vz],       # 速度
            'sensor_fov': [90, 120],        # 水平和垂直视场角
            'detection_history': tensor(10)  # 最近10帧的检测置信度（反映当前感知盲区）
        },
        # 其他车辆...
    ],
    # 通信状态
    'bandwidth': 15.0,          # 当前带宽 (Mbps)
    'latency': 0.05,            # 上一次传输延迟 (秒)
    # 环境信息
    'weather': [0.3, 0.0],      # 天气参数（雨量、雾浓度，影响感知）
    'obstacle_map': tensor(32,32)  # 障碍物分布热力图（可选）
}