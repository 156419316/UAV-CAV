UAV_Feature_Selection_RL/
├── policy/
│   ├── uav_policy.py
│   └── sample.py
├── detector/
│   └── fusion_detector.py
├── dataset/
│   └── nuscenes_uav_loader.py
├── eval/
│   └── nuscenes_eval.py
├── utils/
│   ├── logger.py
│   ├── visualize.py
│   ├── rl_buffer.py
│   └── nuscenes_helper.py
├── config/
│   └── fusion_config.yaml
├── logs/
│   ├── uav_rl/
│   │   └── best_policy.pth
│   └── attention_maps/
├── train_rl.py
├── eval_uav_policy.py
└── train_uav_policy.sh
