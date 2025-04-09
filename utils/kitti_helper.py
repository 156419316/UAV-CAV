def wrap_predictions_for_kitti(pred_dicts):
    """
    将模型预测结果转换为 KITTI 评估格式
    参数:
        pred_dicts: list of dict, 每个预测框包含:
            - name: 类别名
            - translation: [x, y, z]
            - size: [w, l, h]
            - rotation_y: float
            - score: float
    返回:
        dict[sample_token] = list[dict]
    """
    result = {}
    for token, preds in pred_dicts.items():
        result[token] = []
        for det in preds:
            if det['name'] == 'DontCare':
                continue
            box = {
                'name': det['name'],
                'translation': det['translation'],
                'size': det['size'],
                'rotation_y': det['rotation_y'],
                'score': det.get('score', 1.0),
                'bbox': det.get('bbox', [0, 0, 50, 50])  # dummy 2D bbox
            }
            result[token].append(box)
    return result
