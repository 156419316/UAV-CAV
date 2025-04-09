import os
import numpy as np
from collections import defaultdict
from detector.pointpillars.utils.process import compute_ap, match_boxes


class KITTIEvaluator:
    def __init__(self, label_dir, class_names=['Car', 'Pedestrian', 'Cyclist'], iou_thresh=0.5):
        self.label_dir = label_dir
        self.class_names = class_names
        self.iou_thresh = iou_thresh
        self.gt_dict = self._load_all_gt()

    def _load_all_gt(self):
        """
        自动加载所有 ground truth 标签，构造 gt_dict
        返回: dict[sample_token] = list[dict]
        """
        gt_dict = {}
        for fname in sorted(os.listdir(self.label_dir)):
            if not fname.endswith('.txt'):
                continue
            token = fname.split('.')[0]
            gt_path = os.path.join(self.label_dir, fname)
            with open(gt_path, 'r') as f:
                lines = f.readlines()
            boxes = []
            for line in lines:
                parts = line.strip().split()
                name = parts[0]
                if name not in self.class_names:
                    continue
                trunc, occ, alpha = float(parts[1]), int(parts[2]), float(parts[3])
                bbox = list(map(float, parts[4:8]))  # not used here
                h, w, l = map(float, parts[8:11])
                x, y, z = map(float, parts[11:14])
                ry = float(parts[14])
                boxes.append({
                    'name': name,
                    'translation': [x, y, z],
                    'size': [w, l, h],
                    'rotation_y': ry
                })
            gt_dict[token] = boxes
        return gt_dict

    def evaluate(self, pred_dict):
        """
        自动使用已加载的 ground truth 进行评估
        参数:
            pred_dict: dict[sample_token] = list[prediction dict]
        返回:
            mean_ap, ap_dict
        """
        all_tps, all_fps, all_scores, all_gt_counts = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(int)

        for token in pred_dict.keys():
            preds = pred_dict[token]
            gts = self.gt_dict.get(token, [])

            for cls in self.class_names:
                cls_preds = [p for p in preds if p['name'] == cls]
                cls_gts = [g for g in gts if g['name'] == cls]

                tp, fp, scores, num_gt = match_boxes(cls_preds, cls_gts, self.iou_thresh)
                all_tps[cls].extend(tp)
                all_fps[cls].extend(fp)
                all_scores[cls].extend(scores)
                all_gt_counts[cls] += num_gt

        ap_dict = {}
        for cls in self.class_names:
            ap = compute_ap(all_tps[cls], all_fps[cls], all_scores[cls], all_gt_counts[cls])
            ap_dict[cls] = ap
        mean_ap = np.mean(list(ap_dict.values())) if ap_dict else 0.0
        return mean_ap, ap_dict
