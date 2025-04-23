import os
import sys
sys.path.append(os.path.abspath('/home/lancegan/Datas/Codes/Python/P1'))
from detector.pointpillars.utils import keep_bbox_from_image_range, keep_bbox_from_lidar_range
import numpy as np


def wrap2kitti(results:dict,data_dict:dict,pcd_limit_range:list,LABEL2CLASSES:dict):
    """_summary_
    Wrap the single result to kitti format
    arguments:
        result (dict): 由pointpillar得到的结果字典
        data_dict (dict): 数据集的字典
        pcd_limit_range (list): 点云范围
        LABEL2CLASSES (dict): 标签到类别的映射
    """
    format_results = []
    for j, result in enumerate(results):
            format_result = {
                        'name': [],
                        'truncated': [],
                        'occluded': [],
                        'alpha': [],
                        'bbox': [],
                        'dimensions': [],
                        'location': [],
                        'rotation_y': [],
                        'score': []
                    }
            calib_info = data_dict['batched_calib_info'][j]
            tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
            r0_rect = calib_info['R0_rect'].astype(np.float32)
            P2 = calib_info['P2'].astype(np.float32)
            image_shape = data_dict['batched_img_info'][j]['image_shape']
            idx = data_dict['batched_img_info'][j]['image_idx']
            result_filter = keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape)
            result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)

            lidar_bboxes = result_filter['lidar_bboxes']
            labels, scores = result_filter['labels'], result_filter['scores']
            bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
            for lidar_bbox, label, score, bbox2d, camera_bbox in \
                zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                format_result['name'].append(LABEL2CLASSES[label])
                format_result['truncated'].append(0.0)
                format_result['occluded'].append(0)
                alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
                format_result['alpha'].append(alpha)
                format_result['bbox'].append(bbox2d)
                format_result['dimensions'].append(camera_bbox[3:6])
                format_result['location'].append(camera_bbox[:3])
                format_result['rotation_y'].append(camera_bbox[6])
                format_result['score'].append(score)
                
            format_results.append({k:np.array(v) for k, v in format_result.items()} )
               
    return format_results
            