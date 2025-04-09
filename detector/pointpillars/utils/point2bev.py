from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

 
'''
可视化BEV鸟瞰图
'''
def show_lidar_topview(pc_velo, objects, calib):
    '''
    pc_velo: 读取的点云数据，(N, 4)
    '''
    side_range = (-30, 30)  # 左右距离
    fwd_range = (0, 80)  # 后前距离
    
    x_points = pc_velo[:, 0]
    y_points = pc_velo[:, 1]
    z_points = pc_velo[:, 2]
    
    # 2-获得区域内的点
    f_filt = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    s_filt = np.logical_and(y_points > side_range[0], y_points < side_range[1])
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten() 
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    
    # 定义了鸟瞰图中每个像素代表的距离
    res = 0.1   
    # 3-1将点云坐标系 转到 BEV坐标系
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)
    # 3-2调整坐标原点
    x_img -= int(np.floor(side_range[0]) / res)
    y_img += int(np.floor(fwd_range[1]) / res)
    print(x_img.min(), x_img.max(), y_img.min(), y_img.max()) 
    
    # 4-填充像素值, 将点云数据的高度信息（Z坐标）映射到像素值
    height_range = (-3, 1.0)
    pixel_value = np.clip(a=z_points, a_max=height_range[1], a_min=height_range[0])
     
 
    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)
    
    pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
    
    # 创建图像数组
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    
    im2 = Image.fromarray(im)
    #im2.save('save_output/BEV.png')
    im2.show()

if __name__ == '__main__':
    
    velody_root = '/home/lancegan/Datas/Codes/Python/KITTI/training/velodyne/000001.bin'
    point = np.fromfile(velody_root, dtype=np.float32).reshape(-1, 4)
    show_lidar_topview(point, None, None)
    
    
    """ point = np.fromfile('/home/lancegan/Datas/Codes/Python/KITTI/training/velodyne/000001.bin', dtype=np.float32)
    print(point.shape)
    pts = point.reshape(-1, 4)
    print(pts.shape)
    plt.figure(figsize=(12, 8))
    plt.scatter(pts[:,0],pts[:,1],0.5)
    plt.show()  """