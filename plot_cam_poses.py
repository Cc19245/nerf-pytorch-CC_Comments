import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_camera_pose(ax, pose, size=1.0):
    """绘制一个相机的位姿

    Args:
        ax (_type_): 绘图对象
        pose (_type_): (3, 4) or (4, 4) 相机位姿，world系下表示
        size (float, optional): xyz轴的长度
    """

    # 绘制相机的三个轴
    ax.plot3D([pose[0, 3], pose[0, 3] + size * pose[0, 0]],
              [pose[1, 3], pose[1, 3] + size * pose[1, 0]],
              [pose[2, 3], pose[2, 3] + size * pose[2, 0]], 'r-')
    ax.plot3D([pose[0, 3], pose[0, 3] + size * pose[0, 1]],
              [pose[1, 3], pose[1, 3] + size * pose[1, 1]],
              [pose[2, 3], pose[2, 3] + size * pose[2, 1]], 'g-')
    ax.plot3D([pose[0, 3], pose[0, 3] + size * pose[0, 2]],
              [pose[1, 3], pose[1, 3] + size * pose[1, 2]],
              [pose[2, 3], pose[2, 3] + size * pose[2, 2]], 'b-')
    
    # 绘制相机中心
    ax.scatter3D(pose[0, 3], pose[1, 3], pose[2, 3], c='b', marker='o')

def plot_camera_poses(poses, size=1.0):
    """
    绘制多个相机位姿
    """
    # 绘制图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(poses.shape[-1]):
        pose = poses[:, :4, i]  # (3, 4)
        plot_camera_pose(ax, pose, size=size)
    
    # 设置三个坐标轴相等
    ax.set_box_aspect([np.ptp(coord) for coord in \
                    zip(*[pose[:3, 3] for pose in poses.transpose(2, 0, 1)])])


# 测试
if __name__ == '__main__':
    size = 1

    # 示例相机位姿（4x4 变换矩阵）
    camera_poses = np.random.random((3, 5, 20))

    # 绘制多个相机
    plot_camera_poses(camera_poses, size=size)

    # # 设置轴限以便更好地可视化
    # ax.set_xlim([0, 10])
    # ax.set_ylim([0, 10])
    # ax.set_zlim([0, 10])

    plt.show()
