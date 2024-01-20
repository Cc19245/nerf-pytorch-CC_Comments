import numpy as np

# 示例数组形状为 (3, 4, 5)
arr = np.random.random((3, 4, 5))

# 将轴从 0 移动到 2，交换第一个和第三个轴
arr_moved = np.moveaxis(arr, 0, 2)

# 打印移动后的数组形状
print(arr_moved.shape)
