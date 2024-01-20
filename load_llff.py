import numpy as np
import os
import imageio

# 显示相机位姿
from plot_cam_poses import plot_camera_poses

# Slightly modified version of LLFF data loading code
# see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    """_summary_
        输入basedir原始数据集文件路径，输入factors下采样倍数或者resolutions想要缩放到的分辨率，
        然后对创建缩放的文件夹，将原始数据集进行缩放，存储到新的文件夹中
    """
    needtoload = False
    # 判断本地是否已经存有下采样factors或者对应分辨率的图像，如果没有需要重新加载
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        # 如果不存在，那么需要加载，也就是重新生成
        if not os.path.exists(imgdir):
            needtoload = True
    # 这个分值就是使用分辨率来降采样图片
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    
    #; 运行到这里，说明没有降采样的图片，那么需要自己生成降采样的图片
    from shutil import copy
    from subprocess import check_output  # 执行外部命令

    imgdir = os.path.join(basedir, 'images')   # 原始数据集的图片
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    # list N个图像数据路径
    # ; any函数，只要列表中有一个不是false或者空，那么返回就是true
    # ; 所以这里这一步就是把上面读取的原始数据集图片进行一个筛选，让图片的格式必须是下面几个格式之一，否则直接就不加入路径了
    imgs = [f for f in imgs if any(
        [f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    # '/home/cc/nerf-learn/nerf-pytorch_Comments'
    wd = os.getcwd()  # 获取当前代码所在的路径

    for r in factors + resolutions:
        if isinstance(r, int):  # 如果是整型，那么就是缩放几倍
            name = 'images_{}'.format(r)  # 'images_8'
            resizearg = '{}%'.format(100./r)  # '12.5%'
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):  # 如果已经存在images_8这种文件夹，那么直接跳过
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)  # 新建数据路径 images_8
        # check_output是执行外部命令：https://python3-cookbook.readthedocs.io/zh_CN/latest/c13/p06_executing_external_command_and_get_its_output.html
        # 下面的操作就是把原始数据集文件夹下的图片，复制到新建的images_8文件夹下
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]  # 取出第一张图片名称的格式，比如JPG
        args = ' '.join(['mogrify', '-resize', resizearg,
                        '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)  # 修改工作路径，进入images_8文件夹下
        # 注意这里还是调用命令行执行命令，命令是mogrify，这个并不是ubuntu标准命令，而是另外一个包中带有的，
        # 见：https://askubuntu.com/questions/1164/how-to-easily-resize-images-via-command-line
        # 可以简单的认为，这个包可以直接resize图像，而不用使用python读取出来图像resize之后再写入，更加方便
        check_output(args, shell=True)
        os.chdir(wd)  # 再回到当前工程的根目录下

        # 如果原来的图片结尾不是png，那么还要把原来的图片删掉
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    """_summary_
        加载图像数据集，调用时只传入前两个参数，其他都是默认
    Args:
        basedir (string): 数据集路径
        factor (int, optional): 图像缩放因子. Defaults to None.
        width (_type_, optional): _description_. Defaults to None.
        height (_type_, optional): _description_. Defaults to None.
        load_imgs (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # Step 1. 读取位姿和场景深度范围
    # [imagesN,17] [20,17]存放的位姿和bds
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))   # .npy是numpy专用的二进制文件存储格式
    # 注意这里位姿为什么是3x5, 因为3x3是旋转，第4列是平移，第5列分别是h, w, f
    # (20, 15) -> (20, 3, 5) -> (3, 5, 20)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])  # bounds 深度范围, (20, 2) -> (2, 20)

    # 下面的操作首先是一个列表生成，先使用listdir把文件夹下的所有文件列出来，然后sorted对文件名排序，
    # 然后if条件生成，得到图片名称的路径。最后一步[0]取出第一张图像的文件名
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images')))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]  
    sh = imageio.imread(img0).shape  # [h,w,c]  eg.[3024,4032,3]

    sfx = ''

    # Step 2. 如果要对图像进行缩小，则判断是否存在缩小的图像，不存在则进行创建
    if factor is not None:
        sfx = '_{}'.format(factor)  # _8
        # 这里是判断要缩小后的图片数据集中是否,存在如果不存在就在这里用代码手动缩放并存储起来，方便后面读取
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    # 判断是否存在采样后的数据路径，是为了验证前面一步缩小图像是否正确执行了
    imgdir = os.path.join(basedir, 'images' + sfx)  # images_8
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    # pose数量应与图像保持一致
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(
        imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(
            len(imgfiles), poses.shape[-1]))
        return
    
    # Step 3. 根据缩放倍数更新相机的h,w,f（内参）
    # 图像大小，[h/factor,w/factor,c]  [378,504,3]
    sh = imageio.imread(imgfiles[0]).shape
    # 第5列的前2行存储图像h,w，这里就是缩放后的378, 504
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])  
    # 第5列的第3行存储相机的焦距，图像缩小了，则焦距对应缩小
    poses[2, 4, :] = poses[2, 4, :] * 1./factor  # 更新f=f_ori/factor

    # 注意这里默认load_imgs参数是true，所以继续往下走
    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            #; 原始语句运行报错解决：https://zhuanlan.zhihu.com/p/630948914
            # return imageio.imread(f, ignoregamma=True)
            return imageio.imread(f, format="PNG-PIL", ignoregamma=True)
        else:
            return imageio.imread(f)

    # [(378, 504, 3), ...], 也就是每一张图像的数据
    imgs = imgs = [imread(f)[..., :3]/255. for f in imgfiles]
    # (378, 504, 3, 20) 沿着最后一个轴对数据进行stack,从而数据就扩展出一个维度
    imgs = np.stack(imgs, -1)  # [h/factor,w/factor,c,imagesN]
    # 打印结果： (378, 504, 3, 20)  [378, 504, 407.565]就是 h,w,f
    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def normalize(x):
    """ 对传入的numpy数组进行归一化"""
    return x / np.linalg.norm(x)  # np.linalg是线性代数模块


def viewmatrix(z, up, pos):
    vec2 = normalize(z)  # 旋转矩阵最后一列先归一化，这就是Z轴在world系下的表示
    vec1_avg = up  # y轴在world系下表示，没有归一化
    vec0 = normalize(np.cross(vec1_avg, vec2))  # y*z,得到x轴，再归一化。xyzxyzxyz顺序是右手系
    vec1 = normalize(np.cross(vec2, vec0))  # z*x，得到y轴，再归一化
    # [(3,), ...], 沿着维度1拼接，就是扩展出一个维度，变成 (3, 4)
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts-c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    """对所有相机的位姿求均值，以重新定义世界坐标系

    Args:
        poses (_type_): (20, 3, 5)，（相机个数，3x3旋转，3x1平移，3x1 hwf）

    Returns:
        _type_: (3, 5) 所有相机位姿的均值
    """
    # (3, 1) 先把位姿中最后一列存储的hwf，也就是高、宽、焦距取出来
    hwf = poses[0, :3, -1:]
    # (3,) 把位姿中第4列，也就是平移取出来，求平均，得到平移的均值
    center = poses[:, :3, 3].mean(0)
    # (3,) poses[:, :3, 2]: (20, 3)  .sum(0): (3,)    
    #; 对所有相机的旋转向量的最后一列求和，再归一化。注意最后一列其实就是相机的Z轴，所以这里就是求Z轴的平均方向
    vec2 = normalize(poses[:, :3, 2].sum(0)) 
    # (3,)
    #; 旋转向量的第2列求和，其实就是求相机Y轴的平均方向
    up = poses[:, :3, 1].sum(0)  # 对旋转向量的第2列求和
    # [(3, 4), (3, 1)] 沿着维度1拼接 -> (3, 5)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    # (3, 5) 最终返回的结果就是所有相机位姿的均值
    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array(
            [np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def recenter_poses(poses):
    """计算poses 的均值，将所有pose做个该均值逆转换，简单来说重新定义了世界坐标系，
       原点大致在被测物中心，然后world系的姿态也在所有相机姿态的平均值上，这样保证
       所有姿态都不有很大或者很小的值

    Args:
        poses (_type_): (20, 3, 5) 
    Returns:
        _type_: (20, 3, 5) worldb系被重新定义到场景中心
    """
    #; 注意这里+0就是为了新生成一个对象，否则就是引用，仍然是原来的结果
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    # (3, 5)，就是poses中所有位姿的平均，最后一列仍然是hwf
    c2w = poses_avg(poses) 
    # (4, 4) 最下面加一行0001，变成齐次坐标形式，这里最后用的是-2，写成0更好理解
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)  
    # (20, 1, 4) np.tile把数组沿着各个方向复制
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1]) 
    # [(20, 3, 4), (20, 1, 4)] -> (20, 4, 4)
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # 把原来所有的位姿也都改成齐次坐标形式

    #; 矩阵乘法: (4, 4) @ (20, 4, 4)，注意这里有广播操作，首先会把(4, 4)复制变成(20, 4, 4)，再在后面两个维度进行矩阵乘法
    # (20, 4, 4)，注意经过这个操作之后，相机位姿的参考坐标系就被重新定义到了场景中心
    poses = np.linalg.inv(c2w) @ poses  # 4x4矩阵相乘
    poses_[:, :3, :4] = poses[:, :3, :4]  # 取3x4，也就是把其次坐标去掉，只拿有用的部分
    poses = poses_
    return poses


def spherify_poses(poses, bds):

    def p34_to_44(p): return np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i,
                                [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(
        p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1./rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []

    for th in np.linspace(0., 2.*np.pi, 120):

        camorigin = np.array(
            [radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(
        poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:, :3, :4], np.broadcast_to(
        poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds


def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    """ 读取 llff 格式的数据，返回数据集中的所有图像、图像的相机位姿

    Args:
        basedir (string): 数据集路径
        factor (int, optional): 图像缩放因子，用于降低计算量. Defaults to 8.
        recenter (bool, optional): 是否对位姿进行重新计算，让world系位于场景中心. Defaults to True.
        bd_factor (float, optional): 每个相机观察场景深度范围的缩放阈值. Defaults to .75.
        spherify (bool, optional): 是否是360度环视场景. Defaults to False.
        path_zflat (bool, optional):？？？. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Step 1. 读取数据集中的原始图像、每个图像对应的相机位姿、每个图像对应的场景的深度范围
    # 原始数据读取，位姿:poses，采样far,near信息即深度值范围:bds，rgb图像:imgs
    # (3, 5, 20)  (2, 20)  (378, 504, 3, 20)
    poses, bds, imgs = _load_data(basedir, factor=factor)
    print('Loaded', basedir, bds.min(), bds.max())

    # Step 2. 将LLFF的DRB坐标系转成OpenGL(NeRF)的RUB坐标系
    # Correct rotation matrix ordering and move variable dim to axis 0
    #; 源代码实现方式
    # # 选择矩阵变换[x,y,z]->[y,-x,z]，把变量维度就是表示图像个数的移到第一个维度
    # # [(3, 1, 20), (3, 1, 20), (3, 3, 20)]沿着dim=1进行cat,结果还是(3, 5, 20)
    # poses = np.concatenate(
    #     [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    #; 使用旋转矩阵乘法的实现方式
    r_drb_rub = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # (3, 3) 坐标系转换向量
    rot_w_drb = np.moveaxis(poses[:, :3, :], -1, 0)  # (20, 3, 3) 取出DRB相机坐标系的位姿，并调整batch维度位置
    rot_w_rub = rot_w_drb[..., :3] @ r_drb_rub[None, ...]  # (20, 3, 3) 广播实现矩阵乘法，把相机位姿转到RUB坐标系下
    poses[:, :3, :] = np.moveaxis(rot_w_rub, 0, -1)  # 重新赋值到原始变量中
    #? add: 测试绘制相机的位姿
    # plot_camera_poses(poses[:, :4, :])

    # (3, 5, 20) -> (20, 3, 5), np.moveaxis的后两个参数是把原先-1轴移动成0轴
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    # (378, 504, 3, 20) -> (20, 378, 504, 3)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)  # 这个意思就是把最后一个维度移动到第0个维度
    images = imgs
    # (2, 20) -> (20, 2)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    #! 疑问：为什么对边界进行缩放，对pose中的T也要缩放？另外为什么要对边界进行缩放呢？
    # 边界进行缩放，pose中的T也要对应缩放，有点类似归一化操作
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    # Step 2. 重新定义相机位姿的world系，让world系坐标原点位于场景中心
    # 计算poses 的均值，将所有pose做个该均值逆转换，简单来说重新定义了世界坐标系，原点大致在被测物中心
    if recenter:
        poses = recenter_poses(poses)

    # Step 3. 生成最终渲染视频的新视角的相机位姿
    if spherify:  # 是否对位姿进行球形化，好像正常操作没有这样设置
        poses, render_poses, bds = spherify_poses(poses, bds)
    # 如果不是360度的设置
    else:
        # 前面做过recenter pose的均值就是R是单位阵，T是0
        #; 验证前面的重新定义坐标系是否正确执行了，因为之间对旋转和平移进行了去中心化，
        #; 所以这里再次计算平均值的化就是I了，
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3, :4])

        #! 生成螺旋路径？下面的暂时没有看懂
        # Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        # 最近最远深度值
        # 又对设置的边界进行了缩放，这里感觉是为了扩大范围？
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        #! 疑问：为什么这么计算？
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        # 这里默认是False，所以不执行
        if path_zflat:
            # zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        # [(3, 5), ...]，list长度为120，生成用来渲染的螺旋路径的位姿
        render_poses = render_path_spiral(
            c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    # (120, 3, 5)，即新的视角的相机位姿
    render_poses = np.array(render_poses).astype(np.float32)  # [n_views,3,5]
    
    # (3, 5)，其中相机位姿部分仍然是单位阵
    c2w = poses_avg(poses)  
    print('Data:')
    # (20, 3, 5) (20, 378, 504, 3) (20, 2)
    print(poses.shape, images.shape, bds.shape)  

    #; 计算平移距离世界坐标系原点最近的那个相机，作为holdeout view
    # np.square: (20, 3) np.sum: (20, ), 这里其实就是算所有相机到坐标原点的距离的平方和
    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)  # 距离最小值对应的下标
    print('HOLDOUT view is', i_test)  

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    # (20, 378, 504, 3), (20, 3, 5), (20, 2), (120, 3, 5), 一个数字
    return images, poses, bds, render_poses, i_test
