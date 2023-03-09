import numpy as np
import os
import imageio


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
        # ; 如果不存在，那么需要加载，也就是重新生成
        if not os.path.exists(imgdir):
            needtoload = True
    # 这个分值就是使用分辨率来降采样图片
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output  # 执行外部命令

    # ; 运行到这里，说明没有降采样的图片，那么需要自己生成降采样的图片
    imgdir = os.path.join(basedir, 'images')   # 原始数据集的图片
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    # list N个图像数据路径
    # ; any函数，只要列表中有一个不是false或者空，那么返回就是true
    # ; 所以这里这一步就是把上面读取的原始数据集图片进行一个筛选，让图片的格式必须是下面几个格式之一，否则直接就不加入路径了
    imgs = [f for f in imgs if any(
        [f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    # '/home/cc/nerf-learn/nerf-pytorch_Comments'
    wd = os.getcwd()  # 获取代码路径

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
        # ; check_output是执行外部命令：https://python3-cookbook.readthedocs.io/zh_CN/latest/c13/p06_executing_external_command_and_get_its_output.html
        # ; 下面的操作就是把原始数据集文件夹下的图片，复制到新建的images_8文件夹下
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]  # 取出第一张图片名称的格式，比如JPG
        args = ' '.join(['mogrify', '-resize', resizearg,
                        '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)  # 修改工作路径，进入images_8文件夹下
        # ; 注意这里还是调用命令行执行命令，命令是mogrify，这个并不是ubuntu标准命令，而是另外一个包中带有的，
        # ; 见：https://askubuntu.com/questions/1164/how-to-easily-resize-images-via-command-line
        # ; 可以简单的认为，这个包可以直接resize图像，而不用使用python读取出来图像resize之后再写入，更加方便
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
    """
    # ; 注意.npy是numpy专用的二进制文件存储格式
    # [imagesN,17] [20,17]存放的位姿和bds
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    # ; 注意这里位姿为什么是3x5, 因为3x3是旋转，第4列是平移，第5列分别是h, w, f
    # [3,5,imagesN][3,5,20]
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])  # bounds 深度范围[2,imagesN] [2,20]

    # ; 下面的操作首先是一个列表生成，先使用listdir把文件夹下的所有文件列出来，然后sorted对文件名排序，
    # ; 然后if条件生成，得到图片名称的路径。最后一步[0]取出第一张图像的文件名
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images')))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]  # 读取一张图像
    sh = imageio.imread(img0).shape  # [h,w,c]  eg.[3024,4032,3]

    sfx = ''

    # 如果有下采样相关参数输入，则对图像进行下采样
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        #! 疑问：这句是啥意思？自己赋值自己有啥毛病？
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

    # 判断是否存在采样后的数据路径
    imgdir = os.path.join(basedir, 'images' + sfx)
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
    # 获取图像shape
    # [h/factor,w/factor,c]  [378,504,3]
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape(
        [2, 1])  # 3x5第五列存放的是h,w,f,使用下采样图像需要更新h,w
    poses[2, 4, :] = poses[2, 4, :] * 1./factor  # 更新f=f_ori/factor

    # ; 注意这里默认load_imgs参数是true，所以继续往下走
    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    #! 疑问：这里赋值两遍有啥用？
    #! 疑问：这里切片好像就等价于全部取出来？那为什么要写成[..., :3]这种形式呢？
    imgs = imgs = [imread(f)[..., :3]/255. for f in imgfiles]
    # ; np.stack，沿着最后一个轴对数据进行stack,从而数据就扩展出一个维度
    imgs = np.stack(imgs, -1)  # [h/factor,w/factor,c,imagesN]

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)  # np.linalg是线性代数模块


def viewmatrix(z, up, pos):
    vec2 = normalize(z)  # 旋转矩阵最后一列先归一化，这就是Z轴在world系下的表示
    vec1_avg = up  # y轴在world系下表示，没有归一化
    vec0 = normalize(np.cross(vec1_avg, vec2))  # y*z,得到x轴，再归一化。xyzxyzxyz顺序是右手系
    vec1 = normalize(np.cross(vec2, vec0))  # z*x，得到y轴，再归一化
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts-c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    # 先把位姿中最后一列存储的hwf，也就是高、宽、焦距取出来
    hwf = poses[0, :3, -1:]
    # 把位姿中第4列，也就是平移取出来，求平均，得到平移的均值
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0)) # 对旋转向量的最后一列求和，再归一化
    up = poses[:, :3, 1].sum(0)  # 对旋转向量的第2列求和
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

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


# 计算poses 的均值，将所有pose做个该均值逆转换，简单来说重新定义了世界坐标系，
# 原点大致在被测物中心，然后world系的姿态也在所有相机姿态的平均值上，这样保证所有姿态都不有很大或者很小的值
def recenter_poses(poses):
    #; 注意这里+0就是为了新生成一个对象，否则就是引用，仍然是原来的结果
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses) # 结果是(3, 5)，就是poses中所有位姿的平均，最后一列仍然是hwf
    # 最下面加一行0001，变成(4,4)齐次坐标形式，这里最后用的是-2，写成0更好理解
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)  
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1]) # np.tile把数组沿着各个方向复制
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # 把原来所有的位姿也都改成齐次坐标形式

    #; (4,4) * (20, 4, 4)，这里应该是有广播操作？
    poses = np.linalg.inv(c2w) @ poses  # 4x4矩阵相乘
    poses_[:, :3, :4] = poses[:, :3, :4]  # 取3x4，也就是把其次坐标去掉，只拿有用的部分
    poses = poses_
    return poses


#####################


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

    # 原始数据读取，位姿:poses，采样far,near信息即深度值范围:bds，rgb图像:imgs
    # poses[3,5,N] bds [2,N] imgs[h,w,c,N]
    # factor=8 downsamples original imgs by 8x 图像下采样
    poses, bds, imgs = _load_data(basedir, factor=factor)

    print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    # 选择矩阵变换[x,y,z]->[y,-x,z]，把变量维度就是表示图像个数的移到第一个维度poses[N,3,5] bds [N,2] images[N,h,w,c]
    #! 疑问：这里为什么要对旋转矩阵进行变换？
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)  # 这个意思就是把最后一个维度移动到第0个维度
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    #! 疑问：为什么对边界进行缩放，对pose中的T也要缩放？另外为什么要对边界进行缩放呢？
    # 边界进行缩放，pose中的T也要对应缩放，有点类似归一化操作
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    # 计算poses 的均值，将所有pose做个该均值逆转换，简单来说重新定义了世界坐标系，原点大致在被测物中心
    if recenter:
        poses = recenter_poses(poses)

    if spherify:  # 是否对位姿进行球形化，好像正常操作没有这样设置
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        # 前面做过recenter pose的均值就是R是单位阵，T是0
        #; 注意这里如何理解，因为之间对旋转和平移进行了去中心化，所以这里再次计算平均值的化，就是I了
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3, :4])

        # Step 生成螺旋路径？下面的暂时没有看懂
        # Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        # 最近最远深度值
        # 又对设置的边界进行了缩放，这里感觉是为了扩大范围？
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        #! 疑问：这又是啥操作？
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
        if path_zflat:
            # zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        # 生成用来渲染的螺旋路径的位姿
        render_poses = render_path_spiral(
            c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    render_poses = np.array(render_poses).astype(np.float32)  # [n_views,3,5]

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)  # (20, 3, 5) (20, 378, 504, 3) (20, 2)

    #; 计算平移距离世界坐标系原点最近的那个相机，作为holdeout view
    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)  # 距离最小值对应的下标
    print('HOLDOUT view is', i_test)  # llff/fern数据集，结果是12

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test
