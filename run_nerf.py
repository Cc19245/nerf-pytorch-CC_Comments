import os
import sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断device:cuda/cpu
np.random.seed(0)  # 生成随机数
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """输入数据，传入网络，计算输出
        inputs：[N_rays, N_samples, 3], N_rays条光线，每条光线上N_samples个点，每个点3维坐标
        viewdirs：[N_rays, 3]，每条光线的方向
        fn：网络模型的对象
        embed_fn：对输入的三维点的编码函数列表，输入3维向量，编码成63维的向量
        embeddirs_fn：对输入的方向的编码函数列表，输入3维方向，编码成27维的向量。这里其实就和论文中有一些不同了，
            论文中的公式写的是两个方向角，但是代码中直接用的是3维的方向向量
        netchunk：每次
    """
    #; 把输入的N_rays*N_samples个采样点全部展平，变成[N_rays*N_samples, 3]维度
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    #; 对每个点进行位置编码，变成[N_rays*N_smaples, 63]
    embedded = embed_fn(inputs_flat)  

    # 如果也使用方向进行渲染，那么再对方向进行编码
    if viewdirs is not None:
        #; [N_rays, 3] => [N_rays, 1, 3]  => [N_rays, N_samples, 3]
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        #; 展平，变成[N_rays*N_samples, 3]
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]]) 
        #; 对方向进行编码，变成[N_rays*N_samples, 27]
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        #; 沿着最后一个维度进行cat，不会增加维度，变成[N_rays*N_samples, 63+27]
        embedded = torch.cat([embedded, embedded_dirs], -1)
    # 以更小的patch-netchunk送进网络跑前向
    #! 注意：这个写法还挺有意思的，如果netchunk不是None的话，里面会生成很多网络调用的对象，组成一个列表分别计算
    #!  最后再沿着这个列表的第0维进行concat得到输出，也就是相当于网络输入一次数据的数量太大了，比如上面就是
    #!  N_rays*N_samples这么多个数据，这样显存可能不支持同时计算这么大的数据(即这么多的3D采样点)，那么就把
    #!  这些点分成很多小批量计算，把每个小批量计算的结果再cat起来，就得到了这些点一起计算的结果
    #; 这个的结果实际上就是调用NeRF网络的前向函数进行计算，得到预测的RGB+密度
    outputs_flat = batchify(fn, netchunk)(embedded)  # [N_rays*N_samples, 4] => [65536,4]
    # reshpe 为[1024,64,4] 4:rgb alpha
    #; list(inputs.shape[:-1])就是把输入的shape除了最后一维，变成list, 结果是[N_rays, N_samples]
    #; 然后outputs_flat.shape[-1]就是把输出的最后一个维度(也就是一个采样点输入网络得到的计算结果维度4)
    #; 用list形式把他们加起来，就变成[N_rays, N_samples, 4]，也就是对每一个采样点，
    #; 都计算出了他们预测的RGB和不透明度
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    在更小的batch上进行渲染，避免超出内存
    """
    all_ret = {}
    #; 从这里就能看出来这个chunk的作用了，实际上就是一个batch中光线数量可能很大，这个时候可以在外面自己把
    #; batch_size改小，但是这样每次梯度下降就只能使用很小的一部分光线进行计算，不准确。所以这里就把一个
    #; batch的光线分几次算，把最终结果加起来，这样和使用batch_size算一次结果是一样的，只不过由于内存不足多次算而已
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            #; 这里的操作就是每一次计算的小批量的结果都存到字典的对应位置中，然后后面再cat
            all_ret[k].append(ret[k]) 
    # 将所有的结果拼接一起返回
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels. 图像高度
      W: int. Width of image in pixels.  宽度
      focal: float. Focal length of pinhole camera. 针孔相机焦距
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results. 同步处理的最大光线数
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch. 每个batch的ray的原点和方向
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.相机到世界的转换矩阵3x4
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.NDC坐标
      near: float or array of shape [batch_size]. Nearest distance for a ray.光线最近距离
      far: float or array of shape [batch_size]. Farthest distance for a ray.光线最远距离
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.使用view方向
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.变换矩阵
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.预测的rgb图
      disp_map: [batch_size]. Disparity map. Inverse of depth.视差图
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned.其它
    """
    if c2w is not None:
        # special case to render full image
        # render所有图像
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        # 使用提供的ray
        rays_o, rays_d = rays  # 注意rays是[2,B,3]，但是这样分别取之后，2维度就消失了，每一个都是[B,3]

    # 默认使用方向来渲染
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        #; 实测这个不会进入
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            # 特殊case,可视化viewdir的影响
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        #; 此处维度是[B, 3]
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # 归一化
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # [n_batch,3] => [1024,3]
    
    #; sh是shape的缩写，结果是一个元组，大小为(N_rays, 3)
    sh = rays_d.shape  # [..., 3] [1024,3]
    #; 如果使用ndc空间，那么还需要把光线原点和方向进行进一步的处理
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    # 构建batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()  # [1024,3]
    rays_d = torch.reshape(rays_d, [-1, 3]).float()  # [1024,3]
    # 均为[1024,1]
    near, far = near * torch.ones_like(rays_d[..., :1]), \
        far * torch.ones_like(rays_d[..., :1])
    #; torch.cat不会增加维度，而是会在指定的维度上进行concat
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # [1024,8]
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)  # [1024,11]

    # Render and reshape
    # 渲染，reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        # print("before: ", k, all_ret[k].shape)
        #; list(sh[:-1])得到第一个维度，即N_rays；list(all_ret[k].shape[1:])得到输出结果的后面的维度
        #; 不过感觉这里的reshape没有必要？已经是这个shape了。经过验证确实是这样，这里reshape是多余的
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
        # print("after: ", k, all_ret[k].shape)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(
            H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """
        Instantiate NeRF's MLP model.
    """
    # 位置编码函数列表，xyz还有view方向的都进行位置编码
    embed_fn, input_ch = get_embedder(
        args.multires, args.i_embed)  # 输入是xyz三维，输出维度input_ch=63
    
    # 打印输出如右边所示，看不出来什么，距离理解看上面的函数内部注释
    # print(embed_fn)  # <function get_embedder.<locals>.embed at 0x7fe8e2582290> 

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        #; 注意这里方向的编码仍然是3维的，和论文中不一样，具体还要看看。这里input_ch_views就是3+3*4*2=27
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    # 输出通道数
    #; 注意N_importance就是fine采样的时候，采样点的个数
    #! 疑问：输出通道为什么是5？不是RGB+不透明度吗？
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    # 输入网络深度和每层宽度，输入通道数不是5，而是位置编码后的通道数
    # 模型初始化
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    # refine 网络初始化，仅仅是网络深度和每层通道数不同
    #! 疑问：为什么refine网络又要重新定义一遍，而不是使用上面的同样的网络？
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())
    
    # 定义lambda函数，将查询传递给network_fn的函数，模型批量化处理数据，
    def network_query_fn(inputs, viewdirs, network_fn): 
        return run_network(inputs, viewdirs, network_fn,
                            embed_fn=embed_fn,
                            embeddirs_fn=embeddirs_fn,
                            netchunk=args.netchunk)

    # Create optimizer
    # 优化器定义
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    # 加载已有模型参数
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        #; 对之前的ckpts进行排序，放到list中
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    #; 不加载已有模型的话，下面不执行
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    # 训练需要的参数
    #; 注意这个字典中包含了很多东西，层层嵌套传递给渲染的函数，真他妈的服了，写这种代码可读性非常差！
    render_kwargs_train = {
        'network_query_fn': network_query_fn,  # 调用网络的函数
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,   #; 网络模型
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            raw就是网络预测的结果，每个采样点上都预测4个值，其中3个RGB颜色，1个不透明度
        z_vals: [num_rays, num_samples along ray]. Integration time.
            不是积分时间，是从near到far的深度值，也就是上面每个采样点的深度值
        rays_d: [num_rays, 3]. Direction of each ray.
            每条光线的方向向量
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            每条光线(每个像素点)的输出颜色
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
            每条光线(每个像素点)的视差，实际上和深度图成反比
        acc_map: [num_rays]. Sum of weights along each ray.
            每条光线的累积权重
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            每条光线上的每个点的权重
        depth_map: [num_rays]. Estimated distance to object.
            每条光线(每个像素点)的深度
    """
    # 见论文公式3中alpha公式定义
    def raw2alpha(raw, dists, act_fn=F.relu): 
        #! 注意：这里对输出的不透明度又加了relu，那么就是说网络输出的并不是最终的不透明度？那为什么不把这个
        #!   relu放到网络里呢？还是说这里只是为了实现一个max的效果，因为密度必须>0?
        return 1. - torch.exp(-act_fn(raw)*dists)

    # ti+1-ti计算
    #; 这个dists就是论文中的\delta_i，这里就是用后一个点-前一个点，得到相邻两点的距离
    dists = z_vals[..., 1:] - z_vals[..., :-1]  #; 维度[N_rays, N_samples-1]
    # [N_rays, N_samples]
    #; dists[..., :1].shape是(N_rays, 1)，然后前面对维度为(1,)进行expand，得到(N_rays, 1)
    #; 最后对上面的dist沿着最后一个轴进行cat，从而得到[N_rays, N_smaples]维度的输出
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  
    #; [N_rays, 1, 3]最后一个维度求模长，
    #! 疑问：这里求模长干什么？模长不都是1吗？
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # rgb
    #! 疑问：这里为什么要对输出颜色取sigmoid？彩色是为了让输出的颜色在0-1之间？那么为什么不把它放到网络里呢？
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    
    # alpha值计算
    #; 这里可以看到，是对最后输出的密度加一个noise, 然后再和\delta_i相乘算alpha, 就类似透明度
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    # 权重计算见公式3
    #; 是沿着某个维度计算累积的乘积，torch.ones((alpha.shape[0], 1))得到[N_rays, 1]维度全1的结果，
    #; 1.-alpha得到[N_rayse, N_samples]结果，torch.cat([x,y], -1)沿着最后一个维度concat,得到
    #; [N_rays, N_samples+1]结果，然后沿着最后一个维度计算累积乘积，就是论文中的w，最后丢掉最后的维度
    #; 结果仍然是[N_rays, N_samples]
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    # rgb计算，权重加和公式3
    #; [N_rays, N_samples, 1] * [N_rays, N_samples, 3]，广播得到[N_rays, N_samples, 3]
    #; 最后沿着倒数第2个维度求和，得到[N_rays, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    
    # 深度图计算d=Σwizi
    #! 从这里可以理解上面的权重到底是什么，其实就是概率密度函数，所以不论是rgb颜色还是深度图，都是在算期望
    depth_map = torch.sum(weights * z_vals, -1)
    
    # 视差图计算
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))
    #; 沿着最后一个维度求和，输入[N_rays, N_samples]，输出(N_rays,)维度
    acc_map = torch.sum(weights, -1)  # 权重加和[1024]

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map



def render_rays(ray_batch,
                network_fn,         # coarse网络模型，字典中传入
                network_query_fn,   # 调用网络输出的函数，字典中传入
                N_samples,          # 光线上coarse的采样个数
                retraw=False,       # 是否返回模型的原始输出
                lindisp=False,
                perturb=0.,         # 是否添加扰动
                N_importance=0,     # 光线上fine的采样个数
                network_fine=None,  # fine网络模型，字典中传入
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    体素渲染
    Args:
      ray_batch: 用来view ray采样的所有必须数据：ray原点，方向，最大最小距离，单位方向array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.

      network_fn:  NeRF网络，用来预测空间中每个点rgb和透明度 function. Model for predicting RGB and density at each point
        in space.

      network_query_fn: 将查询传递给network_fn的函数function used for passing queries to network_fn.

      N_samples: coarse采样点数，int. Number of different times to sample along each ray.

      retraw: bool. 如果为真，返回数据无压缩，If True, include model's raw, unprocessed predictions.
      lindisp: bool. 如果为true在在深度图的逆上线性采样 If True, sample linearly in inverse depth rather than in depth.
      perturb: 扰动 float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: fine增加的精细采样点数 int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: fine网络 "fine" network with same spec as network_fn.
      white_bkgd: 若为true则认为是白色背景 bool. If True, assume a white background.
      raw_noise_std:噪声 ...
      verbose: 打印debug信息 bool. If True, print more debugging info.
    Returns:
      rgb_map: ray的rgb[num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: 视差[num_rays]. Disparity map. 1 / depth.
      acc_map:累积不透明度 [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: 原始raw数据[num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: 标准差[num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]  # 传入的这一小批数据的光线个数
    #; 相机原点和方向
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    #; 如果使用视角渲染，则这里就是方向 [N_rays, 3]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    #; 渲染深度的范围
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])  # [N_rays, 1, 2]
    #; 注意下面得到的near,far是2dim的，即[N_rays, 1]
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1] each[N_rays, 1]

    # Step 1 corse sample, 得到粗糙的渲染结果
    #; 这里的N_samples就是配置文件中的N_samples，就是在z轴上进行多少个点的采样
    t_vals = torch.linspace(0., 1., steps=N_samples)  # 0-1线性采样N_samples个值
    #; 如果不是在disp视差图上线性采样，那么就是在深度图上进行线性采样，所以直接对z进行线性采样即可
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    #; 维度张量的维度，也就是每一条光线都按照这个来进行采样
    z_vals = z_vals.expand([N_rays, N_samples])  # [N_rays, N_samples]

    #; 如果施加扰动，暂时没看
    if perturb > 0.:
        # get intervals between samples
        # 均值[N_rays, N_samples-1]
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)  # [N_rays, N_samples]
        lower = torch.cat([z_vals[..., :1], mids], -1)  # [N_rays, N_samples]
        # stratified samples in those intervals
        # 在这些间隔中分层采样点
        t_rand = torch.rand(z_vals.shape)  # [N_rays, N_samples]

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)
        z_vals = lower + (upper - lower) * t_rand

    # 每个采样点的3D坐标，o+td
    #; [N,1,3] + [N,1,3] * [N,n_sample,1]，还是广播操作，变成[N_rays, n_samples, 3]
    #; 表示意义是有N_rays条光线，每条光线上有N_samples个点，每个点的坐标都是3维的
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # 送进网络进行预测
    # [ray_batch,N_samples,4]  => [1024,64,4]
    #! 重要：网络预测，这个函数层层往外找，可以看到它实际上调用了最开始创建模型之后定义的函数，实际上就是
    #!   输入采样点，然后经过网络得到输出的颜色值和不透明度的值，也就是每个点都输出4维的结果。但是这个结果
    #!   只是每个采样点的输出颜色和密度，并不是最后得到的一个像素点的颜色值，所以还要调用下面的raw2outputs
    #!   函数，这个函数中才是真正执行论文中的volume rendering公式
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # Step 2 fine sample, 根据粗糙的结果调整采样点，进行精细采样
    # fine 网络部分的运行与上面区别就是采样点不同
    if N_importance > 0:
        #; 重要：先把上面coarse网络的输出结果进行保存备份
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        # 采样点的深度中值
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        #; 每根光线再产生N_importance个新的采样点位置
        #TODO：内部操作暂时没看懂，还需要仔细再看
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, 
                            det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()  #; 不计算梯度
        
        #; 把fine和coarse的采样点拼起来，并且排序，得到[N_rays, N_samples+N_importance]
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        #; 再次计算新的采样点的3D位置
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        #; 选择网络模型，如果有定义的refine的网络模型则使用，否则就是用均匀采样的模型一样的网络
        #! 疑问：这样coarse和fine使用两个网络，那训练的时候梯度反向传播怎么操作？这有点奇怪啊？
        run_fn = network_fn if network_fine is None else network_fine
        #; 网络输出，和coarse网络的输出一样
        raw = network_query_fn(pts, viewdirs, run_fn)
        #; 同理，网络输出并不是最终结果，还要利用体素渲染公式对最终结果进行渲染
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    
    # Step 3 输出结果
    #; 注意如果有fine的采样，那么这里输出的是fine的结果
    ret = {'rgb_map': rgb_map,     # [N_rays, 3]
          'disp_map': disp_map,    # [N_rays,]
           'acc_map': acc_map      # [N_rays,]
        }
    if retraw:
        ret['raw'] = raw           # [N_rays, 4]
    #; 如果有fine采样，那么把coarse网络输出结果也返回，因为此时默认返回的是fine网络的结果
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0    # [N_rays, 3]
        ret['disp0'] = disp_map_0  # [N_rays,]
        ret['acc0'] = acc_map_0    # [N_rays,]
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:  # 判断结果是否有无穷大的非法值，如果有则打印报错提醒
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    """设置参数，并读取配置文件中参数"""
    # configargparse 是一个 Python 库，是对标准库中 argparse 模块的扩展，提供更多的功能和选项
    import configargparse  
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')  # 参数配置文件config.txt
    parser.add_argument("--expname", type=str,
                        help='experiment name')  # 实验名称，会创建不同的文件夹来存储输出结果
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')  # 训练模型和渲染结果保存路径
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')  # 训练数据路径

    # Step 1 training options 训练需要的参数
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')  # 网络深度（层数）
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')  # 每层网络的通道数(实际上对于全连接网络来说就是每一层的宽度)
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')  # fine网络层数
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')  # fine网络每层通道数
    # 光线条数的batch_size，也就是每一次梯度下降的时候随机选择4096个光束进行计算
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')  # batch size光束数量
    # Adam优化器需要配置的两个参数
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')  # 初始学习率
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')  # 指数学习衰减（每1000步）
    
    #; 并行处理的光线数量，如果内存不足则减少
    #  注意：这里的意思是如果选择的光线的batch_size太大，一次计算会超过显存，那么就把一个batch_size的光线
    #    分成几个小的batch来计算，然后把最后的结果cat起来就可以了。注意这个和修改batch_size有区别，
    #    因为这样还是在并行计算一个batch_size, 只不过限于显存限制分成多次计算了，这样和一次计算整个batch_size
    #    对梯度下降没有影响。而如果修改了batch_size来降低计算量，那么梯度下降也会变化
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory') 
    
    #; 通过网络并行光线上的点数，内存不足就减少
    #  注意：这里和上面的意思差不多，但是是为了解决另一个角度造成的显存不足。对于一根光线，需要在他上面采样N个点，
    #    然后把N个点的位置都送入到网络中预测它的颜色和不透明度，然后再把一条光线上的所有点颜色融合起来得到这条
    #    光线的颜色。这里就可以看出来，每一条光线上都有很多点，又增加了计算量，同样还可能超过显存。所以这里就
    #    再次把 N_rays*N_samples 分多次计算，最终的结果在cat起来，得到和一次计算一样的结果
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory') 
    
    #; 是否一次只从一张图像中获取随机光线，默认为false
    #  注意：这里的 batching 指的是从不同图像中获得光线，如果不使用 batch，那么每次就随机选一张图，
    #    然后从这张图中随机选择 N_rays 条光线
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')  
    # 默认不从保存的ckpt加载权重，
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')  
    # 为coarse网络加载指定权重
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')  

    # Step 2 rendering options 渲染相关参数
    # coarse网络每条光线采样的点数
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')  
    # fine网络每条光线增加的精细采样的点数
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray') 
    #! 疑问：这个抖动是什么？
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')  # 是否有抖动
    # 是否使用视角方向，如果使用就是5D输入，不使用就是3D输入
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')  # 默认使用5D输入，而不是3D（x,y,z）
    # 是否使用位置编码，默认使用
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none') 
    # 多分辨率位置编码最大频率的log2（3D位置），也就是使用2^0 ~ 2^9这么多频率的位置编码
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')  
    # 多分辨率位置编码最大频率的log2（2D方向），也就是使用2^0 ~ 2^3这么多频率的方向编码
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')  
    #! 对输出的体密度添加噪声，为什么有这一项？
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended') 
    
    # 仅渲染，加载权重和渲染pose路径
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path') 
    # 渲染测试集而不是根据自己设置的pose来渲染，这样是为了把渲染结果和测试集的真值进行对比
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')  
    # 下采样因子，其实就是缩小因子以加速渲染
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')  

    # training options 训练参数
    #! 这些参数是干嘛的，使用到了吗？
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options 
    # 数据集类型，应该是不同的数据集使用的位姿表示形式不同 
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')  # 数据类型
    # 从训练集的图像中每隔8张图像，就拿出一张作为测试集/验证集
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',  # 在白色背景下渲染合成数据
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',  # 使用一般分辨率
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags llff数据集专用参数
    # 图像缩放系数，原图太大了训练速度太慢
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')  
    # 默认不使用标准化坐标系（为非前向场景设置）
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    
    #! 疑问：下面这几个都没有太明白
    # 是否在时差图（深度图的倒数）上进行均匀采样，默认为False，也就是默认在深度图上进行采样
    parser.add_argument("--lindisp", action='store_true',  
                        help='sampling linearly in disparity rather than depth')
    # 是否是360度场景，这里默认不是
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    # 在训练集中每隔8张图片取出一张图片作为验证集
    parser.add_argument("--llffhold", type=int, default=8,  
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options ng 训练过程n中打印和保存的参数
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():
    # Step 1 读取配置参数
    parser = config_parser()
    args = parser.parse_args()

    # Step 2 读取数据集中的数据，包括图像、相机位姿
    K = None   # 相机内参，默认未知，通过图像的宽、高、焦距手动算出来
    if args.dataset_type == 'llff':
        # Step 2.1. 读取数据集中的图像、相机位姿
        # (20, 378, 504, 3) (20, 3, 5) (20, 2) (120, 3, 5) 12
        images, poses, bds, render_poses, i_test = \
            load_llff_data(args.datadir, args.factor,
                            recenter=True, bd_factor=.75,
                            spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        # Loaded llff (20, 378, 504, 3) (120, 3, 5) [378. 504. 407.5658] ./data/nerf_llff_data/fern
        #; 这里render_poses后面还要看看是干什么的
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        # 生成用来测试的数据id
        if not isinstance(i_test, list):
            i_test = [i_test]  # 转成list，[12]

        # Step 2.2. 如果配置文件中选择了隔N张图像选择一张做测试，则把测试和训练的图像都选择出来
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            # 生成以llffhold为间隔的不超过图像总数的list [0,8,16]
            i_test = np.arange(images.shape[0])[::args.llffhold]
        # [0, 8, 16]，测试图像的索引
        i_val = i_test  
        # 从0-19，除了0，8，16d之外的那些图像的索引
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])  # list len=17
        # 计算最近最远边界值
        print('DEFINING BOUNDS')
        #; 如果不使用ndc空间，那么最近最远就按照参数文件中设置的值来确定，否则就要归一化到[0,1]之间
        if args.no_ndc:  # llff数据集是False 
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
            args.datadir, args.half_res, args.testskip)
        print(
            f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Step 3. 利用数据集中读取的hwf参数手动计算相机内参
    # Cast intrinsics to right types
    # 将内参转为正确的类型内参矩阵
    H, W, focal = hwf  # hwf.shape = (3,)
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # 计算相机内参
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])  # fx fy cx cy

    # 这里默认是False
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Step 4 创建log 路径，保存训练用的所有参数到args,复制config参数并保存
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    # 把argParse中的参数全部写入文件中
    with open(f, 'w') as file:
        # vars是python内置函数，返回对象的__dict__属性，这里就是把args转成dict
        # sorted对上一步的dict进行排序
        for arg in sorted(vars(args)):
            # getattrp是Python内置函数，获取对象的属性值，可以认为就是获取对象的成员变量的值
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # 把这次训练使用的config参数也保存下来
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Step 5 模型构建，训练和测试参数，起始step,优化器
    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = \
        create_nerf(args)
    global_step = start
    # 添加最近最远参数
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)  # len 11
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    # 将测试数据转到device上
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    # 仅仅渲染
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test,
                                  gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(
                testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    # 如果批量处理处理ray，则准备raybatch tensor
    N_rand = args.N_rand
    #; 注意这个batching指的是每次获取光线的时候，是否从所有的训练图片中随机获取指定数量的光线；
    #; 如果是fasle，那么每次都从一张图片上获取制定数量的光线
    use_batching = not args.no_batching  # fern是true
    if use_batching:
        # For random ray batching
        print('get rays')
        # [N, ro+rd, H, W, 3] => [20,2,378,504,3]
        #; 注意stack会扩展维度，也就是把很多array在第0维上进行堆叠，扩展出一个维度
        #; 下面的这个操作还是比较有意思，因为首先for进行列表生成，得到一个len=20的列表；
        #; 列表中的每一个元素都是一个(ro, rd)的元组，而元组的每一个成员(ro或rd)都是(h,w,3)的array
        #; 然后这里stack的时候在0维上对列表的每一个元素进行stack没问题，但是可以发现把每一个元素的元组也自动
        #; stack出一个维度，也就是最终[20, 2, 378, 504, 3]中的那个2
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)
        print('done, concats')

        #! 疑问：这里又摞了一个rgb是干嘛的？
        #! 解答：就是这个点的颜色值
        # [N, ro+rd+rgb, H, W, 3] => [20,3,378,504,3]
        #; images是所有图片，维度(20, 378, 504, 3)；这里images[:, None]维度为(20,1,378,504,3)
        #; 关于[:, None]：https://www.codenong.com/js52dc9bbb16cf/   实际上None和np.newaxis
        #; 是一样的，就是放在哪里，那么就在那个位置新生成一个轴
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        # train images only [17,h,w,ro+rd+rgb,3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        # [(N-1)*H*W, ro+rd+rgb, 3] => [3238704,3,3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        # 打乱排布，或称洗牌 使其不按原来的顺序存储
        #; 注意这里的shuffle就是把所有的图片、图片上的所有像素组成的结果进行shuffle
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    #; 所有图片的位姿都直接搬到device上
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = 200000 + 1  # 迭代20万次
    print('Begin')
    print('TRAIN views are', i_train)  # 哪些图片用来训练
    print('TEST views are', i_test)    # 哪些图片用来测试
    print('VAL views are', i_val)      # 哪些图片用来验证

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # 每次从所有图像的ray中抽取N_rand个ray,每遍历一遍就打乱顺序
            # Random over all images
            #; 注意这里最后两个3的意思，最后一个3代表不同的物理量，倒数第二个3表示不同的物理含义
            #; 比如分别是光束方向、相机中心、颜色值
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)  # [3,B,3]
            # batch_rays=[2,B,3] target_s=[B,3]
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                # 每个epoch后都会重新打乱ray的分布
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            # 每次随机抽取一张图像，抽取一个batch的ray
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            if N_rand is not None:  # 这个判断是废话，肯定不是None啊
                rays_o, rays_d = get_rays(
                    H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(
                        0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0],
                                  select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        # 渲染，核心代码
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        # loss计算L2loss
        #; 计算loss, rgb是网络输出结果再渲染得到的每根光线(每个像素)的颜色值，target_s则是从数据集图片中
        #;   直接读取每根光线(每个像素)得到的值
        img_loss = img2mse(rgb, target_s)  # 直接算像素差的平方
        trans = extras['raw'][..., -1]
        loss = img_loss  
        #! 疑问：又对loss进行了处理？什么操作？
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            #; coarse模型输出的结果也计算loss
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        # 反向传播
        #! 注意：看来这样loss反向传播就会同时更新coarse和fine两个网络模型了！
        loss.backward()
        optimizer.step()  #; 优化一步网络参数

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        # 调整学习率
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        # 保存log
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            # 视频渲染
            with torch.no_grad():
                rgbs, disps = render_path(
                    render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.max(disps)), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(
                    device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
