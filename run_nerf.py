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
    """
        对原始的函数进行修改，变成for循环小批量调用原函数
    Args:
        fn (function): 原函数
        chunk (_type_): 对原函数分成多少次调用

    Returns:
        _type_: 小批量多次调用原函数的新函数
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """
        输入数据，传入网络，计算输出

    Args:
        inputs (_type_): [N_rays, N_samples, 3], N_rays条光线，每条光线上N_samples个点，每个点3维坐标
        viewdirs (_type_): [N_rays, 3]，每条光线的方向
        fn (function): 网络模型的对象
        embed_fn (_type_): 对点的xyz位置的编码函数，输入3维向量，编码成63维的向量
        embeddirs_fn (_type_): 对输入点的方向的编码函数，输入3维方向，编码成27维的向量
        netchunk (_type_, optional): 每次并行计算的点的个数，如果太大超过显存这里就可以分批用for循环计算，但是仍然在一个batch中

    Returns:
        _type_: [N_rays, N_samples, 4], N_rays条光线，每条光线上N_samples个点，每个点的体密度+rgb颜色
    """
    # Step 1. 把输入点的位置和方向进行位置编码，变成全连接神经网络需要的形式
    # 把输入的N_rays*N_samples个采样点全部展平，变成[N_rays*N_samples, 3]维度
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    # 对每个点进行位置编码，变成[N_rays*N_smaples, 63]
    embedded = embed_fn(inputs_flat)  

    # 如果也使用方向进行渲染，那么再对方向进行编码
    if viewdirs is not None:
        # [N_rays, 3] => [N_rays, 1, 3]  => [N_rays, N_samples, 3]
        #; 注意：None是在数组的第2个维度出现，所以扩展的也是第2个维度
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        # 展平，变成[N_rays*N_samples, 3]
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]]) 
        # 对方向进行编码，变成[N_rays*N_samples, 27]
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        # 沿着最后一个维度进行cat，不会增加维度，变成[N_rays*N_samples, 63+27]
        embedded = torch.cat([embedded, embedded_dirs], -1)
    
    # Step 2.以更小的batch-netchunk送进网络，多次计算对应点的体密度和颜色，最后再把结果拼接
    #; 注意：这个写法还挺有意思的，如果netchunk不是None的话，batchify函数里面会把fn函数分成netchunk轮多次
    #   计算，每次都使用更小的数据量进行计算，最后再沿着这个列表的第0维进行concat得到输出，也就是相当于网络输入
    #   一次数据的数量太大了，比如上面就是 N_rays*N_samples这么多个数据，这样显存可能不支持同时计算这么大的
    #   数据(即这么多的3D采样点)，那么就把这些点分成很多小批量计算，把每个小批量计算的结果再cat起来，就得到了
    #   这些点一起计算的结果
    outputs_flat = batchify(fn, netchunk)(embedded)  # [N_rays*N_samples, 4] => [65536,4]
    
    # 把结果维度reshape为[1024, 64, 4] 4:rgb alpha
    # 注意：list(inputs.shape[:-1])就是把输入的shape除了最后一维，变成list, 结果是[N_rays, N_samples]
    #   然后outputs_flat.shape[-1]就是把输出的最后一个维度(也就是一个采样点输入网络得到的计算结果维度4)
    #   用list形式把他们加起来，就变成[N_rays, N_samples, 4]
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """
        Render rays in smaller minibatches to avoid OOM.
        在更小的batch上进行渲染，避免超出内存。也就是对多个光线分批，然后用for循环处理多次

    Args:
        rays_flat (_type_): (B, 11), 表示光线原点、方向、最小距离、最大距离、方向（不明白为什么有两个方向？）
        chunk (_type_, optional): _description_. Defaults to 1024*32.

    Returns:
        _type_: _description_
    """
    all_ret = {}
    #; 注意：从这里就能看出来这个chunk的作用了，实际上就是一个batch中光线数量可能很大，这个时候可以在外面自己把
    #; batch_size改小，但是这样每次梯度下降就只能使用很小的一部分光线进行计算，不准确。所以这里就把一个batch的
    #; 光线分几次算，把最终结果加起来，这样和使用batch_size算一次结果是一样的，只不过由于内存不足多次算而已
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []  # 初始化键对应的值为一个list
            # 这里的操作就是每一次计算的小批量的结果都存到字典的对应位置中，然后后面再cat
            all_ret[k].append(ret[k]) 
    # 将所有的结果拼接一起返回
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """
        输入光线，在上面进行选择点进行采样，然后送入神经网络输出体密度和颜色，最后使用
        体渲染技术得到这条光线的颜色

    Args:
      H: 图像高度
      W: 图像宽度
      K: 相机内参
      chunk: GPU同步处理的最大光线数，主要是为了小显存的GPU也可以以较大的batch_size训练
      rays: [2, batch_size, 3]. 每个batch的ray的原点和方向向量，在world系下表示
      c2w: [3, 4]. 相机到世界的转换矩阵3x4，调用时就是None
      ndc: bool. If True, represent ray origin, direction in NDC coordinates. NDC坐标
      near: float or array of shape [batch_size]. Nearest distance for a ray. 深度最近距离
      far: float or array of shape [batch_size]. Farthest distance for a ray. 深度最远距离
      use_viewdirs: 是否使用方向向量
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
        camera while using other c2w argument for viewing directions. 实际调用的时候就是None
      **kwargs: 可变长度的关键字参数，这里没有定义，在实际调用的时候可以随便传入，如果不是前面显示
                声明的参数都会被保存到这里面，结果就是一个字典，键是参数名，值就是对应的参数值
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays. 预测的rgb图
      disp_map: [batch_size]. Disparity map. Inverse of depth. 视差图
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned.其它
    """
    # Step 1. 把光线原点和光线方向拿出来，并对光线方向rays_d进行归一化得到视角方向viewdirs
    if c2w is not None:  # 实际传入是None，也就是光线方向已经被计算完成了
        # special case to render full image
        # render所有图像
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        # 注意rays是[2,B,3]，但是这样分别取之后，2维度就消失了，每一个都是[B,3]
        rays_o, rays_d = rays  

    # 对光线方向rays_d进行归一化得到视角方向viewdirs
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d  # (B, 3)
        # 实际调用是None
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            # 特殊case,可视化viewdir的影响
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        # (B, 3), 对光线的方向进行归一化，变成单位向量
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  
        # 结果还是(B, 3)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float() 
    
    # Step 2. 如果使用ndc空间，那么需要把光线中心和光线方向转化到ndc空间中
    sh = rays_d.shape   # 元组, (B, 3)
    if ndc:
        # for forward facing scenes
        # (B, 3)  (B, 3), 维度没有变化，只是变成了NDC坐标系的表示
        #! 疑问：这个地方传入的第4个形参near=1.0，但是按照公式推导为什么不是focal?
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    
    # Step 3. 创建一批光线，包括光线起点、光线方向、光线深度的最小最大值
    # Create ray batch
    # 均为 (B, 3)
    rays_o = torch.reshape(rays_o, [-1, 3]).float()  
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    # 均为 (B, 1)
    near, far = near * torch.ones_like(rays_d[..., :1]), \
        far * torch.ones_like(rays_d[..., :1])
    # (B, 3+3+1+1) = (B, 8), torch.cat不会增加维度，而是会在指定的维度上进行concat
    rays = torch.cat([rays_o, rays_d, near, far], -1)  
    #! 疑问: 上面不是已经有光线方向了吗？这里为什么还要cat光线方向？
    #; 解答：上面的光线方向不是归一化向量，是用于计算采样点的公式中的d，这里的viewdirs是归一化向量用于表示视角方向
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)  # (B, 11)

    # Step 4. 调用函数渲染，里面会使用for循环分批多次渲染，以防止超过显存
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        # print("before: ", k, all_ret[k].shape)
        #; 把结果的第一个维度reshape成batch_size，也就是光线的个数
        # 注意：list(sh[:-1])得到第一个维度，即 N_rays；list(all_ret[k].shape[1:])得到输出结果的后面的维度
        #   不过感觉这里的reshape没有必要？已经是这个shape了。经过验证确实是这样，这里reshape是多余的
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
        # print("after: ", k, all_ret[k].shape)

    # 这里就是对字典变成并且要求前三个是如下的三个变量
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
    """创建NeRF训练需要的很多东西，包括网络模型、训练参数、网络训练的优化器等等

    Args:
        args (_type_): _description_

    Returns:
        _type_: 训练时候的对象和参数、测试时候的对象和参数、训练起始步的索引、网络训练的优化器
    """
    # Step 1. 创建位置和视角编码函数
    # 位置编码函数，编码后的向量的长度(63)
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)  
    
    # 视角方向编码
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # 视角方向编码函数，编码后的向量的长度(27)
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    
    # Step 2.输出通道数
    # N_importance就是fine采样的时候，采样点的个数
    #! 疑问：输出通道为什么是5？不是RGB+体密度吗？
    #; 解答：实际上使用x方向向量的时候，这个outpu_ch是无效的参数
    output_ch = 5 if args.N_importance > 0 else 4

    # Step 3.实例化全连接神经网络
    skips = [4]  # 在网络的第4层对输入数据进行跳跃连接
    # Step 3.1. coarse网络实例化
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())  

    # Step 3.2. fine网络实例化，仅仅是网络深度和每层通道数不同
    model_fine = None
    #! 疑问：为什么refine网络又要重新定义一遍，而不是使用上面的同样的网络？
    #; 解答：从技术原理上来说是可以的，但是使用coarse和finew两个网络，可以让coarse网络只估计场景的粗略结构，
    #;      而fine网络专注于细节，这样可以让coarse网络快速收敛，从而指导fine网络的精细采样也快速收敛到正确  
    #;      位置。总的来说，使用coarse和fine两个网络有助于提高网络的收敛速度。
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())
    
    # Step 4. 定义运行网络的函数，输入点的xyz位置、方向向量、神经网络，则此函数调用神经网络
    def network_query_fn(inputs, viewdirs, network_fn): 
        return run_network(inputs, viewdirs, network_fn,
                            embed_fn=embed_fn,
                            embeddirs_fn=embeddirs_fn,
                            netchunk=args.netchunk)

    # Step 5. 定义优化器
    # Create optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Step 6. 如果是断点继续训练，则加载网络的权重
    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        # 把之前所有训练的权重都加载进来，并且按照顺序排列，这样保证最后一个是最新的模型
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    # 这里默认加载已经保存的模型参数
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

    # Step 7. 组合训练的各种对象和参数，作为函数的输出返回
    #; 注意这个字典中包含了很多东西，层层嵌套传递给渲染的函数
    render_kwargs_train = {
        'network_query_fn': network_query_fn,  # 调用网络的函数
        'perturb': args.perturb,
        'N_importance': args.N_importance,  # fine采样点的个数
        'network_fine': model_fine,  # fine网络
        'N_samples': args.N_samples,  # coarse采样点的个数
        'network_fn': model,   # coarse网络
        'use_viewdirs': args.use_viewdirs,  # 是否使用方向向量
        'white_bkgd': args.white_bkgd,   # 渲染结果是否是白色背景
        'raw_noise_std': args.raw_noise_std,  # 对体密度添加的噪声？
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    # Step 8. 组合测试的时候的各种对象和参数，作为函数结果返回
    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
        这里就是真正的体渲染的实现，就是利用网络输出的体密度和颜色值，渲染得到这条光线对应的像素值

    Args:
        raw: (B, N, 4). 网络输出的B条光线、每条光线上N个点，3个rgb颜色+1个体密度
        z_vals: [B, N]. 每条光线上从near到far的采样点深度值
        rays_d: [B, 3]. 每条光线的方向向量
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
    # Step 0. 定义网络输出的体密度sigma和不透明度alpha之间的转换函数
    # 见论文公式3中alpha公式定义，这里就是把网络输出的体密度转换成不透明度
    def raw2alpha(raw, dists, act_fn=F.relu): 
        #! 疑问：这里对输出的不透明度又加了relu，那么就是说网络输出的并不是最终的不透明度？
        #   那为什么不把这个relu放到网络里呢？还是说这里只是为了实现一个max的效果，因为密度必须>0?
        #; 解答：也可以，其实直接放到网络里面代码更加规范一些，这样显得有点乱
        return 1. - torch.exp(-act_fn(raw)*dists)
    
    # Step 1. 计算采样点之间的距离，即论文中的delta
    # (B, N-1), ti+1-ti, 就是论文中的\delta_i，这里就是用后一个点-前一个点，得到相邻两点的距离
    dists = z_vals[..., 1:] - z_vals[..., :-1]  #; 维度[N_rays, N_samples-1]
    # (B, N), 这里就是把最后一个采样点和下一个采样点的距离设置成1e10一个很大的值，保证距离和点的个数一执
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  
    # (B, N), rays_d[..., None, :] = (B, 1, 3)  norm = (B, 1)
    #! 疑问：这里求模长干什么？模长不都是1吗？
    #; 注意：这里的光线方向并不是单位向量，而是从相机光心指向成像平面的向量，所以不同的光线方向的向量长度是不一样的
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # Step 2. 取出网络输出的颜色，并由网络输出的体密度计算不透明度
    # rgb
    rgb = torch.sigmoid(raw[..., :3])  # [B, N, 3]
    noise = 0.
    # 如果有噪声,选需要对体密度添加噪声
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    
    # (B, N), 也就是每个点的alpha值，即不透明度
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    
    # Step 3. 计算光传播到当前点时剩余的光强*当前点的不透明度，也就是论文中的权重weight，公式3
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    # 注意：torch.ones((alpha.shape[0], 1))得到 (B, 1) 全1的结果，1.-alpha得到 (B, N) 
    #   结果，torch.cat([x,y], -1)沿着最后一个维度concat,得到 (B, N+1)结果，然后torch.cumprod
    #   沿着最后一个维度计算累积乘积，得到(B, N+1)。最后丢掉最后的维度，就是论文中的w，结果仍然是(B, N)
    # (B, N), 每条线上每个点的颜色占最终总的颜色的权重值
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    # Step 4. 计算渲染的rgb图、深度图等各种结果
    # (B, N, 3), 每条光线渲染的rgb颜色(B, N, 1) * (B, N, 3) = (B, N, 3), sum(-2) -> (B, 3)
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    
    # (B, ) 深度图计算d=Σwizi，(B, N) * (B, N), sum(-1) -> (B, )
    #; 注意：从这里可以理解上面的权重到底是什么，其实就是概率密度函数，所以不论是渲染rgb颜色还是深度图，都是在算期望
    depth_map = torch.sum(weights * z_vals, -1)
    
    # (B, ) 视差图计算
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))
    
    #! 疑问： (B, ) 这个表示什么？
    #; 解答：这个就是所有点累积的不透明度，如果是0的话，那么说明当前像素位置是没有物体的，
    #;      这个是针对blenders合成数据集背景部分来说的
    acc_map = torch.sum(weights, -1)  # 权重加和[1024]

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map



def render_rays(ray_batch,          # (B, 11), 一个batch的光线
                network_fn,         # coarse网络模型，字典中传入
                network_query_fn,   # 调用网络输出的函数，字典中传入
                N_samples,          # 光线上coarse的采样个数，字典中传入
                retraw=False,       # 是否返回模型的原始输出
                lindisp=False,      # 是否在逆深度上进行采样，还是在深度上进行采样
                perturb=0.,         # 是否添加扰动
                N_importance=0,     # 光线上fine的采样个数，字典中传入
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

      retraw: bool. 是否返回网络输出的原始结果，即每个点的体密度+rgb颜色值
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
    # Step 1. 把光线原点、光线方向、光线上渲染的深度范围都解析出来
    # batch_size, 传入的这一小批数据的光线个数
    N_rays = ray_batch.shape[0]  
    # (B, 3) (B, 3) 光线中心和方向
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  
    # 如果使用视角渲染，则这里就是方向 (B, 3)
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    # (B, 1, 2) 光线上深度的最近、最远距离
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])  
    # (B, 1) (B, 1) 光线上深度的最近、最远距离，如果是NDC空间则范围为[0, 1]，否则按照真实范围来计算
    near, far = bounds[..., 0], bounds[..., 1]  

    # Step 2 corse sample, 得到粗糙的渲染结果
    # Step 2.1. 对光线上的深度采样N个点
    # (N, ) 这里的N_samples就是配置文件中的N_samples，就是在z轴上进行多少个点的采样
    t_vals = torch.linspace(0., 1., steps=N_samples)  # 0-1线性采样N_samples个值
    # 如果不是在disp视差图上线性采样，那么就是在深度图上进行线性采样，所以直接对z进行线性采样即可
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    # (B, N) 维度张量的维度，也就是每一条光线都按照这个来进行采样
    z_vals = z_vals.expand([N_rays, N_samples]) 

    # Step 2.2. 采样点添加扰动，就是论文中说的在每个采样区间内进行均匀采样
    if perturb > 0.:
        # get intervals between samples
        # (B, N-1), 每个采样区间的中点
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # (B, N-1)，以每个采样点的位置为中心，上下偏移的区间范围
        upper = torch.cat([mids, z_vals[..., -1:]], -1)  
        lower = torch.cat([z_vals[..., :1], mids], -1)  
        # stratified samples in those intervals
        # 生成均匀分布的随机数，这样就可以以原始采样点为中心，在采样区间内均匀采样
        t_rand = torch.rand(z_vals.shape)  # [N_rays, N_samples]

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)
        # (B, N-1), 每个采样区间的起始位置 + [0~1] * 区间长度，变成以采样点为中心的均匀分布采样
        z_vals = lower + (upper - lower) * t_rand

    # Step 2.3. 计算B条光线、每个光线上N个采样点的3D坐标，o+td
    # (B, 1, 3) + (B, 1, 3) * (B, N, 1) = (B, N, 3) 广播操作
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] 

    # Step 2.4. 送进网络进行预测，得到每个点的体密度和颜色值
    # (B, N, 4) 每条光线上、每个点的体密度和颜色
    raw = network_query_fn(pts, viewdirs, network_fn)  # 传入调用网络的函数

    # Step 2.5. 利用每条光线上采样点的体密度和颜色，进行体渲染，得到这条光线对应像素的颜色、深度图等
    # (B, 3) (B, ) (B, ) (B, N) (B, )
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)


    # Step 3 fine sample, 根据粗糙的结果调整采样点，进行精细采样
    # fine 网络部分的运行与上面区别就是采样点不同
    if N_importance > 0:
        # Step 3.1. 先把上面coarse网络的输出结果进行保存备份
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # Step 3.2. 根据前面计算的权重，再在每根光线上重新采样N_importance个新的位置
        # (B, N-1), 每个采样区间的深度中值
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # (B, fN), 对每条光线计算fine采样点的深度
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, 
                            det=(perturb == 0.), pytest=pytest)
        
        #! 重要：这里必须切断梯度，因为fine采样点的位置是由weightd计算的，weight是coarse网络输出的
        # 如果这里不切断梯度的话，那么fine网络的梯度也会影响到coarse网络
        z_samples = z_samples.detach()  
        
        # Step 3.3. 把coarse和fine的采样点深度拼起来，并且按照深度值排序
        # (B, N + fN)
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # (B, N + fN, 3), 重新计算的采样点的位置, (B, 1, 3) + (B, 1, 3) * (B, N + fN, 1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  

        # Step 3.4. 把coarse和fine采样点的位置一起送到fine的网络中，计算输出
        run_fn = network_fn if network_fine is None else network_fine
        # (B, N, 4) 每条光线上、每个点的体密度和颜色
        raw = network_query_fn(pts, viewdirs, run_fn)
        # (B, 3) (B, ) (B, ) (B, N) (B, )
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    
    # Step 4 输出结果
    # 如果有fine的采样，那么这里输出的是fine的结果
    ret = {'rgb_map': rgb_map,     # [N_rays, 3]
          'disp_map': disp_map,    # [N_rays,]
           'acc_map': acc_map      # [N_rays,]
        }
    if retraw:
        ret['raw'] = raw           # [N_rays, 4]
    # 如果有fine采样，那么把coarse网络输出结果也返回，因为此时默认返回的是fine网络的结果
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
    parser.add_argument('--config', is_config_file=True,  # 当前参数传入的是一个配置文件
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
    
    #; 在进行渲染的时候并行处理的光线数量，如果内存不足则减少
    #  注意：这里的意思是如果选择的光线的batch_size太大，一次计算会超过显存，那么就把一个batch_size的光线
    #    分成几个小的batch来计算，然后把最后的结果cat起来就可以了。注意这个和修改batch_size有区别，
    #    因为这样还是在并行计算一个batch_size, 只不过限于显存限制分成多次计算了，这样和一次计算整个batch_size
    #    对梯度下降没有影响。而如果修改了batch_size来降低计算量，那么梯度下降也会变化
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory') 
    
    #; 通过网络并行计算光线上的点数，内存不足就减少
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
    # 是否不从保存的权重中重新加载参数，默认False，也就是会从保存的g权重中加载模型参数（默认断点训练）
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
    #; 解答：是在coarse网络中进行采样的时候，本来采样点是在采样范围内线性分布的，这里加入抖动就是每次训练采样点
    #;      的位置都在原来的位置加入一个均匀分布的扰动，这样其实是和论文中的采样点在采样区间内均匀分布相符合的
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
    #! 疑问：对输出的体密度添加噪声，为什么有这一项？
    #; 解答：感觉是为了防止过拟合？让网络的输出更加鲁棒？但是为什么输出不在每个点的rgb上添加噪声呢？ -> 那样就太随机了？
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

    # Step 3. dataset options 数据集的相关参数
    # 数据集类型，应该是不同的数据集使用的位姿表示形式不同 
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')  # 数据类型
    # 从训练集的图像中每隔8张图像，就拿出一张作为测试集/验证集
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags blender类型的数据集相关的参数
    #; blender数据集是合成的数据集，图片是rgba一共四个通道的，其中a是不透明度（打开图像可以看到背景部分没有颜色）。
    #; 这个标志位就是如果设置成true, 那么就在读取数据集的时候把背景变成白色，这样背景也一起被训练。
    parser.add_argument("--white_bkgd", action='store_true',  
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',  # 使用一般分辨率
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags llff数据集专用参数
    # 图像缩放系数，原图太大了训练速度太慢
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')  
    # 是否不使用ndcz空间坐标系，这里默认是false，也就是默认使用ndc空间坐标系
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    
    # 是否在时差图（深度图的倒数）上进行均匀采样，默认为False，也就是默认在深度图上进行采样
    parser.add_argument("--lindisp", action='store_true',  
                        help='sampling linearly in disparity rather than depth')
    # 是否是360度场景，这里默认不是
    #! 疑问：这个参数是干嘛的？
    #; 解答：设置场景是否是360度场景，如果不是，那么就是faceforward场景，也就是相机位姿基本都是一个朝向的场景
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
        # 这里render_poses就是后面生成新视角图像的时候使用的相机位姿
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
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
        if args.no_ndc:  # llff数据集是False，也就是说LLFF数据集默认是使用ndc空间的
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:  # 使用ndc空间的near和far是固定的[0, 1]范围
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

        # 如果使用白色背景，那么就把透明度为0的地方（背景）都设置成,白色这样网络也能使用背景像素进行训练
        if args.white_bkgd:
            # images (N, H, W, 4) 4是rgba
            # if images[..., 1] = 1, 则有物体，不透明：
            #    images = images[..., :3] * 1 + (1-1) = images[..., :3] 即原始像素
            # else, 则没有物体，透明：
            #    images = images[..., :3] * 0 + (1-0) = 1, 也就是白色
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

    #; 计算相机内参, 注意这里默认相机的fx和fy是相等的，也就是水平和竖直两个方向的焦距是相等的
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

    # Step 5 模型构建、训练和测试参数、起始step、网络优化器
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
    # 将渲染新视角的相机位姿转到device上
    render_poses = torch.Tensor(render_poses).to(device)
    
    # Step 6. 如果仅仅从已经训练好的模型进行渲染，则不用进行后面的步骤
    # Short circuit if only rendering out from trained model
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
    
    # Step 7. 运行到这里则需要进行NeRF网络的训练
    # Prepare raybatch tensor if batching random rays
    # 光线条数的batch_size
    N_rand = args.N_rand
    #; 注意这个batching指的是每次获取光线的时候，是否从所有的训练图片中随机获取指定数量的光线；
    #; 如果是fasle，那么每次都从一张图片上获取制定数量的光线
    use_batching = not args.no_batching  # fern是true
    if use_batching:
        # Step 7.1. 生成每张图片的光心位置和每个像素射线的方向向量
        # For random ray batching
        print('get rays')
        # [N, ro+rd, H, W, 3] => [20, 2, 378, 504, 3]，ro是每个相机位置，rd是每条光线的方向
        #; 注意：stack会扩展维度，也就是把很多array在第0维上进行堆叠，扩展出一个维度。下面的这个操作还是
        #    比较有意思，因为首先for进行列表生成，得到一个len=20的列表；列表中的每一个元素都是一个(ro, rd)
        #    的元组，而元组的每一个成员(ro或rd)都是(h,w,3)的array。然后这里stack的时候在0维上对列表的
        #    每一个元素进行stack没问题，但是可以发现把每一个元素的元组也自动stack出一个维度，也就是最终
        #    [20, 2, 378, 504, 3]中的那个2
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)
        print('done, concats')

        # Step 7.2. 把每张图片的每个像素的rgb值也拼起来
        # [N, ro+rd+rgb, H, W, 3] => [20, 3, 378, 504, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        # train images only [17,h,w,ro+rd+rgb,3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        # [(N-3)*H*W, ro+rd+rgb, 3] => [3238704, 3, 3]，把所有图片，所有像素都组合起来
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        
        # Step 7.3. 对所有图片的所有像素进行shuffle
        print('shuffle rays')
        # 打乱排布，或称洗牌 使其不按原来的顺序存储。这里就是把所有的图片、图片上的所有像素组成的结果进行shuffle
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0  # 后面一个epoch训练中取一个batch的光线的起始索引

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    # 迭代20万次，这里的次数指的是训练一个batch的次数，并不是epoch
    #; 注意：200000在python中可以写成 200_000, 不影响正常解析。但是不能写成 2e5, 这样就变成float了，不能for循环
    N_iters = 200000 + 1    
    print('Begin')
    print('TRAIN views are', i_train)  # 哪些图片用来训练
    print('TEST views are', i_test)    # 哪些图片用来测试
    print('VAL views are', i_val)      # 哪些图片用来验证

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    # Step 8. 开始训练过程
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()
        
        # Step 8.1. 随机采样本次训练要使用到的 N 根光线
        # Sample random ray batch
        if use_batching:
            # 每次从所有图像的 ray 中抽取 N_rand 个ray, 每遍历一遍就打乱顺序
            # Random over all images
            #; 后两个维度中，2+1表示相机中心、光线方向、像素颜色，3是因为恰好这三个类变量都是3维的向量
            batch = rays_rgb[i_batch: i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)  # [3, B, 3]
            # batch_rays=[2,B,3]为相机中心+光线方向，target_s=[B,3]为像素颜色
            batch_rays, target_s = batch[:2], batch[2]
            i_batch += N_rand  # 一次训练就把索引+batch_size
            
            # 一个epoch遍历完所有的光线，则重新打乱光线顺序
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                # torch.randperm 生成一个随机排列的整数序列，范围从 0 到 rays_rgb.shape[0]-1
                rand_idx = torch.randperm(rays_rgb.shape[0])
                # 然后利用索引对光线进行切片，实现重新索引的效果
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else:
            # Random from one image
            # 每次随机抽取一张图像，抽取一个batch的ray
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            if N_rand is not None: 
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

        # Step 8.2. 使用网络输出的体密度和rgb值进行渲染
        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)
        
        # Step 8.3. 由渲染颜色和真值对比计算loss，进行网络训练
        optimizer.zero_grad()
        # Step 8.3.1. 计算L2 loss，注意这里是coarse网络(无fine)或者fine网络(有fine)的loss      
        img_loss = img2mse(rgb, target_s) 
        trans = extras['raw'][..., -1]
        loss = img_loss  
        # 根据loss计算psnr的值
        psnr = mse2psnr(img_loss)

        # Step 8.3.2. 如果有fine网络，那么还要计算coarse网络的loss, 这样才能把coarse网络也训练到
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # Step 8.4. 梯度反向传播，优化网络参数
        loss.backward()
        optimizer.step() 

        # Step 8.5 手动调整学习率
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Step 8.6. 隔一定的训练轮次，保存中间结果
        # Step 8.6.1. 保存网络训练过程中的参数
        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # Step 8.6.2. 中间渲染视频结果，位姿是虚拟生成的，验证网络训练效果
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

        # Step 8.6.3. 测试数据结果，从train的数据集中切分的
        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(
                    device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        # Step 8.6.4. 隔一段时间向屏幕上打印一次结果
        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
