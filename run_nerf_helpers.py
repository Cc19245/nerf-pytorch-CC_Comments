import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
def img2mse(x, y): 
    return torch.mean((x - y) ** 2)

#! 疑问：看看这个是怎么回事？对loss进行处理
def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    """_summary_
        
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']  # 输入维度，3
        out_dim = 0
        if self.kwargs['include_input']:  # 调用传入的是true
            #! 疑问：这里向列表中添加的是lambda函数？
            #; 注意：这里先加一个不做任何编码的x自身值
            embed_fns.append(lambda x: x)
            out_dim += d  # 输出维度+3

        # 最高频率，multires-1, 对于xyz，就是10-1=9，对于方位角，就是4-1=3
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']  # 10或4

        if self.kwargs['log_sampling']:  # 默认是true
            # 注意这里用的是linespace，也就是线性的均匀取10个值，即2^0 ~ 2^9, 
            # tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        # 遍历10个频率
        for freq in freq_bands:
            # 遍历编码的sin cos两个配置函数
            for p_fn in self.kwargs['periodic_fns']:
                # 向list中添加一个sin 或者 cos位置编码
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d  # 输出维度+3

        #; 赋值给类成员变量，得到一系列的位置编码函数
        self.embed_fns = embed_fns
        #; 输出维度, 2*10*3 + 3 = 63，因为这里是对位置进行编码，一个位置是xyz三个维度
        self.out_dim = out_dim  

    def embed(self, inputs):
        #; 1.这里的调用很有意思，首先for循环遍历成员变量embed_fns的列表，列表中每个元素fn都是
        #;   lambda函数，然后调用fn(inputs)对输入进行操作
        #; 2.torch.cat不会增加维度，只是对数据沿着某个维度进行拼接。这样的话，就相当于调用这个函数的时候，
        #;   每一个fn函数都会返回一个长度为3的向量，然后这里把所有长度为3的向量拼接起来，变成长度为63的向量
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """_summary_
        对输入进行位置编码，multires是指使用几个频率的位置编码，i=0默认使用位置编码
    """
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    # 类的形参传入关键字参数
    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): 
        return eo.embed(x)

    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, 
                skips=[4], use_viewdirs=False):
        """ 
            D: 网络深度，按照论文来看应该是9层？
            W: 每一层网络的宽度，也就是神经元的个数
            imput_ch：整个网络的输入通道数，是位置编码之后的结果，论文中是60，但是代码中是63，
                    因为还加了一个纯x的编码
            imput_ch_views：方位角的额外输入通道数，也是编码后的结果，同理论文中24，代码中27
            output_ch: 网络最终输出通道数，调用时传入是5，而不是4，为啥？不是RGB+不透明度吗？
            skpis: 在哪个位置进行跳层链接，看论文是序号4（第5层）对输入位置编码进行链接
            use_viewdirs：是否使用方向，也就是论文中有些的，如果不使用方向进行渲染，那么结果没有高光
        """
        super(NeRF, self).__init__()
        self.D = D  # 网络深度
        self.W = W  # 每层设置的通道数
        self.input_ch = input_ch  # xyz的通道数
        self.input_ch_views = input_ch_views  # 方向通道数
        self.skips = skips  # 加入输入的位置
        self.use_viewdirs = use_viewdirs  # 是否使用view信息
        # 生成D层全连接网络，在skip3+1层加入input_pts
        #; 注意，D=8，这里也生成了8层：第1层和输入向量有关，第2、3、4、5层都是线性层，但是注意论文中的图，
        #; 是在第5层的输出上叠加输入向量然后传给下一层，所以说实际上是第6层是256+63，因此这里代码也是第6层，
        #; 然后最后第7、8层还是正常的线性层。
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W)  
                if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])  # 0...6

        # Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # 对view处理的网络层
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W//2)])

        # Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        # 输出特征 alpha和rgb的最后层
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """_summary_
            输入x：[N_rays*N_samples, 63+27]，也就是N_rays*N_samples个点，然后每个点的位置和方向都被
            编码成了高维的向量，分别是63维和27维
        """
        # 将xyz与view分开，沿着最后一个维度，分成63维和27维
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        # 全连接网络
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)  # 线性层，w*x+b
            h = F.relu(h)  # 非线性激活函数
            if i in self.skips:  # 第6层需要扩展输入的维度，所以在最后一个维度上进行cat
                h = torch.cat([input_pts, h], -1)
        # 网络输出计算
        if self.use_viewdirs:   # 使用方向向量
            alpha = self.alpha_linear(h)  # 密度，输入256，输出1
            feature = self.feature_linear(h)  # 如果是预测颜色，那么这个位置输入256，输出256
            h = torch.cat([feature, input_views], -1)  # 把输出256和方向编码cat，变成256+27

            # 这里其实就一个层，所以用不用for循环一样，不过用for循环更加规范
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)  # 输入256+27，输出128
                h = F.relu(h)

            rgb = self.rgb_linear(h)  # 输入128，输出3，得到最终预测的颜色
            outputs = torch.cat([rgb, alpha], -1)  # 把颜色和密度cat起来，输出4
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear+1]))


# Ray helpers
def get_rays(H, W, K, c2w):
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2]) /
                       K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    #; i,j的shape均为(378, 504)，然后i的每一行都是(0,1, ..., 503), j的每一列都是(0, 1, ..., 377)
    #; 也就是i代表每个像素的列坐标，j代表每个像素的行坐标
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), 
        indexing='xy')  # 取像素坐标x,y值大小均为[h,w]
    
    #! 注意：论文中的\theta, \fai就体现在这里，也就是在相机坐标系下连接一个像素点，这个方向可以用两个角度表示，
    #!  但是实际上代码中一直都在使用方向向量来表示，所以方向的维度是3维的而不是2维的
    # 2D点到3D点计算[x,y,z]=[(u-cx)/fx,-(v-cy)/fx,-1]
    # 该公式在y和z位置均乘-1，原因是nerf使用的坐标系是x轴向右，y轴向上，z轴向外
    #; 注意np.stack会扩展维度，也就是把里面的三个[h, w]的数组在扩展的维度上进行堆叠
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)  # [h,w,3]
    
    # Rotate ray directions from camera frame to the world frame
    # 将ray方向从相机坐标系转到世界坐标系,矩阵变换
    # #[h,w,3]dot product, equals to: [c2w.dot(dir) for dir in dirs]
    #; dirs[..., np.newaxis, :]结果从(h,w,3)变成(h,w,1,3)，dirs[..., np.newaxis, :] * c2w[:3, :3]
    #; 的维度为(h,w,3,3)，然后sum(xxx,-1)的结果为(h,w,3)
    #! 注意：这里是使用逐元素乘法来实现矩阵乘法，因为如果是V_c转到V_w，那么应该进行矩阵乘法：c2w @ V_c
    #!  这里使用逐元素乘法，利用广播操作将(h,w,1,3)*(3,3)扩充成(h,w,3,3)*(h,w,3,3)，最后再
    #!  sum(xxx, -1)，最终得到的结果和矩阵乘法一样
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 相机原点在世界坐标系的坐标，也是同一个相机所有ray的起点
    # np.broadcast_to函数将数组广播到新形状，也就是把(3,)的平移向量，广播到(h,w,3)的维度
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))  # [h,w,3]
    return rays_o, rays_d  # 返回元组


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """_summary_
        bins：粗采样的采样深度中值列表
        weights：粗采样中计算出来的每个采样点的权重值
        N_samples：本次精采样要额外采样多少个点
    """
    # Get pdf计算概率密度，见公式5
    #! 注意：下面注释中的N_samples仍然是这个函数外面的粗采样的N_samples，而不是本函数的传入参数N_samples
    #; 维度[N_rays, N_samples-2], 也就是去掉了首位两个点
    weights = weights + 1e-5  # prevent nans [1024,62]
    pdf = weights / torch.sum(weights, -1, keepdim=True)  #; 权重归一化，得到pdf
    #; 计算pdf的累计和
    cdf = torch.cumsum(pdf, -1)  # 返回输入元素累计和[1024,62]
    # (batch, len(bins)) [1024,63]
    #; 对上面的结果在最前面补0，所以变成[N_rays, N_samples-1]。注意[..., :1]的切片方式不会丢失维度
    #; 而如果使用[..., 0]的方式则会丢失最后的维度导致降维
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    #! 注意：上面注释中的N_samples仍然是这个函数外面的粗采样的N_samples，而不是本函数的传入参数N_samples

    #! 注意：下面注释中的N_samples就是本函数的传入参数N_samples
    # Take uniform samples 均匀采样
    if det:
        u = torch.linspace(0., 1., steps=N_samples)  # 0-1之间均匀采样N_samples个数
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]) # 把u扩充成[N_rays, N_sample]的形状
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)


    #TODO 下面具体得到新的采样点的操作没有怎么看懂，后面还需要再仔细看一下
    #! 疑问：下面这一堆操作实在没有怎么看懂，先留着吧，知道大致意思就是根据粗采样计算的权重生成pdf，然后再按照这个pdf采样
    # Invert CDF
    u = u.contiguous()  # 把tensor变成内存连续分布形式[1024,64]
    # 返回和u一样大小tensor, 其元素是cdf中大于等于u中的值的值的索引[1024,64]
    #! 疑问：这个应该是返回cdf中 > u中的值的索引，但是如果没有>的会怎样呢？也没看懂下面的博客说的
    # https://blog.csdn.net/qq_35037684/article/details/125275305
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)  # inds-1
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)  # inds<=62
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    #; [N_rays, N_samples, N_samples-1]
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]  # [1024,64,63]
    #; [N_rays, 1, N_samples-1] => [N_rays, N_samples, N_samples-1]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)  # [1024,64,2]
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)  # [1024,64,2]

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(
        denom < 1e-5, torch.ones_like(denom), denom)  # [1024,64]
    t = (u-cdf_g[..., 0])/denom
    #; 最后输出[N_rays, N_samples]，也就是每根光线，再产生N_samples个新的采样点位置
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])  # [1024,64]

    return samples
