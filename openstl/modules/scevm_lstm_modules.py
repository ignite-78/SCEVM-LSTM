import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
from typing import Optional, Tuple, Callable
from timm.models.swin_transformer import PatchEmbed
from timm.models.layers import to_2tuple
from .vmamba import VSSBlock, SS2D

# ==================== EMA注意力模块 ====================
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        if channels % self.groups != 0:
            self.adaptive_groups = self._find_proper_groups(channels, self.groups)
            print(
                f"⚠️ EMA参数调整: channels({channels})不能被groups({self.groups})整除，调整为groups({self.adaptive_groups})")
        else:
            self.adaptive_groups = self.groups
        self.channels_per_group = channels // self.adaptive_groups
        assert channels % self.adaptive_groups == 0, f"通道数{channels}必须能被分组数{self.adaptive_groups}整除"
        assert self.channels_per_group > 0, f"channels_per_group必须大于0，当前为{self.channels_per_group}"
        self.gn = nn.GroupNorm(num_groups=self.adaptive_groups, num_channels=channels)  # 关键修复
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(self.channels_per_group, self.channels_per_group,
                                 kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(self.channels_per_group, self.channels_per_group,
                                 kernel_size=3, stride=1, padding=1)

    def _find_proper_groups(self, channels, preferred_groups):
        for groups in range(preferred_groups, 0, -1):
            if channels % groups == 0:
                return groups
        return 1

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.adaptive_groups, self.channels_per_group, h, w)
        x_h = self.pool_h(group_x)  # [b*g, c//g, h, 1]
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # [b*g, c//g, 1, w]
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1_temp = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        x1_temp_reshaped = x1_temp.reshape(b, c, h, w)
        x1 = self.gn(x1_temp_reshaped)
        x1 = x1.reshape(b * self.adaptive_groups, self.channels_per_group, h, w)
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.adaptive_groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.adaptive_groups, self.channels_per_group, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.adaptive_groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.adaptive_groups, self.channels_per_group, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.adaptive_groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# ==================== SCConv模块组件 ====================
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class SRU(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5, torch_gn: bool = False):
        super().__init__()
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2,
                 group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze_radio = squeeze_radio
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        gwc_in_channels = self.up_channel // squeeze_radio
        if gwc_in_channels % group_size != 0:
            print(f"⚠️ 参数调整: in_channels({gwc_in_channels})不能被groups({group_size})整除")
            group_size = self.find_proper_divisor(gwc_in_channels, group_size)
            print(f"✅ 自动将groups调整为: {group_size}")
        self.squeeze1 = nn.Conv2d(self.up_channel, self.up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(self.low_channel, self.low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.GWC = nn.Conv2d(gwc_in_channels, op_channel, kernel_size=group_kernel_size,
                             stride=1, padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

    def find_proper_divisor(self, channels, preferred_groups):
        """寻找能整除通道数的合适分组数"""
        for groups in [preferred_groups, 1, 2, 4, 8, 16]:
            if channels % groups == 0:
                return groups
        return 1

class ScConv(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 4, gate_treshold: float = 0.5,
                 alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()
        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio,
                       group_size=group_size, group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

# ==================== SCEMA模块 ====================
class SCEMA(nn.Module):
    def __init__(self, channels, sc_conv_kwargs=None, ema_kwargs=None):
        super(SCEMA, self).__init__()
        if sc_conv_kwargs is None:
            sc_conv_kwargs = {}
        self.sc_conv = ScConv(op_channel=channels, **sc_conv_kwargs)
        if ema_kwargs is None:
            ema_kwargs = {}
        self.ema = EMA(channels=channels, **ema_kwargs)

    def forward(self, x):
        x = self.sc_conv(x)
        x = self.ema(x)
        return x


class VSB(nn.Module):

    def __init__(
            self,
            hidden_dim: int = 0,
            input_resolution: tuple = (224, 224),
            drop_path: float = 0,
            norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs
    ):
        super().__init__()

        self.input_resolution = input_resolution

        self.ln_1 = norm_layer(hidden_dim)

        # Drop path
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

        # 导入VSSBlock
        try:
            from .vmamba import VSSBlock, SS2D
            self.vss_block = VSSBlock(
                hidden_dim=hidden_dim,
                drop_path=drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                **kwargs
            )
            self.has_vss = True
        except ImportError:
            self.has_vss = False
            # 简化版本的VSSBlock
            self.vss_block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(attn_drop_rate),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )

        # VSB输出需要通过tanh和sigmoid产生重加权信号
        # 创建用于生成重加权信号的线性层
        self.tanh_proj = nn.Linear(hidden_dim, hidden_dim)  # 用于生成tanh激活的候选值
        self.sigmoid_proj = nn.Linear(hidden_dim, hidden_dim)  # 用于生成sigmoid激活的门控信号

    def forward(self, x):
        """
        前向传播
        参数:
            x: SCEMA的输出 [B, H*W, C]
        返回:
            tanh_output: 通过tanh激活的候选值 [B, H*W, C]
            sigmoid_output: 通过sigmoid激活的门控信号 [B, H*W, C]
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        # 保存残差连接
        shortcut = x

        # 应用层归一化
        x_norm = self.ln_1(x)

        # 通过VSS块处理
        if self.has_vss:
            # 将序列转换为4D格式 [B, H, W, C] 用于VSSBlock
            x_4d = x_norm.view(B, H, W, C)
            vss_output = self.vss_block(x_4d)
            vss_output = vss_output.view(B, H * W, C)
        else:
            vss_output = self.vss_block(x_norm)

        # 残差连接
        vss_output = shortcut + self.drop_path(vss_output)

        # VSB输出通过tanh和sigmoid产生重加权信号
        # tanh用于生成候选细胞状态，sigmoid用于生成门控信号
        tanh_output = torch.tanh(self.tanh_proj(vss_output))
        sigmoid_output = torch.sigmoid(self.sigmoid_proj(vss_output))

        return tanh_output, sigmoid_output


# ==================== SCEVM-LSTMCell ====================
class SpatioTemporalLSTMCell(nn.Module):
    """
    架构流程:
    1. 输入: X_t 和 H_{t-1}^l
    2. 通过SCEMA模块进行特征优化
    3. 通过VSB模块进行长序列依赖建模
    4. VSB输出tanh和sigmoid信号，用于对细胞状态C进行重加权
    5. 通过标准PredRNN门控计算更新状态
    """

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm,
                 use_scema=True, use_vss=True, vss_depth=1, vss_kwargs=None, scema_kwargs=None):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.use_scema = use_scema
        self.use_vss = use_vss
        self.height = height
        self.width = width
        self.in_channel = in_channel

        # 1. SCEMA模块
        if self.use_scema:
            if scema_kwargs is None:
                scema_kwargs = {}
            self.scema = SCEMA(channels=in_channel + num_hidden, **scema_kwargs)

        # 2. VSS块
        if self.use_vss:
            if vss_kwargs is None:
                vss_kwargs = {}
            vss_kwargs.setdefault('drop_path', 0.1)
            vss_kwargs.setdefault('attn_drop_rate', 0.0)
            vss_kwargs.setdefault('d_state', 16)

            # 创建VSS块列表
            self.vss_blocks = nn.ModuleList([
                VSB(
                    hidden_dim=in_channel + num_hidden,
                    input_resolution=(height, width),
                    drop_path=vss_kwargs['drop_path'],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    attn_drop_rate=vss_kwargs['attn_drop_rate'],
                    d_state=vss_kwargs['d_state'],
                )
                for _ in range(vss_depth)
            ])

            self.vsb_state_proj = nn.Conv2d(in_channel + num_hidden, num_hidden, kernel_size=1, bias=False)

        # 3. 标准PredRNN卷积层
        if layer_norm:
            # 处理X_t的卷积
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            # 处理H_{t-1}的卷积
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            # 处理M_t的卷积
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            # 处理mem的卷积
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                                    stride=stride, padding=self.padding, bias=False)
            self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                                    stride=stride, padding=self.padding, bias=False)
            self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                                    stride=stride, padding=self.padding, bias=False)
            self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                                    stride=stride, padding=self.padding, bias=False)

        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        """
        前向传播
        """
        B, C_in, H, W = x_t.shape

        # ========== 阶段1: SCEMA特征优化 ==========
        if self.use_scema:
            # 拼接当前输入和历史状态
            combined_input = torch.cat([x_t, h_t], dim=1)  # [B, C_in+num_hidden, H, W]

            # 通过SCEMA进行特征优化
            optimized_features = self.scema(combined_input)  # [B, C_in+num_hidden, H, W]
        else:
            optimized_features = torch.cat([x_t, h_t], dim=1)

        # ========== 阶段2: VSS块处理 ==========
        if self.use_vss:
            # 将4D特征转换为序列格式 [B, H*W, C]
            optimized_seq = optimized_features.permute(0, 2, 3, 1).reshape(
                B, H * W, C_in + self.num_hidden
            )

            # 通过VSS块处理
            vss_tanh = None
            vss_sigmoid = None

            for i, vss_block in enumerate(self.vss_blocks):
                if i == 0:
                    vss_tanh, vss_sigmoid = vss_block(optimized_seq)
                else:
                    vss_tanh, vss_sigmoid = vss_block(vss_tanh)

            vss_tanh_4d = vss_tanh.reshape(B, H, W, C_in + self.num_hidden).permute(0, 3, 1, 2)
            vss_sigmoid_4d = vss_sigmoid.reshape(B, H, W, C_in + self.num_hidden).permute(0, 3, 1, 2)

            vss_tanh_proj = torch.tanh(self.vsb_state_proj(vss_tanh_4d))  # [B, num_hidden, H, W]
            vss_sigmoid_proj = torch.sigmoid(self.vsb_state_proj(vss_sigmoid_4d))  # [B, num_hidden, H, W]
        else:
            vss_tanh_proj = torch.zeros(B, self.num_hidden, H, W, device=x_t.device)
            vss_sigmoid_proj = torch.zeros(B, self.num_hidden, H, W, device=x_t.device)

        # ========== 阶段3: 标准PredRNN门控计算，集成VSB输出 ==========
        # 对x_t进行卷积处理
        x_concat = self.conv_x(x_t)  # [B, num_hidden*7, H, W]

        # 对h_t进行卷积处理
        h_concat = self.conv_h(h_t)  # [B, num_hidden*4, H, W]

        # 对m_t进行卷积处理
        m_concat = self.conv_m(m_t)  # [B, num_hidden*3, H, W]

        # 分割卷积结果到各个门控
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        # 计算门控信号，集成VSB的重加权信号
        i_t = torch.sigmoid(i_x + i_h)  # 输入门
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)  # 遗忘门
        g_t = torch.tanh(g_x + g_h)  # 候选细胞状态

        # 更新细胞状态，集成VSB的重加权
        c_new = (f_t + vss_sigmoid_proj) * c_t + i_t * (g_t + vss_tanh_proj)

        # 计算内存相关门控
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        # 更新时空记忆状态
        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        # 计算输出门和新的隐藏状态
        mem = torch.cat((c_new, m_new), 1)  # [B, num_hidden*2, H, W]
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))  # 输出门
        h_new = o_t * torch.tanh(self.conv_last(mem))  # 新的隐藏状态

        return h_new, c_new, m_new