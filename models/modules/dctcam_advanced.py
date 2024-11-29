import torch.nn as nn
import torch
import math
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_freq_indices(method):  # 获取频率分量索引的函数
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:  # 对应三个频率分量选择标准中的两步选择标准
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]  # 得到频率分量的横坐标
        mapper_y = all_top_indices_y[:num_freq]  # 得到频率分量的纵坐标
    elif 'low' in method:  # 对应三个频率分量选择标准中的LF(低频)标准
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]  # 得到频率分量的横坐标
        mapper_y = all_low_indices_y[:num_freq]  # 得到频率分量的纵坐标
    elif 'bot' in method:  # # 对应三个频率分量选择标准中的TS(两步选择)标准
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]  # 得到频率分量的横坐标
        mapper_y = all_bot_indices_y[:num_freq]  # 得到频率分量的纵坐标
    else:
        raise NotImplementedError
    return mapper_x, mapper_y  # 获取频率分量的索引


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel=512, reduction=16, freq_sel_method='low32'):
        super(MultiSpectralAttentionLayer, self).__init__()

        c2wh = dict([(96 // 4, 56), (192 // 4, 28), (384 // 4, 14), (768 // 4, 7)])
        self.reduction = reduction  # Number of channel divisions
        self.dct_h = c2wh[channel]  # Input feature height
        self.dct_w = c2wh[channel]  # Input feature width
        self.channel = channel

        self.mapper_x, self.mapper_y = get_freq_indices(freq_sel_method)  # Get frequency component indices
        self.num_split = len(self.mapper_x)  # Number of splits
        self.mapper_x = [temp_x * (self.dct_h // 7) for temp_x in self.mapper_x]
        self.mapper_y = [temp_y * (self.dct_w // 7) for temp_y in self.mapper_y]

        self.gate_conv_beta = nn.Sequential(
            nn.Conv2d(channel, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 32, 1, 1, 0, bias=True),
        )

        self.gate_ac = nn.Sequential(
            nn.Tanh(),
            nn.ReLU(),
        )

        # Define cb, b, gamma as trainable parameters
        self.cb = nn.Parameter(torch.tensor([48, 64, 96, 160], dtype=torch.float32), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([4, 3, 2, 1], dtype=torch.float32), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(2.0, dtype=torch.float32), requires_grad=True)

        self.beta = nn.Parameter(torch.zeros(1))

        # Initialize parameters using He initialization
        self._initialize_weights()

        t = int(abs(math.log2(channel) - self.b[channel // 96 % 4]) / self.gamma)
        k = t if t % 2 else t + 1

        self.conv1d = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=k, padding=int(k / 2)),
            nn.BatchNorm1d(channel),
            nn.ReLU()
            # nn.Sigmoid(),
        )

    def _initialize_weights(self):
        # He initialization for convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for Conv2d layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                # Kaiming initialization for Conv1d layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm2d parameters to ones and zeros
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize BatchNorm1d parameters to ones and zeros
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)  # Calculate frequency component
        if freq == 0:
            return result  # Return result
        else:
            return result * math.sqrt(2)

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x

        xg = self.gate_conv_beta(x).view(n, 32)
        xg = self.gate_ac(xg) + self.beta  # 0.1

        param = []
        sum = 0
        for i in range(len(xg[0])):
            sum += xg[0][i]
        param.append(0)
        for i in range(len(xg[0]) - 1):
            a = xg[0][i].item() / sum.item()
            param.append(round(a * c))

        sum_n = 0
        for i in range(len(xg[0])):
            sum_n += param[i]
        param.append(c - sum_n)

        dct_filter = torch.zeros(self.channel, self.dct_h, self.dct_w)  # Define a zero tensor of shape [channel, tile_size_x, tile_size_y]

        for i, (u_x, v_y) in enumerate(zip(self.mapper_x, self.mapper_y)):  # Iterate over frequency component indices
            for t_x in range(self.dct_h):
                for t_y in range(self.dct_w):
                    dct_filter[param[i]:param[i+1], t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                   self.dct_h) * self.build_filter(
                        t_y, v_y, self.dct_w)  # Call build_filter

        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))  # Adaptive average pooling
        x_pooled = x_pooled.to(device)
        dct_filter = dct_filter.to(device)
        y = x_pooled * dct_filter
        y = torch.sum(y, dim=[2, 3])
        y = y.unsqueeze(-1)

        y = self.conv1d(y).view(n, c, 1, 1)  # Linear projection of y and change dimensions
        return x * y.expand_as(x) + x

# if __name__ == '__main__':
#     img = torch.randn(2, 384 // 4, 32, 32).to('cuda')
#     model = MultiSpectralAttentionLayer(channel=384 // 4).to('cuda')
#     out = model(img).to('cuda')
#     print(out.shape)

# x_test = torch.randn(1, 512, 7, 7, requires_grad=True).to(device)
# torch.autograd.gradcheck(MultiSpectralAttentionLayer(channel=512), (x_test,))
