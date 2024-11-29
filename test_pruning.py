import torch
from torchvision.models import resnet18, SwinTransformer, Swin_B_Weights
import torch_pruning as tp

model = resnet18(pretrained=True).eval()

# 1. 为resnet18构建依赖图
DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 2. 为model.conv1分组耦合层
group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )

# 3. 执行剪枝
# if DG.check_pruning_group(group): # 避免完全剪枝，即channels=0
#     group.prune()

# 4. 保存与加载
model.zero_grad() # 清除梯度，避免较大的checkpoint
torch.save(model, 'model.pth') # 我们不能使用.state_dict进行存储，这是因为剪枝导致模型结构发生变化。
model = torch.load('model.pth') # 加载剪枝后的模型

# output = model(torch.randn(1,3,224,224))

# print(group.details()) # use print(group) if you are not interested in the full idxs list.

# print(group[0].idxs)
# group.prune()

new_idxs = [1,2,3,4]
group.prune(new_idxs)
# print(group)
for i, (dep, idxs) in enumerate(group):
    print("Dep: ", dep, " Idxs:", idxs)