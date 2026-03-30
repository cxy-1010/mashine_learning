# tensor学习 
官网学习：[张量 — PyTorch 教程 2.11.0+cu130 文档](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- **创建与维度**（从多维到 0 维标量）。
- **切片与索引**（NumPy 兼容性）。
- **内存与设备**（`cat` 的开销与 `to(device)`）。
- **广播与计算**（`mul` 与 `add_(5)`）
[相关代码实现](vscode://file/C:\Users\YYY\Desktop\mashine_learning\jupyter\torch_learning\tensor.ipynb)

# datasets和dataloader 的学习
官网学习：[Datasets & DataLoaders — PyTorch 教程 2.11.0+cu130 文档](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
### 1. 数据抽象层：`Dataset` 类 (仓库管理员)
你学习了如何通过继承 `torch.utils.data.Dataset` 来构建自己的“数据仓库”。
- **三大核心方法（C++ 接口实现）：**
    - `__init__`：**构造函数**。负责初始化“账本”（CSV）和“地图”（路径），并准备好“加工工具”（Transform）。        
    - `__len__`：**Size 获取**。告诉程序一共有多少张图，对应 `std::vector::size()`。
    - `__getitem__`：**操作符重载 `[]`**。这是最核心的逻辑，负责“按索引取货”。
### 2. 索引与寻址逻辑：CSV + `os` + `iloc`
这是将“硬盘文件”转化为“内存张量”的桥梁：
- **CSV 账本**：充当 **vtable（虚表）**。第 0 列存文件名（地址），第 1 列存标签（答案）。
- **`os.path.join`**：**跨平台路径拼接**。解决了 Windows 和 Linux 下斜杠不通用的 Bug，保证代码的工程健壮性
- **`iloc[idx, col]`**：**矩阵寻址**。精准定位到某一个样本的特定信息。
### 3. 图像解码与转化：`torchvision.io`
- **解码器**：替代了传统的 `imread`，直接将二进制图片解码为 **Tensor（张量）**。
- **维度陷阱**：理解了为什么原始图片是 3D 的 `[C, H, W]`（通道、高、宽），以及为什么画图前需要用 `.squeeze()` 压扁成 2D。
### 4. 批量运输层：`DataLoader` (物流车队)
这是为了满足 GPU **并行计算**而设计的迭代器。
- **Batching（分装）**：把 64 个单张图 `[1, 28, 28]` 堆叠成一个 4D 的“大货箱” `[64, 1, 28, 28]`。
- **Shuffle（洗牌）**：通过随机打乱防止模型“死记硬背”题目顺序，是科研训练的基石。
- **Iteration（迭代）**：学习了如何用 `iter()` 和 `next()` 拨动传送带，进行手动的采样调
[相关代码](vscode://file/C:\Users\YYY\Desktop\mashine_learning\jupyter\torch_learning\datasets_and_dataloaders.ipynb)
