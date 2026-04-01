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

# Transform学习
学习网站：[Transforms — PyTorch Tutorials 2.11.0+cu130 documentation](https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html)
在数据进入神经网络这台“机器”之前，必须经过标准化和格式化，这就是 `transform` 的作用。
### 1. 核心算子
- **`ToTensor()`**:
    - **功能**：将 PIL 图片或 Numpy 数组转换为 `torch.Tensor`。
    - **归一化**：自动将像素值从 $[0, 255]$ 缩放到 $[0.0, 1.0]$ 的浮点数。这对于模型收敛至关重要。
- **`Lambda` 变换**:
    - **功能**：定义自定义的转换逻辑。
    - **示例**：将标签（Label）转换为 **One-hot 编码**（一个长度为 10 的向量，正确位置为 1，其余为 0）。
### 2. 数据管道 (Pipeline)
- **`Compose`**: 将多个变换步骤串联起来。
- **`DataLoader`**:
    - **C++ 类比**：一个多线程的迭代器容器。
    - **功能**：负责打乱数据（Shuffle）、分批次（Batching，如一次取 64 张图）以及多线程并行读取
--
[相关代码](vscode://file/C:\Users\YYY\Desktop\mashine_learning\jupyter\torch_learning\transforms.ipynb)




-
# NeuralNetwork学习
学习网站[Build the Neural Network — PyTorch Tutorials 2.11.0+cu130 documentation](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
这是你代码中最核心的 `class NeuralNetwork(nn.Module)` 部分。
### 1. 结构拆解 (Architecture)
- **`nn.Flatten`**:
    - **动作**：将 $28 \times 28$ 的矩阵压扁成 $784$ 维向量。        
    - **注意**：必须保留第 0 维（Batch Size），从第 1 维开始压。
- **`nn.Linear(in, out)`**
    - **本质**：矩阵乘法 $y = xW^T + b$。
    - **参数**：存储了 **Weight**（权重矩阵）和 **Bias**（偏置项），这是模型“记忆”知识的地方。
- **`nn.ReLU` (激活函数)**
    - **映射逻辑**：$f(x) = \max(0, x)$。
    - **关键作用**：引入**非线性**。没有它，多层网络会退化为单层线性回归，无法处理复杂特征。
### 2. 输出与概率映射
- **Logits**: 模型最后一层输出的原始得分，范围是 $(-\infty, +\infty)$。
- **Softmax**:
    - **公式**：$P_i = \frac{e^{z_i}}{\sum e^{z_j}}$。
    - **作用**：将 Logits 映射为概率（范围 $[0, 1]$，总和为 1）。
- **Argmax**:
    - **功能**：提取概率最大的那个**索引 (Index)**。
    - **用途**：将“概率分布”转变为具体的“分类标签”（如：6 代表衬衫），用于计算准确率。
### 3. 底层机制 (PyTorch 特性)
- **`nn.Module`**: 所有网络的基类，提供了参数自动注册（`named_parameters`）和求导追踪功能。
- **`forward` 函数**: 专用函数名。定义了数据从输入到输出的流动路径。调用 `model(X)` 时会自动触发。
- **Device Management**: 使用 `.to(device)` 将模型和数据搬运到 GPU (CUDA)，实现高性能并行计

**加入 ReLU 后**：
由于 ReLU 在 $x < 0$ 时是 0，在 $x > 0$ 时是 $x$，它给模型引入了**“折点”**。
- 每一个 ReLU 就像一个逻辑开关。    
- 数千个 ReLU 组合在一起，就能让模型画出极其复杂的、弯弯曲曲的边界，从而把各种奇形怪状的衣服精准地切分出来
虽然 Softmax 提供了概率映射，但其本身不具备特征提取能力。**ReLU 的核心意义在于引入非线性映射**，使得多层全连接网络能够拟合复杂的函数边界。没有 ReLU，深层网络将退化为单层线性分类器，无法处理 FashionMNIST 这种高维图像数据。
[相关代码](vscode://file/C:\Users\YYY\Desktop\mashine_learning\jupyter\torch_learning\Buildmodel_tutorial.ipynb)

# 第一个模型的训练
[优化模型参数 — PyTorch 教程 2.11.0+cu130 文档](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
深度学习核心：PyTorch 训练与持久化笔记
## 1. 训练主循环 (The Training Loop)

深度学习的训练不是“赋值”，而是**“基于指针的原地修改”**。

### 🔄 核心三部曲

Python

```
pred = model(X)      # 1. 前向传播：模型看题预测
loss = loss_fn(pred, y) # 2. 计算损失：裁判打分 (CrossEntropy)
loss.backward()      # 3. 反向传播：产生梯度 (填充参数的 .grad 存钱罐)

optimizer.step()     # 4. 执行更新：搬运工根据梯度，原地修改模型内的 W 和 b
optimizer.zero_grad()# 5. 清理现场：擦掉梯度，防止累加到下一个 Batch
```
```
### 关键细节 (Refined Insights
- **`model.train()` vs `model.eval()`**：
    - `train()` 开启 **Dropout**（随机丢弃）防止死记硬背。
    - `eval()` 关闭干扰，进入稳定预测模式。
- **`enumerate(dataloader)`**：    
    - 返回 `(batch_index, (data, label))`。
    - **C++ 类比**：带 `i++` 计数器的迭代器包装。
- **`loss.item()`**：
    - 将单元素 Tensor 转为 Python `float`。
    - **避坑**：不加 `.item()` 会导致计算图残留在显存，引发 **OOM (显存溢出)**。
## 2. 优化器 (The Optimizer)
`Optimizer` 是模型参数的“外部管家”
- **存储位置**：参数 W 和 b 物理上储存在 `model` 的内存里。
- **绑定机制**：`SGD(model.parameters())` 让优化器持有了模型参数的**指针列表**。
- **原地更新 (In-place)**：`optimizer.step()` 直接修改指针指向的数值，因此在主循环里看不到 `W = new_W` 的显式赋值，但 `loss` 会因参数变动而下降。
```
  [相关代码](vscode://file/C:\Users\YYY\Desktop\mashine_learning\jupyter\torch_learning\model_train.ipynb)


# 下载学习
[保存并加载模型 — PyTorch 教程 2.11.0+cu130 文档](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)

