# -*- coding: utf-8 -*-
"""
bus_addition_residual_demo.py

版本 2：
  - 一个共享线性 Base，显式设为完美加法器：
        y_base = 0.5 * (a/100) + 0.5 * (b/100)
  - 每个区间训练一个 ResidualExpertMLP，只学习小残差：
        y_hat = y_base + scale * f_i(x)
  - Base 始终冻结不动
  - 训练完之后：
      1) 看 Base-only 在各区间的准确率
      2) 看 Base + 自己区间 Expert 的准确率
      3) 看 Base + 所有 Expert 等权混合 的准确率
"""

import torch
import torch.nn as nn
import torch.optim as optim


# =====================
# 1. 线性 Base
# =====================
class BaseAddLinear(nn.Module):
    """
    线性 Base：理论上完美加法器
      输入 x = [a/100, b/100]
      输出 y_base = 0.5*x1 + 0.5*x2
    """
    def __init__(self):
        super(BaseAddLinear, self).__init__()
        self.fc = nn.Linear(2, 1)
        # 直接设成解析解：W = [0.5, 0.5], b = 0
        with torch.no_grad():
            self.fc.weight[:] = torch.tensor([[0.5, 0.5]])
            self.fc.bias[:] = torch.tensor([0.0])

    def forward(self, x):
        return self.fc(x)


# =====================
# 2. 残差 Expert：小 MLP + scale
# =====================
class ResidualExpertMLP(nn.Module):
    """
    非线性残差 Expert:
      Δy = scale * MLP(x)
    其中 MLP: 2 → 64 → 64 → 1
    """
    def __init__(self, hidden=64, scale=0.1):
        super(ResidualExpertMLP, self).__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.scale * self.net(x)


# =====================
# 3. Bus 容器：Base + 多 Residual Experts
# =====================
class BusAddResidualModel(nn.Module):
    """
    Bus + Base + 多个残差 Expert：
      y = y_base + sum_i w_i * Δy_i

    - base: 共享线性 Base
    - experts: 残差 MLP 列表
    """
    def __init__(self, base):
        super(BusAddResidualModel, self).__init__()
        self.base = base
        self.experts = nn.ModuleList()

    def add_expert(self, expert):
        self.experts.append(expert)

    def forward(self, x, active_indices=None, weights=None):
        y_base = self.base(x)  # (N,1)

        if len(self.experts) == 0 or active_indices is None or len(active_indices) == 0:
            # 只用 base
            return y_base

        if weights is None:
            weights = [1.0] * len(active_indices)

        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        delta = 0.0
        for idx, w in zip(active_indices, weights):
            delta_i = self.experts[idx](x)  # (N,1)
            delta = delta + w * delta_i

        return y_base + delta


# =====================
# 4. 构造数据
# =====================
def make_segment_data(lo, hi):
    """
    构造 [lo, hi] 区间内所有 (a, b) 组合：
      输入：x = [a/100, b/100]
      标签：y = (a + b) / 200
    """
    xs, ys = [], []
    for a in range(lo, hi + 1):
        for b in range(lo, hi + 1):
            xs.append([a / 100.0, b / 100.0])
            ys.append([(a + b) / 200.0])
    X = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    return X, y


# =====================
# 5. 训练某个残差 Expert
# =====================
def train_residual_on_segment(bus_model, expert_index, lo, hi,
                              epochs=4000, lr=0.01):
    """
    只训练指定 expert_index 的残差 Expert，Base 和其他 Expert 冻结。
    """
    X, y = make_segment_data(lo, hi)
    loss_fn = nn.MSELoss()

    params = list(bus_model.experts[expert_index].parameters())
    optimizer = optim.Adam(params, lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        # 使用 base + 当前残差 expert
        pred = bus_model(X, active_indices=[expert_index], weights=[1.0])
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print("    expert {} [{}-{}] epoch {:4d}, loss={:.8f}".format(
                expert_index, lo, hi, epoch + 1, loss.item()))


# =====================
# 6. 测试函数
# =====================
def test_segment(bus_model, lo, hi, active_indices=None, weights=None):
    """
    在 [lo, hi] 区间内测试加法精度：
      - 如果 active_indices=None：只用 Base
      - 否则：Base + 指定残差 Expert 混合
    """
    X, y = make_segment_data(lo, hi)
    with torch.no_grad():
        pred = bus_model(X, active_indices=active_indices, weights=weights)
        pred = pred.squeeze() * 200.0
        true_sum = y.squeeze() * 200.0
        pred_int = torch.round(pred)
        acc = (pred_int == true_sum).float().mean().item()
    return acc


# =====================
# 7. 主流程
# =====================
def main():
    torch.manual_seed(0)

    segments = [
        (1, 20),
        (21, 40),
        (41, 60),
        (61, 80),
        (81, 100),
    ]

    # 共享 Base，直接设为完美加法
    base = BaseAddLinear()
    bus = BusAddResidualModel(base)

    # 先看一下 Base-only 的效果（理论上各区间都是 1.0）
    print("=== Base-only 在各区间上的表现 ===")
    for lo, hi in segments:
        acc_base = test_segment(bus, lo, hi, active_indices=None)
        print("Base-only 在区间 [{}, {}] 上准确率: {:.4f}".format(lo, hi, acc_base))
    print("")

    # 插入并训练 5 个残差 Expert
    for seg_idx, (lo, hi) in enumerate(segments):
        print("=== 插入并训练 residual expert {} 对应区间 [{}, {}] ===".format(seg_idx, lo, hi))

        expert = ResidualExpertMLP(hidden=64, scale=0.1)
        bus.add_expert(expert)

        train_residual_on_segment(bus, seg_idx, lo, hi,
                                  epochs=4000, lr=0.01)

        # Base + 自己这个 expert 在本区间的表现
        acc_single = test_segment(bus, lo, hi, active_indices=[seg_idx])
        print("  Base+expert {} 在区间 [{}, {}] 上准确率: {:.4f}".format(seg_idx, lo, hi, acc_single))
        print("")

    # 所有残差 Expert 等权混合
    print("\n=== Base + 所有 residual expert 等权混合，在各区间上的表现 ===\n")
    all_indices = list(range(len(bus.experts)))
    all_weights = None  # 等权

    for lo, hi in segments:
        acc_mix = test_segment(bus, lo, hi,
                               active_indices=all_indices,
                               weights=all_weights)
        print("Base+AllExperts 在区间 [{}, {}] 上准确率: {:.4f}".format(lo, hi, acc_mix))


if __name__ == "__main__":
    main()
