import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

"""
輸入target，輸出控制衝動，這裏用微分角度表示
"""

# ----- 可微正向運動學 -----
def fk_2dof(theta):
    l1, l2 = 1.0, 1.0
    theta1, theta2 = theta[:, 0], theta[:, 1]
    x = l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
    y = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2)
    return torch.stack([x, y], dim=1)
    

# ----- 策略網絡 -----
"""
簡單的mlp構建神經網絡
"""
class ImpulsePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)  # 輸出 Δθ1, Δθ2
        )

    def forward(self, theta, target_xy):
        x = torch.cat([theta, target_xy], dim=-1)
        return self.net(x)

# ----- 視覺化工具 -----
def plot_arm(theta, target_xy):
    l1, l2 = 1.0, 1.0
    t1, t2 = theta
    x1 = l1 * np.cos(t1)
    y1 = l1 * np.sin(t1)
    x2 = x1 + l2 * np.cos(t1 + t2)
    y2 = y1 + l2 * np.sin(t1 + t2)
    plt.plot([0, x1, x2], [0, y1, y2], 'bo-', linewidth=2)
    plt.plot(target_xy[0], target_xy[1], 'rx')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal')
    plt.grid(True)

# ----- 主訓練流程 -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = ImpulsePolicy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

checkpoint_path = "checkpoint.pt"

# 若存在 checkpoint，載入
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    policy.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("✓ 已加載訓練檢查點")

n_epochs = 3000
n_steps = 20
batch_size = 64

for epoch in range(n_epochs):
    theta = torch.zeros(batch_size, 2).to(device)
    target_xy = (torch.rand(batch_size, 2).to(device) - 0.5) * 3.0
    total_loss = 0.0
    total_energy = 0.0

    theta_seq = []  # 用於能量項計算

    for _ in range(n_steps):
        delta_theta = policy(theta, target_xy)
        theta_seq.append(theta)
        theta = theta + delta_theta
        ee = fk_2dof(theta)

        loss_pos = ((ee - target_xy) ** 2).sum(dim=1).mean()
        loss_reg = 1e-3 * (delta_theta ** 2).sum(dim=1).mean()
        total_loss = total_loss + (loss_pos + loss_reg)

    # 能量項：總角度變化
    theta_seq = torch.stack(theta_seq, dim=1)  # [B, T, 2]
    energy = ((theta_seq[:, 1:] - theta_seq[:, :-1]) ** 2).sum(dim=2).sum(dim=1).mean()
    total_loss = total_loss + 1e-2 * energy

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}, Energy = {energy.item():.4f}")
        # 保存 checkpoint
        torch.save({
            'model': policy.state_dict(),
            'optimizer': optimizer.state_dict()
        }, checkpoint_path)

# ----- 驗證與視覺化 -----
theta = torch.zeros(1, 2).to(device)
target_xy = torch.tensor([[1.8, 0]], device=device)
theta_list = [theta.squeeze(0).cpu().detach().numpy()]

for _ in range(n_steps):
    delta_theta = policy(theta, target_xy)
    theta = theta + delta_theta
    theta_list.append(theta.squeeze(0).cpu().detach().numpy())

plt.figure(figsize=(6, 6))
for th in theta_list:
    plt.cla()
    plot_arm(th, target_xy.squeeze(0).cpu().numpy())
    plt.pause(0.2)

plt.show()
