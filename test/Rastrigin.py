import numpy as np
import matplotlib.pyplot as plt

# 中文设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def four_shallow_pits_flat_base(x,y):
    # 基底：极弱的抛物面，几乎接近平面，不再形成巨大漏斗
    base = 0.006*(x**2 + y**2)

    # 4个独立浅高斯凹陷
    pit_center  = -0.32 * np.exp(-((x - 0.0)**2 + (y - 0.2)**2)/2.2)
    pit_edge1   = -0.32 * np.exp(-((x + 4.0)**2 + (y + 3.0)**2)/2.2)
    pit_edge2   = -0.32 * np.exp(-((x - 3.5)**2 + (y - 3.2)**2)/2.2)
    pit_edge3   = -0.32 * np.exp(-((x + 3.3)**2 + (y - 3.6)**2)/2.2)

    z_raw = base + pit_center + pit_edge1 + pit_edge2 + pit_edge3
    # 简单压缩，保证数值稳定，不破坏坑形态
    z_compress = np.log1p(z_raw + 1)
    return z_compress

# 生成网格
x = np.linspace(-5, 5, 220)
y = np.linspace(-5, 5, 220)
X, Y = np.meshgrid(x, y)
Z = four_shallow_pits_flat_base(X,Y)

# 3D绘图
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(projection="3d")

surf = ax.plot_surface(
    X, Y, Z,
    cmap="coolwarm",
    alpha=0.85,
    linewidth=0,
    antialiased=True
)
cbar = fig.colorbar(surf, ax=ax, shrink=0.6)
cbar.set_label("log1p(代价函数)")

ax.set_xlabel("参数 x")
ax.set_ylabel("参数 y")
ax.set_zlabel("压缩代价 Z")
ax.set_title("代价曲面：近平面基底 + 4处分散浅坑（无主导大漏斗）")

plt.tight_layout()
plt.show()
# plt.savefig("four_shallow_pits_flat.png", dpi=300, bbox_inches="tight")
# plt.close()
