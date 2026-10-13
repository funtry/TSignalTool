import numpy as np
import matplotlib
matplotlib.use("Agg")  # 后台渲染，不弹出图形窗口，导出GIF专用
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========== 中文支持 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== DE 超参数 =====================
pop_size = 25
dim = 2
F = 0.75
CR = 0.7
max_iter = 80
bounds = (-5, 5)

# ===================== Rastrigin 多坑代价函数 =====================
def cost_func(x):
    A = 10
    return A*2 + x[0]**2 + x[1]**2 - A * np.cos(2 * np.pi * x[0]) - A * np.cos(2 * np.pi * x[1])

# ===================== DE预计算所有世代数据 =====================
pop = np.random.uniform(low=bounds[0], high=bounds[1], size=(pop_size, dim))
history = []
history.append({
    "parent": pop.copy(),
    "mutate": pop.copy(),
    "trial": pop.copy(),
    "newpop": pop.copy()
})

for gen in range(max_iter):
    parent_pop = pop.copy()
    v_list = []
    u_list = []
    new_pop = []

    for i in range(pop_size):
        xi = parent_pop[i]
        # DE/rand/1 变异
        idx_pool = [j for j in range(pop_size) if j != i]
        r1, r2, r3 = np.random.choice(idx_pool, 3, replace=False)
        xr1, xr2, xr3 = parent_pop[r1], parent_pop[r2], parent_pop[r3]
        v = xr1 + F * (xr2 - xr3)
        v = np.clip(v, bounds[0], bounds[1])
        v_list.append(v)

        # 交叉
        u = np.zeros_like(xi)
        for d in range(dim):
            if np.random.rand() < CR or d == np.random.randint(0, dim):
                u[d] = v[d]
            else:
                u[d] = xi[d]
        u_list.append(u)

        # 选择
        if cost_func(u) < cost_func(xi):
            new_pop.append(u)
        else:
            new_pop.append(xi)

    pop = np.array(new_pop)
    history.append({
        "parent": parent_pop,
        "mutate": np.array(v_list),
        "trial": np.array(u_list),
        "newpop": pop.copy()
    })

# ===================== 画布 =====================
fig, ax = plt.subplots(figsize=(9,7), dpi=100)
x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-5, 5, 200)
Xg, Yg = np.meshgrid(x_grid, y_grid)
A = 10
Zg = A*2 + Xg**2 + Yg**2 - A*np.cos(2*np.pi*Xg) - A*np.cos(2*np.pi*Yg)

contour = ax.contourf(Xg, Yg, Zg, levels=60, cmap="coolwarm", alpha=0.7)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label("代价 Rastrigin(x,y)")

# 散点
scatter_parent = ax.scatter([], [], s=50, c="lime", edgecolors="black", label="父代个体")
scatter_v = ax.scatter([], [], s=50, c="royalblue", edgecolors="black", label="变异向量 v")
scatter_u = ax.scatter([], [], s=50, c="gold", edgecolors="black", label="交叉试验个体 u")
ax.plot(0, 0, marker="*", markersize=20, color="red", label="全局最优 (0,0)")

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel("参数 x")
ax.set_ylabel("参数 y")
ax.legend(loc="upper right")

arrow_list = []

def draw_frame(frame):
    global arrow_list
    # 清除上一轮所有箭头
    for arr in arrow_list:
        arr.remove()
    arrow_list.clear()

    data = history[frame]
    parent = data["parent"]
    v_arr = data["mutate"]
    u_arr = data["trial"]
    newpop = data["newpop"]

    scatter_parent.set_offsets(parent)
    scatter_v.set_offsets(v_arr)
    scatter_u.set_offsets(u_arr)

    # 父代最优个体
    parent_costs = [cost_func(p) for p in parent]
    best_idx = int(np.argmin(parent_costs))
    xi_best = parent[best_idx]
    v_best = v_arr[best_idx]
    u_best = u_arr[best_idx]

    # 绘制箭头 xi -> v , v -> u
    arr1 = ax.arrow(xi_best[0], xi_best[1],
                    v_best[0]-xi_best[0], v_best[1]-xi_best[1],
                    width=0.015, head_width=0.06, fc="gray", ec="gray", alpha=0.7)
    arr2 = ax.arrow(v_best[0], v_best[1],
                    u_best[0]-v_best[0], u_best[1]-v_best[1],
                    width=0.015, head_width=0.06, fc="dimgray", ec="dimgray", alpha=0.85)
    arrow_list.append(arr1)
    arrow_list.append(arr2)

    # 外圈标记最优父个体
    ax.plot(xi_best[0], xi_best[1], marker="o", markersize=11,
            markerfacecolor="none", markeredgecolor="black", markeredgewidth=2.2)

    newpop_costs = [cost_func(p) for p in newpop]
    best_new_idx = int(np.argmin(newpop_costs))
    best_xy = newpop[best_new_idx]
    best_cost = newpop_costs[best_new_idx]

    ax.set_title(
        f"差分进化DE 多坑Rastrigin函数寻优\n"
        f"第 {frame} 代 | 当前最优 x={best_xy[0]:.3f}, y={best_xy[1]:.3f}, 代价={best_cost:.3f}\n"
        "大量局部极小坑，DE依靠种群跳出局部洼地，寻找全局最低点"
    )
    return scatter_parent, scatter_v, scatter_u

ani = FuncAnimation(
    fig,
    draw_frame,
    frames=len(history),
    interval=400,
    blit=False,
    repeat=True
)

ani.save(
    "de_rastrigin_multi_opt.gif",
    writer="pillow",
    fps=4,
    dpi=100
)
plt.close(fig)
print("GIF导出完成: de_rastrigin_multi_opt.gif")
