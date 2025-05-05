# BellmanSolver

基于阻变存储器(Memristor)的贝尔曼方程解算器(MBS - Memristor-Based Solver)

## 项目简介

BellmanSolver是一个开源项目，实现了基于阻变存储器的贝尔曼方程解算器(MBS)，根据论文"Memristive Bellman solver for decision-making"设计与实现。该项目利用阻变存储器的物理特性加速强化学习中贝尔曼方程的求解过程，显著提高大规模状态空间下的计算效率。

项目核心创新点：
1. **引入时间维度**：将传统贝尔曼方程迭代求解过程转换为循环点积形式，实现与阻变存储器CIM(Computing-in-Memory)技术的完美结合
2. **利用阻变存储器内在噪声**：将硬件固有噪声δintrinsic~N(0,φ²)转化为加速收敛的优势，实现近似最优解
3. **软硬件协同优化**：通过MBdot和MBr核心操作，实现贝尔曼方程求解与硬件特性的完美匹配

项目支持两种模式：
1. **纯CPU模式** - 使用NumPy实现的仿真模式（支持精确解和近似解）
2. **硬件加速模式** - 利用阻变存储器硬件加速矩阵运算（天然支持近似解）

## 核心概念

### 时间维度引入

传统贝尔曼方程：
```
V(Sn) = R(Sn) + γ∑P(Sn-1|Sn)V(Sn-1)
```

MBS引入时间维度后的点积形式：
```
V(St) = [R(St) + γV(St-1)]·∑P(St-1|St)
```

通过这种转换，使贝尔曼方程求解过程与阻变存储器的计算特性（点积运算）相匹配，极大提高了计算效率。实际实现中，时间维度是通过对状态进行距离分组实现的，使得每个状态的更新只需考虑前一时间步的状态。

### MBS核心操作

1. **MBdot操作** - 在阻变存储器上执行向量矩阵乘法
   - 输入向量: R(St) + γV(St-1) （当前奖励与折扣后的前一步值函数）
   - 权重矩阵: 状态转移概率 P(St-1|St)
   - 数学表达: V(St) = [R(St) + γV(St-1)]·∑P(St-1|St)

2. **MBr操作** - 迭代执行MBdot直到值函数收敛
   - 通过阈值ε判断收敛性：||V(n+1) - V(n)|| < ε
   - 迭代执行直到值函数稳定
   - 返回收敛后的值函数

### 近似解与噪声利用

论文中的关键创新之一是利用阻变存储器的内在噪声加速收敛。阻变存储器的读取噪声遵循高斯分布：
```
δintrinsic ~ N(0, φ²)
```
其中φ是噪声标准差，通常在0.01-0.1之间。

该噪声可以在值函数迭代过程中帮助算法更快地"跳出"局部最优解，加速收敛到全局最优解。虽然结果是近似的，但大多数情况下产生的策略与精确解非常接近，同时迭代次数显著减少。

项目同时支持精确解和近似解，便于比较分析和根据实际需求选择合适的方法。

## 项目结构

```
BellmanSolver/
├── mbs_core/                  # 核心算法实现
│   ├── bellman_solver.py      # MBS算法实现
│   ├── state_manager.py       # 状态空间管理
│   ├── matrix_operations.py   # 矩阵操作功能
│   ├── distance_calculator.py # 距离计算
│   └── hardware_interface.py  # MBSolver硬件接口
│
├── hardware_interface/        # 硬件接口层
│   ├── device_manager.py      # 设备管理
│   ├── memristor_operations.py# 阻变存储器操作
│   ├── data_converter.py      # 数据格式转换
│   ├── memory_manager.py      # 内存管理
│   └── hardware_accelerator.py# 硬件加速器接口
│
├── grid_world/                # 网格世界环境
│   ├── grid_environment.py    # 网格环境实现
│   └── utils.py               # 环境工具函数
│
├── campus_routing/            # 校园路径规划
│   ├── campus_map.py          # 校园地图表示
│   └── campus_routing.py      # 路径规划算法
│
├── utils/                     # 工具函数
│   └── visualization.py       # 可视化工具
│
├── demos/                     # 示例程序
│   └── hardware_accelerated_solver_demo.py  # 硬件加速演示
│
├── tests/                     # 测试代码
│   └── test_hardware_accelerator.py  # 硬件加速器测试
│   └── test_noise_impact.py          # 噪声影响测试
│
└── README.md                  # 项目说明文档
```

## 核心功能

1. **MBS核心算法** - 实现基于时间维度的贝尔曼方程求解器
2. **硬件加速接口** - 提供与阻变存储器硬件交互的统一接口，支持噪声模拟
3. **网格世界环境** - 实现5×5方格世界环境及可视化功能，对应论文Figure 4a
4. **校园路径规划** - 基于实际地图的路径规划应用，对应论文Figure 4g
5. **噪声测试分析** - 提供噪声水平影响分析功能，验证论文中的核心创新点

## 使用指南

### 安装依赖

本项目依赖以下Python库：
```bash
pip install numpy matplotlib pandas
```

对于硬件加速功能，需要安装特定的阻变存储器硬件驱动（根据实际硬件而定）。

### 基本使用方法

#### 创建和使用MBSolver

```python
from BellmanSolver.mbs_core.bellman_solver import MBSolver

# 创建求解器（精确解模式）
solver = MBSolver(
    state_size,         # 状态空间大小
    gamma,              # 折扣因子（论文中的γ）
    epsilon,            # MBr收敛阈值（论文中的ε）
    tau,                # 权重矩阵收敛阈值（论文中的τ）
    use_hardware=False,     # 不使用硬件加速
    enable_noise=False      # 不启用噪声（精确解）
)

# 预处理：初始化并分析状态空间
solver.preprocess(reward_vector, transition_matrix)

# 求解贝尔曼方程
V, W, stats = solver.solve(max_iterations=100)

print(f"MBr迭代次数: {stats['mbr_iterations']}")
print(f"MBdot操作次数: {stats['mbdot_operations']}")
print(f"求解时间: {stats['solution_time']:.4f}秒")
```



#### 使用硬件加速

```python
# 创建硬件加速的MBS求解器
solver = MBSolver(
    state_size=100,
    gamma=0.9,
    epsilon=0.1,
    tau=0.1,
    use_hardware=True,       # 启用硬件加速
    enable_noise=True,       # 启用噪声（近似解）
    noise_level=0.05,        # 噪声级别（φ值）
    config_path='path/to/hardware_config.json'  # 硬件配置文件
)

# 预处理与求解
solver.preprocess(reward_vector, transition_matrix)
V, W, stats = solver.solve(max_iterations=100)
```

### 演示程序使用

项目提供了演示程序，展示了硬件加速求解器在路径规划任务中的性能：

```bash
# 运行演示程序 - 5×5迷宫任务(论文Figure 4a)
python demos/hardware_accelerated_solver_demo.py --task maze --compare --visualize

# 运行演示程序 - 道路地图任务(论文Figure 4g)
python demos/hardware_accelerated_solver_demo.py --task road --compare --visualize
```

#### 噪声影响测试

```bash
# 测试不同噪声水平对求解性能的影响
python demos/hardware_accelerated_solver_demo.py --task maze --noise_test --visualize --output_file results.csv
```

#### 重要参数说明

- `--task`: 选择任务类型（maze: 5×5迷宫, road: 道路地图）
- `--hardware`: 启用硬件加速模式
- `--compare`: 比较精确解和近似解的性能
- `--visualize`: 可视化结果
- `--noise_level`: 设置噪声级别（φ值）
- `--noise_test`: 执行噪声影响测试
- `--output_file`: 结果输出文件路径

## 性能对比

在大规模状态空间下，硬件加速的近似解相比纯CPU的精确解可获得显著的性能提升：

| 方法 | 迭代减少 | 求解速度提升 | 能耗优势 | 决策质量 |
|-----|---------|------------|--------|---------|
| 近似解(φ=0.01) | 约30% | 1.5-3倍 | - | 极小影响 |
| 近似解(φ=0.05) | 约40% | 2-4倍 | - | 小幅影响 |
| 硬件加速+近似解 | 30%-50% | 5-10倍 | 约10³倍 | 极小影响 |

根据论文估计，使用阻变存储器实现的MBS相比传统CPU实现可获得约10³倍的能耗优势。

## 噪声影响分析

![噪声影响曲线](noise_impact_example.png)

根据噪声测试结果，一般在φ值为0.01-0.05范围内可以获得最佳的平衡点：
- 当φ值过小时，噪声加速效果不明显
- 当φ值过大时，虽然收敛加速明显，但决策质量可能下降
- φ=0.01是大多数情况下的理想选择，能在保持决策质量的同时明显减少迭代次数

## 论文参考

本项目基于论文"Memristive Bellman solver for decision-making"实现，该论文提出了利用阻变存储器特性加速贝尔曼方程求解的创新方法。论文中的核心思想包括：

1. 引入时间维度转换贝尔曼方程
2. 利用阻变存储器内在噪声加速收敛
3. 软硬件协同优化设计

## 贡献指南

欢迎对本项目提出改进建议和代码贡献。请遵循以下贡献流程：
1. Fork本仓库
2. 创建功能分支(git checkout -b feature/your-feature)
3. 提交更改(git commit -am 'Add your feature')
4. 推送到分支(git push origin feature/your-feature)
5. 创建Pull Request 
