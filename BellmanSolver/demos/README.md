# BellmanSolver 演示示例

本目录包含多个演示BellmanSolver功能的示例脚本，用于展示MBS（Memristor-Based Solver）的性能和特性。

## 硬件加速贝尔曼求解器演示

`hardware_accelerated_solver_demo.py` 脚本演示了使用阻变存储器硬件加速的MBS算法与纯CPU版本的性能比较，主要展示了论文中的两个关键创新点：
1. 通过引入时间维度，将贝尔曼方程转换为循环点积形式，降低计算复杂度
2. 利用阻变存储器的内在噪声特性，加速方程收敛

## 演示任务

演示脚本支持两种路径规划任务：

1. **5x5迷宫任务** (对应论文Figure 4a)
   - 包含起点(Start)、终点(End)、奖励点(Bonus)和陷阱(Trap)
   - 解决方案需要找到通过奖励点到达终点的最优路径

2. **道路地图任务** (对应论文Figure 4g)
   - 模拟有障碍物(湖泊)的道路网络
   - 解决方案需要找到绕过障碍物的最短路径

## 使用方法

### 基本用法

```bash
# 基本用法 - 5×5迷宫任务(精确解)
python hardware_accelerated_solver_demo.py --task maze

# 基本用法 - 道路地图任务(精确解)
python hardware_accelerated_solver_demo.py --task road

# 使用硬件加速(如可用)
python hardware_accelerated_solver_demo.py --task maze --hardware

# 使用近似解（启用噪声模拟）
python hardware_accelerated_solver_demo.py --task maze --approximate

# 硬件加速 + 近似解
python hardware_accelerated_solver_demo.py --task maze --hardware --approximate

# 比较精确解和近似解的性能差异
python hardware_accelerated_solver_demo.py --task maze --compare
```

### 进阶选项

```bash
# 使用定制配置文件
python hardware_accelerated_solver_demo.py --task road --hardware --config path/to/config.json

# 生成随机障碍物(仅迷宫任务)
python hardware_accelerated_solver_demo.py --task maze --random

# 设置最大迭代次数
python hardware_accelerated_solver_demo.py --task road --iterations 50

# 可视化转移概率矩阵
python hardware_accelerated_solver_demo.py --task maze --visualize
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--task` | 选择任务类型: `maze`(5×5迷宫) 或 `road`(道路地图) |
| `--hardware` | 启用硬件加速(如果可用) |
| `--approximate` | 使用近似解(启用阻变存储器噪声) |
| `--compare` | 比较精确解和近似解的性能 |
| `--config` | 指定硬件配置文件路径 |
| `--iterations` | 设置最大迭代次数(默认为20) |
| `--random` | 生成随机障碍物(仅迷宫任务) |
| `--visualize` | 可视化状态转移概率矩阵 |

## 输出解读

在比较模式(`--compare`)下，演示将输出以下指标：

1. **迭代次数比较**
   - MBr迭代次数: MBr操作的执行次数(对应论文Figure 2e)
   - MBdot操作次数: MBdot操作的执行次数(对应论文Figure 2d)
   - 迭代减少率: 近似解相比精确解减少的迭代百分比

2. **求解时间比较**
   - 精确解求解时间(秒)
   - 近似解求解时间(秒)
   - 时间减少率

3. **解的质量比较**
   - 值函数最大差异: 精确解与近似解之间的最大绝对误差
   - 路径是否相同: 判断两种方法生成的路径是否一致

## 近似解特性

阻变存储器的内在读取噪声遵循高斯分布(δintrinsic ~ N(0, φ²))，硬件演示中模拟了这种特性。近似解有如下特点：

1. 加速收敛: 噪声有助于打破相似值之间的平衡，加速迭代收敛
2. 保持精度: 即使使用近似解，最终路径规划结果通常与精确解相同
3. 更低能耗: 根据论文估计，硬件加速的近似解可节省约10³倍能耗

## 示例输出

```
已创建5×5迷宫世界环境

比较精确解和近似解的性能...

使用精确解（无噪声）求解...
精确解求解时间: 0.025680 秒
精确解MBr迭代次数: 8
精确解MBdot操作次数: 40

使用近似解（有噪声）求解...
近似解求解时间: 0.015371 秒
近似解MBr迭代次数: 5
近似解MBdot操作次数: 25

值函数最大差异: 0.089763
迭代次数减少比例: 37.50%
MBdot操作次数减少比例: 37.50%
求解时间减少比例: 40.14%
路径是否相同: 是
```

## 其他演示

更多演示程序正在开发中，将陆续添加到本目录。 