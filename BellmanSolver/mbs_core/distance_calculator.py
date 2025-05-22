"""
距离计算模块

该模块提供计算状态之间距离的功能，用于MBS算法中的预处理阶段，
帮助将状态分组并确定时间步数。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import heapq


