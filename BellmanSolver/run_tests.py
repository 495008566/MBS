#!/usr/bin/env python
"""
测试运行脚本

运行BellmanSolver项目的所有测试。
"""

import os
import sys
import unittest
import argparse
import subprocess

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def filter_hardware_tests(suite):
    """
    从测试套件中过滤掉与硬件相关的测试
    
    参数:
    suite: 测试套件
    
    返回:
    unittest.TestSuite: 过滤后的测试套件
    """
    filtered_suite = unittest.TestSuite()
    
    # 处理可能的TestCase对象
    if isinstance(suite, unittest.TestCase):
        # 如果它不是硬件测试，则保留
        if not 'hardware' in suite._testMethodName.lower():
            filtered_suite.addTest(suite)
        return filtered_suite
    
    # 遍历测试套件
    for item in suite:
        # 如果是TestSuite，递归处理
        if isinstance(item, unittest.TestSuite):
            sub_suite = filter_hardware_tests(item)
            if sub_suite.countTestCases() > 0:
                filtered_suite.addTest(sub_suite)
        # 如果是TestCase，检查是否是硬件测试
        elif isinstance(item, unittest.TestCase):
            if not 'hardware' in item._testMethodName.lower():
                filtered_suite.addTest(item)
        # 对于失败的测试，处理FailedTest等特殊情况
        else:
            try:
                # 尝试提取测试名称
                test_name = str(item).lower()
                if not 'hardware' in test_name:
                    filtered_suite.addTest(item)
            except:
                # 保守处理，保留无法判断的测试
                filtered_suite.addTest(item)
    
    return filtered_suite


def run_noise_impact_tests(env_size=5, trials=3, output_file=None, no_visual=False):
    """
    运行噪声影响测试
    
    参数:
    env_size: 环境大小
    trials: 每个噪声水平的测试次数
    output_file: 输出文件名
    no_visual: 是否禁用可视化
    
    返回:
    int: 返回代码
    """
    print("\n" + "="*60)
    print("运行噪声影响测试")
    print("="*60)
    
    test_script = os.path.join(PROJECT_ROOT, 'tests', 'test_noise_impact.py')
    
    # 构建命令行参数
    cmd = [sys.executable, test_script, '--size', str(env_size), '--trials', str(trials)]
    
    if output_file:
        cmd.extend(['--output', output_file])
    
    if no_visual:
        cmd.append('--no-visual')
    
    # 运行测试
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"运行噪声影响测试失败: {e}")
        return e.returncode
    except Exception as e:
        print(f"运行噪声影响测试时发生错误: {e}")
        return 1


def main():
    """主函数，运行测试"""
    parser = argparse.ArgumentParser(description='运行BellmanSolver测试')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                      help='增加详细程度')
    parser.add_argument('--hardware', action='store_true',
                      help='包含硬件测试')
    parser.add_argument('--pattern', '-p', type=str, default='test_*.py',
                      help='测试文件匹配模式')
    parser.add_argument('--test', '-t', type=str, default=None,
                      help='指定运行特定测试模块')
    parser.add_argument('--noise-test', action='store_true',
                      help='运行噪声影响测试')
    parser.add_argument('--env-size', type=int, default=5,
                      help='噪声测试的环境大小')
    parser.add_argument('--trials', type=int, default=3,
                      help='每个噪声水平的测试次数')
    parser.add_argument('--output', type=str, default=None,
                      help='噪声测试结果输出文件')
    parser.add_argument('--no-visual', action='store_true',
                      help='禁用噪声测试的可视化结果')
    
    args = parser.parse_args()
    
    # 如果指定了噪声测试，直接运行噪声测试
    if args.noise_test:
        return run_noise_impact_tests(
            env_size=args.env_size,
            trials=args.trials,
            output_file=args.output,
            no_visual=args.no_visual
        )
    
    # 确保tests目录存在
    tests_dir = os.path.join(PROJECT_ROOT, 'tests')
    if not os.path.exists(tests_dir):
        print(f"警告: 测试目录 {tests_dir} 不存在")
        return 1
    
    try:
        # 发现测试文件
        if args.test:
            # 运行特定测试模块
            test_suite = unittest.defaultTestLoader.discover('tests', 
                                                         pattern=f'{args.test}.py')
        else:
            test_suite = unittest.defaultTestLoader.discover('tests', 
                                                         pattern=args.pattern)
        
        # 如果不包含硬件测试，过滤掉硬件测试
        if not args.hardware:
            print("跳过硬件测试（使用 --hardware 运行硬件测试）")
            test_suite = filter_hardware_tests(test_suite)
        
        if test_suite.countTestCases() == 0:
            print("警告: 没有找到测试用例")
            return 0
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=args.verbose+1)
        result = runner.run(test_suite)
        
        # 返回结果状态
        return 0 if result.wasSuccessful() else 1
    
    except Exception as e:
        print(f"运行测试时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 