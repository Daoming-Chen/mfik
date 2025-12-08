"""
Robot 模块单元测试

测试URDF解析、正运动学和逆运动学功能。
"""

import unittest
import os
import sys
import numpy as np
import torch

# 设置随机种子以确保测试可重复
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mfik.robot import (
    parse_urdf,
    get_kinematic_chain,
    ForwardKinematics,
    InverseKinematics,
    check_joint_limits,
    clip_to_limits,
    shortest_angular_distance,
    normalize_angles,
    interpolate_angles,
)


class TestURDFParsing(unittest.TestCase):
    """测试URDF解析功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.robot_dir = os.path.join(os.path.dirname(__file__), '..', 'robots')
        self.panda_urdf = os.path.join(self.robot_dir, 'panda_arm.urdf')
        self.ur10_urdf = os.path.join(self.robot_dir, 'ur10.urdf')
    
    def test_parse_panda_arm(self):
        """测试解析Panda Arm URDF"""
        chain = parse_urdf(self.panda_urdf)
        
        # Panda Arm有7个关节
        self.assertEqual(chain.n_joints, 7, "Panda Arm应该有7个关节")
        
        # 检查关节类型
        for joint_type in chain.joint_types:
            self.assertIn(joint_type, ['revolute', 'continuous', 'prismatic'])
        
        # 检查关节限位存在
        self.assertEqual(len(chain.joint_limits), 7)
        
        print(f"✓ Panda Arm解析成功: {chain.n_joints} 个关节")
    
    def test_parse_ur10(self):
        """测试解析UR10 URDF"""
        chain = parse_urdf(self.ur10_urdf)
        
        # UR10有6个关节
        self.assertEqual(chain.n_joints, 6, "UR10应该有6个关节")
        
        # 检查关节类型
        for joint_type in chain.joint_types:
            self.assertIn(joint_type, ['revolute', 'continuous', 'prismatic'])
        
        print(f"✓ UR10解析成功: {chain.n_joints} 个关节")
    
    def test_file_not_found(self):
        """测试文件不存在的情况"""
        with self.assertRaises(FileNotFoundError):
            parse_urdf('nonexistent.urdf')
    
    def test_kinematic_chain_properties(self):
        """测试运动学链属性"""
        chain = parse_urdf(self.panda_urdf)
        
        # 检查下限和上限数组
        lower = chain.lower_limits
        upper = chain.upper_limits
        
        self.assertEqual(len(lower), chain.n_joints)
        self.assertEqual(len(upper), chain.n_joints)
        
        # 上限应该大于下限(对于有限位的关节)
        finite_mask = np.isfinite(lower) & np.isfinite(upper)
        if np.any(finite_mask):
            self.assertTrue(np.all(upper[finite_mask] >= lower[finite_mask]))
        
        print(f"✓ 运动学链属性验证通过")


class TestForwardKinematics(unittest.TestCase):
    """测试正运动学计算"""
    
    def setUp(self):
        """设置测试环境"""
        # 重置随机种子确保每个测试可重复
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        self.robot_dir = os.path.join(os.path.dirname(__file__), '..', 'robots')
        self.panda_urdf = os.path.join(self.robot_dir, 'panda_arm.urdf')
        self.ur10_urdf = os.path.join(self.robot_dir, 'ur10.urdf')
        self.device = 'cpu'
    
    def test_fk_single_config_panda(self):
        """测试单个配置的FK计算(Panda Arm)"""
        chain = parse_urdf(self.panda_urdf)
        fk = ForwardKinematics(chain, self.device)
        
        # 测试零位置
        q = torch.zeros(chain.n_joints)
        pos, quat = fk.compute(q)
        
        # 检查输出形状
        self.assertEqual(pos.shape, (3,), "位置应该是3维向量")
        self.assertEqual(quat.shape, (4,), "四元数应该是4维向量")
        
        # 四元数应该是单位四元数
        quat_norm = torch.norm(quat)
        self.assertAlmostEqual(quat_norm.item(), 1.0, places=5, 
                             msg="四元数应该是单位四元数")
        
        print(f"✓ Panda单配置FK: pos={pos.numpy()}, quat_norm={quat_norm.item():.6f}")
    
    def test_fk_batch_panda(self):
        """测试批量FK计算(Panda Arm)"""
        chain = parse_urdf(self.panda_urdf)
        fk = ForwardKinematics(chain, self.device)
        
        # 生成10个随机配置
        batch_size = 10
        q = torch.randn(batch_size, chain.n_joints) * 0.5
        
        pos, quat = fk.compute(q)
        
        # 检查输出形状
        self.assertEqual(pos.shape, (batch_size, 3))
        self.assertEqual(quat.shape, (batch_size, 4))
        
        # 所有四元数应该是单位四元数
        quat_norms = torch.norm(quat, dim=1)
        for i, norm in enumerate(quat_norms):
            self.assertAlmostEqual(norm.item(), 1.0, places=5,
                                 msg=f"四元数{i}应该是单位四元数")
        
        print(f"✓ Panda批量FK ({batch_size}个配置)验证通过")
    
    def test_fk_single_config_ur10(self):
        """测试单个配置的FK计算(UR10)"""
        chain = parse_urdf(self.ur10_urdf)
        fk = ForwardKinematics(chain, self.device)
        
        q = torch.zeros(chain.n_joints)
        pos, quat = fk.compute(q)
        
        self.assertEqual(pos.shape, (3,))
        self.assertEqual(quat.shape, (4,))
        
        quat_norm = torch.norm(quat)
        self.assertAlmostEqual(quat_norm.item(), 1.0, places=5)
        
        print(f"✓ UR10单配置FK验证通过")
    
    def test_fk_gradient(self):
        """测试FK的梯度计算(自动微分)"""
        chain = parse_urdf(self.panda_urdf)
        fk = ForwardKinematics(chain, self.device)
        
        q = torch.zeros(chain.n_joints, requires_grad=True)
        pos, quat = fk.compute(q)
        
        # 对位置的某个分量求梯度
        loss = pos[0]  # x坐标
        loss.backward()
        
        # 梯度应该存在且非零(对于大多数配置)
        self.assertIsNotNone(q.grad, "FK应该支持自动微分")
        
        print(f"✓ FK自动微分验证通过, grad shape={q.grad.shape}")


class TestInverseKinematics(unittest.TestCase):
    """测试逆运动学求解"""
    
    def setUp(self):
        """设置测试环境"""
        # 重置随机种子确保每个测试可重复
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        self.robot_dir = os.path.join(os.path.dirname(__file__), '..', 'robots')
        self.panda_urdf = os.path.join(self.robot_dir, 'panda_arm.urdf')
        self.ur10_urdf = os.path.join(self.robot_dir, 'ur10.urdf')
        self.device = 'cpu'
    
    def test_ik_position_only_panda(self):
        """测试仅位置的IK求解(Panda Arm)"""
        chain = parse_urdf(self.panda_urdf)
        fk = ForwardKinematics(chain, self.device)
        ik = InverseKinematics(chain, self.device)
        
        # 生成一个目标配置
        q_target = torch.randn(chain.n_joints) * 0.3
        target_pos, _ = fk.compute(q_target)
        
        # 从不同的初始点求解IK
        q_init = torch.zeros(chain.n_joints)
        q_solved = ik.solve(target_pos, target_quat=None, q_init=q_init,
                           method='jacobian', max_iter=500, tolerance=1e-6)
        
        # 验证求解结果
        solved_pos, _ = fk.compute(q_solved)
        position_error = torch.norm(solved_pos - target_pos).item()
        
        # 精度要求 0.001mm
        self.assertLess(position_error, 1e-6, 
                       f"位置误差应小于0.001mm, 实际: {position_error*1000:.4f}mm")
        
        print(f"✓ Panda位置IK: 误差={position_error*1000:.4f}mm")
    
    def test_ik_jacobian_method_panda(self):
        """测试雅可比伪逆法(Panda Arm)"""
        chain = parse_urdf(self.panda_urdf)
        fk = ForwardKinematics(chain, self.device)
        ik = InverseKinematics(chain, self.device)
        
        # 测试目标
        q_target = torch.randn(chain.n_joints) * 0.2
        target_pos, _ = fk.compute(q_target)
        
        q_solved = ik.solve(target_pos, method='jacobian', 
                           max_iter=500, tolerance=1e-6)
        
        solved_pos, _ = fk.compute(q_solved)
        position_error = torch.norm(solved_pos - target_pos).item()
        
        # Jacobian方法应收敛到1e-3mm
        self.assertLess(position_error, 1e-6,
                       f"Jacobian方法误差应小于0.001mm, 实际: {position_error*1000:.4f}mm")
        
        print(f"✓ Panda Jacobian IK: 误差={position_error*1000:.4f}mm")
    
    def test_ik_ur10(self):
        """测试UR10的IK求解"""
        chain = parse_urdf(self.ur10_urdf)
        fk = ForwardKinematics(chain, self.device)
        ik = InverseKinematics(chain, self.device)
        
        q_target = torch.randn(chain.n_joints) * 0.3
        target_pos, _ = fk.compute(q_target)
        
        q_solved = ik.solve(target_pos, method='jacobian', 
                           max_iter=500, tolerance=1e-6)
        
        solved_pos, _ = fk.compute(q_solved)
        position_error = torch.norm(solved_pos - target_pos).item()
        
        self.assertLess(position_error, 1e-6)
        
        print(f"✓ UR10 IK: 误差={position_error*1000:.4f}mm")
    
    def test_ik_batch_solve(self):
        """测试批量IK求解"""
        chain = parse_urdf(self.panda_urdf)
        fk = ForwardKinematics(chain, self.device)
        ik = InverseKinematics(chain, self.device)
        
        # 生成多个目标
        batch_size = 5
        q_targets = torch.randn(batch_size, chain.n_joints) * 0.3
        target_positions, _ = fk.compute(q_targets)
        
        # 批量求解
        q_solved = ik.solve_batch(target_positions, method='jacobian',
                                  max_iter=500, tolerance=1e-6)
        
        # 验证所有解
        solved_positions, _ = fk.compute(q_solved)
        errors = torch.norm(solved_positions - target_positions, dim=1)
        
        for i, error in enumerate(errors):
            self.assertLess(error.item(), 1e-6,
                           f"批量IK第{i}个解误差过大: {error.item()*1000:.4f}mm")
        
        mean_error = errors.mean().item()
        print(f"✓ 批量IK ({batch_size}个目标): 平均误差={mean_error*1000:.4f}mm")


class TestJointUtilities(unittest.TestCase):
    """测试关节工具函数"""
    
    def setUp(self):
        """设置测试环境"""
        self.robot_dir = os.path.join(os.path.dirname(__file__), '..', 'robots')
        self.panda_urdf = os.path.join(self.robot_dir, 'panda_arm.urdf')
    
    def test_check_joint_limits(self):
        """测试关节限位检查"""
        chain = parse_urdf(self.panda_urdf)
        
        # 零位置应该在限位内(对于大多数机器人)
        q = torch.zeros(chain.n_joints)
        within_limits = check_joint_limits(q, chain)
        
        # 检查限位逻辑是否正常工作
        # 测试超出限位的情况
        q_over = torch.ones(chain.n_joints) * 100.0  # 很大的值
        within_limits_over = check_joint_limits(q_over, chain)
        
        # 至少应该有一些关节超出限位
        has_finite_limits = torch.any(torch.isfinite(torch.from_numpy(chain.upper_limits)))
        if has_finite_limits:
            self.assertFalse(torch.all(within_limits_over),
                           "超大值应该超出某些关节限位")
        
        print(f"✓ 关节限位检查验证通过")
    
    def test_clip_to_limits(self):
        """测试关节限位裁剪"""
        chain = parse_urdf(self.panda_urdf)
        
        # 创建超出限位的关节角度
        q = torch.ones(chain.n_joints) * 10.0  # 很大的值
        q_clipped = clip_to_limits(q, chain)
        
        # 裁剪后应该在限位内
        within_limits = check_joint_limits(q_clipped, chain)
        self.assertTrue(torch.all(within_limits),
                       "裁剪后应该在限位内")
        
        print(f"✓ 关节限位裁剪验证通过")
    
    def test_shortest_angular_distance(self):
        """测试最短角度差计算"""
        # 测试正常情况
        q1 = torch.tensor(0.0)
        q2 = torch.tensor(np.pi / 2)
        dist = shortest_angular_distance(q1, q2)
        self.assertAlmostEqual(dist.item(), np.pi / 2, places=5)
        
        # 测试跨越±π边界的情况
        q1 = torch.tensor(3.0)
        q2 = torch.tensor(-3.0)
        dist = shortest_angular_distance(q1, q2)
        # 最短路径应该是走短的那一边
        self.assertLess(abs(dist.item()), np.pi,
                       "跨越边界时应该走最短路径")
        
        print(f"✓ 最短角度差计算验证通过: dist={dist.item():.3f}")
    
    def test_normalize_angles(self):
        """测试角度归一化"""
        # 测试大角度
        q = torch.tensor([3*np.pi, -3*np.pi, 0.0])
        q_normalized = normalize_angles(q)
        
        # 归一化后应该在[-π, π]范围内
        self.assertTrue(torch.all(q_normalized >= -np.pi))
        self.assertTrue(torch.all(q_normalized <= np.pi))
        
        print(f"✓ 角度归一化验证通过")
    
    def test_interpolate_angles(self):
        """测试角度插值"""
        q0 = torch.tensor([0.0, 0.0, 0.0])
        q1 = torch.tensor([np.pi/2, np.pi, -np.pi/2])
        
        # t=0应该返回q0
        q_t0 = interpolate_angles(q0, q1, 0.0)
        self.assertTrue(torch.allclose(q_t0, q0, atol=1e-6))
        
        # t=1应该返回q1(考虑周期性)
        q_t1 = interpolate_angles(q0, q1, 1.0)
        # 使用最短角度差比较
        diff = shortest_angular_distance(q_t1, q1)
        self.assertTrue(torch.all(torch.abs(diff) < 1e-5))
        
        # t=0.5应该在中间
        q_t05 = interpolate_angles(q0, q1, 0.5)
        
        print(f"✓ 角度插值验证通过")


class TestIntegration(unittest.TestCase):
    """集成测试:端到端工作流"""
    
    def setUp(self):
        """设置测试环境"""
        # 重置随机种子确保每个测试可重复
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        self.robot_dir = os.path.join(os.path.dirname(__file__), '..', 'robots')
        self.panda_urdf = os.path.join(self.robot_dir, 'panda_arm.urdf')
        self.device = 'cpu'
    
    def test_fk_ik_consistency(self):
        """测试FK和IK的一致性"""
        chain = parse_urdf(self.panda_urdf)
        fk = ForwardKinematics(chain, self.device)
        ik = InverseKinematics(chain, self.device)
        
        # 随机生成几个配置
        n_tests = 5
        for i in range(n_tests):
            # 生成随机配置
            q_original = torch.randn(chain.n_joints) * 0.3
            
            # FK: 计算位姿
            target_pos, target_quat = fk.compute(q_original)
            
            # IK: 从不同初值求解
            q_init = torch.zeros(chain.n_joints)
            q_solved = ik.solve(target_pos, target_quat=None, q_init=q_init,
                               method='jacobian', max_iter=500, tolerance=1e-6)
            
            # 再次FK验证
            solved_pos, solved_quat = fk.compute(q_solved)
            
            # 计算误差
            pos_error = torch.norm(solved_pos - target_pos).item()
            
            # 数值IK精度要求(0.001mm)
            self.assertLess(pos_error, 1e-6,
                           f"测试{i}: FK-IK一致性误差过大: {pos_error*1000:.4f}mm")
        
        print(f"✓ FK-IK一致性测试通过 ({n_tests}次测试)")
    
    def test_continuous_trajectory(self):
        """测试连续轨迹跟踪"""
        chain = parse_urdf(self.panda_urdf)
        fk = ForwardKinematics(chain, self.device)
        ik = InverseKinematics(chain, self.device)
        
        # 生成一条简单的直线轨迹
        start_q = torch.randn(chain.n_joints) * 0.2
        end_q = torch.randn(chain.n_joints) * 0.2
        
        n_waypoints = 10
        trajectory_q = []
        
        for t in np.linspace(0, 1, n_waypoints):
            q_t = interpolate_angles(start_q, end_q, float(t))
            trajectory_q.append(q_t)
        
        trajectory_q = torch.stack(trajectory_q)
        
        # 计算轨迹位置
        positions, quaternions = fk.compute(trajectory_q)
        
        # 验证轨迹连续性
        for i in range(1, n_waypoints):
            # 相邻点之间的距离应该相近
            dist = torch.norm(positions[i] - positions[i-1])
            # 由于是插值轨迹,距离应该相对稳定
            self.assertGreater(dist.item(), 0.0, "轨迹点不应重合")
        
        print(f"✓ 连续轨迹跟踪测试通过 ({n_waypoints}个航点)")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestURDFParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestForwardKinematics))
    suite.addTests(loader.loadTestsFromTestCase(TestInverseKinematics))
    suite.addTests(loader.loadTestsFromTestCase(TestJointUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回成功/失败
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

