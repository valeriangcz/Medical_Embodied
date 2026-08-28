#!/usr/bin/env python3
"""
bed_detection_server 新增逻辑单元测试（不依赖ROS/模型/GPU）

覆盖:
  1. waypoints.yaml 解析 (bed_{patrol_id}_{seq} -> waypoint index)
  2. VLM prompt 构建 (左右布局)
  3. VLM 返回床名 -> waypoint index 映射
  4. occupied_beds 解析 (JSON / 正则回退)
"""
import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch

# 让测试可以 import monitor 包
SRC_MONITOR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'monitor'
)
sys.path.insert(0, os.path.abspath(SRC_MONITOR))

from bed_detection_server import BedDetectionNode  # noqa: E402


def make_node():
    """构造一个不执行 __init__ 的实例, 手动设置所需属性"""
    node = BedDetectionNode.__new__(BedDetectionNode)
    node.get_logger = lambda: MagicMock()
    node.patrol_bed_map = {}
    node.waypoints_config_loaded = False
    node.patrol_bed_indexes = {}
    node.patrol_bed_names = {}
    node.bed_name_to_index = {}
    node.vlm_client = MagicMock()
    node.vlm_model = 'qwen3.8-flash'
    node.vlm_area_retries = 2
    node.vlm_area_retry_delay = 0.0
    node.capture_dir = '/tmp/monitor_test_captures'
    node.save_detection_capture = MagicMock(return_value=(None, None))
    return node


class TestWaypointsParsing(unittest.TestCase):
    def setUp(self):
        self.node = make_node()

    def test_load_waypoints_config(self):
        waypoints_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'nav', 'xjrobot_bridge', 'config', 'waypoints.yaml'
        )
        ok = self.node._load_waypoints_config(waypoints_path)
        self.assertTrue(ok)
        # patrol 0: 4 张床, 左侧 bed_0_0, bed_0_1; 右侧 bed_0_2, bed_0_3
        self.assertEqual(self.node.patrol_bed_names[0],
                         ['bed_0_0', 'bed_0_1', 'bed_0_2', 'bed_0_3'])
        self.assertEqual(self.node.patrol_bed_indexes[0], [2, 3, 4, 5])
        self.assertEqual(self.node.patrol_bed_names[1],
                         ['bed_1_0', 'bed_1_1', 'bed_1_2', 'bed_1_3'])
        self.assertEqual(self.node.patrol_bed_indexes[1], [6, 7, 8, 9])
        # 名称 -> index 映射
        self.assertEqual(self.node.bed_name_to_index['bed_0_0'], 2)
        self.assertEqual(self.node.bed_name_to_index['bed_1_3'], 9)

    def test_get_beds_for_patrol(self):
        self.node.waypoints_config_loaded = True
        self.node.patrol_bed_names = {0: ['bed_0_0', 'bed_0_1', 'bed_0_2', 'bed_0_3']}
        self.node.patrol_bed_indexes = {0: [2, 3, 4, 5]}
        names, indexes = self.node._get_beds_for_patrol(0)
        self.assertEqual(names, ['bed_0_0', 'bed_0_1', 'bed_0_2', 'bed_0_3'])
        self.assertEqual(indexes, [2, 3, 4, 5])
        # 不存在的 patrol
        self.assertIsNone(self.node._get_beds_for_patrol(99)[0])

    def test_build_vlm_area_prompt(self):
        self.node.waypoints_config_loaded = True
        self.node.patrol_bed_names = {0: ['bed_0_0', 'bed_0_1', 'bed_0_2', 'bed_0_3']}
        self.node.patrol_bed_indexes = {0: [2, 3, 4, 5]}
        sys_prompt, user_prompt = self.node._build_vlm_area_prompt(0)
        self.assertIsNotNone(sys_prompt)
        self.assertIn('4 张病床', sys_prompt)
        self.assertIn('左右各 2 张', sys_prompt)
        self.assertIn('bed_0_0, bed_0_1', sys_prompt)   # 左侧 近->远
        self.assertIn('bed_0_2, bed_0_3', sys_prompt)   # 右侧 近->远
        self.assertIn('occupied_beds', sys_prompt)


class TestVlmParsing(unittest.TestCase):
    def setUp(self):
        self.node = make_node()
        self.node.waypoints_config_loaded = True
        self.node.patrol_bed_names = {0: ['bed_0_0', 'bed_0_1', 'bed_0_2', 'bed_0_3']}
        self.node.patrol_bed_indexes = {0: [2, 3, 4, 5]}
        self.node.bed_name_to_index = {
            'bed_0_0': 2, 'bed_0_1': 3, 'bed_0_2': 4, 'bed_0_3': 5,
        }

    def test_parse_occupied_beds_json(self):
        reply = '{"occupied_beds": ["bed_0_0", "bed_0_2"]}'
        self.assertEqual(self.node._parse_occupied_beds(reply),
                         ['bed_0_0', 'bed_0_2'])

    def test_parse_occupied_beds_empty(self):
        reply = '{"occupied_beds": []}'
        self.assertEqual(self.node._parse_occupied_beds(reply), [])

    def test_parse_occupied_beds_regex_fallback(self):
        reply = '床位上有人: bed_0_1 和 bed_0_3 有人'
        self.assertEqual(self.node._parse_occupied_beds(reply),
                         ['bed_0_1', 'bed_0_3'])

    def test_parse_occupied_beds_garbage(self):
        self.assertEqual(self.node._parse_occupied_beds('无法识别'), [])

    def test_detect_beds_with_vlm_area_occupied(self):
        # VLM 返回有人的床
        self.node._call_vlm_once = MagicMock(
            return_value='{"occupied_beds": ["bed_0_0", "bed_0_2"]}'
        )
        img = MagicMock()  # 不会真正被编码, 因为 _call_vlm_once 被替换
        idxs, urgencies, details = self.node._detect_beds_with_vlm_area(img, 0)
        self.assertEqual(idxs, [2, 4])
        self.assertEqual(urgencies, [1, 1])
        self.assertIn('VLM', details)
        # 保存识别结果被调用
        self.node.save_detection_capture.assert_called_once()

    def test_detect_beds_with_vlm_area_empty(self):
        self.node._call_vlm_once = MagicMock(
            return_value='{"occupied_beds": []}'
        )
        img = MagicMock()
        idxs, urgencies, details = self.node._detect_beds_with_vlm_area(img, 0)
        self.assertEqual(idxs, [])
        self.assertEqual(urgencies, [])
        self.assertIn('No occupied', details)

    def test_detect_beds_with_vlm_area_failure_retries(self):
        # VLM 连续失败, 验证重试次数
        self.node._call_vlm_once = MagicMock(return_value=None)
        img = MagicMock()
        idxs, urgencies, details = self.node._detect_beds_with_vlm_area(img, 0)
        self.assertEqual(idxs, [])
        # 重试 2 次 + 首次 = 3 次调用
        self.assertEqual(self.node._call_vlm_once.call_count, 3)

    def test_detect_beds_with_vlm_area_unknown_name(self):
        # VLM 返回未知床名, 应被过滤
        self.node._call_vlm_once = MagicMock(
            return_value='{"occupied_beds": ["bed_0_0", "bed_9_9"]}'
        )
        img = MagicMock()
        idxs, _, _ = self.node._detect_beds_with_vlm_area(img, 0)
        self.assertEqual(idxs, [2])

    def test_detect_beds_with_vlm_area_no_patrol_data(self):
        idxs, _, _ = self.node._detect_beds_with_vlm_area(MagicMock(), 99)
        self.assertEqual(idxs, [])

    def test_retry_success_after_failure(self):
        # 第一次失败, 第二次成功
        calls = [None, '{"occupied_beds": ["bed_0_1"]}']
        self.node._call_vlm_once = MagicMock(side_effect=calls)
        idxs, _, _ = self.node._detect_beds_with_vlm_area(MagicMock(), 0)
        self.assertEqual(idxs, [3])


if __name__ == '__main__':
    unittest.main()
