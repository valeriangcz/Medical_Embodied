#!/usr/bin/env python3
"""
床位检测节点 - Area/Bed模式实现
整合版本: 用于Medical_Embodied项目的monitor包

功能: 
  Area模式: 调用VLM API一次性识别画面中所有床位的有无人状态, 返回有人床位ID列表
  Bed模式:  检测指定床位是否有人(异常),返回is_anomaly

Area模式识别逻辑 (VLM替代YOLO+CLIP):
  1. 从导航 waypoints.yaml 解析 bed_{patrol_id}_{seq} 点位, 获取每个巡诊点的床位数
     及左右布局 (左侧近->远 bed号从小到大, 右侧近->远 bed号从小到大)
  2. 构建VLM prompt, 由VLM输出有人床位的 bed_{patrol_id}_{seq} 名称
  3. 通过 waypoints.yaml 将 bed 名称映射为导航 waypoint index, 返回 bed_ids
  4. 无 waypoints.yaml 时回退 patrol_bed_mapping.json / 顺序编号

Bed模式: VLM判断坠床风险, 失败回退CLIP (保持原逻辑不变)

服务接口: /detect_anomaly (继承自interfaces/DetectAnomaly.srv)
相机话题: 通过camera_topic参数配置 (默认RealSense L515: /camera/camera/color/image_raw)
"""
import os
import re
import time
import json
import base64
import logging
from pathlib import Path
import yaml
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from interfaces.srv import DetectAnomaly

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from ultralytics import YOLO
import open_clip

# 模式枚举
MODE_AREA = 0  # 区域扫描模式
MODE_BED = 1   # 单床检测模式


def _find_dashscope_key_file():
    """在多个候选路径中查找 DashScope key 文件"""
    current_file = Path(__file__).resolve()
    candidates = []
    # 1) monitor 包自身目录: src/monitor/key/dashscope.key
    candidates.append(current_file.parents[1] / "key" / "dashscope.key")
    # 2) 复用 llm_node_py 的 key: workspace/src/llm_node_py/key/dashscope.key
    for parent in current_file.parents:
        if (parent / "src").exists() and (parent / "install").exists():
            candidates.append(parent / "src" / "llm_node_py" / "key" / "dashscope.key")
            candidates.append(
                parent / "install" / "llm_node_py" / "share" / "llm_node_py" / "key" / "dashscope.key"
            )
    for path in candidates:
        if path.exists():
            return path
    return None


def load_dashscope_api_key():
    """读取 DashScope API key（与 llm_node_py 保持一致的文件格式）"""
    key_file = _find_dashscope_key_file()
    if key_file is None:
        raise FileNotFoundError(
            "未找到 DashScope key 文件，期望路径如 src/llm_node_py/key/dashscope.key"
        )
    api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"DashScope key 文件为空: {key_file}")
    return api_key

class BedDetectionNode(Node):
    def __init__(self):
        super().__init__('bed_detection_node')

        # 创建CV桥
        self.bridge = CvBridge()

        # 声明参数用于配置模型路径
        self.declare_parameter('yolo_model_path', '')
        self.declare_parameter('clip_model_path', '')
        self.declare_parameter('max_beds', 10)
        self.declare_parameter('vlm_model', 'qwen3.8-flash')
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('capture_dir',
                               os.path.join(os.path.expanduser('~'),
                                            'Medical_Embodied', 'captures'))
        # waypoints.yaml 路径（留空自动查找 xjrobot_bridge 包内路径）
        self.declare_parameter('waypoints_config_path', '')
        # VLM Area 模式重试次数与间隔
        self.declare_parameter('vlm_area_retries', 2)
        self.declare_parameter('vlm_area_retry_delay', 1.0)

        # 获取参数值
        yolo_model_param = self.get_parameter('yolo_model_path').value
        clip_model_param = self.get_parameter('clip_model_path').value
        self.max_beds = self.get_parameter('max_beds').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.capture_dir = self.get_parameter('capture_dir').value

        # 解析资源目录（优先ROS安装share目录，回退源码目录）
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        src_monitor_dir = os.path.dirname(current_file_dir)
        self.share_dir, self.src_root_dir = self._resolve_resource_roots(current_file_dir)
        self.src_models_dir = os.path.join(self.share_dir, 'models')

        # 加载巡诊点-床位映射配置
        self.declare_parameter('patrol_bed_mapping_path', '')
        mapping_param = self.get_parameter('patrol_bed_mapping_path').value
        if mapping_param:
            mapping_path = mapping_param
        else:
            mapping_path = os.path.join(self.share_dir, 'config', 'patrol_bed_mapping.json')
        self.patrol_bed_map = self._load_patrol_bed_mapping(mapping_path)

        # 加载导航 waypoints.yaml (bed 点信息: bed_{patrol_id}_{seq} -> waypoint index)
        waypoints_param = self.get_parameter('waypoints_config_path').value
        self.patrol_bed_indexes = {}    # patrol_id -> [waypoint_index, ...] 按 seq 排序
        self.patrol_bed_names = {}      # patrol_id -> ["bed_p_s", ...] 按 seq 排序
        self.bed_name_to_index = {}     # "bed_p_s" -> waypoint index
        self.waypoints_config_loaded = self._load_waypoints_config(waypoints_param)
        if not self.waypoints_config_loaded:
            self.get_logger().warn(
                'waypoints.yaml 未加载成功, Area 模式将回退到 patrol_bed_mapping.json 顺序编号'
            )

        # VLM Area 模式重试参数
        self.vlm_area_retries = int(self.get_parameter('vlm_area_retries').value)
        self.vlm_area_retry_delay = float(self.get_parameter('vlm_area_retry_delay').value)
        
        # 记录路径信息
        self.get_logger().info(f'Python文件目录: {current_file_dir}')
        self.get_logger().info(f'监控包资源目录(share/src): {self.share_dir}')
        self.get_logger().info(f'监控包源码根目录回退路径: {self.src_root_dir}')
        self.get_logger().info(f'模型目录: {self.src_models_dir}')

        # YOLO模型路径 - 优先使用参数配置，否则使用源代码目录路径
        if yolo_model_param:
            yolo_model_path = yolo_model_param
        else:
            # 直接使用源代码目录下的models
            yolo_model_path = os.path.join(self.src_models_dir, 'best.pt')
            self.get_logger().info(f'使用默认YOLO模型路径: {yolo_model_path}')
        
        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.get_logger().info(f'YOLOv8模型加载成功: {yolo_model_path}')
        except Exception as e:
            self.get_logger().error(f'YOLOv8模型加载失败: {e}')
            # 尝试备用路径（同样是源代码目录）
            fallback_path = os.path.join(self.src_models_dir, 'bed_detector.pt')
            try:
                self.yolo_model = YOLO(fallback_path)
                self.get_logger().info(f'使用备用YOLOv8模型: {fallback_path}')
            except Exception as e2:
                self.get_logger().error(f'备用YOLOv8模型也失败: {e2}')
                raise

        # CLIP模型路径 - 优先使用参数配置，否则使用源代码目录路径
        if clip_model_param:
            clip_model_path = clip_model_param
        else:
            # 直接使用源代码目录下的models
            clip_model_path = os.path.join(self.src_models_dir, 'best_model.pt')
            self.get_logger().info(f'使用默认CLIP模型路径: {clip_model_path}')
        
        try:
            # 加载检查点
            checkpoint = torch.load(clip_model_path, map_location='cpu')
            state_dict = checkpoint['clip_model']
            
            # 基于模型检查结果,使用ViT-L-14架构
            model_name = 'ViT-L-14'
            
            # 首先加载模型架构,但不打印其内部警告
            clip_logger = logging.getLogger('root')
            original_level = clip_logger.level
            clip_logger.setLevel(logging.ERROR)
            
            try:
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=None
                )
            finally:
                clip_logger.setLevel(original_level)
            
            # 安全地加载权重,确保键匹配
            model_state_dict = self.clip_model.state_dict()
            
            # 过滤掉不匹配的键
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key in model_state_dict and value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
                else:
                    self.get_logger().warn(f"跳过不匹配的权重: {key}")
            
            # 加载权重
            missing_keys, unexpected_keys = self.clip_model.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                self.get_logger().warn(f"缺失的权重键: {missing_keys}")
            if unexpected_keys:
                self.get_logger().warn(f"意外的权重键: {unexpected_keys}")
            
            self.clip_tokenizer = open_clip.get_tokenizer(model_name)
            
            # 将模型设置为评估模式
            self.clip_model.eval()
            
            # 将模型移动到合适的设备
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.clip_model = self.clip_model.to(device)
            
            # 检查训练结果信息
            if 'accuracy' in checkpoint:
                self.get_logger().info(f'训练准确率: {checkpoint["accuracy"]:.4f}')
            if 'class_names' in checkpoint:
                self.get_logger().info(f'训练类别: {checkpoint["class_names"]}')
            
            self.get_logger().info(f'CLIP模型加载成功(设备: {device}): {clip_model_path}')
            self.get_logger().info(f'加载了 {len(filtered_state_dict)}/{len(state_dict)} 个权重参数')
            
        except Exception as e:
            self.get_logger().error(f'CLIP模型加载失败: {e}')
            raise

        # 存储最新图像
        self.current_image = None
        self.image_received = False

        # 订阅相机话题
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f'已订阅相机话题: {self.camera_topic}')

        # 发布模式切换指令到模拟相机
        self.mode_pub = self.create_publisher(String, '/camera/mode', 10)

        # 创建床位检测服务 - 使用与anomaly_detect_server相同的服务名称
        self.detect_service = self.create_service(
            DetectAnomaly,
            '/detect_anomaly',
            self.handle_detect_request
        )
        self.get_logger().info('床位检测服务已创建: /detect_anomaly')

        # 初始化外部VLM客户端(DashScope Qwen-VL), 用于Bed模式坠床风险判断
        self.vlm_client = None
        self.vlm_model = self.get_parameter('vlm_model').value
        try:
            # 清除代理, 避免本地代理影响外网API调用
            for var in ('http_proxy', 'https_proxy', 'HTTP_PROXY',
                        'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY'):
                os.environ.pop(var, None)
            from openai import OpenAI
            self.vlm_client = OpenAI(
                api_key=load_dashscope_api_key(),
                base_url="https://llm-wd6fnilu0sk3blx6.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            )
            self.get_logger().info(f'VLM客户端初始化成功 (DashScope, 模型: {self.vlm_model})')
        except Exception as e:
            self.get_logger().warn(
                f'VLM客户端初始化失败, Bed模式将回退到CLIP: {e}'
            )

    def _resolve_resource_roots(self, current_file_dir):
        """返回 (resource_root, source_root_fallback)"""
        source_root = os.path.dirname(current_file_dir)
        try:
            share_dir = get_package_share_directory('monitor')
            return share_dir, source_root
        except Exception as e:
            self.get_logger().warn(f'获取monitor share目录失败: {e}, 使用源码目录')
            return source_root, source_root

    def _load_patrol_bed_mapping(self, path):
        """加载巡诊点-床位映射JSON"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            mapping = {}
            for item in data.get('patrol_points', []):
                pid = item['patrol_id']
                # 新格式: beds = [{"detection_index": 0, "bed_id": 1}, ...]
                # 旧格式兼容: bed_ids = [1, 2, 3, ...]
                if 'beds' in item:
                    bed_map = {}
                    for bed_info in item['beds']:
                        bed_map[bed_info['detection_index']] = bed_info['bed_id']
                    mapping[pid] = bed_map
                elif 'bed_ids' in item:
                    bed_map = {i: bid for i, bid in enumerate(item['bed_ids'])}
                    mapping[pid] = bed_map
            self.get_logger().info(f'加载巡诊点-床位映射: {path}, {len(mapping)} 个巡诊点')
            for pid, bed_map in mapping.items():
                self.get_logger().info(
                    f'  巡诊点 {pid} -> ' +
                    ', '.join(f'det[{k}]=bed{v}' for k, v in sorted(bed_map.items()))
                )
            return mapping
        except Exception as e:
            self.get_logger().warn(f'加载巡诊点-床位映射失败: {e}, 使用默认顺序编号')
            return {}

    def _find_waypoints_config(self, param_path):
        """
        查找 waypoints.yaml 路径。

        优先级:
          1. 参数显式指定 waypoints_config_path
          2. ament index 中的 xjrobot_bridge 包 share 目录 config/waypoints.yaml
          3. 工作区源码目录 src/nav/xjrobot_bridge/config/waypoints.yaml

        Returns:
            str 路径或 None
        """
        if param_path:
            if os.path.exists(param_path):
                return param_path
            self.get_logger().warn(f'指定的 waypoints 配置不存在: {param_path}')

        candidates = []
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory('xjrobot_bridge')
            candidates.append(os.path.join(share_dir, 'config', 'waypoints.yaml'))
        except Exception:
            pass

        for parent in Path(__file__).resolve().parents:
            if (parent / 'src').exists() and (parent / 'install').exists():
                candidates.append(
                    parent / 'src' / 'nav' / 'xjrobot_bridge' / 'config' / 'waypoints.yaml'
                )
                candidates.append(
                    parent / 'install' / 'xjrobot_bridge' / 'share' /
                    'xjrobot_bridge' / 'config' / 'waypoints.yaml'
                )
        for path in candidates:
            if os.path.exists(path):
                return str(path)
        return None

    def _load_waypoints_config(self, param_path=''):
        """
        解析导航 waypoints.yaml，提取所有 bed_{patrol_id}_{seq} 点位。

        填充:
          self.patrol_bed_indexes: patrol_id -> [waypoint index, ...] (按 seq 升序)
          self.patrol_bed_names:   patrol_id -> ["bed_p_s", ...] (按 seq 升序)
          self.bed_name_to_index:  "bed_p_s" -> waypoint index

        Returns:
            bool 是否加载成功
        """
        path = self._find_waypoints_config(param_path)
        if not path:
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            # 兼容三种结构:
            #   1. {xjrobot_bridge_node: {ros__parameters: {...}}}
            #   2. {xjrobot_bridge_node: {...}}
            #   3. 平铺: {...}
            node_params = data.get('xjrobot_bridge_node', data)
            if 'ros__parameters' in node_params:
                node_params = node_params['ros__parameters']
            waypoint_ids = node_params.get('waypoint_ids', [])
            waypoints = node_params.get('waypoints', {})

            import re
            bed_pattern = re.compile(r'^bed_(\d+)_(\d+)$')

            # 收集所有 bed 条目: (patrol_id, seq, waypoint_id, index)
            beds = []
            for wid in waypoint_ids:
                m = bed_pattern.match(str(wid))
                if not m:
                    continue
                patrol_id = int(m.group(1))
                seq = int(m.group(2))
                entry = waypoints.get(wid)
                if not entry:
                    continue
                index = int(entry.get('index', -1))
                if index < 0:
                    continue
                beds.append((patrol_id, seq, wid, index))

            if not beds:
                self.get_logger().warn(f'waypoints.yaml 中没有 bed_{{patrol}}_{{seq}} 条目: {path}')
                return False

            beds.sort(key=lambda x: (x[0], x[1]))
            self.patrol_bed_indexes = {}
            self.patrol_bed_names = {}
            self.bed_name_to_index = {}
            for patrol_id, seq, wid, index in beds:
                self.patrol_bed_indexes.setdefault(patrol_id, []).append(index)
                self.patrol_bed_names.setdefault(patrol_id, []).append(wid)
                self.bed_name_to_index[wid] = index

            self.get_logger().info(f'加载导航 waypoints: {path}')
            for pid in sorted(self.patrol_bed_names):
                self.get_logger().info(
                    f'  巡诊点 {pid} -> ' +
                    ', '.join(
                        f'{name}(idx={self.bed_name_to_index[name]})'
                        for name in self.patrol_bed_names[pid]
                    )
                )
            return True
        except Exception as e:
            self.get_logger().error(f'解析 waypoints.yaml 失败: {e}')
            return False

    def _get_beds_for_patrol(self, patrol_id):
        """
        获取指定巡诊点下的床位信息。

        Returns:
            (bed_names, bed_indexes) 或 (None, None)（无数据）
            其中 bed_names/bed_indexes 均按序号升序, 前 N/2 为左侧(近->远), 后 N/2 为右侧(近->远)
        """
        if self.waypoints_config_loaded and patrol_id in self.patrol_bed_names:
            return self.patrol_bed_names[patrol_id], self.patrol_bed_indexes[patrol_id]

        # 回退: patrol_bed_mapping.json
        if patrol_id in self.patrol_bed_map:
            bed_map = self.patrol_bed_map[patrol_id]
            indexes = [bed_map[k] for k in sorted(bed_map.keys())]
            names = [f'bed_{patrol_id}_{i}' for i in range(len(indexes))]
            return names, indexes

        return None, None

    def _build_vlm_area_prompt(self, patrol_id):
        """
        构建 Area 模式 VLM prompt。

        Args:
            patrol_id (int): 巡诊点ID

        Returns:
            (system_prompt, user_prompt) 或 (None, None)（无床位数据）
        """
        bed_names, bed_indexes = self._get_beds_for_patrol(patrol_id)
        if not bed_names:
            return None, None

        total = len(bed_names)
        half = total // 2
        left_names = bed_names[:half]
        right_names = bed_names[half:]

        system_prompt = (
            f"你是一个医院病房床位检测助手。画面中固定位置有 {total} 张病床，"
            f"左右各 {half} 张。\n"
            f"左侧病床从近到远依次为: {', '.join(left_names)}\n"
            f"右侧病床从近到远依次为: {', '.join(right_names)}\n"
            "请判断每张床上是否有人。只输出一个JSON对象, 格式如下:\n"
            '{"occupied_beds": ["bed_X_Y", ...]}\n'
            '其中 occupied_beds 只包含有人病床的名称(必须是上面列出的名称)。'
            "如果所有病床都无人, 输出 {\"occupied_beds\": []}。\n"
            "不要输出任何其他内容。"
        )
        user_prompt = "请判断这张病房照片中哪些病床有人。"
        return system_prompt, user_prompt

    def _get_bed_id(self, patrol_id, detection_index):
        """根据巡诊点ID和YOLO检测框索引获取对应的导航床位编号"""
        if patrol_id in self.patrol_bed_map:
            bed_map = self.patrol_bed_map[patrol_id]
            if detection_index in bed_map:
                return bed_map[detection_index]
        # 无映射时按顺序编号 (index 0 -> bed_id 1)
        return detection_index + 1

    def image_callback(self, msg):
        """接收相机图像"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_received = True
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')

    def detect_beds_with_yolo(self, image):
        """
        使用YOLOv8检测图像中的床位

        Returns:
            List[dict]: [{'bbox': [x1,y1,x2,y2], 'class_id': 0, 'confidence': 0.95}, ...]
        """
        results = self.yolo_model(image, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class_id': class_id,
                    'confidence': confidence
                })

        return detections

    def detect_person_with_clip(self, bed_image):
        """
        使用CLIP判断床位上是否有人 (Area模式)

        Args:
            bed_image (np.ndarray): 床位区域图像

        Returns:
            tuple: (bool, float) - (是否有人, 置信度)
        """
        text = ["a patient lying on a hospital bed", "an empty hospital bed"]
        return self._clip_classify(bed_image, text)

    def detect_fall_risk_with_clip(self, bed_image):
        """
        使用CLIP判断病人是否有坠床风险 (Bed模式)

        Args:
            bed_image (np.ndarray): 床位区域图像

        Returns:
            tuple: (bool, float) - (是否有坠床风险, 置信度)
        """
        text = [
            "a patient about to fall off a hospital bed",
            "a patient safely lying in a hospital bed"
        ]
        return self._clip_classify(bed_image, text)

    def save_detection_capture(self, image, question, reply):
        """
        保存识别图片 及 同名的txt文件(含提问与回复)

        Args:
            image (np.ndarray): BGR 图像
            question (str): 发给VLM的提问/prompt
            reply (str): VLM的回复
        Returns:
            (img_path, txt_path) 或 (None, None)
        """
        try:
            os.makedirs(self.capture_dir, exist_ok=True)
            ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            base = f"{ts}_{int(time.time() * 1000) % 1000:03d}"
            img_path = os.path.join(self.capture_dir, base + ".png")
            txt_path = os.path.join(self.capture_dir, base + ".txt")
            cv2.imwrite(img_path, image)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"{question}\n\n")
                f.write(f"【回复】\n{reply}\n")
            self.get_logger().info(f'识别结果已保存: {img_path}')
            return img_path, txt_path
        except Exception as e:
            self.get_logger().error(f'保存识别结果失败: {e}')
            return None, None

    def detect_fall_risk_with_vlm(self, bed_image):
        """
        使用外部VLM API判断病人是否有坠床风险 (Bed模式, 替代CLIP)

        Args:
            bed_image (np.ndarray): 床位区域图像(BGR)

        Returns:
            tuple: (is_risk, risk_score, description)
                调用失败时返回 None, 由调用方回退到CLIP
        """
        if self.vlm_client is None:
            return None

        try:
            # 压缩图片, 限制最大边长, 控制传输体积
            h, w = bed_image.shape[:2]
            if max(h, w) > 1024:
                scale = 1024.0 / max(h, w)
                bed_image = cv2.resize(
                    bed_image,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            _, buf = cv2.imencode('.jpg', bed_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

            system_prompt = (
                "只输出一个JSON对象, 格式如下: "
                '{"has_fall_risk": "y" 或 "n", "posture": "不超过30字的人体姿态简单描述"}'
                "其中 y 表示有坠床风险, n 表示无坠床风险。"
                "posture 必须为非空字符串, 用不超过30字简单描述当前人体姿态。"
                "不要输出任何其他内容。"
            )

            completion = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                top_p=0.8,
                temperature=0.7,
                stream=False,
                max_tokens=300,
                extra_body={"enable_thinking": True, "thinking_budget": 2048},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                            {"type": "text", "text": "请判断这张病床照片中病人的坠床风险，并用不超过30字简单描述当前人体姿态。"},
                        ],
                    },
                ],
            )

            reply = completion.choices[0].message.content
            if not reply:
                self.get_logger().warn('VLM返回为空')
                return None

            # 保存识别图片 + 同名txt(提问与回复)
            question = system_prompt + "\n请判断这张病床照片中病人的坠床风险，并用不超过30字简单描述当前人体姿态。"
            self.save_detection_capture(bed_image, question, reply)

            # 解析 y/n 二分类 + 姿态描述
            try:
                j = json.loads(reply)
            except Exception:
                j = {}
            ans = str(j.get('has_fall_risk', reply)).strip().strip("'\"“”‘’ \t\n").lower()
            is_risk = (
                ans[:1] in ("y", "1", "有", "是")
                or ans in ("true", "yes", "risk", "有风险", "是风险")
            )
            posture = str(j.get('posture') or j.get('description') or "").strip()
            if not posture:
                posture = "有坠床风险" if is_risk else "姿态稳定"
            if len(posture) > 30:
                posture = posture[:30]
            risk_score = 0.9 if is_risk else 0.0
            description = posture

            self.get_logger().info(
                f'VLM判断: has_fall_risk={is_risk} ({"y" if is_risk else "n"}), '
                f'posture={posture}, score={risk_score:.3f}'
            )
            return is_risk, risk_score, description

        except Exception as e:
            self.get_logger().error(f'VLM检测失败, 回退CLIP: {e}')
            return None

    def _encode_image_b64(self, image):
        """将BGR图像压缩编码为base64 JPEG字符串"""
        h, w = image.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    def _call_vlm_once(self, system_prompt, user_prompt, image):
        """
        调用一次 VLM API。

        Returns:
            reply (str) 或 None（失败）
        """
        if self.vlm_client is None:
            self.get_logger().warn('VLM客户端不可用, 无法调用VLM')
            return None
        try:
            img_b64 = self._encode_image_b64(image)
            completion = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                top_p=0.8,
                temperature=0.7,
                stream=False,
                max_tokens=500,
                extra_body={"enable_thinking": True, "thinking_budget": 2048},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
            )
            reply = completion.choices[0].message.content
            if not reply:
                self.get_logger().warn('VLM返回为空')
                return None
            return reply
        except Exception as e:
            self.get_logger().error(f'VLM API调用失败: {e}')
            return None

    def _parse_occupied_beds(self, reply):
        """
        解析VLM返回的 occupied_beds JSON。

        Returns:
            list[str]: 有人床位名称列表（已验证存在于 bed_name_to_index）
        """
        if not reply:
            return []
        try:
            j = json.loads(reply)
            if isinstance(j, dict):
                raw = j.get('occupied_beds', [])
                if isinstance(raw, list):
                    return [str(x).strip() for x in raw if str(x).strip()]
        except Exception:
            pass
        # JSON 解析失败: 尝试正则提取 bed_X_Y
        matched = re.findall(r'bed_\d+_\d+', reply)
        return list(dict.fromkeys(matched))

    def _detect_beds_with_vlm_area(self, image, patrol_id):
        """
        Area模式: 使用VLM一次性识别画面中所有床位的有人/无人状态。

        VLM根据固定画面布局直接输出有人床位的 bed_{patrol}_{seq} 名称,
        再通过 waypoints.yaml 的 bed_name_to_index 映射为导航 waypoint index。

        Args:
            image (np.ndarray): BGR图像
            patrol_id (int): 巡诊点ID

        Returns:
            (occupied_indexes, urgencies, details)
              occupied_indexes: list[int] 有人床位的 waypoint index
              urgencies: list[int] 对应紧急程度(有人=1)
              details: str 描述信息
        """
        bed_names, bed_indexes = self._get_beds_for_patrol(patrol_id)
        if not bed_names:
            self.get_logger().warn(f'巡诊点 {patrol_id} 没有床位数据')
            return [], [], f"No bed data for patrol {patrol_id}"

        system_prompt, user_prompt = self._build_vlm_area_prompt(patrol_id)
        if system_prompt is None:
            return [], [], f"No bed data for patrol {patrol_id}"

        reply = None
        for attempt in range(self.vlm_area_retries + 1):
            reply = self._call_vlm_once(system_prompt, user_prompt, image)
            if reply is not None:
                break
            if attempt < self.vlm_area_retries:
                time.sleep(self.vlm_area_retry_delay)

        if reply is None:
            self.get_logger().error(f'VLM Area检测失败(巡诊点 {patrol_id}), 重试 {self.vlm_area_retries} 次后放弃')
            return [], [], f"VLM call failed for patrol {patrol_id}"

        # 保存识别图片 + 同名txt
        question = system_prompt + "\n" + user_prompt
        self.save_detection_capture(image, question, reply)

        occupied_names = self._parse_occupied_beds(reply)
        if not occupied_names:
            self.get_logger().info(
                f'巡诊点 {patrol_id}: VLM判定所有床位无人 ({len(bed_names)} 张床)'
            )
            return [], [], f"No occupied beds (patrol {patrol_id})"

        occupied_indexes = []
        unknown = []
        for name in occupied_names:
            if name in self.bed_name_to_index:
                occupied_indexes.append(self.bed_name_to_index[name])
            else:
                unknown.append(name)
        if unknown:
            self.get_logger().warn(
                f'VLM返回的床位名称不在waypoints映射中: {unknown}'
            )
        # 去重并保持顺序
        seen = set()
        occupied_indexes = [i for i in occupied_indexes if not (i in seen or seen.add(i))]

        urgencies = [1] * len(occupied_indexes)
        details = (
            f"Patrol {patrol_id}: VLM detected occupied beds "
            f"{occupied_names} -> indexes {occupied_indexes}"
        )
        self.get_logger().info(
            f'Area模式VLM检测完成: 巡诊点 {patrol_id}, '
            f'{len(occupied_indexes)} 个有人床位 -> {occupied_indexes}'
        )
        return occupied_indexes, urgencies, details

    def _clip_classify(self, bed_image, text_prompts):
        """
        CLIP通用分类方法

        Args:
            bed_image (np.ndarray): 床位区域图像
            text_prompts (list[str]): 两个文本提示, 第一个为"异常/正向", 第二个为"正常/负向"

        Returns:
            tuple: (bool, float) - (第一个提示得分更高, 第一个提示的置信度)
        """
        try:
            bed_image_rgb = cv2.cvtColor(bed_image, cv2.COLOR_BGR2RGB)
            bed_image_pil = PILImage.fromarray(bed_image_rgb)
            image_tensor = self.clip_preprocess(bed_image_pil).unsqueeze(0)

            device = next(self.clip_model.parameters()).device
            text_tokens = self.clip_tokenizer(text_prompts).to(device)
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                image_features = self.clip_model.encode_image(image_tensor)

            similarity = (image_features @ text_features.T).softmax(dim=-1)
            positive_score = similarity[0][0].item()
            negative_score = similarity[0][1].item()

            is_positive = positive_score > negative_score
            return is_positive, positive_score

        except Exception as e:
            self.get_logger().error(f'CLIP检测失败: {e}')
            return False, 0.0

    def handle_detect_request(self, request, response):
        """
        处理床位检测服务请求 - Area和Bed模式
        """
        if not self.image_received or self.current_image is None:
            self.get_logger().warn('未收到相机图像')
            response.is_anomaly = False
            response.details = "No image available"
            response.bed_ids = []
            response.urgencies = []
            return response

        if request.mode == MODE_AREA:
            return self._handle_area_mode(request, response)
        elif request.mode == MODE_BED:
            return self._handle_bed_mode(request, response)
        else:
            self.get_logger().warn(f'未知模式: {request.mode}, 支持Area(0)/Bed(1)')
            response.is_anomaly = False
            response.details = f"Unknown mode: {request.mode}"
            response.bed_ids = []
            response.urgencies = []
            return response

    def _switch_camera_mode(self, mode_name):
        """通知模拟相机切换图片模式"""
        msg = String()
        msg.data = mode_name
        self.mode_pub.publish(msg)
        self.get_logger().info(f'已发送相机模式切换: {mode_name}')

    def _handle_area_mode(self, request, response):
        """
        Area模式处理逻辑
        使用VLM一次性识别画面中所有床位的有无人状态, 返回有人床位的导航 index 列表。

        识别位置固定: VLM 根据固定的左右床位布局直接输出有人床位的
        bed_{patrol_id}_{seq} 名称, 再经 waypoints.yaml 映射为导航 waypoint index。
        """
        self._switch_camera_mode('area')
        patrol_id = request.area_bed_id
        self.get_logger().info(f'开始Area模式检测, 巡诊点ID: {patrol_id}')

        occupied_indexes, urgencies, details = self._detect_beds_with_vlm_area(
            self.current_image, patrol_id
        )

        response.is_anomaly = len(occupied_indexes) > 0
        response.details = details
        response.bed_ids = occupied_indexes
        response.urgencies = urgencies

        self.get_logger().info(
            f'Area模式检测完成: {len(occupied_indexes)} 个有人床位 -> {occupied_indexes}'
        )

        return response

    def _handle_bed_mode(self, request, response):
        """
        Bed模式处理逻辑
        相机已对准目标床位, 直接对整帧图像做坠床风险判断(VLM优先, CLIP回退),
        不做YOLO全量床位检测

        Args:
            request.area_bed_id: 目标床位ID (仅用于记录/返回)
        """
        self._switch_camera_mode('bed')
        target_bed_id = request.area_bed_id
        self.get_logger().info(f'开始Bed模式检测, 目标床位ID: {target_bed_id}')

        # Bed模式下相机画面即目标床位, 直接使用整帧图像
        bed_image = self.current_image

        # 使用VLM判断是否有坠床风险, 失败时回退到CLIP
        vlm_result = self.detect_fall_risk_with_vlm(bed_image)
        if vlm_result is not None:
            is_risk, risk_score, vlm_desc = vlm_result
            judge_source = 'VLM'
        else:
            is_risk, risk_score = self.detect_fall_risk_with_clip(bed_image)
            vlm_desc = 'y' if is_risk else 'n'
            judge_source = 'CLIP'

        # Bed模式: 有坠床风险即为异常
        is_anomaly = is_risk
        response.is_anomaly = is_anomaly
        response.details = (
            f"Bed {target_bed_id}: {'fall risk detected' if is_anomaly else 'patient safe'}, "
            f"source: {judge_source}, "
            f"score: {risk_score:.3f}, VLM: {vlm_desc}"
        )
        response.bed_ids = [target_bed_id] if is_anomaly else []
        response.urgencies = [1 if risk_score > 0.7 else 0] if is_anomaly else []

        self.get_logger().info(
            f'Bed模式检测完成: 床位 {target_bed_id} '
            f'{"有坠床风险(异常)" if is_anomaly else "病人安全(正常)"}, '
            f'判断来源: {judge_source}, 置信度: {risk_score:.3f}'
        )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = BedDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()