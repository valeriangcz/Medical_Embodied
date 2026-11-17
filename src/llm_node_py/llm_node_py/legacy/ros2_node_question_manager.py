"""

这个测试的时候要把 ros2_node_llm_ui.py 启动

# 测试多个从 ui 输入的问题
ros2 topic pub --once /ui_send_question_to_queue std_msgs/msg/String "{data: '肾病患者运动时有哪些注意事项？'}"

ros2 topic pub --once /ui_send_question_to_queue std_msgs/msg/String "{data: '慢性肾病患者如何合理安排蛋白质摄入？'}"

ros2 topic pub --once /ui_send_question_to_queue std_msgs/msg/String "{data: '透析患者饮水量应该如何控制？'}"

"""

import queue
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LLMQuestionNode(Node):
    def __init__(self):
        super().__init__("question_manager_node")

        self.lock = threading.Lock()
        self.busy = False
        self.question_queue = queue.Queue()
        self.cur_question = None

        # 订阅从 ui 发送过来的问题
        self.sub_question = self.create_subscription(
            String,
            "/ui_send_question_to_queue",
            self.question_ui_callback,
            10,
        )

        # 订阅 TTS 完成信号
        self.sub_finished = self.create_subscription(
            String,
            "/tts_session_finished",
            self.finished_callback,
            10,
        )

        # 发布给大模型
        self.pub_llm_input = self.create_publisher(
            String,
            "/ui_question_to_llm",
            10,
        )

        self.get_logger().info("LLM 问题管理节点已启动，当前处于空闲状态。")

    # 把问题发送给大模型
    def send_question_to_llm(self, queue_item) -> None:
        self.cur_question = queue_item

        message = String()
        message.data = queue_item["question"]
        self.pub_llm_input.publish(message)

    # ==========================================================
    # 接收来自 ui 的问题
    # ==========================================================
    def question_ui_callback(self, msg: String) -> None:
        self.get_logger().info(f"q:{msg.data}, source: ui")

        queue_item = {
            "question": msg.data,
            "source": "ui",
            "patient_id": -1,
        }

        with self.lock:
            if not self.busy:
                self.busy = True
                self.send_question_to_llm(queue_item=queue_item)
            else:
                self.get_logger().info("当前模型忙碌，问题加入队列。")
                self.question_queue.put(queue_item)

    # ==========================================================
    # TTS 播放完成回调
    # ==========================================================
    def finished_callback(self, msg: String) -> None:
        self.get_logger().info("收到 TTS 播放完成信号。")

        with self.lock:
            if not self.question_queue.empty():
                next_question = self.question_queue.get()
                self.get_logger().info("处理下一个排队问题。")
                self.send_question_to_llm(next_question)
            else:
                self.get_logger().info("队列为空，模型空闲。")
                self.busy = False
                self.cur_question = None


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None
    try:
        node = LLMQuestionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
