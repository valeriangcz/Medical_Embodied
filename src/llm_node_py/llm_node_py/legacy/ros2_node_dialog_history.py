import rclpy
from threading import Lock
from rclpy.node import Node

"""
测试：
发送消息给 history manager:
ros2 topic pub /send_to_dialog_history llm_node_comm/msg/LlmDialogHistoryMsg "{question: '你好，今天天气怎么样？', response: '今天天气晴朗，适合出行。'}" --once

从 history manager 中获取消息：
ros2 service call /get_llm_dialog_history llm_node_comm/srv/LlmDialogHistorysSrv "{}"

"""

import sys, os
# 把 ros2 生成的 srv 文件的地址加上
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "../../../install/llm_node_comm/lib/python3.12/site-packages"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from llm_node_comm.msg import LlmDialogHistoryMsg
from llm_node_comm.srv import (
    LlmDialogHistorysSrv,
)


class DialogHistoryNode(Node):
    """
    LLM 对话历史管理节点

    Topic:
        /send_to_dialog_history
            接收 question + response

    Service:
        /get_llm_dialog_history
            返回历史问答列表

    Param:
        ~max_history_size   (int)
            最大历史条数，默认 20
    """

    def __init__(self, max_history_size:int = 5):
        super().__init__("dialog_history_node")

        # 参数：最大历史长度
        self.max_history_size = max_history_size

        self.history = []  # 是一个 LlmDialogHistoryMsg 的 list
        self.lock = Lock()

        # 订阅问答输入
        self.sub = self.create_subscription(
            LlmDialogHistoryMsg,
            "/send_to_dialog_history",
            self.dialog_callback,
            100
        )

        # 服务：获取历史
        self.service = self.create_service(
            LlmDialogHistorysSrv,
            "/get_llm_dialog_history",
            self.handle_get_history
        )

        self.get_logger().info(
            f"DialogHistoryNode started. Max history size: {self.max_history_size}"
        )

    # ==============================
    # 接收新的问答
    # ==============================
    def dialog_callback(self, msg: LlmDialogHistoryMsg):
        with self.lock:
            # 如果达到最大长度，删除最老的一条
            if len(self.history) >= self.max_history_size:
                removed = self.history.pop(0)
                self.get_logger().info(
                    f"History limit reached. Removed oldest dialog:\nQ: {removed.question}\nA: {removed.response}"
                )

            self.history.append(msg)

            self.get_logger().info(
                f"Added dialog:\nQ: {msg.question}\nA: {msg.response}"
            )
            # ==============================
            # 输出当前完整 history
            # ==============================
            self.get_logger().info(
                f"------ Current Dialog History ({len(self.history)} items) ------"
            )

            for idx, item in enumerate(self.history):
                self.get_logger().info(
                    f"[{idx}]\nQ: {item.question}\nA: {item.response}"
                )

            self.get_logger().info("------------------------------------------------")

    # ==============================
    # 处理 Service 请求
    # ==============================
    def handle_get_history(self, req, res):
        with self.lock:
            history_copy = list(self.history)

            # if req.clear_after_get:
            #     self.history.clear()
            #     self.get_logger().info("Dialog history cleared after service call.")

        res.history = history_copy
        return res


if __name__ == "__main__":
    try:
        rclpy.init()
        node = DialogHistoryNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
