# 用来判断用户是否想结束对话
"""
使用说明（ROS2）:
1) 启动节点:
   先在工作空间根目录编译并加载环境:
   colcon build --packages-select llm_node_py llm_node_comm
   source install/setup.bash

   启动方式（二选一）:
   ros2 run llm_node_py ros2_node_check_end_session
   或
   python3 src/llm_node_py/llm_node_py/ros2_node_check_end_session.py

2) 命令行测试服务:
   ros2 service call /check_end_dialog llm_node_comm/srv/EndSession "{question: '没什么事了，先这样吧'}"
   
   ros2 service call /check_end_dialog llm_node_comm/srv/EndSession "{question: '我想问一下肾病饮食注意事项'}"

3) 查看/监听通信信息:
   本节点提供的是 Service，不发布自定义 Topic。
   可用以下命令查看服务信息:
   ros2 service list | grep check_end_dialog
   ros2 service type /check_end_dialog
   ros2 service info /check_end_dialog

   若需看节点日志（日志会发到 /rosout）:
   ros2 topic echo /rosout
"""

import json
import os
from pathlib import Path

import rclpy
from openai import OpenAI
from rclpy.node import Node


from llm_node_comm.srv import EndSession


def load_dashscope_api_key() -> str:
    current_dir = Path(__file__).resolve().parent
    key_file = (current_dir / "../key/dashscope.key").resolve()
    if not key_file.exists():
        raise FileNotFoundError(f"未找到 DashScope key 文件: {key_file}")
    api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"DashScope key 文件为空: {key_file}")
    return api_key


class EndSessionService(Node):
    """
    ROS2 Node 提供服务，用于判断用户是否想结束对话
    """

    def __init__(self):
        super().__init__("end_session_service_node")

        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("all_proxy", None)
        os.environ.pop("ALL_PROXY", None)

        self.client = OpenAI(
            api_key=load_dashscope_api_key(),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        self.history = [
            {
                "role": "system",
                "content": (
                    "你是聊天助手，负责监测用户输入，"
                    "判断用户是否想结束本次对话。"
                    "你只判断意图，不执行实际操作。"
                    "返回 JSON 格式："
                    "need_end（bool，是否想结束对话），"
                    "reason（string，判断原因），"
                    "response（string，给用户的回复内容，如果不需要结束则为空）。"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "好的，我会根据用户的输入判断是否想结束对话，"
                    "并且我会严格输出 JSON 格式，如果不想结束对话则 response 为空。"
                ),
            },
            {
                "role": "user",
                "content": "没什么事儿了，你走吧。",
            },
            {
                "role": "assistant",
                "content": (
                    '{"need_end": true, '
                    '"reason": "用户明确表示想结束聊天。", '
                    '"response": "好的，那我们下次再聊。"}'
                ),
            },
            {
                "role": "user",
                "content": "没事儿了，你滚蛋吧。",
            },
            {
                "role": "assistant",
                "content": (
                    '{"need_end": true, '
                    '"reason": "用户明确表示想结束聊天。", '
                    '"response": "好的，那我们下次再聊。"}'
                ),
            },
            {
                "role": "user",
                "content": "肾病患者应该注意些什么？",
            },
            {
                "role": "assistant",
                "content": (
                    '{"need_end": false, '
                    '"reason": "用户仍在进行咨询或聊天，没有表示结束意图。", '
                    '"response": ""}'
                ),
            },
        ]

        self.service = self.create_service(
            EndSession,
            "check_end_dialog",
            self.handle_service,
        )
        self.get_logger().info(
            "End Session Service is ready. Service name: /check_end_dialog"
        )

    def handle_service(
        self,
        request: EndSession.Request,
        response: EndSession.Response,
    ) -> EndSession.Response:
        """
        接收 EndSession 请求，返回用户是否想结束对话
        """
        question = request.question

        try:
            prompt = [{"role": "user", "content": question}]
            completion = self.client.chat.completions.create(
                model="qwen3.5-flash",
                messages=self.history + prompt,  # type: ignore[arg-type]
                stream=False,
                extra_body={"enable_thinking": False},
            )

            reply_text = completion.choices[0].message.content
            if reply_text is None:
                raise ValueError("模型返回为空")

            result = json.loads(reply_text)

            need_end = result.get("need_end", False)
            reason = result.get("reason", "")
            response_text = result.get("response", "")

            self.get_logger().info(
                f"Q: {question} | need_end: {need_end} | reason: {reason} | response: {response_text}"
            )

            response.need_end = need_end
            response.comment = reason
            response.response = response_text
            return response

        except Exception as exc:
            self.get_logger().error(f"Error in end session service: {exc}")
            response.need_end = False
            response.comment = "[Error] Failed to query LLM"
            response.response = ""
            return response


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None
    try:
        node = EndSessionService()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if "node" in locals() and node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
