# 给 ui 单独开一个 llm, 不要让它污染了 对话历史

"""
2026-4-3 测试没啥问题
"""

"""
测试

1. 发送

ros2 topic pub -1 /ui_question_to_llm std_msgs/String "data: '你好，LLM！'"

2. 接收

ros2 topic echo /ali_llm_output

"""

from http import HTTPStatus

import rclpy
from dashscope import Application
from rclpy.node import Node
from std_msgs.msg import String

import os, sys
# 当前目录
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "./"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from get_dashscope_key import get_dashscope_key

# input  -> llm_to_dashapp

# dialog history input -> get_llm_dialog_history

# output -> ali_llm_output(tts), huzhou_llm_res(ui), send_to_dialog_history



class KnowledgeBaseLLMNode(Node):
    def __init__(self):
        super().__init__("ros_node_llm_ui")

        self.sub = self.create_subscription(
            String,
            "/ui_question_to_llm",
            self.query_callback,
            10,
        )

        self.pub = self.create_publisher(
            String,
            "/ali_llm_output",
            10,
        )

        # self.ui_pub = self.create_publisher(String, "/huzhou_llm_res", 10)

        self.get_logger().info("KnowledgeBase LLM Node started.")

    # 从多问题管理节点发过来的问题
    def query_callback(self, msg: String) -> None:
        question = msg.data.strip()
        self.get_logger().info(f"Received question: {question}")

        try:
            history_list = [
                {
                    "user": "有几个血透中心的病人，现在在血透中心做透析，你是巡诊的机器人，名字叫小医。你的大脑跟病人的测量数据是实时联系在一起的，下面几个病人开始对话。对话的时候不要长篇大论，像真人一样，使用对话句子，也就是我们一句来一句去这样。语气需要轻柔舒缓。所有使用单位名称转化为中文。尽可能不出现英文。你只需要监测，禁止输出执行实际动作,例如倒水。不要增加小动作。不要输出表情符号。在需要时，给出数据偏高或偏低，不需要具体数据。不要给出实际的执行动作，而是不用通过类似读取的方式。讲话必须亲切，温柔。科普类的内容可以丰富一些。叫护士或医生时回答好的，我马上帮您联系护士或医生。您先安心休息一会儿。禁止在回答中出现着急两个字。问题不严重时，你来提出解决办法，不要经常输出寻找医生。询问天气/温度等关键词，仅输出杭州本地核心信息.",
                    "bot": "好的，我记住了,我的名字是小医，我会采用真人对话风格那样进行对话，并使用轻柔温和的语气，我不会输出执行动作，不会输出表情符合，不会给出具体的数值，问题不严重时我会自己提出解决办法不会寻求医生的帮助，不会使用'着急'等急迫性质的词语。",
                }
            ]

            responses = Application.call(
                api_key=get_dashscope_key(),
                app_id='b7ed8dad5c0c483d8e865a3be0b4e088',
                prompt=question,
                history=history_list,
                stream=True,
                incremental_output=True,
            )

            start_msg = String()
            start_msg.data = "[START]"
            self.pub.publish(start_msg)

            for response in responses:
                if response.status_code == HTTPStatus.OK:
                    chunk = response.output.text
                    chunk_msg = String()
                    chunk_msg.data = chunk
                    self.pub.publish(chunk_msg)
                else:
                    self.get_logger().error(f"LLM response error: {response}")
                    break

            done_msg = String()
            done_msg.data = "[DONE]"
            self.pub.publish(done_msg)

            self.get_logger().info("Dialog pair sent to dialog_history.")

        except Exception as exc:
            self.get_logger().error(f"Error querying KnowledgeBase LLM: {exc}")
            error_msg = String()
            error_msg.data = "[ERROR] Failed to get response."
            self.pub.publish(error_msg)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None
    try:
        node = KnowledgeBaseLLMNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
