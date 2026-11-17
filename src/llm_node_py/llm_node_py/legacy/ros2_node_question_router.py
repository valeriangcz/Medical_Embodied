import re
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

KEYWORD_LIST = [
    "天气",
    # 可追加："温度", "湿度", "空气质量", "杭州"
]



# 总体的问题输入topic
"""
要测试这个节点的话，要启动 llm_medical, llm_network, dialog_history_manager

测试：

ros2 topic pub --once /question_asr std_msgs/msg/String "{data: '天气今天如何？'}"

监听输出消息：

ros2 topic echo /ali_llm_output

"""



TOPIC_QUESTION_INPUT = "/question_asr"

# 把 question 路由到 联网大模型
TOPIC_PUBLISH_TO_OPENAI = "/llm_to_openai"
# 把 question 路由到 知识库大模型
TOPIC_PUBLISH_TO_DASHAPP = "/llm_to_dashapp"


class LLMAPIRouterNode(Node):
    def __init__(self):
        super().__init__("llm_api_router_node")

        self.keyword_set = {kw.lower() for kw in KEYWORD_LIST}

        self.typo_map = {
            "火龙观": "火龙罐",
            "内露": "内瘘",
            "内漏": "内瘘",
            "血林": "血磷",
            "首卫生": "手卫生",
            # 唤醒词"小医"的 ASR 常见同音错字（实测出现过 小一/小伊/小易/小姨）
            # 注："小姨"已由句首正则处理（见 _wake_re），字典不收以免误伤
            "小一": "小医",
            "小伊": "小医",
            "小易": "小医",
            # 血透语境下"每周"被误识别为"美洲"（实测 2026-11）
            "美洲": "每周",
        }
        # 句首"小X"称呼 → 小医（含小姨；后随称谓/动词/标点或句尾才命中）
        self._wake_re = re.compile(
            r"^\s*小[一二伊姨易](?=[你好我请麻烦帮带想，,。？?！!]|$)")
        self.typo_pattern = re.compile("|".join(map(re.escape, self.typo_map.keys())))

        self.sub_question = self.create_subscription(
            String,
            TOPIC_QUESTION_INPUT,
            self.question_callback,
            10,
        )

        self.pub_to_openai = self.create_publisher(
            String,
            TOPIC_PUBLISH_TO_OPENAI,
            10,
        )
        self.pub_to_dashapp = self.create_publisher(
            String,
            TOPIC_PUBLISH_TO_DASHAPP,
            10,
        )

        self.get_logger().info("LLM API 分流节点启动成功")
        self.get_logger().info(f"统一问题入口话题：{TOPIC_QUESTION_INPUT}")
        self.get_logger().info(
            f"关键词列表：{KEYWORD_LIST}（匹配转发到 {TOPIC_PUBLISH_TO_OPENAI}）"
        )
        self.get_logger().info(f"未匹配关键词转发到：{TOPIC_PUBLISH_TO_DASHAPP}")
        self.get_logger().info("等待用户问题输入...")

    def fix_typos(self, text: str) -> str:
        """一次性替换已知错别字，返回修正后的字符串"""
        if not text or not isinstance(text, str):
            return text
        text = self.typo_pattern.sub(lambda m: self.typo_map[m.group(0)], text)
        # 唤醒词"小医"的句首同音错字（实测 ASR 输出过 小一/小伊/小易/小姨）。
        # 只替换句首称呼位置，避免误伤"我小姨也透析"这类真实称谓。
        text = self._wake_re.sub("小医", text)
        return text

    def question_callback(self, msg: String) -> None:
        """接收统一入口的问题，预处理后分发"""
        raw_question = msg.data.strip()
        if not raw_question:
            self.get_logger().warning("接收到空问题，忽略处理")
            return

        fixed_question = self.fix_typos(raw_question)
        if raw_question != fixed_question:
            self.get_logger().info(f"错别字修正：{raw_question} -> {fixed_question}")

        self.get_logger().info(f"接收到用户问题：{fixed_question}")

        threading.Thread(
            target=self.route_question,
            args=(fixed_question,),
            daemon=True,
        ).start()

    def route_question(self, question: str) -> None:
        """核心分流逻辑：关键词匹配后发布到对应话题"""
        try:
            message = String()
            message.data = question

            if any(kw in question.lower() for kw in self.keyword_set):
                self.pub_to_openai.publish(message)
                self.get_logger().info(
                    f"匹配关键词，转发到OpenAI接口话题：{TOPIC_PUBLISH_TO_OPENAI}"
                )
            else:
                self.pub_to_dashapp.publish(message)
                self.get_logger().info(
                    f"未匹配关键词，转发到DashScope接口话题：{TOPIC_PUBLISH_TO_DASHAPP}"
                )
        except Exception as exc:
            self.get_logger().error(f"问题分发失败：{exc}")


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None
    try:
        node = LLMAPIRouterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if rclpy.ok():
            print(f"分流节点启动失败：{exc}")
        raise
    finally:
        if "node" in locals() and node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
