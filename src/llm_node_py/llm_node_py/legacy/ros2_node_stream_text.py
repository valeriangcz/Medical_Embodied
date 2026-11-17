
"""
2026-4-3 测试没啥问题
"""


"""
发布消息给 stream text

ros2 topic pub -1 /ali_llm_output std_msgs/String "data: '你'"
ros2 topic pub -1 /ali_llm_output std_msgs/String "data: '好，'"

ros2 topic pub -1 /ali_llm_output std_msgs/String "data: '这个是一条'"
ros2 topic pub -1 /ali_llm_output std_msgs/String "data: '测试消息。'"

监听：

ros2 topic echo /tts_realtime_data
"""

import os
import re

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SentenceSplitterNode(Node):
    """
    ROS 节点: 接收大模型的零碎文本，并按中文逗号、句号、分号拆分成完整句子
    支持 [START] / [DONE] 控制
    """

    def __init__(self):
        super().__init__("sentence_splitter_node")

        self.buffer = ""
        # 首句长度上限：>0 时第一段在累计到该字数就立即送 TTS（缩短首音延迟），
        # 默认 0 = 关闭（保持原断句行为）
        self.first_fragment_max = int(os.environ.get("FIRST_FRAGMENT_MAX_CHARS", "0"))
        self.first_flushed = False
        # 扩展断句标点（：、！？）：默认开启；=0 时恢复原规则（仅 ，。；）
        self.extra_punct = os.environ.get("SPLITTER_EXTRA_PUNCT", "1") != "0"
        if self.extra_punct:
            self.pattern = r"[^，。；：、！？]*[，。；：、！？]?"
            self.sentence_marks = "。；！？"
            self.short_marks = "，：、"
        else:
            self.pattern = r"[^，。；]*[，。；]?"
            self.sentence_marks = "。；"
            self.short_marks = "，"
        if self.first_fragment_max > 0:
            self.get_logger().info(
                f"首句提前切分已启用：FIRST_FRAGMENT_MAX_CHARS={self.first_fragment_max}"
            )

        self.sub = self.create_subscription(
            String,
            "/ali_llm_output",
            self.on_text,
            500,
        )

        self.pub = self.create_publisher(
            String,
            "/tts_realtime_data",
            100,
        )

        self.get_logger().info("Sentence Splitter Node ready")

    def on_text(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return

        if text == "[START]":
            self.buffer = ""
            self.first_flushed = False
            start_msg = String()
            start_msg.data = "[START]"
            self.pub.publish(start_msg)
            self.get_logger().info("[CONTROL] [START] received, buffer cleared")
            return

        if text == "[DONE]":
            if self.buffer:
                remaining_msg = String()
                remaining_msg.data = self.buffer
                self.pub.publish(remaining_msg)
                self.get_logger().info(
                    f"[BUFFER] Remaining buffer sent: {self.buffer}"
                )
                self.buffer = ""

            done_msg = String()
            done_msg.data = "[DONE]"
            self.pub.publish(done_msg)
            self.get_logger().info("[CONTROL] [DONE] received, buffer flushed")
            return

        self.buffer += text

        parts = re.findall(self.pattern, self.buffer)

        new_buffer = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if part[-1] in self.sentence_marks:
                self.first_flushed = True
                sentence_msg = String()
                sentence_msg.data = part
                self.pub.publish(sentence_msg)
                self.get_logger().info(f"[SENTENCE] {part}")
            elif part[-1] == "，":
                # 首段过长时按上限切开，让 TTS 尽早开始合成（默认关闭）
                if (not self.first_flushed and self.first_fragment_max > 0
                        and len(part) > self.first_fragment_max):
                    cut = self.first_fragment_max
                    fragment_msg = String()
                    fragment_msg.data = part[:cut]
                    self.pub.publish(fragment_msg)
                    self.get_logger().info(f"[FIRST-CUT] {fragment_msg.data}")
                    part = part[cut:]
                self.first_flushed = True
                fragment_msg = String()
                fragment_msg.data = part
                self.pub.publish(fragment_msg)
                self.get_logger().info(f"[FRAGMENT] {part}")
            elif part[-1] in self.short_marks:
                self.first_flushed = True
                fragment_msg = String()
                fragment_msg.data = part
                self.pub.publish(fragment_msg)
                self.get_logger().info(f"[FRAGMENT] {part}")
            else:
                new_buffer += part

        self.buffer = new_buffer

def main(args=None) -> None:
    rclpy.init(args=args)

    node = None
    try:
        node = SentenceSplitterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
