



"""
监听放缓的文本消息，发布给 ui 显示

ros2 topic echo /huzhou_llm_res


不断的发布消息给这个节点：

ros2 topic pub tts_realtime_data std_msgs/String "data: '连续测试消息'" --rate 2

"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LLMUIBridge(Node):
    def __init__(self):
        super().__init__("llm_ui_bridge")

        self.ui_pub = self.create_publisher(
            String,
            "/huzhou_llm_res",
            100,
        )

        self.sub = self.create_subscription(
            String,
            "tts_realtime_data",
            self.llm_callback,
            200,
        )

        self.char_time = 0.185 / 0.85  # 这里的 0.85 说的是速度比原来的速度放缓了 0.85 倍

        self.get_logger().info("llm_ui_bridge started")

    def llm_callback(self, msg: String) -> None:
        text = msg.data.strip()

        if not text:
            return

        self.get_logger().info(f"Receive LLM text: {text}")

        out = String()
        out.data = text
        self.ui_pub.publish(out)

        length = len(text)
        sleep_time = length * self.char_time

        self.get_logger().info(f"Sleep {sleep_time:.2f} sec for TTS")
        time.sleep(sleep_time)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None
    try:
        node = LLMUIBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
