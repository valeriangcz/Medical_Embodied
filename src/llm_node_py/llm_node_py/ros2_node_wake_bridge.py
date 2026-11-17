#!/usr/bin/env python
"""唤醒桥：把 C++ ASR 节点的 /call_signal 转成 /start_asr。

生产环境里这一环由行为树/蓝牙节点完成（收到 KWS 信号后发 /start_asr 开启收音）。
精简栈里没有它们，所以真人说话时麦克风会一直停在 pause 状态。
本节点只做转发：/call_signal(Bool true) -> /start_asr(Bool true)。
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class WakeBridge(Node):
    def __init__(self):
        super().__init__("wake_bridge")
        self.pub = self.create_publisher(Bool, "/start_asr", 10)
        self.create_subscription(Bool, "/call_signal", self.on_call_signal, 10)
        self.get_logger().info("wake_bridge ready: /call_signal -> /start_asr")

    def on_call_signal(self, msg: Bool):
        if not msg.data:
            return
        self.pub.publish(Bool(data=True))
        self.get_logger().info("kws signal received -> publish /start_asr")


def main():
    rclpy.init()
    node = WakeBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
