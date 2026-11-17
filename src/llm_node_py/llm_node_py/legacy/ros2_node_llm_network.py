
from openai import OpenAI
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

"""
测试这个节点的时候也要把 dialog history 节点启动

向 llm medical 节点发送消息：

ros2 topic pub -1 /llm_to_openai std_msgs/msg/String "data: '今天的天气怎么样？'"

 

监听发送过来的消息：

ros2 topic echo /ali_llm_output

"""
import os, sys
# 把 ros 生成的 srv 文件的地址加上
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "../../../install/llm_node_comm/lib/python3.12/site-packages"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 当前目录
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "./"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from llm_node_comm.msg import LlmDialogHistoryMsg
from llm_node_comm.srv import LlmDialogHistorysSrv
from get_dashscope_key import get_dashscope_key


class NetworkedLLMNode(Node):
    def __init__(self):
        super().__init__('networked_llm_node')

        # 订阅问题输入
        self.sub = self.create_subscription(String, "/llm_to_openai", self.query_callback, 10)
        # 发布模型输出
        self.pub = self.create_publisher(String, "/ali_llm_output", 10)
        # 发布内容到 ui 界面
        # self.ui_pub = self.create_publisher(String, "/huzhou_llm_res", 10)
        # 发布到 dialog history 节点
        self.dialog_pub = self.create_publisher(LlmDialogHistoryMsg, "/send_to_dialog_history", 10)

        # 这里必须取消所有代理，下面的 OpenAI 才可以正常初始化
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("all_proxy", None)
        os.environ.pop("ALL_PROXY", None)

        # 初始化 OpenAI 客户端（联网模型）
        self.client = OpenAI(
            api_key=get_dashscope_key(),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 初始化获取 dialog history 的 ROS 服务 client
        self.get_history_client = self.create_client(
            LlmDialogHistorysSrv,
            "/get_llm_dialog_history",
        )

        self.get_logger().info("Networked LLM Node started. Waiting for input...")


    def query_callback(self, msg: String):
        question = msg.data.strip()
        self.get_logger().info(f"Received question: {question}")
        threading.Thread(target=self.handle_query, args=(question,), daemon=True).start()


    def handle_query(self, question: str):
        try:
            if not self.get_history_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("Service /get_llm_dialog_history not available")
                return

            # -------------------------------
            # 获取对话历史
            # -------------------------------
            request = LlmDialogHistorysSrv.Request()
            request.clear_after_get = False
            future = self.get_history_client.call_async(request)
            while rclpy.ok() and not future.done():
                time.sleep(0.05)

            srv_response = future.result()
            if srv_response is None:
                raise RuntimeError("Failed to get dialog history")
            history_msgs = srv_response.history  # list of llm_dialog_history_msg

            # 构造 prompt（user/assistant 格式）
            prompt = []
            prompt.append(
                {
                    "role":"system",
                    "content":
"当前地点是杭州，有几个血透中心的病人，现在在血透中心做透析，你是巡诊的机器人，名字叫小医。你的大脑跟病人的测量数据是实时联系在一起的，下面几个病人开始对话。对话的时候不要长篇大论，像真人一样，使用对话句子，也就是我们一句来一句去这样。语气需要轻柔舒缓。所有使用单位名称转化为中文。尽可能不出现英文。你只需要监测，禁止输出执行实际动作,例如倒水。不要增加小动作。不要输出表情符号。在需要时，给出数据偏高或偏低，不需要具体数据。不要给出实际的执行动作，而是不用通过类似读取的方式。讲话必须亲切，温柔。科普类的内容可以丰富一些。叫护士或医生时回答好的，我马上帮您联系护士或医生。您先安心休息一会儿。禁止在回答中出现着急两个字。问题不严重时，你来提出解决办法，不要经常输出寻找医生。询问天气/温度等关键词，仅输出杭州本地核心信息."
                }
            )
            
            prompt.append(
                {
                    "role": "assistant", 
                    "content": "好的，我记住了,我的名字是小医，在杭州市，我会采用真人对话风格那样进行对话，并使用轻柔温和的语气，我不会输出执行动作，不会输出表情符合，不会给出具体的数值，问题不严重时我会自己提出解决办法不会寻求医生的帮助，不会使用'着急'等急迫性质的词语。"
                }
            )
            for item in history_msgs:
                prompt.append({"role": "user", "content": item.question})
                prompt.append({"role": "assistant", "content": item.response})

            # 加上当前问题
            prompt.append({"role": "user", "content": question})

            # -------------------------------
            # 调用 LLM 流式生成
            # -------------------------------
            completion = self.client.chat.completions.create(
                model="qwen-plus-latest",
                messages=prompt,
                stream=True,
                top_p=0.8,
                temperature=0.7,
                extra_body={
                    "thinking_budget": 4055,
                    "enable_search": True
                }
            )

            assistant_reply = ""
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        assistant_reply += delta.content
                        self.pub.publish(String(data=delta.content))
                        # self.ui_pub.publish(String(data=delta.content))
                        print(delta.content)

            # 流式输出结束标记
            self.pub.publish(String(data="[DONE]"))
            # self.ui_pub.publish(String("[DONE]"))

            # -------------------------------
            # 保存当前对话到 dialog history
            # -------------------------------
            dialog_msg = LlmDialogHistoryMsg()
            dialog_msg.question = question
            dialog_msg.response = assistant_reply
            self.dialog_pub.publish(dialog_msg)
            self.get_logger().info(f"Dialog saved to history. Q: {question} | A: {assistant_reply}")

        except Exception as e:
            self.get_logger().error(f"Error querying networked LLM: {str(e)}")
            self.pub.publish(String(data="[Error] Failed to get response from LLM."))


def main(args=None):
    rclpy.init(args=args)
    node = NetworkedLLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
