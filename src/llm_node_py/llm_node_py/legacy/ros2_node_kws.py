
# 进行 keywords 检测

"""
测试命令:
1. 发送 start_kws 信号:
ros2 topic pub -1 /start_kws std_msgs/msg/Bool "{data: true}"

2. 监听 kws_detected 信号:
ros2 topic echo /start_kws
"""



import sys
import os
import subprocess
from pathlib import Path

import rclpy
import sounddevice as sd
import sherpa_onnx
from rclpy.node import Node
from std_msgs.msg import Bool

# 把 ros2 生成的 srv 文件地址加上
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../install/llm_node_comm/lib/python3.12/site-packages"
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from llm_node_comm.srv import TtsOneshot  # type: ignore


AUDIO_ASSETS_DIR = (
    Path(__file__).resolve().parents[2] / "llm_node_comm" / "audio_assets"
)


def play_mp3_non_blocking(filename: str) -> None:
    subprocess.Popen(
        [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            str(AUDIO_ASSETS_DIR / filename),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def play_mp3_blocking(filename: str) -> None:
    subprocess.run(
        [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            str(AUDIO_ASSETS_DIR / filename),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class RealTimeKeywordSpotter:

    def __init__(
        self,
        tokens,
        encoder,
        decoder,
        joiner,
        keywords_file,
        provider="cpu",
        num_threads=1,
        max_active_paths=4,
        keywords_score=1.0,
        keywords_threshold=0.25,
        num_trailing_blanks=1,
        sample_rate=16000,
    ):

        self._assert_file_exists(tokens)
        self._assert_file_exists(encoder)
        self._assert_file_exists(decoder)
        self._assert_file_exists(joiner)

        if not Path(keywords_file).is_file():
            raise ValueError(f"{keywords_file} does not exist")

        self.sample_rate = sample_rate
        self.samples_per_read = int(0.1 * sample_rate)

        self.keyword_spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=num_threads,
            max_active_paths=max_active_paths,
            keywords_file=keywords_file,
            keywords_score=keywords_score,
            keywords_threshold=keywords_threshold,
            num_trailing_blanks=num_trailing_blanks,
            provider=provider,
        )

        self.stream = self.keyword_spotter.create_stream()

    def _assert_file_exists(self, filename):
        if not Path(filename).is_file():
            raise FileNotFoundError(f"{filename} does not exist")

    def list_devices(self):
        devices = sd.query_devices()
        if len(devices) == 0:
            print("No microphone devices found")
            sys.exit(0)

        print(devices)
        default_input_device_idx = sd.default.device[0]
        print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    def detect(self):

        print("Started! Please speak")

        idx = 0

        with sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=self.sample_rate,
        ) as s:
            samples, _ = s.read(self.samples_per_read)
            samples = samples.reshape(-1)

            self.stream.accept_waveform(self.sample_rate, samples)

            while self.keyword_spotter.is_ready(self.stream):

                self.keyword_spotter.decode_stream(self.stream)

                result = self.keyword_spotter.get_result(self.stream)

                if result:
                    # print(f"{idx}: {result}")
                    idx += 1

                    self.keyword_spotter.reset_stream(self.stream)

                    return True
        
        # end of 'with sd'

        print("detect keywords.")
        return False



class Ros2KeywordSpotterNode(Node):

    def __init__(self):
        super().__init__("ros2_node_kws")

        base_kws_path = "/opt/sherpa-onnx/kws-zipformer-wenetspeech/"

        self.kws = RealTimeKeywordSpotter(
            num_threads=4,
            tokens=base_kws_path + "tokens.txt",
            encoder=base_kws_path + "encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            decoder=base_kws_path + "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            joiner=base_kws_path + "joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
            keywords_file=base_kws_path + "keywords.txt",
        )

        self.kws_enabled = False
        self.kws_pub = self.create_publisher(
            Bool, "/kws_detected", 10
        )
        self.kws_start_sub = self.create_subscription(
            Bool,
            "/start_kws",
            self.start_kws_callback,
            10,
        )
        self.tts_oneshot_client = self.create_client(TtsOneshot, "/tts_one_shot")

        self.get_logger().info("ros2 node kws started")

    def list_devices(self):
        self.kws.list_devices()

    def start_kws_callback(self, msg: Bool):
        if msg.data:
            self.kws_enabled = True
            self.get_logger().info("receive /start_kws, enable kws once")

    def call_oneshot_tts(self, tts_text: str, block: bool):
        req = TtsOneshot.Request()
        req.tts_text = tts_text
        req.block = block

        while not self.tts_oneshot_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("/tts_one_shot service not available, wait again...")

        future = self.tts_oneshot_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def run_once_when_enabled(self):
        if not self.kws_enabled:
            return

        self.kws_enabled = False
        self.get_logger().info("Started! Please speak")

        detected = self.kws.detect()
        if detected:

            # 这里添加 tts once, 发送一个阻塞的语音表示自己听到了
            # self.call_oneshot_tts("你好！我在呢！", True)
            play_mp3_blocking("hello_i_am_here.mp3")

            # 向外面发送一个消息，表示已经完成 kws 检测了
            msg = Bool()
            msg.data = True
            self.kws_pub.publish(msg)

            self.get_logger().info("kws detected, published")


if __name__ == "__main__":
    rclpy.init()
    node = Ros2KeywordSpotterNode()
    node.list_devices()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.run_once_when_enabled()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
    finally:
        node.destroy_node()
        rclpy.shutdown()
