import sys
import queue
import threading
import time
from pathlib import Path

import numpy as np

import sounddevice as sd
import sherpa_onnx

"""
我也忘了这个我呢件到底是参考哪个写出来的
github 上是有文件写这个的：
https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/streaming-paraformer-asr-microphone.py


"""

class RealTimeParaformerASR:

    def __init__(
        self,
        silero_vad_model: str,
        paraformer_model: str,
        tokens: str,
        num_threads: int = 2,
        hr_lexicon: str = "",
        hr_rule_fsts: str = "",
        sample_rate: int = 16000,
    ):

        self._assert_file_exists(silero_vad_model)
        self._assert_file_exists(paraformer_model)
        self._assert_file_exists(tokens)

        self.sample_rate = sample_rate
        self.killed = False
        self.samples_queue = queue.Queue()

        print("Creating recognizer...")

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=paraformer_model,
            tokens=tokens,
            num_threads=num_threads,
            debug=False,
            hr_rule_fsts=hr_rule_fsts,
            hr_lexicon=hr_lexicon,
        )

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = silero_vad_model
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_silence_duration = 0.1
        config.silero_vad.min_speech_duration = 0.25
        config.silero_vad.max_speech_duration = 8
        config.sample_rate = sample_rate

        self.window_size = config.silero_vad.window_size

        self.vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=100
        )

        self.display = sherpa_onnx.Display()

        self.recording_thread = None

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

    def _start_recording(self):

        samples_per_read = int(0.1 * self.sample_rate)

        with sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=self.sample_rate,
        ) as s:

            while not self.killed:

                samples, _ = s.read(samples_per_read)

                samples = samples.reshape(-1)
                samples = np.copy(samples)

                self.samples_queue.put(samples)

    def start(self):

        print("Started! Please speak")

        buffer = []
        offset = 0

        started = False
        started_time = None

        self.recording_thread = threading.Thread(target=self._start_recording)
        self.recording_thread.start()

        while not self.killed:

            samples = self.samples_queue.get()

            buffer = np.concatenate([buffer, samples])

            while offset + self.window_size < len(buffer):

                self.vad.accept_waveform(
                    buffer[offset : offset + self.window_size]
                )

                if not started and self.vad.is_speech_detected():
                    started = True
                    started_time = time.time()

                offset += self.window_size

            if not started:
                if len(buffer) > 10 * self.window_size:
                    offset -= len(buffer) - 10 * self.window_size
                    buffer = buffer[-10 * self.window_size :]

            if started and time.time() - started_time > 0.2:

                stream = self.recognizer.create_stream()
                stream.accept_waveform(self.sample_rate, buffer)

                self.recognizer.decode_stream(stream)

                text = stream.result.text.strip()

                if text:
                    self.display.update_text(text)
                    self.display.display()

                started_time = time.time()

            while not self.vad.empty():

                stream = self.recognizer.create_stream()

                stream.accept_waveform(
                    self.sample_rate, self.vad.front.samples
                )

                self.vad.pop()

                self.recognizer.decode_stream(stream)

                text = stream.result.text.strip()

                self.display.update_text(text)

                buffer = []
                offset = 0
                started = False
                started_time = None

                self.display.finalize_current_sentence()
                self.display.display()

    def stop(self):

        self.killed = True

        if self.recording_thread:
            self.recording_thread.join()

    def recognize_once(self):

        print("Please speak...")

        buffer = []
        offset = 0

        started = False
        started_time = None

        samples_per_read = int(0.1 * self.sample_rate)

        with sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=self.sample_rate,
        ) as s:

            while True:

                samples, _ = s.read(samples_per_read)
                samples = samples.reshape(-1)

                buffer = np.concatenate([buffer, samples])

                while offset + self.window_size < len(buffer):

                    self.vad.accept_waveform(
                        buffer[offset : offset + self.window_size]
                    )

                    if not started and self.vad.is_speech_detected():
                        started = True
                        started_time = time.time()

                    offset += self.window_size

                if started:

                    # 如果VAD检测到完整语音
                    while not self.vad.empty():

                        stream = self.recognizer.create_stream()

                        stream.accept_waveform(
                            self.sample_rate,
                            self.vad.front.samples,
                        )

                        self.vad.pop()

                        self.recognizer.decode_stream(stream)

                        text = stream.result.text.strip()

                        return text
                    

if __name__ == "__main__":
    
    base_path = "/opt/sherpa-onnx/asr_paraformer/"
    model_path = base_path + "sherpa-onnx-paraformer-zh-int8-2025-10-07/"

    asr = RealTimeParaformerASR(
        silero_vad_model=base_path + "silero_vad.onnx",
        paraformer_model=model_path + "model.int8.onnx",
        tokens=model_path + "tokens.txt",
    )

    # asr.list_devices()

    # try:
    #     asr.start()
    # except KeyboardInterrupt:
    #     asr.stop()
    #     print("Exiting")

    text = asr.recognize_once()

    print("识别结果:", text)
