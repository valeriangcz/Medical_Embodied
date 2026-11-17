#ifndef ASR_STREAM_H
#define ASR_STREAM_H

#include <string>
#include <chrono>
#include <portaudio.h>
#include <functional>

#include <speex/speex_resampler.h>

#include "keywords.h"

#include "asr_punctuation.h"
#include "sherpa-onnx/c-api/c-api.h"

class AsrStream {
public:
    enum asrContinueEnum{
        asrContinue,
        asrPause
    };

    AsrStream(const std::string& asrBasePath,
              const std::string& provider = "cpu");

    ~AsrStream();

    void start();
    void stop();
    void pause() {
        is_paused_ = true;
    }

    void resume() {
        is_paused_ = false;
    }

    // 回调函数的输入参数是识别出的字符串，
    // 回调函数的返回值：如果返回asrStop，那么停止识别，返回其他无效果
    void setAsrCallback(std::function<enum asrContinueEnum(std::string&)> func) {
        asr_callback = func;
    }

    void enablePunctuation(void) {
        use_punctuation_ = true;
    }
    
    void disablePunctuation(void) {
        use_punctuation_ = false;
    }

    void setReadInStreamCallback(std::function<int(const float* audioData, int numSamples)> func) {
        readInStreamCallback = func;
    }

    void setInStreamCallbackProcessAudio(void) {
        readInStreamCallback = std::bind(&AsrStream::ProcessAudio, this, std::placeholders::_1, std::placeholders::_2);
    }

    double sample_rate(void) {
        return sample_rate_;
    }

    void notifySkipNextAudioRead(void) {
        skipNextAudioReadFlag = true;
    }

    

private:
    static int PortAudioCallback(const void* inputBuffer, void* outputBuffer,
                                 unsigned long framesPerBuffer,
                                 const PaStreamCallbackTimeInfo* timeInfo,
                                 PaStreamCallbackFlags statusFlags,
                                 void* userData);

    static size_t ReadFile(const char* filename, const char** buffer_out);

    void selectInputDevice(void);

    int ProcessAudio(const float* audioData, int numSamples);

    const SherpaOnnxOnlineRecognizer* recognizer_;
    const SherpaOnnxOnlineStream* stream_;
    const SherpaOnnxDisplay* display_;

    AsrPunctuation * asr_punc = nullptr;

    std::function<enum asrContinueEnum(std::string&)> asr_callback;

    std::function<int(const float* audioData, int numSamples)> readInStreamCallback;

    PaStream* audio_stream_;

    bool is_paused_;
    bool use_punctuation_ = true;

    SpeexResamplerState *resampler;
    int resample_in_rate;
    int resample_out_rate;

    double sample_rate_; // 硬件设备的硬件采样率

    int dev_max_channels_; // 硬件设备的通道数

    bool skipNextAudioReadFlag = false;

    // 语义端点：识别文本稳定 + 句末语气词时，比库默认 rule2 更早判停
    std::string last_partial_text_;
    std::chrono::steady_clock::time_point last_partial_change_{};
    bool semantic_endpoint_enabled_ = false;
    float semantic_silence_s_ = 0.55f;

    bool is_running_;

    // SpeexPreprocessState *st = nullptr;

    // bool useSoftwareResampling;

};

#endif // ASR_STREAM_H
