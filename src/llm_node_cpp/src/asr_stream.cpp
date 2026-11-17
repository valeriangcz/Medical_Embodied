#include "llm_node_cpp/asr_stream.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

#include <vector>

#include <chrono>             // 用于 std::chrono
#include <ctime>              // 用于 std::time_t 和 std::localtime
#include <iomanip>            // 用于 std::put_time 和 std::setw

#include <string_view>

#include <speex/speex_preprocess.h>

// 里面是一些负责打印的函数
#include "llm_node_cpp/conf.h"
#include "llm_node_cpp/latency_probe.h"


#define ASR_STREAM_DEBUG_ENABLE       0
// 10 → 4（2026-11 资源优化）：流式 zipformer 4 线程实时率仍有大余量，
// 把 CPU 让给导航等算法。流式解码实测单题 <0.6s（7.8s 音频）。
#define ASR_STREAM_NUM_THREADS        4


#define ASR_ENDPOINT_RULE1_DEFAULT  2.4
// rule2 维持 1.2s（2026-11 实测结论）：仿真显示降到 0.8s 时，句中 0.4~0.7s 的
// 自然停顿（解码滞后使有效尾静音长于声学静音）会提前误触发端点、截断问句；
// 老人语速停顿更长，风险更高。语气词结尾的提速由语义端点承担
//（ASR_SEMANTIC_RULE=0.55s 文本稳定 + ends_with_sentence_final）。
// 如需调整，运行时用 ASR_ENDPOINT_RULE2 环境变量覆盖，勿直接降默认值。
#define ASR_ENDPOINT_RULE2_DEFAULT  1.2
#define ASR_ENDPOINT_RULE3_DEFAULT  20

// 端点静音阈值支持环境变量覆盖（rule2 决定“说完话→判端点”的固定延迟）：
//   ASR_ENDPOINT_RULE2=0.8  可把固定延迟从 ~1.0-1.3s 降到 ~0.7-0.9s
namespace {
    // 句末语气词（UTF-8）——出现这些字且识别文本稳定，通常表示这句话说完了
    bool ends_with_sentence_final(const std::string &text) {
        static const char *kFinals[] = {
            "\xE4\xBA\x86",  // 了
            "\xE5\x90\x97",  // 吗
            "\xE5\x91\xA2",  // 呢
            "\xE5\x90\xA7",  // 吧
            "\xE5\x95\x8A",  // 啊
            "\xE5\x91\x80",  // 呀
            "\xE5\x98\x9B",  // 嘛
            "\xE4\xB9\x88",  // 么
            "\xE7\x9A\x84",  // 的
            "\xE5\x93\xA6",  // 哦
            "\xE5\x97\xAF",  // 嗯
        };
        if (text.size() < 3) return false;
        for (const char *f : kFinals) {
            if (text.compare(text.size() - 3, 3, f) == 0) return true;
        }
        return false;
    }

    float env_float(const char *name, float fallback) {
        const char *v = std::getenv(name);
        if (v == nullptr || *v == '\0') return fallback;
        try { return std::stof(v); } catch (...) { return fallback; }
    }
}

// 采购的一体式的麦克风
// #define MICROPHONE_DEVICE_NAME "UACDemoV1.0: USB Audio"

// 台子麦克风
// #define MICROPHONE_DEVICE_NAME "USB Audio Device"

// 圆形的四个阵列的麦克风
// #define MICROPHONE_DEVICE_NAME "4-mic Microphone: USB Audio"

// #define MICROPHONE_DEVICE_NAME "spdif"

#define MICROPHONE_DEVICE_NAME "default"

namespace {
    void print_time(void) {
        // auto now = std::chrono::system_clock::now();
        
        // // 转换为 time_t（秒级）
        // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        
        // // 计算毫秒部分
        // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        
        // // 使用 printf 输出格式化时间（带毫秒）
        // std::tm* now_tm = std::localtime(&now_c);  // 转换为 tm 结构体
        // sys_log("time: %04d-%02d-%02d %02d:%02d:%02d.%03d\n",
        //        now_tm->tm_year + 1900,  // 年份要加上 1900
        //        now_tm->tm_mon + 1,      // 月份从 0 开始，所以加 1
        //        now_tm->tm_mday,
        //        now_tm->tm_hour,
        //        now_tm->tm_min,
        //        now_tm->tm_sec,
        //        ms.count());
    }

    // 标点模型路径，允许通过环境变量覆盖（默认与生产部署路径一致）
    std::string punct_model_path(void) {
        const char * env = std::getenv("PUNCT_MODEL_PATH");
        if (env != nullptr && *env != '\0') {
            return std::string(env);
        }
        return std::string("/opt/sherpa-onnx/sherpa-onnx-punct-model/model.onnx");
    }
};


void AsrStream::selectInputDevice(void)
{
    // 获取输入设备数量
    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        std::cerr << "Pa_GetDeviceCount returned error: " << Pa_GetErrorText(numDevices) << std::endl;
        return;
    }

    int selectedDevice = -1;  // 先初始化一个无用的默认值

    // 打印所有输入设备
    for (int i = 0; i < numDevices; i++) {

        const PaDeviceInfo *deviceInfo = Pa_GetDeviceInfo(i);

        // 用于筛选支持音频输入的设备（因为输出设备的 maxInputChannels 为 0）
        if (deviceInfo->maxInputChannels > 0) {

            sys_log("dev[%d]: %s -> (Max Input Channels:  %d)", i, 
                    deviceInfo->name, deviceInfo->maxInputChannels);

            // if (selectedDevice == -1) {
            //     selectedDevice = i; // 选择第一个可用的输入设备
            // }

            std::string_view str(deviceInfo->name);
            if (str.starts_with( MICROPHONE_DEVICE_NAME )) {
                selectedDevice = i;

                dev_max_channels_ = deviceInfo->maxInputChannels;
            }
        }
    }

    if (selectedDevice == -1) {
        err_log("No suitable input device found.");
        return;
    }

    sys_log("selected device: %d", selectedDevice);

    // channels count > 2, realese a warning
    if (dev_max_channels_ > 2) {
        warn_log("channels count > 2, channel count: %d", dev_max_channels_);
        dev_max_channels_ = 2;
    }

    // ！！这里强制是单通道打开：
    dev_max_channels_ = 1;

    sys_log("selected channel count: %d", dev_max_channels_);

    // 设置音频流参数
    PaStreamParameters inputParams;
    inputParams.device           = selectedDevice;
    inputParams.channelCount     = dev_max_channels_;
    inputParams.sampleFormat     = paFloat32;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(selectedDevice)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    // 获取硬件采样率
    sample_rate_ = Pa_GetDeviceInfo(selectedDevice)->defaultSampleRate;
    // 固定 帧buf长度是0.2秒
    int frames_per_buffer = static_cast<int>(sample_rate_ * 0.05);
    // int frames_per_buffer = static_cast<int>(sample_rate_ * 0.05);

    sys_log("select dev: %d, sample rate: %.2f, buf frames: %d", 
            selectedDevice, sample_rate_, frames_per_buffer);

    // 打开音频流
    auto err = Pa_OpenStream(
        &audio_stream_,   // 指向音频流指针的地址
        &inputParams,     // 输入设备参数
        nullptr,          // 输出设备参数 (这里为 `nullptr`，表示不使用输出)
        sample_rate_,     // 采样率
        frames_per_buffer,// 每个缓冲区的帧数 (frames per buffer)
        paClipOff,        // 音频流标志 (这里选择了 `paClipOff`)
        PortAudioCallback,// 回调函数
        this              // 传递给回调函数的用户数据
    );
    if (err != paNoError) {
        std::cerr << "PortAudio stream opening failed: " << Pa_GetErrorText(err) << std::endl;
        return;
    }

    // ===== 进行带通滤波器设计 ==================================================================
    // 设计带通滤波器 人声大概的范围是 80Hz ~ 4000Hz
    // unsigned int order = 64; // 滤波器阶数
    // float fc_low = 80.0f / (sample_rate_/2);  // 归一化截止频率(低)
    // float fc_high = 4000.0f / (sample_rate_/2); // 归一化截止频率(高)
    
    // // 创建带通滤波器
    // firfilt_crcf bp_filter = firfilt_crcf_create_kaiser(order, fc_low, fc_high, 60.0f, 0.0f);


    // ==== 频谱减法降噪设计 ======================================================================

    // SpeexPreprocessState *st = speex_preprocess_state_init(frames_per_buffer, sample_rate_);
    // int denoise = 1; // 开启噪声抑制
    // speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_DENOISE, &denoise);

}


AsrStream::AsrStream(const std::string& asrBasePath,
                     const std::string& provider)
    : recognizer_(nullptr), 
      asr_punc(new AsrPunctuation(punct_model_path())),
      stream_(nullptr),
      display_(nullptr),
      audio_stream_(nullptr),
      is_running_(false),
      is_paused_(false) {


    setInStreamCallbackProcessAudio();

    // 语义端点开关（默认开）：识别文本稳定 0.55s 且以句末语气词结尾时提前判停
    {
        const char *denv = std::getenv("ASR_SEMANTIC_ENDPOINT");
        semantic_endpoint_enabled_ = !(denv != nullptr && std::string(denv) == "0");
        semantic_silence_s_ = env_float("ASR_SEMANTIC_RULE", 0.55f);
        sys_log("semantic endpoint: %s (silence=%.2fs)",
                semantic_endpoint_enabled_ ? "on" : "off", semantic_silence_s_);
    }

    /* init zipformer start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    // Initialize Zipformer config
    // 初始化 zipformer 的 encoder, decoder, joiner


    // Paraformer config
    // SherpaOnnxOnlineParaformerModelConfig paraformer_config;

    // std::string encoderPath = asrBasePath + "/encoder.onnx";
    // std::string decoderPath = asrBasePath + "/decoder.onnx";

    // memset(&paraformer_config, 0, sizeof(paraformer_config));
    // paraformer_config.encoder = encoderPath.c_str();
    // paraformer_config.decoder = decoderPath.c_str();

    std::string encoder_filename = asrBasePath + "/encoder-epoch-99-avg-1.onnx";
    std::string decoder_filename = asrBasePath + "/decoder-epoch-99-avg-1.onnx";
    std::string joiner_filename = asrBasePath + "/joiner-epoch-99-avg-1.onnx";

    SherpaOnnxOnlineTransducerModelConfig zipformer_config;
    memset(&zipformer_config, 0, sizeof(zipformer_config));
    zipformer_config.encoder = encoder_filename.c_str();
    zipformer_config.decoder = decoder_filename.c_str();
    zipformer_config.joiner  = joiner_filename.c_str();

/////////////////////////////////////////////////

    // SherpaOnnxOnlineTransducerModelConfig zipformer_config_;
    // memset(&zipformer_config_, 0, sizeof(zipformer_config_));
    // zipformer_config_.encoder = encoder_filename.c_str();
    // zipformer_config_.decoder = decoder_filename.c_str();
    // zipformer_config_.joiner = joiner_filename.c_str();

    std::string bep_vocab_path = asrBasePath + "/bpe.vocab";

    // Initialize Online model config
    SherpaOnnxOnlineModelConfig online_model_config_;
    memset(&online_model_config_, 0, sizeof(online_model_config_));
    online_model_config_.debug         = ASR_STREAM_DEBUG_ENABLE;
    online_model_config_.num_threads   = ASR_STREAM_NUM_THREADS;
    online_model_config_.provider      = provider.c_str();  // 是 cpu 还是 gpu
    // online_model_config_.paraformer    = paraformer_config;
    online_model_config_.transducer    = zipformer_config;

    // 开了 bpe 之后好像 hotwords 就不管用了
    // online_model_config_.bpe_vocab = bep_vocab_path.c_str();
    // online_model_config_.modeling_unit = "bpe";
    

    // Read tokens and hotwords to buffers
    // 将 tokens 文件 和 hotwords 文件全部读入然后全部设置好
    const char* tokens_buf;
    std::string tokens_filename = asrBasePath + "/tokens.txt";
    size_t token_buf_size = ReadFile(tokens_filename.c_str(), &tokens_buf);
    if (token_buf_size < 1) {
        std::cerr << "Please check your tokens.txt!" << std::endl;
        free((void*)tokens_buf);
        exit(1);
    }

    online_model_config_.tokens_buf = tokens_buf;
    online_model_config_.tokens_buf_size = token_buf_size;



    const char* hotwords_buf;
    std::string hotwords_filename = asrBasePath + "/hotwords_cn.txt";
    size_t hotwords_buf_size = ReadFile(hotwords_filename.c_str(), &hotwords_buf);
    if (hotwords_buf_size < 1) {
        std::cerr << "Please check your hotwords.txt!" << std::endl;
        free((void*)hotwords_buf);
        exit(1);
    }
    // Initialize Recognizer config
    SherpaOnnxOnlineRecognizerConfig recognizer_config_;
    memset(&recognizer_config_, 0, sizeof(recognizer_config_));
    recognizer_config_.decoding_method   = "modified_beam_search";
    recognizer_config_.model_config      = online_model_config_;

    // 和 hotwords 相关的
    sys_log("hotwords file path: %s", hotwords_filename.c_str());
    sys_log("hotwords file content: %s", hotwords_buf);
    recognizer_config_.hotwords_buf      = hotwords_buf;
    recognizer_config_.hotwords_buf_size = hotwords_buf_size;

    // 和 endpoint 检测相关的
    recognizer_config_.enable_endpoint   = 1;
    const float rule1 = env_float("ASR_ENDPOINT_RULE1", ASR_ENDPOINT_RULE1_DEFAULT);
    const float rule2 = env_float("ASR_ENDPOINT_RULE2", ASR_ENDPOINT_RULE2_DEFAULT);
    const float rule3 = env_float("ASR_ENDPOINT_RULE3", ASR_ENDPOINT_RULE3_DEFAULT);
    recognizer_config_.rule1_min_trailing_silence = rule1;
    recognizer_config_.rule2_min_trailing_silence = rule2;
    recognizer_config_.rule3_min_utterance_length = rule3;
    sys_log("endpoint thresholds: rule1=%.2fs rule2=%.2fs rule3=%.0fs", rule1, rule2, rule3);

    // recognizer_config_.rule1_min_trailing_silence = ;


    recognizer_ = SherpaOnnxCreateOnlineRecognizer(&recognizer_config_);
    if (recognizer_ == nullptr) {
        free((void*)tokens_buf);
        // free((void*)hotwords_buf);
        
        std::cerr << "Failed to create recognizer!" << std::endl;
        exit(1);
    }

    stream_ = SherpaOnnxCreateOnlineStream(recognizer_);

    free((void*)tokens_buf);
    free((void*)hotwords_buf);
    /* init zipformer end >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */

    /* open portAudio stream */
    // 1 表示输入通道数， 0 表示输出通道数量
    // 16000 是采样率， 3200 是每次回调处理的样本量
    // PortAudioCallback 是绑定的回调函数， this 是传入回调函数的参数
    // 调用频率 = 采样率 / 样本数 = 5次/秒
    // PaError err = Pa_OpenDefaultStream(&audio_stream_, 2, 0, paFloat32, 
    //               16000, 3200, PortAudioCallback, this);
    // if (err != paNoError) {
    //     err_log("PortAudio stream opening failed: %s", Pa_GetErrorText(err));
    //     exit(1);
    // }

    selectInputDevice();

    pause();
    start();
}


AsrStream::~AsrStream() {
    sys_log("asr is been destoryed. start......");
    if (is_running_) {
        sys_log("stop asr stream read...");
        stop();
    }

    if (stream_) {
        sys_log("destory sherpa stream...");
        SherpaOnnxDestroyOnlineStream(stream_);
    }

    if (recognizer_) {
        sys_log("destory sherpa recognizer...");
        SherpaOnnxDestroyOnlineRecognizer(recognizer_);
    }

    if (asr_punc) {
        sys_log("delete asr punc...");
        delete asr_punc;
    }

    // if (st) {
    //     speex_preprocess_state_destroy(st);
    // }
    
    // if (resampler) {
    //     sys_log("destory resampler...");
    //     speex_resampler_destroy(resampler);
    // }
    sys_log("asr is been destoryed. end......");
}


// 读取 filename 中的全部内容到动态分配的缓冲区 buffer_out 中
// 返回文件的大小
size_t AsrStream::ReadFile(const char* filename, const char** buffer_out) {

    // 打印 filename 文件
    FILE* file = fopen(filename, "rb");
    if (file == nullptr) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 0;
    }

    /* 获取文件的大小 */
    fseek(file, 0, SEEK_END);  // 将文件指针移动到文件末尾。
    size_t size = ftell(file); // 获取当前文件指针的位置（即文件大小）。
    rewind(file);              // 将文件指针重新移动到文件开头

    /* 动态分配一块内存来存放文件内容 */
    *buffer_out = (const char*)malloc(size);
    if (*buffer_out == nullptr) {
        std::cerr << "Failed to allocate memory for file: " << filename << std::endl;
        fclose(file);
        return 0;
    }

    /* 读取全部的文件内容到缓冲区 */
    size_t read_bytes = fread((void*)*buffer_out, 1, size, file);
    if (read_bytes != size) {  /* 要保证读取全部的文件内容 */
        std::cerr << "Failed to read file: " << filename << std::endl;
        free((void*)*buffer_out);
        *buffer_out = nullptr;
        fclose(file);
        return 0;
    }

    fclose(file);
    return read_bytes;
}


void AsrStream::start() {
    is_running_ = true;

    /* 启动音频流的读取 */
    // if (!audio_stream_) {
        sys_log("asr stream start......");
        
        PaError err = Pa_StartStream(audio_stream_);
        if (err != paNoError) {
            err_log("PortAudio stream start failed: %s", Pa_GetErrorText(err));
        }

    // } else {
    //     warn_log("asr stream is running, can not start again.");
    // }
}

void AsrStream::stop() {
    
    is_running_ = false;

    pause();
    
    /* 停止音频流的读取 */
    if (audio_stream_) {
        PaError err = Pa_StopStream(audio_stream_);
        // PaError err = Pa_AbortStream(audio_stream_);
        if (err != paNoError) {
            warn_log("try stop stream faild, start abort stream...");

            PaError err = Pa_AbortStream(audio_stream_);
            if (err != paNoError) {
                err_log("PortAudio stream stop failed: %s", Pa_GetErrorText(err));
                return;
            }

        }
    } else {
        warn_log("asr stream has been already stopped.");
    }
}


int AsrStream::PortAudioCallback(const void* inputBuffer, void* outputBuffer,
                                unsigned long framesPerBuffer,
                                const PaStreamCallbackTimeInfo* timeInfo,
                                PaStreamCallbackFlags statusFlags,
                                void* userData) {

    // print_time();

    if (!userData) {
        return paContinue;
    }
    AsrStream* asr = static_cast<AsrStream*>(userData);

    // 处理是否跳过这次的音频读取
    if (asr->skipNextAudioReadFlag) {
        asr->skipNextAudioReadFlag = false;
        return paContinue;
    }

    // 处理音频读取是否被暂停
    if (asr->is_paused_) {
        if (!asr->is_running_) {
            sys_log("asr is stopping by callback");
            return paAbort;
        }
        return paContinue;
    }

    // sys_log("port audio callback.");

    // 将输入数据转换为 float 类型
    const float* audioData = static_cast<const float*>(inputBuffer);

    if (!asr->is_running_) return paComplete;  // 人为的退出点函数

    // 如果是双声道的话，提取右声道数据
    if (asr->dev_max_channels_ == 2) {
        std::vector<float> rightChannelData(framesPerBuffer);
        for (unsigned long i = 0; i < framesPerBuffer; i++) {
            rightChannelData[i] = audioData[2 * i + 1]; // 右声道数据
        }
    
        // 处理声道数据
        if (asr->readInStreamCallback) {
            asr->readInStreamCallback(rightChannelData.data(), framesPerBuffer);
        }
    }
    else if (asr->dev_max_channels_ == 1) {

        if (asr->readInStreamCallback) {
            asr->readInStreamCallback(audioData, framesPerBuffer);
        }

    } else {
        err_log("invaild channels: %d", asr->dev_max_channels_);
        return paAbort;
    }

    if (!asr->is_running_) return paComplete;  // 人为的退出点函数

    return paContinue;
}


int AsrStream::ProcessAudio(const float* audioData, int numSamples) {

    // do not use software resample
    SherpaOnnxOnlineStreamAcceptWaveform(stream_, sample_rate_, audioData, numSamples);

    // 检查音频流是否准备好进行解码
    while (SherpaOnnxIsOnlineStreamReady(recognizer_, stream_)) {
        // 如果准备好，调用 SherpaOnnxDecodeOnlineStream 对音频流进行解码
        SherpaOnnxDecodeOnlineStream(recognizer_, stream_);
    }

    // 获取识别结果，result 是一个指向 SherpaOnnxOnlineRecognizerResult 结构体的指针，
    // 包含识别出的文本等信息
    const SherpaOnnxOnlineRecognizerResult* result = SherpaOnnxGetOnlineStreamResult(recognizer_, stream_);

    // ---- 语义端点：跟踪“识别文本最后一次变化”的时刻 ----
    const std::string partial_text =
        (result != nullptr && result->text != nullptr) ? std::string(result->text) : std::string();
    const auto now_tp = std::chrono::steady_clock::now();
    if (partial_text != last_partial_text_) {
        last_partial_text_ = partial_text;
        last_partial_change_ = now_tp;
    }
    const float stable_silence_s = last_partial_change_.time_since_epoch().count() == 0
        ? 0.0f
        : std::chrono::duration<float>(now_tp - last_partial_change_).count();

    bool lib_endpoint = SherpaOnnxOnlineStreamIsEndpoint(recognizer_, stream_);
    bool semantic_endpoint = false;
    if (!lib_endpoint && semantic_endpoint_enabled_ &&
        partial_text.size() >= 12 /* 约 4 个汉字，太短不提前切 */ &&
        stable_silence_s >= semantic_silence_s_ &&
        ends_with_sentence_final(partial_text)) {
        semantic_endpoint = true;
    }

    // 检查是否到达音频流终点：如，检测到静音或语音结束）。
    asrContinueEnum asrC = asrContinue;
    if (lib_endpoint || semantic_endpoint) {
        const char *reason = semantic_endpoint ? "semantic" : "lib";
        sys_log("endpoint detect (%s, silence=%.2fs) ......", reason, stable_silence_s);
        // 埋点：端点检测时刻 + 实际尾部静音（probe 用它还原“说话结束”时刻）
        latency_probe::emit(
            "asr", "endpoint",
            latency_probe::detail_str("reason", reason) + "," +
                latency_probe::detail_int("silence_ms", (long long)(stable_silence_s * 1000.0f)));

        /* 调用用户设置的回调函数 */
        if (result && strlen(result->text) && asr_callback) {
            std::string res(result->text);

            // 判断是否需要添加标点
            if (use_punctuation_ && asr_punc) {
                /* 添加标点符号 */
                std::string res_punc = asr_punc->AddPunctuation(res);
                asrC = asr_callback(res_punc);
            } else {
                /* 使用原始的数据 */
                asrC = asr_callback(res);
            }
        }

        // 如果到达终点，调用 SherpaOnnxOnlineStreamReset 重置音频流，以便处理下一段音频
        SherpaOnnxOnlineStreamReset(recognizer_, stream_);
        last_partial_text_.clear();
        last_partial_change_ = std::chrono::steady_clock::time_point{};
    }

    // 释放识别结果占用的资源
    SherpaOnnxDestroyOnlineRecognizerResult(result);

    if (asrC == asrPause) {
        sys_log("asr stream pause <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ");
        pause();
    }

    return paContinue;
}
