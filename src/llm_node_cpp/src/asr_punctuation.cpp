#include "llm_node_cpp/asr_punctuation.h"
#include <iostream>
#include <cstring>
#include <filesystem>

#include "llm_node_cpp/conf.h"

#define  ASR_PUNCTUATION_DEBUG_ENBAL    1
#define  ASR_PUNCTUATION_NUM_THREADS    6

AsrPunctuation::AsrPunctuation(const std::string& model_path) {

    // 模型文件缺失时降级为"不加标点"，不再让整个 ASR 节点启动失败。
    // （部署机上可能只装了识别模型，没有标点模型）
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        warn_log("punctuation model not found: %s, ASR 将输出不带标点的文本",
                 model_path.c_str());
        punct_ = nullptr;
        return;
    }

    SherpaOnnxOfflinePunctuationConfig config;
    std::memset(&config, 0, sizeof(config));

    config.model.ct_transformer = model_path.c_str();
    config.model.num_threads = ASR_PUNCTUATION_NUM_THREADS;
    config.model.debug = ASR_PUNCTUATION_DEBUG_ENBAL;
    config.model.provider = "cpu";

    punct_ = (SherpaOnnxOfflinePunctuation*)SherpaOnnxCreateOfflinePunctuation(&config);
    if (!punct_) {
        // 同样降级处理，避免因为标点模型损坏导致语音链路不可用
        warn_log("failed to create OfflinePunctuation with %s, 继续但不加标点",
                 model_path.c_str());
        punct_ = nullptr;
    }
}

AsrPunctuation::~AsrPunctuation() {
    if (punct_) {
        SherpaOnnxDestroyOfflinePunctuation(punct_);
    }
}

std::string AsrPunctuation::AddPunctuation(const std::string& text) {
    if (!punct_) {
        return text;
    }

    const char* text_with_punct = SherpaOfflinePunctuationAddPunct(punct_, text.c_str());
    if (!text_with_punct) {
        return text;
    }

    std::string result(text_with_punct);
    SherpaOfflinePunctuationFreeText(text_with_punct);
    return result;
}
