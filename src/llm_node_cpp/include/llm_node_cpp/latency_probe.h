#ifndef LLM_NODE_CPP_LATENCY_PROBE_H
#define LLM_NODE_CPP_LATENCY_PROBE_H

// 轻量级链路耗时埋点工具。
//
// 默认完全不生效（零行为变化）；只有环境变量 LATENCY_PROBE=1 时才会：
//   1) 往 stdout 打印 [LAT] {json} 行；
//   2) 通过注册的 publisher 把同一份 JSON 发布到 /latency_event（std_msgs/String）。
//
// JSON 格式：
//   {"node":"asr","stage":"endpoint","t_ns":<epoch 纳秒>,"detail":{"k":v,...}}
// detail 可省略。时间戳统一用 system_clock 的 epoch 纳秒，跨进程可对齐。

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

namespace latency_probe {

inline bool enabled() {
    static const bool value = []() {
        const char * env = std::getenv("LATENCY_PROBE");
        return env != nullptr && std::string(env) == "1";
    }();
    return value;
}

inline long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// 由 main 注册：把 JSON 字符串发布到 /latency_event。
inline std::function<void(const std::string &)> & publisher() {
    static std::function<void(const std::string &)> pub;
    return pub;
}

inline std::string json_escape(const std::string & in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

inline std::string detail_int(const char * key, long long v) {
    char b[128];
    std::snprintf(b, sizeof(b), "\"%s\":%lld", key, v);
    return b;
}

inline std::string detail_double(const char * key, double v) {
    char b[128];
    std::snprintf(b, sizeof(b), "\"%s\":%.3f", key, v);
    return b;
}

inline std::string detail_str(const char * key, const std::string & v) {
    return std::string("\"") + key + "\":\"" + json_escape(v) + "\"";
}

inline void emit(const std::string & node, const std::string & stage,
                 const std::string & detail = "") {
    if (!enabled()) {
        return;
    }
    char buf[2048];
    const long long t = now_ns();
    if (detail.empty()) {
        std::snprintf(buf, sizeof(buf),
                      "{\"node\":\"%s\",\"stage\":\"%s\",\"t_ns\":%lld}",
                      node.c_str(), stage.c_str(), t);
    } else {
        std::snprintf(buf, sizeof(buf),
                      "{\"node\":\"%s\",\"stage\":\"%s\",\"t_ns\":%lld,\"detail\":{%s}}",
                      node.c_str(), stage.c_str(), t, detail.c_str());
    }
    std::printf("[LAT] %s\n", buf);
    std::fflush(stdout);
    if (publisher()) {
        publisher()(buf);
    }
}

}  // namespace latency_probe

#endif  // LLM_NODE_CPP_LATENCY_PROBE_H
