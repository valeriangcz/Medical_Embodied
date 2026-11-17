
#include "rclcpp/rclcpp.hpp"

#include <string>
#include <memory>
#include <future>
#include <signal.h>
#include <chrono>
#include <thread>
#include <filesystem>
#include <cstdlib>
#include <portaudio.h>

#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"
#include "llm_node_comm/msg/dialog_session_finished.hpp"
#include "llm_node_comm/srv/tts_oneshot.hpp"
#include "llm_node_comm/srv/end_session.hpp"
#include "llm_node_comm/srv/nurse_alert.hpp"

#include "llm_node_cpp/conf.h"
#include "llm_node_cpp/asr_stream.h"
#include "llm_node_cpp/latency_probe.h"




// PortAudioManager 用来管理 Pa_Initialize 和 Pa_Terminate
// PortAudioManager 需要在程序的生命周期内存在，因此需要使用 static 修饰
// e.g. static PortAudioManager pam;
// 利用 static 的生命周期特性来自动管理 portAudio 库的 init 和 deinit
class PortAudioManager{
public:
    PortAudioManager(){
        sys_log("PortAudioManager init start.");
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            err_log("PortAudio init failed: %s", Pa_GetErrorText(err));
        }
        sys_log("PortAudioManager init end.");
    }

    ~PortAudioManager() {
        Pa_Terminate();
    }
};



namespace hzc {

// ======== ASR / KWS / Audio ========
inline std::unique_ptr<AsrStream> asr;
inline std::unique_ptr<KeywordSpotter> kws;
inline std::unique_ptr<PortAudioManager> pam;

// ======== control flags ========
inline bool exitFlag = false;
inline constexpr int sleepPeriodMs = 300;

// ======== audio state ========
inline bool audio_buffer_empty = false;

// ============================================================
// signal handler
// ============================================================
inline void signalHandler(int signum) {
    exitFlag = true;
    sys_log("signal handler .... exit ......");

    if (asr) {
        sys_log("asr pause.");
        asr->pause();
        asr->stop();
    }

    if (kws) kws.reset();
    if (asr) asr.reset();
    if (pam) pam.reset();

    exit(signum);
}

// ============================================================
// ASR input callback
// ============================================================
inline int asr_inputStream_callback(const float* audioData, int numSamples) {
    if (hzc::kws && hzc::asr) {
        hzc::kws->DetectKeyword(
            static_cast<int>(hzc::asr->sample_rate()),
            audioData,
            numSamples
        );
    }
    return 0;
}

// ============================================================
// init
// ============================================================
inline std::string env_or(const char * name, const std::string & fallback) {
    const char * value = std::getenv(name);
    if (value != nullptr && *value != '\0') {
        return std::string(value);
    }
    return fallback;
}

inline void init(void) {
    signal(SIGINT, signalHandler);

    // init PortAudio
    pam = std::make_unique<PortAudioManager>();

    // 模型根目录和各模型子目录都允许用环境变量覆盖；
    // 不设置环境变量时与生产部署路径完全一致。
    const std::string model_root =
        env_or("SHERPA_MODEL_ROOT", "/opt/sherpa-onnx");
    const std::string asr_dir = env_or(
        "ASR_MODEL_DIR",
        model_root + "/sherpa-onnx-streaming-zipformer-bilingual-zh-en");
    const std::string kws_dir = env_or(
        "KWS_MODEL_DIR",
        model_root + "/kws-zipformer-wenetspeech");
    const std::string kws_keywords = env_or(
        "KWS_KEYWORDS_FILE",
        kws_dir + "/outputMykeywords.txt");

    sys_log("asr model dir: %s", asr_dir.c_str());
    sys_log("kws model dir: %s", kws_dir.c_str());
    sys_log("kws keywords file: %s", kws_keywords.c_str());

    // init ASR
    asr = std::make_unique<AsrStream>(asr_dir);

    // init keyword spotting
    kws = std::make_unique<KeywordSpotter>(kws_dir, kws_keywords);
}

}; // namespace hzc



namespace {
    std::shared_ptr<rclcpp::Node> node;

    // nurse_alert 与主 LLM 并行（乐观路径），默认开启；=0 回到原同步逻辑
    inline bool nurse_parallel = true;

    // “小医听到了”语音反馈：默认关闭（连续对话时每轮播报太吵）；ASR_HEARD_FEEDBACK=1 打开
    inline bool heard_feedback = false;

    // 唤醒问候语（hello_i_am_here.mp3）：默认开启；ASR_WAKE_GREETING=0 关闭
    inline bool wake_greeting = true;

    // 呼叫护士后是否保持对话（继续收音等下一句）：默认开，NURSE_KEEP_DIALOG=0 回到原行为（结束会话回 KWS）
    inline bool nurse_keep_dialog = true;

    // 记录当前麦克风的状态是 kws 检测 还是 asr 检测
    std::string mic_status = "kws";

    // 变成 关键词检测
    void change_to_kws(void) {
        hzc::asr->setReadInStreamCallback(hzc::asr_inputStream_callback);
        mic_status = "kws";
    }

    // 变成 语音识别
    void change_to_asr(void) {
        hzc::asr->setInStreamCallbackProcessAudio();
        mic_status = "asr";
    }

    std::string audio_asset_path(const std::string &filename) {
        const auto base =
            std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
        return (base / "llm_node_comm" / "audio_assets" / filename).string();
    }

    void play_mp3_non_blocking(const std::string &filename) {
        const auto path = audio_asset_path(filename);
        std::thread([path]() {
            const std::string cmd =
                "ffplay -nodisp -autoexit -loglevel quiet \"" + path + "\" >/dev/null 2>&1";
            std::system(cmd.c_str());
        }).detach();
    }

    void play_mp3_blocking(const std::string &filename) {
        const auto path = audio_asset_path(filename);
        const std::string cmd =
            "ffplay -nodisp -autoexit -loglevel quiet \"" + path + "\" >/dev/null 2>&1";
        std::system(cmd.c_str());
    }
}

namespace {

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr req_pub, req_ui_pub, video_pub;
    rclcpp::Publisher<llm_node_comm::msg::DialogSessionFinished>::SharedPtr dialog_session_finished_pub;
    rclcpp::Client<llm_node_comm::srv::TtsOneshot>::SharedPtr client;
    rclcpp::Client<llm_node_comm::srv::EndSession>::SharedPtr end_session_client;
    rclcpp::Client<llm_node_comm::srv::NurseAlert>::SharedPtr nurse_call_client;

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr kws_pub;  // 发布检测到 kws 的信号

    rclcpp::TimerBase::SharedPtr asr_timer;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_asr_sub;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr video_status_sub;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr llm_queue_empty_sub;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr latency_pub;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tts_cancel_pub;

    // 血透语境同音字归一：护士告警在网关纠错之前调用，必须在这里先修，
    // 否则“内漏怎么处理”会被当成内出血急症，误 call 护士并结束会话。
    static std::string normalize_asr_typos(std::string text) {
        static const std::vector<std::pair<std::string, std::string>> kMap = {
            {"内漏", "内瘘"}, {"内露", "内瘘"}, {"血林", "血磷"},
            {"火龙观", "火龙罐"}, {"美洲", "每周"},
        };
        for (const auto &kv : kMap) {
            size_t pos = 0;
            while ((pos = text.find(kv.first, pos)) != std::string::npos) {
                text.replace(pos, kv.first.size(), kv.second);
                pos += kv.second.size();
            }
        }
        return text;
    }

    // 急症关键词：命中则走「同步等 nurse_alert」的保守路径（保证先叫护士）
    inline bool is_emergency(const std::string &text) {
        static const std::vector<std::string> kEmergencyWords = {
            "胸痛", "胸口疼", "胸口痛", "胸闷", "呼吸困难", "喘不上", "憋气",
            "说不出话", "意识", "昏迷", "晕倒", "抽搐", "心慌", "心跳",
            "出血不止", "止不住", "震颤", "高热", "寒战", "剧烈头痛",
            "手脚无力", "肢体无力", "麻木", "眼前发黑", "大汗淋漓",
        };
        for (const auto &w : kEmergencyWords) {
            if (text.find(w) != std::string::npos) return true;
        }
        return false;
    }

    // 请求 TTS 节点立即停止播报并清空待播队列（用于护士判断晚到但需中断时）
    inline void cancel_tts(const std::string &reason) {
        if (!tts_cancel_pub) return;
        std_msgs::msg::String msg;
        msg.data = reason;
        tts_cancel_pub->publish(msg);
        sys_log("published /tts_cancel (%s)", reason.c_str());
        latency_probe::emit("asr", "tts_cancel",
                            latency_probe::detail_str("reason", reason));
    }


    void call_oneshot_tts(const std::string &tts_text, bool block) {
        // 构造请求
        auto request = std::make_shared<llm_node_comm::srv::TtsOneshot::Request>();
        request->tts_text = tts_text;
        request->block = block;
        // 这个 call 是阻塞的，会一直阻塞到 语音播放完成
        auto future = client->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
            err_log("call /tts_one_shot timeout.");
        }
    }

    // 接收 start asr 信号,开始进行语音识别
    void start_asr_callback(const std_msgs::msg::Bool::SharedPtr msg) {
        latency_probe::emit("asr", "start_asr");
        // 开启 asr timer 计时器
        asr_timer->reset();

        std::thread([=]() {
            (void)msg;

            // 播放完成后恢复 ASR
            sys_log("switch to asr.");
            change_to_asr();
            hzc::asr->resume();

        }).detach();
    }

    void dialog_session_finished(bool call_nurse, std::string reason) {
        // 手动停止
        asr_timer->cancel();

        // 调整成 kws 检测
        change_to_kws();
        // 恢复语音输入
        hzc::asr->resume();

        // 发送 dialog finished 的消息和原因
        llm_node_comm::msg::DialogSessionFinished msg;
        msg.call_nurse = call_nurse;
        msg.reason = reason;
        dialog_session_finished_pub->publish(msg);

        sys_log("send dialog finished msg (beacuse of 'call nurse').");
    }



    void asr_timerCallback()
    {
        RCLCPP_INFO(node->get_logger(), "Asr Timer triggered, Switch to KWS.");

        // 结束 session 对话，原因是 时间到了
        dialog_session_finished(
            false, std::string("timeout")
        );

        // 说句客套话表示已经结束对话了（阻塞）
        // call_oneshot_tts("没有其他事情的话，小医先走了，有问题记得叫小医。", true);
    }

    
    void OnKeywordDetected(const std::string &keyword) {
        latency_probe::emit("asr", "kws_detected",
                            latency_probe::detail_str("keyword", keyword));
        hzc::asr->pause();

        sys_log("Detected keyword: %s, pause asr audio stream.", keyword.c_str());

        std::thread([=]() {
            // 播放客套话（阻塞）
            // call_oneshot_tts("你好！我在呢！", true);
            if (wake_greeting) {
                play_mp3_blocking("hello_i_am_here.mp3");
            }

            // 发送一个 topic 信号，告诉行为树 kws 开启了
            std_msgs::msg::Bool msg;
            msg.data = true;   // 表示检测到 KWS
            kws_pub->publish(msg);

            sys_log("send kws signal to bt.");


            // 这里不能进行 resume, 要靠行为树来发送 action 来开启 asr
            // 播放完成后恢复 ASR
            // sys_log("switch to asr.");
            // change_to_asr();

            // 测试用，有的时候一次呼叫没有用
            // hzc::asr->resume();
        }).detach();
    }

    // 所有问题播放完成了之后，恢复到关键词检测
    void llm_question_finished_callback(const std_msgs::msg::String::SharedPtr msg_p) {
        (void)msg_p;
        
        // 判断当前是处于 kws 状态还是 asr 状态
        // 如果是 kws 状态的话,应该是 ui 在发问题,这里其实不应该接收
        if (mic_status == "asr") {
            hzc::asr->resume();  // 恢复语音输入
            asr_timer->reset();  // 开始计时，超过一定时间就恢复成 kws
        } else {
            sys_log("ignore question finished callback, because mic_status != asr.");
        }
    }


    // 向 question manager 和 ui 节点发送消息
    void send_question(std::string & question, bool call_nurse,
                       bool nurse_decided = true) {
        (void)call_nurse;
        (void)nurse_decided;

        // 网关已直接消费 /question_asr；旧版拼接的
        // "。提示:请用简短口语回答" 会被本地纠错模型当正文改写，
        // 也会污染 LLM 上下文，这里只发纯问题文本。
        std_msgs::msg::String msg;
        msg.data = question;
        req_pub->publish(msg);
        latency_probe::emit(
            "asr", "publish_question",
            latency_probe::detail_int("text_len", (long long)msg.data.size()));

        // 向 ui 发送问题
        msg.data = question;
        req_ui_pub->publish(msg);
        // 向ui节点发送结束符
        msg.data = "[DONE]";
        req_ui_pub->publish(msg);
    }


    // 同步调用 nurse_alert（定义在下方）；返回 true 表示已结束本轮对话
    bool handle_nurse_sync(const std::string &res);

    // 语音识别识别到了一些内容，发送给大模型
    AsrStream::asrContinueEnum asr_callback(std::string &res)
    {
        sys_log("%s", res.c_str());
        // 同音字归一必须在 nurse_alert 与网关之前：护士服务拿到的是原始识别文本，
        // “内漏/内露”会被当成内出血急症而误呼护士并结束会话。
        res = normalize_asr_typos(res);
        latency_probe::emit("asr", "asr_callback",
                            latency_probe::detail_int("text_len", (long long)res.size()));

        if (res.size() <= 7) { // 一个中文字符占 3 个字节
            std::cout << "res 太短，直接返回" << std::endl;
            return AsrStream::asrContinue;
        }

        hzc::asr->pause(); // 暂停音频数据的输入

        // 停止 timer 计时，stop 方法同时也会重置 timer 的计时
        asr_timer->cancel();

        // 判断是否要播放视频
        if (res.find("播放") != std::string::npos && res.find("视频") != std::string::npos) {
            // 提取“播放”和“视频”之间的内容
            size_t startPos = res.find("播放");
            size_t endPos = res.find("视频");
            std::string content;
            if (endPos > startPos) {
                content = res.substr(
                    startPos + std::string("播放").length(), 
                    endPos - startPos - std::string("播放").length()
                );
            }

            std::cout << "提取内容: " << content << std::endl;

            // 重新恢复语音输入
            hzc::asr->resume();
            // 重新变成 kws 检测
            hzc::asr->setReadInStreamCallback(hzc::asr_inputStream_callback); 
            
            // 发送给 ui 节点，让它播放视频
            std_msgs::msg::String msg; 
            msg.data = content;
            video_pub->publish(msg);
            return AsrStream::asrContinue;
        }



        // deprecated: 原先是想要获取全部的对话历史综合判断护士呼叫的
        // 但是历史记录的影响会影响到后面的每一个新的提问,所以这个删掉历史

        // 调用 service 获取 dialog history 
        // std::string all_questions;
        // llm_node::llm_dialog_historys_srv h_srv;
        // h_srv.request.clear_after_get = false;
        // if (dialog_history_client.call(h_srv)) {
        //     // 遍历返回的 history 列表,把过往的问题全部串起来
        //     for (const auto& item : h_srv.response.history) {
        //         all_questions += item.question + ";";
        //     }
        // }

        // 在进行后续判断前，先给出收到语音的反馈，不进行阻塞
        // call_oneshot_tts("小医听到了", false);
        if (heard_feedback) {
            play_mp3_non_blocking("xiaoyi_heard_you.mp3");
        }

        // ---------- 急症：同步等 nurse 结论，先叫护士再回答 ----------
        if (is_emergency(res)) {
            sys_log("emergency keyword detected, use synchronous nurse path.");
            latency_probe::emit("asr", "emergency_path");
            if (handle_nurse_sync(res)) {
                latency_probe::emit("asr", "emergency_handled");
                return AsrStream::asrContinue;
            }
            std::string new_msg = res;
            send_question(new_msg, false);
            return AsrStream::asrContinue;
        }

        // ---------- 普通问题：nurse 判断与主 LLM 并行（乐观路径） ----------
        if (nurse_parallel) {
            latency_probe::emit("asr", "nurse_parallel_start");
            const std::string question_copy = res;
            const long long nurse_req_ns = latency_probe::now_ns();
            latency_probe::emit("asr", "nurse_req",
                                latency_probe::detail_int("text_len",
                                                          (long long)question_copy.size()));
            std::thread([question_copy, nurse_req_ns]() {
                auto request = std::make_shared<llm_node_comm::srv::NurseAlert::Request>();
                request->question = question_copy;
                auto future = nurse_call_client->async_send_request(request);
                if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
                    latency_probe::emit(
                        "asr", "nurse_resp",
                        std::string("\"timeout\":true,\"dur_ms\":") +
                            std::to_string((latency_probe::now_ns() - nurse_req_ns) / 1000000));
                    err_log("async call /check_need_call_nurse timeout.");
                    return;
                }
                latency_probe::emit(
                    "asr", "nurse_resp",
                    latency_probe::detail_double(
                        "dur_ms", (latency_probe::now_ns() - nurse_req_ns) / 1e6));
                const auto response = future.get();
                const bool need_call = response->need_call;
                const bool need_end = response->need_end;
                sys_log("async nurse result: need_call=%d need_end=%d",
                        (int)need_call, (int)need_end);
                if (!need_call && !need_end) return;

                // 护士判断需要中断：停止当前播报，改为呼叫护士 / 结束会话
                cancel_tts(need_call ? "need_call" : "need_end");
                if (need_call) {
                    call_oneshot_tts(response->response, false);
                    if (nurse_keep_dialog) {
                        latency_probe::emit("asr", "nurse_call_keep_dialog");
                        sys_log("nurse call: keep dialog open (no session end).");
                    } else {
                        dialog_session_finished(true, response->comment);
                    }
                } else {
                    dialog_session_finished(false, response->comment);
                }
            }).detach();

            // 立即把问题送进 RAG+LLM（不等 nurse）
            std::string new_msg = res;
            send_question(new_msg, false, false);
            return AsrStream::asrContinue;
        }

        // ---------- 兼容路径：同步等 nurse（NURSE_PARALLEL=0） ----------
        if (handle_nurse_sync(res)) {
            return AsrStream::asrContinue;
        }
        std::string new_msg = res;
        send_question(new_msg, false);
        return AsrStream::asrContinue;
    }

    // 同步调用 nurse_alert：需要呼叫护士 / 结束对话时直接处理并返回 true 表示已结束
    bool handle_nurse_sync(const std::string &res) {
        const long long nurse_req_ns = latency_probe::now_ns();
        latency_probe::emit("asr", "nurse_req",
                            latency_probe::detail_int("text_len", (long long)res.size()));
        auto request = std::make_shared<llm_node_comm::srv::NurseAlert::Request>();
        request->question = res;
        auto future = nurse_call_client->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
            latency_probe::emit(
                "asr", "nurse_resp",
                latency_probe::detail_double(
                    "dur_ms", (latency_probe::now_ns() - nurse_req_ns) / 1e6));
            const auto response = future.get();
            bool need_call = response->need_call;
            bool need_end = response->need_end;
            std::string reason = response->comment;
            std::string response_text = response->response;

            if (need_call) {
                // 模型回复的内容说一下
                call_oneshot_tts(response_text, false);

                sys_log("need call nurse: True.");

                // 对话结束，需要呼叫护士，原因是 模型返回的 reason
                dialog_session_finished(
                    true, reason
                );

                return true;
            }

            if (need_end) {
                sys_log("用户想要结束对话.");

                dialog_session_finished(
                    false, reason
                );

                return true;
            }

            RCLCPP_INFO(node->get_logger(), "need_call: %s", need_call ? "true" : "false");
            RCLCPP_INFO(node->get_logger(), "need_end: %s", need_end ? "true" : "false");
            RCLCPP_INFO(node->get_logger(), "reason: %s", reason.c_str());
            RCLCPP_INFO(node->get_logger(), "response: %s", response_text.c_str());
        } else {
            latency_probe::emit(
                "asr", "nurse_resp",
                std::string("\"timeout\":true,\"dur_ms\":") +
                    std::to_string((latency_probe::now_ns() - nurse_req_ns) / 1000000));
            err_log("call /check_need_call_nurse timeout.");
        }
        return false;
    }

    // UI 节点视频相关
    void recVideoStatusFromUI(const std_msgs::msg::String::SharedPtr msg_p)
    {
        sys_log("rec video status from ui: %s", msg_p->data.c_str());

        std::string status = msg_p->data;

        if (status.find("start") != std::string::npos) {
            sys_log("system paused for palying video... ...");

            // 调整成 kws 检测
            hzc::asr->setReadInStreamCallback(hzc::asr_inputStream_callback);
            // 暂停语音输入
            hzc::asr->pause();

            return;
        }

        if (status.find("end") != std::string::npos) {
            sys_log("system resumed.");
            hzc::asr->resume();

            return;
        }

        sys_log("error: 发的什么东西，又不是 start 又不是 end: %s", status.c_str());
    }


}


int main(int argc, char ** argv)
{
    hzc::init();  // 初始化 kws, asr, tts

    rclcpp::init(argc, argv);
    node = rclcpp::Node::make_shared("ros_node_keyword_asr");

    {
        const char *env = std::getenv("NURSE_PARALLEL");
        nurse_parallel = (env == nullptr) ? true : (std::string(env) != "0");
        sys_log("nurse_parallel = %s", nurse_parallel ? "true" : "false");
    {
        const char *hf = std::getenv("ASR_HEARD_FEEDBACK");
        heard_feedback = hf && (std::string(hf) == "1" || std::string(hf) == "true");
        const char *wg = std::getenv("ASR_WAKE_GREETING");
        wake_greeting = !(wg && (std::string(wg) == "0" || std::string(wg) == "false"));
        const char *nkd = std::getenv("NURSE_KEEP_DIALOG");
        nurse_keep_dialog = !(nkd && (std::string(nkd) == "0" || std::string(nkd) == "false"));
        sys_log("heard_feedback = %s, wake_greeting = %s",
                heard_feedback ? "true" : "false", wake_greeting ? "true" : "false");
        sys_log("nurse_keep_dialog = %s", nurse_keep_dialog ? "true" : "false");
    }
    }

    // ===== 链路耗时埋点（仅 LATENCY_PROBE=1 时创建话题/发布）=====
    if (latency_probe::enabled()) {
        latency_pub = node->create_publisher<std_msgs::msg::String>("/latency_event", 200);
        latency_probe::publisher() = [](const std::string & payload) {
            if (latency_pub) {
                std_msgs::msg::String latency_msg;
                latency_msg.data = payload;
                latency_pub->publish(latency_msg);
            }
        };
        sys_log("latency probe enabled, publishing /latency_event");
    }
    latency_probe::emit("asr", "init_done");

    // asr 定期检查，如果 15 秒 没有应答就走开
    // 会话静默计时器：默认 15s，环境变量 ASR_SESSION_TIMEOUT 可覆盖（秒）。
    // 注意：长回答播放超过该值时，会话会在播放中途被判超时结束（BT 视角），
    // 需要支持长回答时调大（如 30~60s）。
    {
        const char *to_env = std::getenv("ASR_SESSION_TIMEOUT");
        int session_timeout_s = 15;
        if (to_env != nullptr && *to_env != '\0') {
            try { session_timeout_s = std::max(5, std::stoi(to_env)); } catch (...) {}
        }
        sys_log("session timeout: %d s", session_timeout_s);
        asr_timer = node->create_wall_timer(
            std::chrono::seconds(session_timeout_s), asr_timerCallback);
    }
    asr_timer->cancel();

    // 向 question manager 发布语音识别到的问题
    req_pub = node->create_publisher<std_msgs::msg::String>("question_asr", 10);

    // 发布检测到 kws 的信号
    kws_pub = node->create_publisher<std_msgs::msg::Bool>("/call_signal", 10);

    // 打断 TTS 播报（护士判断晚到、需立即改为“呼叫护士”时使用）
    tts_cancel_pub = node->create_publisher<std_msgs::msg::String>("/tts_cancel", 10);

    // 发布 dialog session 已经结束了
    dialog_session_finished_pub = node->create_publisher<llm_node_comm::msg::DialogSessionFinished>(
        "dialog_session_finished", 10
    );

    // 接收 topic, 这个 topic 是用来开启 asr 的
    start_asr_sub = node->create_subscription<std_msgs::msg::Bool>(
        "start_asr", 10, start_asr_callback);


    // 是否需要呼叫护士
    nurse_call_client = node->create_client<llm_node_comm::srv::NurseAlert>("check_need_call_nurse");

    // 机器人播放视频，ui节点向我发送 start 和 end 标志
    // rostopic pub /huzhou_llm_end std_msgs/String "data: 'start'" -1
    // rostopic pub /huzhou_llm_end std_msgs/String "data: 'end'" -1
    video_status_sub = node->create_subscription<std_msgs::msg::String>(
        "huzhou_llm_end", 10, recVideoStatusFromUI);

    // 一个问题的回答语音已经播放完成了
    llm_queue_empty_sub = node->create_subscription<std_msgs::msg::String>(
        "tts_session_finished", 10, llm_question_finished_callback);

    // 一次性合成的 tts 语音信息
    client = node->create_client<llm_node_comm::srv::TtsOneshot>("/tts_one_shot");

    // 检查用户是否想要结束对话
    end_session_client = node->create_client<llm_node_comm::srv::EndSession>("/check_end_dialog");

    // 向 ui 节点发送信息
    req_ui_pub = node->create_publisher<std_msgs::msg::String>("send_question_to_ui", 10);

    // 检测到播放视频的关键词，把这个发送给 ui
    video_pub = node->create_publisher<std_msgs::msg::String>("huzhou_llm_video", 10);

    // 设置 kws 的回调函数 和 asr 的回调函数
    hzc::kws->SetKeywordDetectedCallback(OnKeywordDetected);
    hzc::asr->setAsrCallback(asr_callback);

    // 切换到关键词识别模型
    change_to_kws();

    // 恢复语音输入
    hzc::asr->resume();

    sys_log("ros node keyword asr init done.");

    // 发布已经初始化完成的 topic
    // ros::Publisher asr_init_pub = nh.advertise<std_msgs::Bool>("asr_init_done", 10);
    // std_msgs::Bool msg;
    // msg.data = true;
    // asr_init_pub.publish(msg);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    hzc::signalHandler(0); // ??? 感觉应该是  exit(0)
}
