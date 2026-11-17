#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""医疗知识库 LLM 节点（本地 RAG 检索 + 云端大模型直连）。

链路：/llm_to_dashapp -> 本地 RAG 检索 -> 组装带参考资料的消息 -> 云端流式生成
      -> /ali_llm_output（分块文本，[START]/[DONE] 控制符）-> 断句 -> 本地 TTS

与旧版的区别
------------
- 旧版：dashscope Application.call(app_id=...)，检索与生成都在云端应用内，首块 5–11s。
- 新版：本地 RAG（http://127.0.0.1:8765）取 top-k 片段注入 prompt，再直连大模型流式生成。
- 接入方式默认 dashscope SDK 的 MultiModalConversation（MaaS 专有部署地址，
  model=deepseek-v4.1-flash-1017，enable_thinking=False）；设 MED_LLM_PROVIDER=openai
  可退回 OpenAI 兼容接口（qwen-plus）。
- 保留 `MED_LLM_BACKEND=app` 可切回旧百炼应用做 A/B 对比；直连失败可选回退。

环境变量
--------
MED_LLM_BACKEND      direct | app | auto，默认 direct（auto=直连失败回退 app）
MED_LLM_PROVIDER     dashscope（默认，SDK 原生）| openai（兼容接口）| ollama（本机 11434）
MED_LLM_MODEL        默认 deepseek-v4.1-flash-1017（dashscope）/ qwen-plus-latest（openai）/ qwen3:8b（ollama）
MED_LLM_BASE_URL     默认 MaaS 地址（dashscope）/ dashscope 兼容地址（openai）/ http://127.0.0.1:11434/v1（ollama）
MED_LLM_ENABLE_THINKING 默认 0：关闭思考（否则先吐 reasoning_content，首句会晚数秒）
MED_LLM_NO_THINK     provider=ollama 时默认 1：系统提示追加 /no_think，关闭 qwen3 的思考
MED_LLM_CHAT_TEMPLATE_KWARGS  本地 llama.cpp/vLLM 关思考用（JSON），例
                     '{"enable_thinking": false}'；云端顶层 enable_thinking 对 llama-server 无效
                     本地 Qwen3.5 接入示例：MED_LLM_PROVIDER=openai
                     MED_LLM_BASE_URL=http://127.0.0.1:8080/v1 MED_LLM_MODEL=qwen-local
MED_AUX_LOCAL        1：ASR 语义纠错 + RAG 查询扩展优先走本地 llama.cpp（失败回退云端）
MED_ASR_CORRECT_LOCAL / MED_RAG_EXPAND_LOCAL  单项覆盖（默认跟随 MED_AUX_LOCAL）
MED_LLM_LOCAL_BASE_URL / MED_LLM_LOCAL_MODEL  本地服务地址/模型（默认 8080/qwen-local）
DASHSCOPE_API_KEY    优先读环境变量，没有则读 src/llm_node_py/key/dashscope.key
MED_LLM_APP_ID       百炼应用 id（app 模式用）
MED_LLM_MAX_TOKENS   默认 400
MED_LLM_HISTORY_PAIRS 带入的历史问答对数，默认 3
RAG_ENABLE           默认 1
RAG_URL              默认 http://127.0.0.1:8765
RAG_TOP_K            默认 4
RAG_TIMEOUT          默认 3.0 秒
MED_CHAT_ENABLE      默认 1：非医学问题正常闲聊回答；0=所有问题都只依据资料（旧行为）
MED_INTENT_JUDGE     默认 1：关键词判闲聊后再让快模型复核（防生僻说法的医学问题漏判）
MED_INTENT_MODEL     默认 qwen-flash（实测 0.23–0.33 s）
MED_INTENT_TIMEOUT   默认 1.2 秒，超时/报错回退关键词结果（寒暄/闲聊词直接跳过 judge）
LATENCY_PROBE        1 时把内部时间戳发到 /latency_event

测试
----
ros2 topic pub --once /llm_to_dashapp std_msgs/msg/String "{data: '透析患者饮水量怎么控制？'}"
ros2 topic echo /ali_llm_output
"""

import difflib
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import dashscope
from dashscope import Application, MultiModalConversation
from openai import OpenAI
import httpx

# 把 ros 生成的 srv 文件的地址加上（ROS2中可能需要调整路径，保留原逻辑）
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../install/llm_node_comm/lib/python3.12/site-packages",
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 当前目录
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from llm_node_comm.msg import LlmDialogHistoryMsg, DialogSessionFinished
from llm_node_comm.srv import LlmDialogHistorysSrv, EndSession
from get_dashscope_key import get_dashscope_key
import rag_bm25

# ---------- 语音链网关化（2026-11 架构精简） ----------
# 本节点现在是 ASR→LLM→TTS 的网关：直接消费 /question_asr（含纠错），
# 内嵌 dialog_history 与 /check_end_dialog 服务，替代独立的
# question_router / llm_network / dialog_history / check_end_session 四个进程。
_TYPO_MAP = {
    "火龙观": "火龙罐",
    "内露": "内瘘",
    "内漏": "内瘘",
    "血林": "血磷",
    "首卫生": "手卫生",
    # 唤醒词"小医"的 ASR 常见同音错字（实测出现过 小一/小伊/小易/小姨）
    "小一": "小医",
    "小伊": "小医",
    "小易": "小医",
    # 血透语境下"每周"被误识别为"美洲"（实测 2026-11）
    "美洲": "每周",
}
_TYPO_PATTERN = re.compile("|".join(re.escape(k) for k in _TYPO_MAP))
# 句首"小X"称呼 → 小医（含小姨；仅句首，避免误伤"我小姨也透析"）
_WAKE_RE = re.compile(r"^\s*小[一二伊姨易](?=[你好我请麻烦帮带想，,。？?！!]|$)")
# C++ ASR 节点旧版会在问题后拼接给老 router 的提示语
# （"。提示:请用简短口语回答" / "。提示:模型判断出此时不需要呼叫护士"）。
# 新网关不解析它，且本地 2B 纠错会把提示当正文改写——必须在入口剥掉。
_HINT_RE = re.compile(r"[。\.]?\s*提示\s*[:：].*$", re.S)


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "")
    return value if value != "" else default


# 患者教育场景的固定人设 + 基于资料的硬约束（尽量短，减少首块延迟）
MED_SYSTEM_PROMPT = """你是血透中心巡诊机器人「小医」，用轻柔亲切的语气和患者对话。
必须遵守：
1. 医学与护理问题，优先依据【参考资料】回答：
   - 资料里有相关内容（哪怕不是逐字回答），就用自己的话讲清楚，给患者可执行的建议；
   - 只有资料完全没提到这个话题、确实无法回答时，才说“这个问题我资料里没有。建议问护士或医生。”；
   - 不要编造资料以外的具体数值、剂量或做法，不要重复“资料里没有”。
2. 口语化短句，一次说 1-3 句，总长不超过 80 字，适合语音播报；不要列出资料编号。
2.1 第一句必须极短：先给一句不超过 8 个字的独立短句，并以句号结尾（如“先别沾水。”“这个要注意。”），再补充细节；首句必须是完整短句，不要把两个信息合并进第一句，不要带逗号。
3. 不给具体用药剂量、不诊断；涉及具体数值时只做一般性提醒，并建议咨询医生。
4. 患者出现胸痛、呼吸困难、意识改变、内瘘出血不止、明显心慌乏力等紧急情况时，立即说“我马上帮您呼叫护士”，不要展开长篇内容。
5. 禁止输出表情符号、动作描写；回答中不要出现“着急”两个字。

【示例】
问：透析完针眼什么时候能沾水？
答：先别沾水。透析后 24 小时内穿刺点不能清洗，当天也别洗澡。
问：内瘘摸不到震颤了怎么办？
答：先别慌。摸不到震颤要立即就医，可能是内瘘堵塞。
问：透析的时候能吃东西吗？
答：尽量别吃。透析中进食容易引起低血压、抽筋，实在饿了先问护士。"""


# 非医学问题（闲聊/常识/陪伴）用的轻量人设：正常回答，不做资料约束
MED_CHAT_PROMPT = """你是血透中心巡诊机器人「小医」，说话轻柔亲切，像身边的陪护一样。
现在患者问的不是医学或护理问题（可能是闲聊、家常、常识或心情），请正常、自然地回答：
1. 直接回答，不要拒答，不要说“资料里没有”“这个问题我资料里没有”之类的话，也不要提参考资料。
2. 口语化短句，1-2 句，不超过 40 字，像聊天一样；不要列编号、不要写长段落。
3. 如果你是开玩笑/讲故事，控制在两三句内就好；禁止使用任何表情符号或颜文字（如 😄😊^_^），不要动作描写，不要出现“着急”两个字。
4. 如果患者说到身体不舒服、治疗、用药，只用一句“这个我不太确定，建议问护士或医生”带过，不要给医学建议。
5. 不要主动给健康建议（比如“多喝水”“多运动”“注意饮食”），也不要评价患者的身体状况——这些交给医学问答去处理。
6. 用户要你讲笑话/故事/诗、唱歌、说谜语时，直接把内容说出来，不要反问“想不想听”，不要说“我刚学会”“我还在学”这类话；同一个请求每次换一个新内容，不要重复上一轮的回答。
7. 用户说“讲个笑话/再来一个/我要听”时，你直接讲一个完整的短笑话（一到两句话、40 字以内说完）；“再来一个”必须换新的，不能重复上一轮讲过的内容，也不要把本提示里的任何句子当成回答照搬。"""


# 关键词判为"闲聊"后的轻量 LLM 复核（只判医学/闲聊，不再生成内容，约 0.3 s）
INTENT_JUDGE_PROMPT = (
    "判断以下问题是否与医学、健康、护理、透析、用药有关。只回答一个字：医 或 聊。"
)

# 流式输出里的表情符号/颜文字直接过滤（TTS 念不出来，也会破坏播报节奏）
_EMOJI_RE = re.compile(
    "[" 
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U00002190-\U000021FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+"

)
# 中文引号在 TTS 里会被切出单独的““””片段
_QUOTE_RE = re.compile("[\u201c\u201d\u300c\u300d\u300e\u300f]")

# 明显的寒暄/闲聊：直接走闲聊路径，不再花一次 judge（0.4–1.3 s）
_CHITCHAT_RE = re.compile(
    "|".join([
        r"^(你好|您好|嗨|哈喽|早上好|早安|中午好|下午好|晚上好|晚安|再见|拜拜|谢谢|感谢|不客气)[呀啊哦呢！!。～~ ]*$",
        r"(讲|说|来|换个|再来)(个|一个)?(睡前)?(笑话|故事|段子)",
        r"谢谢你的?陪伴|感谢你的?陪伴",
        r"(唱|哼)(首|一首)?歌",
        r"(背|念)(首|一首)?(诗|古诗)",
        r"(出|猜)(个|一个)?谜语",
        r"绕口令|歇后语",
        r"(陪|陪陪|和|跟)(我)?聊",
        r"聊聊天|聊会儿|聊一会",
        r"你是谁|你叫什么|你几岁|你是机器人|你会做什么|你会什么|你能做什么",
        r"夸夸我|夸我|鼓励我",
        r"我喜欢你|我爱你",
        r"无聊|好无聊",
        r"你会(下棋|唱歌|写诗|讲故事)",
        r"^(嗯+|好+|可以|行|要|想|对|是|听|继续|再来)[呀啊哦呢哈！!。～~ ]*$",
        r"(我要听|我想听|想听|要听|继续讲|继续|再讲一个|再来一个|再唱|再背|换一个|下一个|还有吗|接着讲)",
    ])
)


def _looks_like_chitchat(text: str) -> bool:
    """明显的寒暄/闲聊问题：跳过 judge 复核，直接走闲聊。"""
    return bool(_CHITCHAT_RE.search(text or ""))


# 判定“是否医学/护理相关问题”的关键词表（宁可漏判成闲聊，也不要误判成医学）
_MEDICAL_TERMS = [
    "透析", "血透", "血液净化", "腹透", "尿毒", "肾脏", "肌酐", "尿素氮",
    "内瘘", "瘘管", "瘘", "导管", "置管", "穿刺", "针眼", "针口", "扎针",
    "血压", "血糖", "贫血", "输血", "化验", "检查结果", "指标", "血钾", "血磷",
    "血钙", "白蛋白", "蛋白", "营养", "药", "服药", "用药", "剂量", "头孢",
    "消炎药", "抗生素", "感冒", "咳嗽", "发烧", "发热", "感染", "炎症",
    "疼痛", "疼", "痛", "麻木", "无力", "乏力", "没劲", "头晕", "头痛",
    "恶心", "呕吐", "想吐", "难受", "不舒服", "消瘦", "瘦了",
    "抽筋", "痉挛", "瘙痒", "痒", "水肿", "浮肿", "肿", "失眠", "睡不着",
    "心慌", "胸闷", "呼吸", "出血", "渗血", "凝血", "体重", "涨水", "控水",
    "饮水量", "喝水", "饮食", "忌口", "运动", "锻炼", "热敷", "洗澡", "沾水",
    "水果", "蔬菜", "主食", "胃口", "饭量", "零食", "咸", "辣", "油腻",
    "护士", "医生", "医院", "住院", "手术", "治疗", "症状", "副作用", "并发",
    "中医", "艾灸", "贴敷", "情志", "心理", "焦虑", "抑郁", "睡眠", "怀孕",
    "月经", "疫苗", "献血", "晕针", "盐", "钾", "磷", "钙", "透析器", "干体重",
]

_MEDICAL_PATTERN = re.compile("|".join(re.escape(t) for t in _MEDICAL_TERMS))


def _is_medical(text: str) -> bool:
    """粗判是否医学/护理相关。关键词命中即按医学处理（走 RAG + 严格话术）。"""
    return bool(text) and _MEDICAL_PATTERN.search(text) is not None


class KnowledgeBaseLLMNode(Node):

    def __init__(self):
        super().__init__("knowledgebase_llm_node")

        # ---------- 配置 ----------
        self.backend = _env("MED_LLM_BACKEND", "direct").strip().lower()
        # provider=dashscope: SDK 原生接口（MaaS 专有部署）；openai: 兼容接口（旧 qwen-plus 路径）
        # 默认走 openai 兼容 (qwen-plus-latest)：MaaS deepseek 专有部署 2026-11 起 404 失效
        # 需要恢复 MaaS 时设 MED_LLM_PROVIDER=dashscope
        self.provider = _env("MED_LLM_PROVIDER", "openai").strip().lower()
        if self.provider == "ollama":
            # 本机 ollama（OpenAI 兼容接口），默认 qwen3:8b
            self.model_name = _env("MED_LLM_MODEL", "qwen3:8b")
            default_base = "http://127.0.0.1:11434/v1"
        elif self.provider == "dashscope":
            self.model_name = _env("MED_LLM_MODEL", "deepseek-v4.1-flash-1017")
            default_base = (
                "https://llm-wd6fnilu0sk3blx6.cn-beijing.maas.aliyuncs.com/api/v1"
            )
        else:
            # qwen3.8-flash 实测（2026-11-12, 23题质量评测）：TTFT 555ms vs 608ms
            # 但要点覆盖均值 49.7% vs 58.7%、1 题边界编造——医疗场景质量优先，
            # 默认仍用 qwen-plus-latest。需切换时 MED_LLM_MODEL=qwen3.8-flash
            # （注意：qwen3 系列混合思考模型，openai 路径已显式关思考）
            self.model_name = _env("MED_LLM_MODEL", "qwen-plus-latest")
            default_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.base_url = _env("MED_LLM_BASE_URL", default_base)
        # 思考模型必须关思考：否则首块是 reasoning_content，TTS 要等推理完才有话可播
        # 闲聊可以走更小的本地模型（如 qwen3:4b）：生成约 2x，显存减半
        self.chat_model = _env("MED_CHAT_MODEL", "")
        self._request_model = self.model_name
        self.enable_thinking = _env("MED_LLM_ENABLE_THINKING", "0") == "1"
        # llama.cpp / vLLM 等本地 OpenAI 兼容服务：思考开关走 chat_template_kwargs，
        # 顶层 enable_thinking 会被忽略（llama-server 会继续吐 reasoning_content）。
        # 例：MED_LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
        self.chat_template_kwargs = {}
        _ctk_raw = _env("MED_LLM_CHAT_TEMPLATE_KWARGS", "").strip()
        if _ctk_raw:
            try:
                self.chat_template_kwargs = json.loads(_ctk_raw)
            except Exception as exc:
                self.get_logger().warn(
                    f"MED_LLM_CHAT_TEMPLATE_KWARGS 解析失败({exc})，已忽略")
        # ollama 上的 qwen3 默认会先思考（reasoning）；系统提示加 /no_think 可关掉
        self.no_think = _env("MED_LLM_NO_THINK", "1") == "1"
        self.api_key = os.getenv("DASHSCOPE_API_KEY") or get_dashscope_key()
        # judge 走公网兼容模式（快模型），与主模型接入地址解耦
        self.judge_base_url = _env(
            "MED_INTENT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.app_id = _env("MED_LLM_APP_ID", "b7ed8dad5c0c483d8e865a3be0b4e088")
        # 本地 8B 生成慢（约 1 s / 80 字），默认压到 220 token；云端仍用 400
        default_max_tokens = "220" if self.provider == "ollama" else "400"
        self.max_tokens = int(_env("MED_LLM_MAX_TOKENS", default_max_tokens))
        self.chat_max_tokens = int(_env("MED_CHAT_MAX_TOKENS", "120"))
        # 单次 LLM 请求超时（秒）：超时后播兜底话术并正常结束本段
        self.llm_timeout = float(_env("MED_LLM_TIMEOUT", "10"))
        self.history_pairs = int(_env("MED_LLM_HISTORY_PAIRS", "3"))
        self.app_fallback = _env("MED_LLM_APP_FALLBACK", "0") == "1"

        self.rag_enabled = _env("RAG_ENABLE", "1") == "1"
        # 非医学问题是否正常回答（1=闲聊直答，0=所有问题都走资料约束，回到旧行为）
        self.chat_enabled = _env("MED_CHAT_ENABLE", "1") == "1"
        self.rag_url = _env("RAG_URL", "http://127.0.0.1:8765").rstrip("/")
        self.rag_top_k = int(_env("RAG_TOP_K", "4"))
        # 每块资料注入 prompt 的最大字数（压缩输入可显著降低 LLM 首 token 延迟）
        self.rag_chunk_max_chars = int(_env("RAG_CHUNK_MAX_CHARS", "350"))
        self.rag_timeout = float(_env("RAG_TIMEOUT", "3.0"))

        # 意图复核：关键词判"闲聊"后再让快模型确认一次，防止生僻说法的医学问题漏判
        self.intent_judge_enabled = _env("MED_INTENT_JUDGE", "1") == "1"
        # judge 用哪个模型：auto=主模型是 ollama 就用本机（约 50 ms），否则云端 qwen-flash（400–800 ms）
        self.intent_provider = _env("MED_INTENT_PROVIDER", "auto").strip().lower()
        if self.intent_provider == "auto":
            self.intent_provider = "ollama" if self.provider == "ollama" else "cloud"
        default_intent_model = "qwen3:8b" if self.intent_provider == "ollama" else "qwen-flash"
        self.intent_model = _env("MED_INTENT_MODEL", default_intent_model)
        # 本机 judge 的 HTTP 根地址（去掉 OpenAI 兼容的 /v1 后缀）
        self.ollama_root = self.base_url.rstrip("/")
        if self.ollama_root.endswith("/v1"):
            self.ollama_root = self.ollama_root[:-3]
        self.intent_judge_timeout = float(_env("MED_INTENT_TIMEOUT", "1.2"))

        # 本地辅助模型：ASR 纠错 + RAG 查询扩展（两者每题都调用，云端约 0.4~0.7s/次）。
        # MED_AUX_LOCAL=1 时优先走本机 llama.cpp（实测纠错 ~120ms、扩展 ~160ms），
        # 本地失败自动回退云端 qwen-flash；judge 仍保持云端（准确率优先）。
        aux_local = _env("MED_AUX_LOCAL", "0") == "1"
        self.asr_correct_local = _env(
            "MED_ASR_CORRECT_LOCAL", "1" if aux_local else "0") == "1"
        self.rag_expand_local = _env(
            "MED_RAG_EXPAND_LOCAL", "1" if aux_local else "0") == "1"
        self.local_llm_base_url = _env(
            "MED_LLM_LOCAL_BASE_URL", "http://127.0.0.1:8080/v1")
        self.local_llm_model = _env("MED_LLM_LOCAL_MODEL", "qwen-local")
        self.aux_local_timeout = float(_env("MED_AUX_LOCAL_TIMEOUT", "1.2"))
        self.local_client = None

        # OpenAI 客户端（DashScope 兼容模式）前先清掉代理，避免初始化失败
        for key in (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
        ):
            os.environ.pop(key, None)
        if self.provider == "dashscope":
            dashscope.base_http_api_url = self.base_url
        # max_retries=0：语音链路宁可带着兜底话术快速失败，也不要在后台静默重试 3 次
        main_base = (
            self.base_url if self.provider in ("openai", "ollama") else self.judge_base_url
        )
        # 分项超时：读超时 4s（流式中途卡死快速失败→切备用；正常长回答不受影响）
        self._http_timeout = httpx.Timeout(
            connect=float(os.environ.get("MED_LLM_CONNECT_TIMEOUT", "3.0")),
            read=float(os.environ.get("MED_LLM_READ_TIMEOUT", "4.0")),
            write=10.0, pool=3.0)
        self.client = OpenAI(api_key=self.api_key, base_url=main_base,
                             max_retries=0, timeout=self._http_timeout)
        self.judge_client = OpenAI(
            api_key=self.api_key, base_url=self.judge_base_url, max_retries=0
        )
        if self.asr_correct_local or self.rag_expand_local:
            # 本地辅助调用客户端（llama.cpp/vLLM 的 OpenAI 兼容接口）
            self.local_client = OpenAI(
                api_key="local", base_url=self.local_llm_base_url, max_retries=0
            )

        # ---------- 云端连接预热（工业优化） ----------
        # 唤醒（/start_asr）后用户还在说话，微小请求先把 TLS/连接池建好——
        # 实测云端建连/响应头等待中位 1.44s / P90 3.2s / 最大 10s
        self._last_prewarm = 0.0
        self.start_asr_sub = self.create_subscription(
            Bool, "/start_asr", self._on_start_asr, 10
        )
        if os.environ.get("MED_LLM_PREWARM", "1") == "1":
            threading.Thread(target=self._prewarm, daemon=True).start()
        # 连接保活：定期 1-token 请求维持连接池热态（keep-alive 超时后不再冷连）
        keepalive_s = int(os.environ.get("MED_LLM_KEEPALIVE", "60") or 0)
        if keepalive_s > 0:
            def _keepalive():
                while True:
                    time.sleep(keepalive_s)
                    self._prewarm()
            threading.Thread(target=_keepalive, daemon=True).start()

        # ---------- ROS 接口 ----------
        # 网关入口：直接消费 ASR 输出（含同音错字纠正），不再经过独立 router 进程
        self.sub_asr = self.create_subscription(
            String, "/question_asr", self.question_asr_callback, 10
        )
        # 兼容保留：bt/调试仍可直发 /llm_to_dashapp（原 router 输出话题）
        self.sub = self.create_subscription(
            String, "/llm_to_dashapp", self.query_callback, 10
        )
        self.pub = self.create_publisher(String, "/ali_llm_output", 10)

        # ---------- 内嵌 dialog_history（原独立节点，list + 服务，口径不变） ----------
        self._history = []          # LlmDialogHistoryMsg 列表，最多保留最近 5 组
        self._history_max = 5
        # 会话代次：结束信号到达时 +1；旧回答线程若晚于结束信号才 DONE，
        # 不允许把"结束语那轮"再写回已清空的历史（否则下次唤醒上下文仍脏）。
        self._session_epoch = 0
        self._tls = threading.local()   # 每轮线程记录自己的 epoch
        self.dialog_pub = self.create_publisher(
            LlmDialogHistoryMsg, "/send_to_dialog_history", 40
        )
        self.dialog_history_sub = self.create_subscription(
            LlmDialogHistoryMsg, "/send_to_dialog_history", self._on_dialog_msg, 40
        )
        self.dialog_history_service = self.create_service(
            LlmDialogHistorysSrv, "/get_llm_dialog_history", self._handle_get_history
        )

        # ---------- 内嵌 /check_end_dialog 服务（原 check_end_session 节点） ----------
        self.end_session_service = self.create_service(
            EndSession, "check_end_dialog", self._handle_end_session
        )
        self._end_session_history = [
            {"role": "system", "content": (
                "你是聊天助手，负责监测用户输入，判断用户是否想结束本次对话。"
                "你只判断意图，不执行实际操作。返回 JSON 格式："
                "need_end（bool，是否想结束对话），reason（string，判断原因），"
                "response（string，给用户的回复内容，如果不需要结束则为空）。")},
            {"role": "assistant", "content": (
                "好的，我会根据用户的输入判断是否想结束对话，"
                "并且我会严格输出 JSON 格式，如果不想结束对话则 response 为空。")},
            {"role": "user", "content": "没什么事儿了，你走吧。"},
            {"role": "assistant", "content": (
                '{"need_end": true, "reason": "用户明确表示想结束聊天。", '
                '"response": "好的，那我们下次再聊。"}')},
            {"role": "user", "content": "肾病患者应该注意些什么？"},
            {"role": "assistant", "content": (
                '{"need_end": false, "reason": "用户仍在进行咨询或聊天，没有表示结束意图。", '
                '"response": ""}')},
        ]

        # 会话结束（need_end / need_call / 超时）由 ASR 节点发布；清空内嵌历史，
        # 否则重新唤醒后仍带着"结束对话"的上下文，回答会跑偏（实测 2026-11-16）。
        self.dialog_finished_sub = self.create_subscription(
            DialogSessionFinished, "dialog_session_finished",
            self._on_dialog_finished, 10
        )

        # ---------- 埋点（仅 LATENCY_PROBE=1 时创建话题） ----------
        self._latency_probe = _env("LATENCY_PROBE", "0") == "1"
        self._latency_pub = None
        if self._latency_probe:
            self._latency_pub = self.create_publisher(String, "/latency_event", 200)
            self.get_logger().info("latency probe enabled, publishing /latency_event")

        self.get_logger().info(
            f"KnowledgeBase LLM Node started. backend={self.backend}, "
            f"provider={self.provider}, model={self.model_name}, "
            f"chat_model={self.chat_model or '-'}, "
            f"base={self.base_url}, thinking={self.enable_thinking}, "
            f"rag={self.rag_enabled} ({self.rag_url}), "
            f"top_k={self.rag_top_k}, chat={self.chat_enabled}, "
            f"intent_judge={self.intent_judge_enabled}({self.intent_model}, {self.intent_provider}), "
            f"aux_local={self.asr_correct_local}/{self.rag_expand_local}"
            f"({self.local_llm_model}@{self.local_llm_base_url})"
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _ensure_start_published(self) -> None:
        """每轮只发一次 [START]（TTS 轮次起点：清队列+填充词）。
        问题一进入就发，填充词不必等纠错/RAG扩展/云端建连。"""
        if not getattr(self, "_round_start_published", False):
            self.pub.publish(String(data="[START]"))
            self._round_start_published = True

    def _publish_stream(self, text: str) -> bool:
        """流式文本发布：丢弃已结束轮次的迟到内容（避免串进下一轮）。

        handle_query 在自己的线程里设置 _tls.epoch；会话结束 epoch 变化后，
        旧线程即使还在收云端流，也不能再往 TTS 发文字。
        """
        epoch = getattr(self._tls, "epoch", None)
        if epoch is not None and epoch != self._session_epoch:
            return False
        self.pub.publish(String(data=text))
        return True

    def _aux_chat(self, messages, max_tokens: int, timeout: float,
                  prefer_local: bool) -> str:
        """辅助模型调用（ASR 纠错 / RAG 扩展）：本地优先，失败回退云端。

        prefer_local=True 且配置了本地服务时，先调 llama.cpp（关思考）；
        本地超时/报错再调云端 qwen-flash（保证断网外的异常也能降级）。
        返回去掉首尾空白的文本；全部失败时抛最后一个异常。
        """
        providers = []
        if prefer_local and self.local_client is not None:
            providers.append((self.local_client, self.local_llm_model, True))
        providers.append((self.judge_client, self.intent_model, False))
        last_exc = None
        for client, model, is_local in providers:
            t0 = time.perf_counter()
            try:
                extra = (
                    {"chat_template_kwargs": {"enable_thinking": False}}
                    if is_local else {"enable_thinking": False}
                )
                r = client.chat.completions.create(
                    model=model, messages=messages, temperature=0.0,
                    max_tokens=max_tokens, timeout=timeout, extra_body=extra)
                txt = (r.choices[0].message.content or "").strip()
                self._lat(
                    "aux_llm", provider="local" if is_local else "cloud",
                    model=model, ms=round((time.perf_counter() - t0) * 1000, 1),
                    max_tokens=max_tokens, chars=len(txt))
                return txt
            except Exception as exc:
                last_exc = exc
                self._lat(
                    "aux_llm_error", provider="local" if is_local else "cloud",
                    model=model, ms=round((time.perf_counter() - t0) * 1000, 1),
                    error=str(exc)[:100])
        raise last_exc if last_exc else RuntimeError("no aux provider")

    def _llm_correct_asr(self, text: str) -> str:
        """ASR 语义纠错层（第二层，静态 typo_map 之后）：上下文纠错，
        修同音错字与丢字。防呆：空结果、长度突变、与原句编辑相似度 <0.65
        → 用原文（本地 2B 曾把"再讲个笑话"续写成医学句子）。
        闲聊/指令类不做纠错（讲笑话/天气等，避免模型"接着写"）。
        MED_ASR_CORRECT=0 关闭；超时 MED_ASR_CORRECT_TIMEOUT（默认 1.0s）。"""
        if os.environ.get("MED_ASR_CORRECT", "1") != "1":
            return text
        if _looks_like_chitchat(text):
            return text
        if len(text) < 6 or len(text) > 80:
            return text
        try:
            timeout = float(os.environ.get("MED_ASR_CORRECT_TIMEOUT", "1.0"))
            fixed = self._aux_chat(
                [
                    {"role": "system", "content": (
                        "你是血透病房语音识别纠错器。只修正同音错字和明显丢字，"
                        "保持原意、原语气和标点结构；不要回答问题、不要续写、不要增删语义。"
                        "领域词汇：内瘘、透析、血钾、血磷、干体重、穿刺、震颤、"
                        "导管、人工血管。无错误就原样输出。只输出修正后的文本。\n"
                        "示例：'我内露摸不到擅抖' -> '我内瘘摸不到震颤'；"
                        "'再讲个笑话' -> '再讲个笑话'（绝不续写笑话内容）。")},
                    {"role": "user", "content": text},
                ],
                max_tokens=100, timeout=timeout,
                prefer_local=self.asr_correct_local)
            max_delta = max(2, len(text) // 4)
            ratio = (difflib.SequenceMatcher(None, text, fixed).ratio()
                     if fixed else 0.0)
            if (fixed and abs(len(fixed) - len(text)) <= max_delta
                    and ratio >= 0.65):
                if fixed != text:
                    self._lat("asr_correct", fixed=fixed[:24],
                              provider="local" if self.asr_correct_local else "cloud",
                              ratio=round(ratio, 3))
                return fixed
            self.get_logger().warn(
                f"语义纠错被防呆拦截（ratio={ratio:.2f}, "
                f"len {len(text)}->{len(fixed)}），用原文")
            self._lat("asr_correct_reject", ratio=round(ratio, 3),
                      in_len=len(text), out_len=len(fixed))
        except Exception as exc:
            self.get_logger().warn(f"语义纠错失败（用原文）: {str(exc)[:60]}")
        return text

    def _prewarm(self) -> None:
        """云端连接预热：1-token 微请求建立 TLS/连接池。30s 防抖。"""
        if time.time() - self._last_prewarm < 30:
            return
        try:
            t0 = time.perf_counter()
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "1"}],
                max_tokens=1, stream=False, timeout=8,
            )
            dt = (time.perf_counter() - t0) * 1000
            self._last_prewarm = time.time()
            self.get_logger().info(f"LLM 连接预热完成 {dt:.0f}ms")
            self._lat("llm_prewarm", ms=round(dt))
        except Exception as exc:
            self.get_logger().warn(f"LLM 连接预热失败: {str(exc)[:80]}")

    def _on_start_asr(self, msg: Bool) -> None:
        """收音开启（唤醒桥/行为树）：预热云端连接，赶在问题说完之前。"""
        if not msg.data:
            return
        if time.time() - self._last_prewarm < 30:
            return
        threading.Thread(target=self._prewarm, daemon=True).start()

    def _lat(self, stage: str, **detail) -> None:
        """把内部时间戳发布到 /latency_event（仅 LATENCY_PROBE=1）。"""
        if self._latency_pub is None:
            return
        try:
            payload = {
                "node": "llm_medical",
                "stage": stage,
                "t_ns": time.time_ns(),
                "detail": detail,
            }
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self._latency_pub.publish(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def query_callback(self, msg: String):
        question = msg.data.strip()
        if not getattr(self, "_round_start_published", False):
            self._ensure_start_published()
        self.get_logger().info(f"Received question: {question}")
        threading.Thread(
            target=self.handle_query, args=(question,), daemon=True
        ).start()

    # ------------------------------------------------------------------
    def _judge_intent_local(self, text: str):
        """本机 ollama judge：/api/chat + think=False + /no_think，实测 46–113 ms。"""
        t0 = time.perf_counter()
        body = {
            "model": self.intent_model,
            "messages": [
                {"role": "system", "content": INTENT_JUDGE_PROMPT},
                {"role": "user", "content": text.rstrip() + "/no_think"},
            ],
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {"num_predict": 2, "temperature": 0.0},
        }
        try:
            req = urllib.request.Request(
                self.ollama_root + "/api/chat",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.intent_judge_timeout + 1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            answer = (data.get("message", {}).get("content") or "").strip()
            ms = (time.perf_counter() - t0) * 1000
            kind = "medical" if "医" in answer else ("chat" if "聊" in answer else None)
            self._lat("intent_judge", model=self.intent_model, ms=round(ms, 1),
                      answer=answer, kind=kind, provider="ollama")
            return kind, ms
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            self._lat("intent_judge_error", model=self.intent_model, ms=round(ms, 1),
                      error=str(exc)[:120], provider="ollama")
            return None, ms

    def _judge_intent(self, text: str):
        """轻量模型复核意图，返回 ("medical" | "chat" | None, 耗时 ms)。

        只在关键词判为"闲聊"时调用；超时或报错返回 (None, ms)，调用方回退关键词结果。
        """
        if self.intent_provider == "ollama":
            return self._judge_intent_local(text)
        t0 = time.perf_counter()
        try:
            resp = self.judge_client.chat.completions.create(
                model=self.intent_model,
                messages=[
                    {"role": "system", "content": INTENT_JUDGE_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=2,
                timeout=self.intent_judge_timeout,
            )
            answer = (resp.choices[0].message.content or "").strip()
            took_ms = (time.perf_counter() - t0) * 1000.0
            kind = None
            if "医" in answer:
                kind = "medical"
            elif "聊" in answer or "无关" in answer:
                kind = "chat"
            self._lat(
                "intent_judge",
                model=self.intent_model,
                ms=round(took_ms, 1),
                answer=answer[:6],
                kind=kind,
            )
            return kind, took_ms
        except Exception as exc:
            took_ms = (time.perf_counter() - t0) * 1000.0
            self._lat(
                "intent_judge_error",
                model=self.intent_model,
                ms=round(took_ms, 1),
                error=str(exc)[:80],
            )
            return None, took_ms

    # ------------------------------------------------------------------
    # ---------- 网关入口与内嵌服务（替代 router/network/history/end_session） ----------
    # ------------------------------------------------------------------
    def question_asr_callback(self, msg: String) -> None:
        """/question_asr 入口：剥掉 ASR 旧提示后缀、同音错字纠正后进原有流程。"""
        incoming = (msg.data or "").strip()
        raw = _HINT_RE.sub("", incoming).strip()
        if raw != incoming:
            self.get_logger().info(
                f"剥离 ASR 提示后缀：{incoming[:40]} -> {raw[:40]}")
        if not raw:
            return
        self._round_start_published = False
        self._ensure_start_published()   # 填充词立即响（纠错/扩展与播放并行）
        fixed = _TYPO_PATTERN.sub(lambda m: _TYPO_MAP[m.group(0)], raw)
        fixed = _WAKE_RE.sub("小医", fixed)
        # 第二层：LLM 语义纠错（qwen-flash，~0.3-0.5s，MED_ASR_CORRECT=0 可关）
        corrected = self._llm_correct_asr(fixed)
        if corrected != fixed:
            self.get_logger().info(f"语义纠错：{fixed} -> {corrected}")
            fixed = corrected
        if fixed != raw:
            self.get_logger().info(f"错别字修正：{raw} -> {fixed}")
        self.query_callback(String(data=fixed))

    def _append_history(self, question: str, response: str) -> None:
        msg = LlmDialogHistoryMsg()
        msg.question = question
        msg.response = response
        self._on_dialog_msg(msg)

    def _on_dialog_msg(self, msg: LlmDialogHistoryMsg) -> None:
        """内嵌历史存储（原 dialog_history 节点逻辑）：保留最近 5 组。"""
        if msg.question or msg.response:
            self._history.append(msg)
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max:]

    def _on_dialog_finished(self, msg: DialogSessionFinished) -> None:
        """ASR 判定会话结束（结束语/呼叫护士/超时）：清空上下文与轮次状态。"""
        n = len(self._history)
        self._history = []
        self._session_epoch += 1
        self._round_start_published = False
        self._partial_text = ""
        self.get_logger().info(
            f"收到会话结束(call_nurse={bool(msg.call_nurse)}, "
            f"reason={str(msg.reason)[:30]})：清空历史 {n} 组并重置轮次"
            f"(epoch={self._session_epoch})")

    def _handle_get_history(self, req, res):
        """服务口径与原 dialog_history 节点一致（clear_after_get 语义保留）。"""
        res.history = list(self._history)
        if getattr(req, "clear_after_get", False):
            self._history = []
        return res

    def _handle_end_session(self, request, response):
        """内嵌 /check_end_dialog（原 check_end_session 节点逻辑，云端判结束意图）。"""
        question = request.question
        try:
            completion = self.client.chat.completions.create(
                model="qwen3.5-flash",
                messages=self._end_session_history + [
                    {"role": "user", "content": question}],
                stream=False,
                extra_body={"enable_thinking": False},
            )
            reply = completion.choices[0].message.content
            if reply is None:
                raise ValueError("模型返回为空")
            result = json.loads(reply)
            response.need_end = result.get("need_end", False)
            response.comment = result.get("reason", "")
            response.response = result.get("response", "")
            if response.need_end:
                # 结束对话后清空上下文，下一次唤醒从干净状态开始
                self._history = []
                self._session_epoch += 1
                self._round_start_published = False
                self.get_logger().info("check_end_dialog=结束，已清空对话历史")
            self.get_logger().info(
                f"check_end_dialog Q: {question} | need_end: {response.need_end} "
                f"| {response.comment}")
        except Exception as exc:
            self.get_logger().error(f"check_end_dialog 失败: {exc}")
            response.need_end = False
            response.comment = "[Error] Failed to query LLM"
            response.response = ""
        return response

    # ------------------------------------------------------------------
    def _rag_expand(self, question: str) -> str:
        """LLM 查询扩展：口语问句 → 检索关键词（BM25 词面无重叠的语义改述
        如 睡不着→失眠/情绪、小豆子→耳穴压豆）。超时/失败返回空串。
        MED_RAG_EXPAND=0 关闭；MED_AUX_LOCAL=1 时走本地 llama.cpp（~160ms），
        失败自动回退云端；延迟与填充词播放重叠，不增加首音。"""
        if os.environ.get("MED_RAG_EXPAND", "1") != "1":
            return ""
        if len(question) > 60:
            return ""
        try:
            kw = self._aux_chat(
                [
                    {"role": "system", "content": (
                        "把血透患者的问题改写成 2~4 个知识库检索关键词（逗号分隔）。"
                        "只输出关键词。领域：内瘘、透析、血钾、血磷、干体重、穿刺、"
                        "震颤、导管、人工血管、耳穴压豆、中医、情绪、失眠、运动、饮食。"
                        "示例：'总睡不着心里烦' → '失眠, 情绪, 心理'")},
                    {"role": "user", "content": question},
                ],
                max_tokens=30, timeout=1.0,
                prefer_local=self.rag_expand_local)
            return kw.replace("\n", " ")[:40]
        except Exception:
            return ""

    # ------------------------------------------------------------------
    def _rag_retrieve(self, question: str):
        """调用本地 RAG，返回 (chunks, took_ms)；失败返回 ([], None)。

        RAG_MODE=bm25（默认）：进程内纯 Python BM25 检索（rag_bm25.py），
        实测 Recall@4=100%（23 题评测），省掉 rag_service 的 BGE-M3 GPU
        进程（~1.3GB 显存）。RAG_MODE=http：回退旧 127.0.0.1:8765 服务。
        """
        if not self.rag_enabled:
            return [], None
        if os.environ.get("RAG_MODE", "bm25") == "bm25":
            t0 = time.perf_counter()
            try:
                chunks = rag_bm25.retrieve(question, top_k=self.rag_top_k)
                # LLM 查询扩展：口语改述与知识库词面无重叠时（睡不着→失眠）
                # 用扩展词补检索，合并去重（原词结果优先保序）
                expanded = self._rag_expand(question)
                if expanded and expanded != question:
                    extra = rag_bm25.retrieve(expanded, top_k=max(2, self.rag_top_k - 2))
                    seen = {(c.get("doc_id"), c.get("heading_path")) for c in chunks}
                    for c in extra:
                        key = (c.get("doc_id"), c.get("heading_path"))
                        if key not in seen:
                            chunks.append(c)
                            seen.add(key)
                    chunks = chunks[: self.rag_top_k]
                took_ms = (time.perf_counter() - t0) * 1000.0
                return chunks, took_ms
            except Exception as exc:
                self._lat("rag_error", error=str(exc)[:120])
                self.get_logger().warning(f"BM25 检索失败（将不带资料回答）: {exc}")
                return [], None
        url = self.rag_url
        if not url.endswith("/retrieve"):
            url = url + "/retrieve"
        payload = json.dumps(
            {"query": question, "top_k": self.rag_top_k}, ensure_ascii=False
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.rag_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            took_ms = (time.perf_counter() - t0) * 1000.0
            chunks = data.get("chunks") or []
            return chunks, took_ms
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            self._lat("rag_error", error=str(exc)[:120])
            self.get_logger().warning(f"RAG 检索失败（将不带资料回答）: {exc}")
            return [], None

    # ------------------------------------------------------------------
    def _get_history(self):
        """读取内嵌对话历史，返回 [(question, response), ...]（原为跨进程服务调用）。"""
        return [(item.question, item.response) for item in self._history]

    # ------------------------------------------------------------------
    def _build_user_prompt(self, question: str, chunks) -> str:
        if not chunks:
            return (
                "【参考资料】\n（本轮没有检索到相关资料）\n\n"
                f"【患者问题】{question}\n\n"
                "注意：没有资料时不要编造，按系统规则回答（说明资料里没有、建议咨询护士或医生）。"
            )
        lines = ["【参考资料】"]
        for idx, chunk in enumerate(chunks, 1):
            title = chunk.get("title") or chunk.get("doc_id") or "资料"
            text = (chunk.get("text") or "").strip().replace("\n", " ")
            if len(text) > self.rag_chunk_max_chars:
                text = text[: self.rag_chunk_max_chars] + "…"
            lines.append(f"[{idx}]（来源：{title}）{text}")
        lines.append("")
        lines.append(f"【患者问题】{question}")
        lines.append("")
        lines.append(
            "注意：只依据上述资料回答；资料和问题无关时按系统规则说明资料里没有。"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _stream_direct(self, messages, model: str = None, enable_search: bool = False,
                       **gen_kwargs) -> str:
        """按 MED_LLM_PROVIDER 选择接入方式，流式生成并逐块发布到 /ali_llm_output。

        model 不为空时可临时覆盖（闲聊用更小的本地模型）。
        """
        model_used = model or self.model_name
        self._request_model = model_used
        backup = os.environ.get("MED_LLM_BACKUP_MODEL", "qwen-flash")
        failover = (os.environ.get("MED_LLM_FAILOVER", "1") == "1" and bool(backup)
                    and backup != model_used and self.provider == "openai")
        try:
            if self.provider == "dashscope":
                return self._stream_dashscope(messages, enable_search=enable_search)
            return self._stream_openai(messages, allow_fallback=not failover,
                                       **gen_kwargs)
        except Exception as primary_exc:
            if not failover:
                raise
            self.get_logger().warn(
                f"主模型 {model_used} 失败({str(primary_exc)[:80]})，"
                f"自动切换备用模型 {backup}")
            try:
                self._request_model = backup
                out = self._stream_openai(messages, allow_fallback=False,
                                          publish_start=False,
                                          skip_prefix=getattr(self, "_partial_text", ""),
                                          **gen_kwargs)
                self._lat("llm_failover", model=backup)
                self.get_logger().info(f"备用模型 {backup} 回答成功 chars={len(out)}")
                return out
            except Exception as backup_exc:
                self.get_logger().error(
                    f"备用模型 {backup} 也失败({str(backup_exc)[:80]})")
                raise primary_exc

    # ------------------------------------------------------------------
    def _stream_dashscope(self, messages, enable_search: bool = False) -> str:
        """dashscope SDK 原生接口（MaaS 专有部署，MultiModalConversation，流式增量输出）。

        注意：这是思考模型，必须 enable_thinking=False，否则先吐 reasoning_content，
        首句要等推理结束（实测 80 token 内甚至一个字内容都没有）。
        """
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        ds_messages = [
            {"role": m.get("role", "user"), "content": [{"text": m.get("content", "")}]}
            for m in messages
        ]
        q: "queue.Queue" = queue.Queue()

        def worker():
            try:
                responses = MultiModalConversation.call(
                    model=self._request_model,
                    api_key=self.api_key,
                    messages=ds_messages,
                    stream=True,
                    incremental_output=True,
                    enable_thinking=self.enable_thinking,
                    temperature=0.6,
                    top_p=0.8,
                    max_tokens=self.max_tokens,
                    **({"enable_search": True} if enable_search else {}),
                )
                for resp in responses:
                    status = getattr(resp, "status_code", 200)
                    if status != 200:
                        q.put(
                            (
                                "err",
                                f"status={status} "
                                f"{getattr(resp, 'code', '')} "
                                f"{getattr(resp, 'message', '')}",
                            )
                        )
                        return
                    msg = resp.output.choices[0].message
                    content = msg.get("content") or []
                    if isinstance(content, str):
                        text = content
                    else:
                        text = "".join(
                            c.get("text", "")
                            for c in content
                            if isinstance(c, dict)
                        )
                    if text:
                        q.put(("data", text))
                q.put(("done", None))
            except Exception as exc:  # 网络/鉴权/参数错误都在这里收口
                q.put(("err", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()
        self._lat(
            "llm_request",
            provider="dashscope",
            model=self._request_model,
            prompt_chars=prompt_chars,
            n_messages=len(messages),
        )

        self._ensure_start_published()
        full_response = ""
        first_chunk = True
        error_detail = None
        while True:
            # 首块等 llm_timeout；之后每块最多 5 s，避免半截卡死
            wait_s = self.llm_timeout if first_chunk else 5.0
            try:
                kind, payload = q.get(timeout=wait_s)
            except queue.Empty:
                error_detail = f"timeout({wait_s:.0f}s 无新块)"
                break
            if kind == "data":
                content = _EMOJI_RE.sub("", payload)
                content = _QUOTE_RE.sub("", content)
                if not content or not content.strip():
                    continue
                if first_chunk:
                    self._lat(
                        "llm_first_chunk",
                        provider="dashscope",
                        prompt_chars=prompt_chars,
                    )
                    first_chunk = False
                full_response += content
                if _diverged:
                    self._publish_stream(content)
                    continue
                # 前缀去重：备用通道从与已播前缀的分歧处续播（避免重播开头）
                cp = 0
                m = min(len(full_response), len(skip_prefix))
                while cp < m and full_response[cp] == skip_prefix[cp]:
                    cp += 1
                if cp == len(full_response) and len(full_response) <= len(skip_prefix):
                    continue
                _diverged = True
                out = (full_response[len(skip_prefix):] if cp >= len(skip_prefix)
                       else full_response[cp:])
                if out:
                    self._publish_stream(out)
            elif kind == "err":
                error_detail = payload
                break
            else:  # done
                break

        if error_detail:
            self._lat("llm_error", backend="dashscope", error=str(error_detail)[:160])
            self.get_logger().error(f"dashscope 流式失败: {error_detail}")
            if not full_response:
                fallback = "抱歉，我这会儿有点忙，请稍后再问一次。"
                self._publish_stream(fallback)
                full_response = fallback
        self._lat("llm_done", chars=len(full_response))
        return full_response

    # ------------------------------------------------------------------
    def _stream_openai(self, messages, temperature: float = 0.6,
                       frequency_penalty: float = None, max_tokens: int = None,
                       allow_fallback: bool = True,
                       publish_start: bool = True,
                       skip_prefix: str = "") -> str:
        """OpenAI 兼容接口路径（qwen-plus 直连）。

        allow_fallback: 失败时是否就地播兜底话术（False=抛异常，由双通道切换接管）
        publish_start: 是否发布 [START]（备用通道重试时主通道已发过，避免重复）"""
        if self.provider == "ollama" and self.no_think:
            # qwen3 的思考开关：系统提示带上 /no_think 才不推理（实测首块 107 ms vs 无内容）
            messages = [dict(m) for m in messages]
            for m in messages:
                if m.get("role") == "system":
                    m["content"] = m["content"] + "\n/no_think"
                    break
            else:
                messages.insert(0, {"role": "system", "content": "/no_think"})
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        full_response = ""
        started = False
        # llm_request 必须记在发起请求之前：ollama 会先把首 token 备好再回响应头，
        # 若记在 create() 之后，TTFT 会被算成 ~0 ms。
        self._lat(
            "llm_request",
            provider=self.provider,
            prompt_chars=prompt_chars,
            n_messages=len(messages),
        )
        try:
            # 先发 [START] 再发云端请求：TTS 的填充词"好的，"不必等云端建连
            #（实测云端建连等待中位 1.44s、P90 3.2s、最大 10s——提前发布让
            # 首响从 ~1.4s 降到 ~0.1-0.3s；请求失败时错误路径会补发兜底话术）
            if publish_start:
                self._ensure_start_published()
            started = True
            completion = self.client.chat.completions.create(
                model=self._request_model,
                messages=messages,
                stream=True,
                temperature=temperature,
                **({"frequency_penalty": frequency_penalty} if frequency_penalty is not None else {}),
                top_p=0.8,
                max_tokens=self.max_tokens,
                timeout=self._http_timeout,
                # qwen3 系列混合思考模型必须显式关思考（否则先吐 reasoning 内容）。
                # 云端认顶层 enable_thinking；本地 llama.cpp 认 chat_template_kwargs。
                extra_body={
                    "enable_thinking": self.enable_thinking,
                    **({"chat_template_kwargs": self.chat_template_kwargs}
                       if self.chat_template_kwargs else {}),
                },
            )
            first_chunk = True
            _diverged = not skip_prefix
            for chunk in completion:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if not content:
                    continue
                content = _EMOJI_RE.sub("", content)
                content = _QUOTE_RE.sub("", content)
                if not content or not content.strip():
                    continue
                if first_chunk:
                    self._lat("llm_first_chunk", prompt_chars=prompt_chars)
                    first_chunk = False
                full_response += content
                self._publish_stream(content)
            self._lat("llm_done", chars=len(full_response))
            return full_response
        except Exception as exc:
            # 网络超时/中断：播兜底话术并正常结束本段，避免 TTS 等不到 [DONE] 卡满超时
            self._lat("llm_error", backend="openai", error=str(exc)[:120])
            self.get_logger().error(f"LLM 流式失败({type(exc).__name__}): {exc}")
            self._partial_text = full_response  # 供备用通道前缀去重
            if not allow_fallback:
                raise
            fallback = "抱歉，我这会儿有点忙，请稍后再问一次。"
            if not started:
                self._ensure_start_published()
            if not full_response:
                self._publish_stream(fallback)
                full_response = fallback
            self._lat("llm_done", chars=len(full_response))
            return full_response

    # ------------------------------------------------------------------
    def _stream_app(self, question: str, history):
        """旧版百炼应用路径（A/B 对比 / 回退用）。"""
        app_history = [
            {
                "user": (
                    "有几个血透中心的病人，现在在血透中心做透析，你是巡诊的机器人，名字叫小医。"
                    "你的大脑跟病人的测量数据是实时联系在一起的，下面几个病人开始对话。"
                    "对话的时候不要长篇大论，像真人一样，使用对话句子，也就是我们一句来一句去这样。"
                    "语气需要轻柔舒缓。所有使用单位名称转化为中文。尽可能不出现英文。"
                    "你只需要监测，禁止输出执行实际动作,例如倒水。不要增加小动作。不要输出表情符号。"
                    "在需要时，给出数据偏高或偏低，不需要具体数据。不要给出实际的执行动作，"
                    "而是不用通过类似读取的方式。讲话必须亲切，温柔。科普类的内容可以丰富一些。"
                    "叫护士或医生时回答好的，我马上帮您联系护士或医生。您先安心休息一会儿。"
                    "禁止在回答中出现着急两个字。问题不严重时，你来提出解决办法，不要经常输出寻找医生。"
                    "询问天气/温度等关键词，仅输出杭州本地核心信息."
                ),
                "bot": (
                    "好的，我记住了,我的名字是小医，我会采用真人对话风格那样进行对话，"
                    "并使用轻柔温和的语气，我不会输出执行动作，不会输出表情符合，"
                    "不会给出具体的数值，问题不严重时我会自己提出解决办法不会寻求医生的帮助，"
                    "不会使用'着急'等急迫性质的词语。"
                ),
            }
        ]
        for item in history:
            app_history.append({"user": item[0], "bot": item[1]})

        # 百炼应用走公网原生接口；主模型若接在 MaaS 地址上，这里临时切回来
        prev_base = dashscope.base_http_api_url
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
        try:
            responses = Application.call(
                api_key=self.api_key,
                app_id=self.app_id,
                prompt=question,
                history=app_history,
                stream=True,
                incremental_output=True,
            )
            responses = list(responses)
        finally:
            dashscope.base_http_api_url = prev_base

        self._ensure_start_published()
        full_response = ""
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                chunk = response.output.text
                full_response += chunk
                self._publish_stream(chunk)
            else:
                self.get_logger().error(f"LLM response error: {response}")
                break
        return full_response

    # ------------------------------------------------------------------
    def handle_query(self, question: str):
        full_response = ""
        round_epoch = self._session_epoch
        self._tls.epoch = round_epoch
        try:
            # ---------- 意图分流：关键词粗判 + 快模型复核 ----------
            # 关键词判"闲聊"时，用 qwen-flash 再确认一次（与历史获取并行，约 0.3 s），
            # 避免"那条管能不能碰水"这类不含标准术语的医学问题被当闲聊放过去。
            keyword_medical = _is_medical(question)
            chitchat = (not keyword_medical) and _looks_like_chitchat(question)
            judge_holder = {}
            judge_thread = None
            if self.intent_judge_enabled and not keyword_medical and not chitchat:
                def _run_judge():
                    judge_holder["kind"], judge_holder["ms"] = self._judge_intent(question)

                judge_thread = threading.Thread(target=_run_judge, daemon=True)
                judge_thread.start()

            history = self._get_history()

            if judge_thread is not None:
                judge_thread.join(timeout=self.intent_judge_timeout + 0.5)
            judge_kind = judge_holder.get("kind")
            judge_ms = judge_holder.get("ms", 0.0) or 0.0

            is_medical = keyword_medical or (judge_kind == "medical")
            if keyword_medical:
                intent_source = "keyword"
            elif chitchat:
                intent_source = "chitchat"
            elif judge_kind:
                intent_source = f"llm:{self.intent_model}"
            else:
                intent_source = "keyword-fallback"
            chat_mode = (not is_medical) and self.chat_enabled
            self._lat(
                "intent",
                kind="medical" if is_medical else "chat",
                chat_mode=chat_mode,
                source=intent_source,
                keyword_medical=keyword_medical,
                chitchat=chitchat,
                judge_kind=judge_kind,
                judge_ms=round(judge_ms, 1),
            )
            self.get_logger().info(
                f"intent={'medical' if is_medical else 'chat'} chat_mode={chat_mode} "
                f"source={intent_source} keyword={keyword_medical} chitchat={chitchat} "
                f"judge={judge_kind} "
                f"judge_ms={judge_ms:.0f}"
            )

            # ---------- 本地 RAG 检索（仅医学模式） ----------
            chunks, rag_ms = [], None
            if self.rag_enabled and not chat_mode:
                self._lat("rag_start")
                chunks, rag_ms = self._rag_retrieve(question)
                self._lat(
                    "rag_end",
                    n_chunks=len(chunks),
                    took_ms=round(rag_ms, 2) if rag_ms is not None else None,
                    top1=(
                        (chunks[0].get("title") or chunks[0].get("doc_id"))
                        if chunks
                        else None
                    ),
                )
                top_titles = [
                    (c.get("title") or c.get("doc_id") or "") for c in chunks[: self.rag_top_k]
                ]
                self.get_logger().info(
                    f"RAG: {len(chunks)} chunks, {rag_ms:.1f}ms, top={top_titles}"
                )

            if self.backend == "app":
                full_response = self._stream_app(question, history)
            else:
                # ---------- 直连 qwen-plus ----------
                # 闲聊模式：换轻量人设、不带参考资料；医学模式：严格按资料回答
                system_prompt = MED_CHAT_PROMPT if chat_mode else MED_SYSTEM_PROMPT
                messages = [{"role": "system", "content": system_prompt}]
                for q, a in history[-self.history_pairs:]:
                    a = (a or "")[:120]
                    if q:
                        messages.append({"role": "user", "content": q})
                    if a:
                        messages.append({"role": "assistant", "content": a})
                if chat_mode:
                    user_content = question
                else:
                    user_content = self._build_user_prompt(question, chunks)
                messages.append({"role": "user", "content": user_content})
                try:
                    model_override = self.chat_model if (chat_mode and self.chat_model) else None
                    gen = {"temperature": 0.85, "frequency_penalty": 0.4,
                           "max_tokens": self.chat_max_tokens} if chat_mode else {}
                    # 天气类闲聊走 dashscope 联网搜索（原 llm_network 节点的独有能力）
                    if chat_mode and self.provider == "dashscope" and (
                            "天气" in question or "气温" in question):
                        gen["enable_search"] = True
                    full_response = self._stream_direct(messages, model=model_override, **gen)
                except Exception as direct_exc:
                    self._lat("llm_error", backend="direct", error=str(direct_exc)[:120])
                    if self.backend == "auto" or self.app_fallback:
                        self.get_logger().error(
                            f"直连失败，回退百炼应用: {direct_exc}"
                        )
                        full_response = self._stream_app(question, history)
                    else:
                        raise

            if round_epoch == self._session_epoch:
                self.pub.publish(String(data="[DONE]"))
                self._round_start_published = False
            else:
                # 旧轮次的 [DONE] 不能发进新会话，否则 TTS 会把新一轮提前收尾
                self.get_logger().info("[DONE] 已过期（会话已结束），丢弃")

            # 完整问答写入内嵌对话历史（原 dialog_history 节点职责）。
            # 若本轮进行中会话已结束（epoch 变化），丢弃，避免结束语那轮
            # 在清空后又写回历史，污染下一次唤醒。
            if round_epoch == self._session_epoch:
                self._append_history(question, full_response.strip())
                self.get_logger().info(
                    f"Dialog pair appended to history. backend={self.backend} chars={len(full_response)}"
                )
            else:
                self.get_logger().info(
                    f"会话已结束(epoch {round_epoch}->{self._session_epoch})，"
                    f"丢弃本轮历史 chars={len(full_response)}")

        except Exception as exc:
            self.get_logger().error(f"Error querying KnowledgeBase LLM: {exc}")
            self._lat("llm_error", error=str(exc)[:120])
            if round_epoch == self._session_epoch:
                self._ensure_start_published()
                self.pub.publish(String(data="抱歉，我这会儿有点忙，请稍后再问一次。"))
                self.pub.publish(String(data="[DONE]"))
                self._round_start_published = False
            else:
                self.get_logger().info("旧轮次异常，会话已结束，丢弃兜底播报")


def main(args=None):
    rclpy.init(args=args)
    node = KnowledgeBaseLLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
