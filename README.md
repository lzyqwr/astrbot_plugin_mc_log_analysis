# MC日志分析（astrbot_plugin_mc_log_analysis）

自动接收并分析 Minecraft 错误报告/日志文件，调用 LLM 输出结构化排障结论，并以图片或文本方式返回。

## 功能特性

- 自动识别并处理 QQ/OneBot 文件消息。
- 支持压缩包：`.zip`、`.gz`。
- 支持文本日志：`.txt`、`.log`。
- 三种提取策略（A/B/C）自动选择，避免把整份原始日志全量喂给模型。
- Map/最终分析可分别选择不同 Provider（`select_provider`）。
- 内置工具：
  - `search_mcmod`（mcmod.cn）
  - `search_minecraft_wiki`（minecraft.wiki）
- 支持两种渲染模式：
  - `html_to_image`（模板渲染）
  - `text_to_image`（Pillow）
- 输出前自动脱敏（IP、邮箱、路径、token 等）。
- 默认按“证据优先”输出（结论与步骤均要求证据编号，证据不足时标注 `UNCERTAIN`）。
- 调试日志落盘前自动脱敏，避免敏感原文持久化。
- 脱敏占位符采用确定性规则：`<IP>` / `<LAN_IP>` / `<HOST>` / `<PATH>` / `<USER>` / `<UUID>` / `<EMAIL>` / `<TOKEN>`。
- 输出的“解决步骤”默认要求：可逆性提醒、风险等级（低/中/高）与端区分（Client/Server/Both），并按低→高风险排序。
- 工具失败或超时时，仍会基于日志证据继续分析，并明确提示工具不可用/未搜到。

## 触发规则

插件仅在收到文件消息且文件名命中规则时触发：

- 压缩包：扩展名为 `.zip` 或 `.gz`，且文件名包含 `错误报告` / `日志` / `log`
- 文本文件：扩展名为 `.txt` 或 `.log`，且文件名包含 `crash` / `hs_err` / `latest` / `debug` / `fcl` / `pcl`

## 快速开始

1. 安装插件到 AstrBot 插件目录并重载。
2. 在插件配置中至少填写：
   - `map_select_provider`
   - `analyze_select_provider`
3. 发送符合规则的日志文件到机器人会话。

## 配置说明（重点）

### 主配置

| 配置项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `chunk_size` | `int` | `100000` | Map-Reduce 分块字符数 |
| `global_timeout_sec` | `int` | `180` | 全流程超时秒数（从命中文件消息开始计时） |
| `rate_limit_user_sec` | `int` | `60` | 按用户维度限频间隔（秒），0 表示不限制 |
| `queue_wait_sec` | `int` | `8` | 全局并发队列等待时间（秒） |
| `max_concurrent_jobs` | `int` | `2` | 全局并发处理上限 |
| `max_concurrent_io` | `int` | `1` | 重 I/O 阶段并发上限（解压/大文件读写/清理） |
| `max_input_file_bytes` | `int` | `33554432` | 单次上传/下载文件最大字节数 |
| `max_archive_file_count` | `int` | `160` | 压缩包内最大文件数量 |
| `max_archive_single_file_bytes` | `int` | `12582912` | 压缩包内单文件最大字节数 |
| `max_archive_total_bytes` | `int` | `33554432` | 压缩包解压后总字节数上限 |
| `max_gz_output_bytes` | `int` | `16777216` | gz 解压后最大字节数 |
| `must_keep_window_lines` | `int` | `30` | 强制保留证据窗口行数（上下各 N 行） |
| `diag_version` | `string` | `1.0.0` | 诊断系统版本号 |
| `metrics_enabled` | `bool` | `true` | 是否启用结构化指标落库 |
| `metrics_path` | `string` | `audit_metrics.jsonl` | 指标落库文件名（相对 data 目录） |
| `map_select_provider` | `string(select_provider)` | `""` | Map 阶段模型提供商（必填） |
| `analyze_select_provider` | `string(select_provider)` | `""` | 最终分析阶段模型提供商（必填） |
| `render_mode` | `string` | `html_to_image` | 渲染模式：`html_to_image` / `text_to_image` |
| `image_width` | `int` | `640` | 渲染图片宽度（像素） |
| `full_read_char_limit` | `int` | `140000` | hs_err/crash 全量读取阈值 |
| `total_char_limit` | `int` | `140000` | 进入 LLM 的总字符上限 |
| `max_tool_calls` | `int` | `6` | 单次分析最大工具调用次数 |
| `tool_timeout_sec` | `int` | `120` | 单次工具调用超时秒数 |
| `tool_retry_limit` | `int` | `1` | 单次工具调用失败重试上限 |
| `tool_snippet_chars` | `int` | `600` | 工具结果进入模型前的截断字符数 |
| `map_llm_timeout_sec` | `int` | `120` | Map 阶段单片段 LLM 超时秒数 |
| `analyze_llm_timeout_sec` | `int` | `240` | 最终分析 LLM 超时秒数 |
| `max_map_chunks` | `int` | `10` | Map 阶段最多送入 LLM 的片段数 |
| `map_timeout_break_threshold` | `int` | `2` | Map 阶段 LLM 降级累计阈值 |
| `skip_final_analyze_on_map_timeout` | `bool` | `true` | 触发阈值后是否跳过最终分析 |
| `html_render_timeout_sec` | `int` | `30` | HTML 渲染超时秒数 |
| `file_download_timeout_sec` | `int` | `30` | 文件下载超时秒数 |
| `messages` | `object` | `{...}` | 用户可见文案配置 |

### `messages` 子项

| 配置项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `messages.accepted_notice` | `string` | `已接收文件，正在分析，请稍候。` | 命中文件后立即提示 |
| `messages.download_failed` | `string` | `日志文件下载失败，请稍后重试。` | 文件下载失败提示 |
| `messages.no_extractable_content` | `string` | `未在文件中识别到可分析的日志内容。` | 未识别到可分析内容提示 |
| `messages.analyze_failed_logged` | `string` | `日志分析失败，已记录日志，请联系管理员检查检查。` | LLM 分析失败提示 |
| `messages.analyze_failed_retry` | `string` | `日志分析失败，请联系管理员检查。` | 流程异常提示 |
| `messages.prompt_missing` | `string` | `日志分析模板缺失，请联系管理员检查 assets 目录。` | Prompt 缺失提示 |
| `messages.provider_not_configured` | `string` | `请先在插件配置中填写 map_select_provider 与 analyze_select_provider。` | Provider 未配置提示 |
| `messages.global_timeout` | `string` | `日志分析超时，请稍后重试。` | 全流程超时提示 |
| `messages.html_render_fallback_notice` | `string` | `[提示] HTML 渲染失败，已降级发送原始文本。` | HTML 渲染失败附加提示 |
| `messages.text_render_fallback_notice` | `string` | `[提示] 渲染不可用，已降级为纯文本发送。` | 文本图渲染失败附加提示 |
| `messages.forward_sender_name` | `string` | `MC日志分析` | 合并转发节点显示名称 |
| `messages.summary_template` | `string` | `分析完成，耗时 {elapsed:.2f} 秒\n文件: {source_name}\n策略: {strategy}` | 分析简报模板 |

完整配置字段见：`_conf_schema.json`

## 可自定义提示词与模板

- 最终分析提示词：
  - `assets/analyze_system.txt`
  - `assets/analyze_user.txt`
- Map 提示词：
  - `assets/map_system.txt`
  - `assets/map_user.txt`
- HTML 渲染模板：
  - `assets/html_to_image.html.j2`

## 返回内容

默认返回一个合并转发消息，包含：

1. 分析简报（耗时、文件名、策略）
2. 分析结果（图片或文本）

## 常见问题

- 提示“请先在插件配置中填写 map_select_provider 与 analyze_select_provider”
  - 说明两个 Provider 未配置或 ID 不存在。

- 提示模板缺失
  - 检查 `assets` 下 4 个 prompt 文件是否存在且非空。

- HTML 渲染失败
  - 会自动降级为文本发送，并写入日志。
  - 若你使用本地 text2img 服务，建议确认服务版本与截图参数兼容。
