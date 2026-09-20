# 实战失败教训(Pitfalls)

本文件汇总训推一致性采集/分析流程中踩过的坑,按阶段组织。每条给「症状 → 根因 → 修复 → 预防」。真实案例来源:2026-09-10 GLM-4.7 失败尝试、2026-09-11 卡 6,7 成功重跑、2026-09-14 卡 6,7(裁剪后 loss 崩溃)、2026-09-15 卡 6,7 确定性单样本(补丁嵌套引号 / prompt_logprobs 嵌套结构 / 漏 rm-type / ray job 拆 JSON)、2026-09-15 卡 6,7 5 步重跑(log_rollout_data 诊断字段 sum 崩 / train_rollout_logprob_abs_diff 形状不匹配)、2026-09-15 晚 5 步重跑(让 train_rollout_logprob_abs_diff 可算:P17 改对齐 prompt 段计算,替代"跳过")、2026-09-16 卡 1,2(ray job 未解析工作目录 vime_plugins → AutoBridge 拒架构,先于 OOM)、2026-09-17 卡 0,1(Megatron-Bridge TE 硬依赖假阻塞 P21 / msprobe 替换 F.silu 的引用缓存连锁 P20 / 孤儿 EngineCore 清理 P6 补充)、2026-09-20 卡 0,1 打桩闭环 r1-r3(ray Session name mismatch P22 / No available agent to submit job P23 / AF_UNIX socket 路径 107 字节上限 P24)。后续各框架/版本出现同类症状时优先对照这里。

## 阶段 4:输入对齐(PROMPTS_ONLY 裁剪)

### P1. 裁剪副本不能替换整个 rollout_data
- **症状**:`compute_advantages_and_returns` 崩 `split_with_sizes ... got split_sizes=[0]`;或 `log_rollout_data` 崩。
- **根因**:把 `rollout_data` 整体替换成 prompt-only 副本后,`response_lengths` 全为 0,下游按 response_lengths split 的统计 tensor 变成空。
- **修复**:裁剪副本**只喂给 `get_data_iterator`**(compute_log_prob 主前向的输入),原始 `rollout_data` 保留给下游;adv/correction 与统计路径按官方改法处理——`rollout_corr_helper` 在 `PROMPTS_ONLY=1` 时直接 `return batch, {}` 早退,loss/metrics 同步兜底,**训练照常走完**(见 P6b);`log_rollout_data` 这类依赖完整 response 段的日志调用可跳过。
- **预防**:PROMPTS_ONLY 裁剪只允许影响「log_prob 前向输入」,禁止动 advantages/统计路径;改动后先 `git diff` 看清改了几处。

### P2. loss_masks 裁剪必须 `m[:0]` 再依赖 pad 展开
- **症状**:`assert loss_masks.shape == tokens.shape` 不通过,shape [1,260] vs [1,256]。
- **根因**:data.py 里 `F.pad(loss_mask, (prompt_length-1, 1), value=0)` 会把 loss_mask 展开成 prompt_length 长度;若裁剪时没把 response 段去掉(保留原始 260),pad 后超长。
- **修复**:`loss_masks = [m[:0] for m in loss_masks]`(裁成空),再依赖 pad 展开成 prompt_length。校验数学:`prompt_length = total_length - response_length`,`F.pad(mask, (prompt_length-1, 1))` 长度 = prompt_length。
- **预防**:改裁剪逻辑时先把长度数学算一遍,再对照 data.py 的 pad 行。

### P3. response_lengths=0 时 log_prob 返回空 tensor
- **症状**:compute_log_prob 返回的空 tensor(`log_prob_full[255:255]`),下游以为有值。
- **修复**:采集模式显式跳过依赖 response 段的下游(log_rollout_data),不要把空 tensor 当正常产物。
- **预防**:裁剪方案设计时把「哪些下游会崩」列出来,提前跳过。

### P18. 训练端 padding 必须去掉,输入 shape 两侧对齐(190 vs 256)
- **症状**:两侧 dump 逐边界比对时,训练侧边界 shape [256,...]、推理侧同一边界 [190,...],形状对不上;或训练侧 prefill logprob 带 padding 位,与推理侧真实 token 数错位。
- **根因**:训练侧(Megatron 系)把序列 pad 到对齐长度(如 256)才进 prefill,dump 出来就是 pad 后形状;推理侧是真实 prompt 长度(如 190)。padding 是**训练侧单边**的,不去掉则两侧 dump 输入形状天然不等,逐 token 比对无从谈起。
- **修复**:**在输入对齐阶段就去掉 padding,不是分析阶段再处理**——裁剪成 prompt-only 后,按真实 prompt token 数截取训练侧输入与 logprob(`log_probs[:, :-1]` 前先按真实长度切),保证两侧 dump 输入 shape = 真实 prompt 长度、逐 token 1:1;TP>1 时两侧按同一 pad 规则对齐(见 SKILL 阶段 4 条目 3)。
- **预防**:采完先验两侧同一边界的输入 shape 是否相等且等于真实 prompt 长度;出现"一边 pad 后形状、一边真实长度"(如 [256,...] vs [190,...])回阶段 4 修掉重采,别把带 padding 的 dump 交接给分析环节。分析环节若已拿到带 padding 的 dump,对应边界只能标 `evidence_missing`,由 consistency-dump-analysis 处理。

## 阶段 5:接入补丁(改代码的代码)

### P11. 补丁脚本自身的字符串嵌套陷阱
- **症状**(2026-09-15):`apply_patch.py` 一跑就语法错误 / 字符串提前终止,反复 `PATCH_RESULT=FAIL`(本次 6 次)。
- **根因**:补丁脚本里用 `"""` 三引号包 old/new 字符串,而补丁内容本身也含 `"""`(被接入的目标函数有 docstring)或 f-string 花括号,字符串边界错乱,Python 提前截断/花括号求值报错。
- **修复**:补丁脚本所有 old/new 定界符统一用 `'''`(选内容里出现最少的引号类型);应用前先对本脚本 `py_compile` 再执行。
- **预防**:"修改代码的代码"必须先自检语法;写补丁脚本时,定界符避开目标代码中最常见的引号类型,别用默认 `"""`。

### P12. 推理侧 prompt_logprobs 返回 top-k 嵌套 dict,不是单层
- **症状**(2026-09-15):训练侧 `KeyError: 'logprob'` 崩溃(vllm_rollout 解析 prompt_logprobs 处)。
- **根因**:vLLM-ascend 的 `prompt_logprobs` 每个 position 的 entry 是 `{token_id: {"logprob": float, "rank": int, "decoded_token": str}, ...}`(**top-k 嵌套字典**,逻辑在 Rust 二进制 `vllm_router_rs.abi3.so`),position 0 无前序 token;不是文档示例里的单层 `{logprob: float}`。
- **修复**:容错解析——entry 是 dict 且含 `"logprob"` 直接取;是 dict-of-dicts 取概率最大的项(或首项)的 logprob;`None`/数值兜底为 0;解析前打一行 `[DUMP_DIAG]` 打印原始结构核对。
- **预防**:任何"引擎返回体"字段先打一行诊断确认结构再写解析,别照抄文档;KEY 假设错误时看原始返回(本次靠 `[DUMP_DIAG]` 定位)。

## 阶段 6:执行采集

### P4. msprobe step 语义:训练侧有效 dump 在 step0,不是 step1
- **症状**:训练 dump 的 step1 `debug.json` 数据为空/不完整。
- **根因**:训练侧 step0 = compute_log_prob **主前向**(完整),step1 = train 前向(未 flush,数据空)。另外,只有 `can_reuse_log_probs_in_loss == False` 时 compute_log_prob 才执行——不强制的话主前向被跳过,什么都采不到。
- **修复**:用 `--get-mismatch-metrics --custom-tis-function-path <vanilla_tis_function>` 强制 actor 主前向;验证时认准训练侧 step0 的 dump。
- **预防**:采集后先验证训练侧 step0 dump 非空、53 条(或预期数),再进分析。

### P5. docker exec 卡死 / exit 137(SIGKILL)
- **症状**:`docker exec` 超时或被 SIGKILL;pkill 清理时把自己杀了。
- **根因**:`pkill -f` 的匹配串匹配到执行命令的 bash 自身(命令字符串里含同样关键字)。
- **修复**:用正则技巧防自匹配,如 `pkill -f "[r]ay"`;长任务用 `docker exec -d` 分离模式;对特定进程(EngineCore)用精确 PID kill,别用宽泛 `pkill -f vllm`。
- **预防**:清理命令一律写成不自匹配的形式;能按 PID 就不按名字。

### P6. 残留 VLLM::EngineCore 占用显存 → 二次采集 OOM
- **症状**:第一次 job 崩溃后 EngineCore 残留,第二次 vLLM 引擎 OOM(free 只有几百 MiB)。
- **修复**:kill 残留 EngineCore + 清 ray,确认目标卡 HBM 回到基线值(如 910B4 空闲 ~2.9G/32G)再重跑。注意独立 PID namespace 环境:`npu-smi info` 显示宿主 PID,须进对应容器按 `comm=VLLM::EngineCor` 找容器内 PID 再 kill。
- **补充(2026-09-17)**:job 崩后 EngineCore 可能变**孤儿进程(PPID=1)**且不再带 ray 前缀,`ps aux | grep ray` 找不到它——但 npu-smi 显示目标卡 HBM 明显高于基线(如 21G/32G)。识别法:**全量扫持 davinci 句柄的进程**(★必须 readlink 看 fd 目标——`ls /proc/<pid>/fd` 目录条目是纯数字、`grep davinci` 查不到(2026-09-17 实测漏检两次);正确做法:`for fd in /proc/<pid>/fd/*; do readlink "$fd"; done | grep davinci`)→ `ps -o pid,ppid,etime,cmd -p <pid>` 确认身份(`VLLM::EngineCore`)→ 精确 `kill -9`。宽泛 pkill 对孤儿进程未必命中(命令行只剩 `VLLM::EngineCore`)。清完再验 HBM 回基线(910B4 空闲 ~2.9G)。
- **预防**:每次采集前/后检查目标卡 HBM 占用,有残留先清;把「卡基线占用」写进 task_card。

### P19. ray job 没解析工作目录的 vime_plugins → AutoBridge 拒架构(桥早已存在)
- **症状**(2026-09-16,卡 1,2):`ValueError: Model architecture 'XxxLiteForCausalLM' is not yet supported`(Megatron-Bridge `AutoBridge`),错误提示「去定义 bridge」。但工作目录 `/workspace/vime/vime-zjc/vime_plugins/megatron_bridge/` 里**已有**自定义 lite bridge 文件(会注册该架构),且 09-15 同一个 job 明明成功过(workdir 该目录 `__pycache__` 有 09-15 的 `glm4_7_moe_lite.cpython-312.pyc` 为证)。用户直觉「之前都没用到什么桥接」——对:桥早就存在,不是要新建桥,是 job 没加载到它。
- **根因**:vime 是 **editable 安装**(`site-packages/__editable__.vime-0.3.0.pth` + `__editable___vime_0_3_0_finder.py`),finder 的 `MAPPING` 把 `vime_plugins` 固定映射到已安装路径 `/root/vime/vime_plugins`(其 `megatron_bridge/__init__.py` 只 import `glm4v_moe`,不注册 lite)。job driver 进程的 `PYTHONPATH` 里只有 `/root/vime`、**没有工作目录**,于是 `import vime_plugins` 落到已安装版;AutoBridge 在已安装版里找不到 lite 架构注册 → 拒绝。09-15 成功是因为当时 driver 的 cwd/PYTHONPATH 能命中工作目录(证据:workdir `vime_plugins/__pycache__` 有当时的编译产物)。这不是「缺桥」,是 **import 解析漂移**。
- **修复**:把工作目录放到 job 环境 `PYTHONPATH` 最前(如 `export PYTHONPATH="/workspace/vime/vime-zjc:${PYTHONPATH:-}"`),确保 `import vime_plugins` 优先命中工作目录里的 bridge。验证:`python3 -c "import vime_plugins; print(vime_plugins.__file__)"`(cwd=工作目录时命中 `/workspace/vime/vime-zjc/vime_plugins/__init__.py`)。
- **预防**:① 复现时先确认「该 bridge 文件之前被 job 用过吗」——看 workdir `vime_plugins/megatron_bridge/__pycache__` 的 `.pyc` 时间戳,有即证明不是缺文件;② 遇到「not yet supported」别急着写新 bridge,先核对两侧 `vime_plugins` 的 `__init__.py` import 列表 + job 的 `PYTHONPATH`;③ 重建采集脚本时把「工作目录在 PYTHONPATH 最前」当必备段,与 P13 的「逐段 diff 基线」并列。

### P20. msprobe(level=mix)替换 F.silu 后,引用缓存/引用相等校验连锁崩(2026-09-17)
- **症状**:`DUMP_ON=1` 时训练侧报错按 debugger 实例化时机分两层:
  1. debugger 晚于模型构造:`mlp.py:198` `Only support fusion of gelu and swiglu`(`self.activation_func == F.silu` 引用相等失败);
  2. debugger 提前到模块导入期后仍崩:`transformer_config.py:1436-1445` `When bias_activation_fusion is True, activation function should be either gelu, swiglu, or quick_geglu`(`self.activation_func not in [F.gelu, F.silu, quick_gelu]`)。
- **根因**:msprobe `PrecisionDebugger(level=mix)` 在实例化时经 `_register_api_hook` **全局替换 `torch.nn.functional.silu` 为包装函数**。凡是**模块导入期缓存原生 `F.silu` 引用、替换后与动态访问的 `F.silu`(包装函数)做引用相等/成员判断**的代码都会失败。Megatron-Bridge 具体链:`model_bridge.py:322` `ACTIVATION_MAPPING = {"silu": F.silu, ...}` 在模块导入期缓存原生引用 → debugger 提前到模块导入期后,`model_provider → import megatron.bridge` 在 debugger 实例化**之前**跑完,缓存仍是原生 → 模型构造时 `hf_to_megatron_activation("silu")` 返回原生 silu,校验对照动态 F.silu(包装)不匹配 → 崩。**时序敏感**:debugger 时机一变,连锁点就变。
- **修复**(采集态补丁,采完还原):
  1. **debugger 提前到模块导入期**(训练 `model.py` 顶部,env-gated)——先于模型构造,config.activation_func 捕获包装函数本身,MLP 判断 `self.activation_func == F.silu` 通过;
  2. **激活函数映射改调用期动态求值**(`model_bridge.py`):`hf_to_megatron_activation` 对 `"silu"/"gelu"` 动态访问 `F.silu`/`F.gelu`(此时即包装函数,数值与原生一致),不再返回 `ACTIVATION_MAPPING` 里缓存的原生引用;`megatron_to_hf_activation` 反向加 `is F.silu` → `"silu"` 兜底。
- **预防**:① 接入 msprobe 前 `grep -rn "F.silu\|F.gelu"` 扫模型/转换代码,列出所有"导入期缓存 + 替换后引用相等/成员判断"的点;② 改 debugger 实例化时机后重新扫调用链(时序一变连锁就变);③ 遇到 `activation_func not in [...]` 校验崩,别改 msprobe 排除 silu——排除了就采不到 silu 边界,正确做法是让校验两边的引用一致(动态求值)。

### P21. Megatron-Bridge 的 TE 硬依赖假阻塞 + Python 3.12 注解求值(2026-09-17)
- **症状**:孤立环境(无 transformer_engine)`import megatron.bridge` 崩 `ModuleNotFoundError: transformer_engine`;连锁假象是 AutoBridge 报 `'XxxLiteForCausalLM' is not yet supported`。
- **根因**:`peft/lora_layers.py:20` 顶层**无条件** `import transformer_engine.pytorch as te`,TE 只服务 LoRA 但 `import megatron.bridge` 必经;TE 缺失 → 导入期崩 → vime_plugins 里 bridge 的**注册语句从未执行** → AutoBridge 找不到该架构 → "not yet supported"。**"不支持"是假象,真因是导入链硬依赖崩在注册之前**。孤立环境无 TE ≠ 作业进程无 TE(modelopt 警告显示作业里 TE 存在但缺 RMSNorm,注入源不明,别耗)。
- **修复**:TE 可选化(采集态补丁,采完还原):① TE import 改 `try/except ModuleNotFoundError`(`te=None, HAVE_TE=False`);② `class TELinearAdapter(te.Linear):` → `class TELinearAdapter(te.Linear if HAVE_TE else nn.Module):`(类定义期唯一 TE 依赖);③ 文件顶部加 `from __future__ import annotations`——**Python 3.12 无 future 时,类体内方法返回注解在类定义期求值**,`-> tuple[te.ops.Sequential, ...]` 在 te=None 时直接崩。
- **预防**:① 遇 "not yet supported" 先查 bridge 导入链有没有硬依赖在导入期炸(注册没执行到),别只盯着 AutoBridge 校验;② 孤立环境报 ModuleNotFoundError 先判断该模块是否真被这条路径用到(TE 只服务 LoRA,模型加载/前向不碰);③ 给导入链加 try/except 时,同步检查类定义期会求值的注解(3.12)。

- **症状**(2026-09-14,卡 6,7):两侧 dump **已完整落盘**,随后训练崩:`policy_loss_function` 里 `torch.cat(batch["advantages"])` 收到 `None` → `TypeError: cat() received ... got (NoneType, dim=int)`(或 `split_sizes=[0]`)。指标 `train_rollout_logprob_abs_diff` 在 loss 更靠后的行,根本执行不到(日志 0 命中)。
- **根因**:PROMPTS_ONLY 裁剪把训练侧输入裁成 prompt-only(为对齐推理 prefill),但此前我方方案为防崩而**把 `advantages` 设成 `None`、跳过 advantages 路径,却没按官方改 loss 兜底**——loss 仍要按 response 段切、要 `advantages`/`log_probs`,收到 None 即崩。
- **修复(官方做法,对齐 `*_consistency_preprocess_dump.md`,训练照常走完)**:**不是跳过训练**,而是改三处让裁剪后的输入能正常算 loss:
  1. **裁剪**(`compute_log_prob`):有 `responses` 时按 `response_length` 截掉 `input_ids/attention_mask/position_ids` 的 response 段,`pop("responses"/"rollout_log_probs"/"response_mask")`;log_probs 提取分支改为 prompt-only(`log_probs[:, :-1]`)。
  2. **改 loss**(`loss_func`):`response_length==0` 时不再按 response 段切 logits,改用 prompt 段;`response_mask=None` 时用全 True prompt mask 兜底;`rollout_corr_helper` 在 `PROMPTS_ONLY=1` 时直接 `return batch, {}` 跳过 correction。
  3. **改 metrics**(`calculate_debug_metrics`):`responses`/`rollout_log_probs`/`old_log_probs` 缺任一 → 主动返回 `{rollout_probs_diff_valid:0, _max/_mean/_std:0.0, ...}`,不崩。
  训练照常 forward→loss→backward→step 走完;官方明确该模式下 loss 和梯度**不代表正常训练结果**。采集完取消 `DUMP_ON`/`PROMPTS_ONLY` 恢复正常训练配置。
- **预防**:不要在采集开关下跳过训练、也不要把 `advantages` 设 `None` 绕过——正确做法是让裁剪后的 loss 也能算。response 段 E2E 指标(`rollout_probs_diff`)锚定 response 段,裁剪下 metrics 主动返回 0 值(是"没算"不是"一致",见 `references/probs-diff-monitoring.md`);`train_rollout_logprob_abs_diff` 已改为裁剪下可算(prompt 段对齐推理侧 prompt_logprobs),**不要**再对它返回 0 或跳过,见 P17;要实测须补跑 `PROMPTS_ONLY=0` 标准训练 step,没抓到标 `evidence_missing`。

### P16. 诊断字段进 log_rollout_data 会崩 `sum(val)`(int + list)
- **症状**(2026-09-15 5 步重跑):第 1 个 step 两侧 dump 已完整落盘、prefill_logprob_diff 已打印,随后 `log_rollout_data` 崩:`data.py:380 sum_value = sum(val)` → `TypeError: unsupported operand type(s) for +: 'int' and 'list'`。
- **根因**:`log_rollout_data` 遍历 `rollout_data.items()` 全量聚合,而新增的诊断 key `prompt_logprobs`(list of list:每样本一个 token 列表)不在此前预置的跳过列表里,落入 `sum(val)` 分支——`sum([list, list])` 的初始值 0(int) + list 即崩。`lp_train_prompt`(list of tensor)虽不会崩 sum,也会被当成指标聚合污染日志。
- **修复**:把新增的诊断 key `prompt_logprobs`、`lp_train_prompt` 一并加进 `log_rollout_data` 的跳过列表(与 `tokens`/`loss_masks`/`micro_batch_indices` 等并列)。
- **预防**:凡是往 `rollout_data` 里加的"仅供诊断/打印用"key,先查 `log_rollout_data` 的跳过列表——它全量遍历且对非 tensor-list 走 `sum(val)`,任何新增非标量 key 都会崩。

### P17. PROMPTS_ONLY 下让 `train_rollout_logprob_abs_diff` 可算(替代"跳过")
- **症状**(2026-09-15 晚,用户要求"这个指标我要能算出来,前面计算方式是错误的"):此前在 PROMPTS_ONLY 下直接跳过该指标,日志里没有 `train/train_rollout_logprob_abs_diff`。原始实现 `(old_log_probs - rollout_log_probs).abs()` 在裁剪下崩 `The size of tensor a ... must match ...`(train prompt 段 vs rollout response 段,长度不一致),直接跳过是错误收尾。
- **根因**:① `old_log_probs` 裁剪后是 prompt-only prefill 段(prompt_len-1 tokens),`rollout_log_probs` 是 response 段,长度不匹配不能减;② 即便对齐,`sum_of_sample_mean` 在 `response_lengths` 全 0(PROMPTS_ONLY)时是**常量-0 stub**(cp_utils 的 `get_sum_of_sample_mean` 早退分支),任何传入值都返回 0——这就是"算不出来/算出恒 0"的根子。
- **修复(两处,缺一不可)**:
  1. `model.py` 训练前向 `get_batch` 的 keys 列表**必须加 `"prompt_logprobs"`**——否则 `policy_loss_function` 的 `batch.get("prompt_logprobs")` 恒为 None,推理侧 prompt 段 logprob 根本进不了 loss。
  2. `loss.py` 的 `train_rollout_logprob_abs_diff` 分支:PROMPTS_ONLY 下把 `old_log_probs`(train prompt 段)与 `batch["prompt_logprobs"]`(推理侧同一 prompt 段)对齐,`torch.cat` 后算 `(old_log_probs - rollout_log_probs).abs().mean()`——**绕过** `sum_of_sample_mean`(它是 0 stub),直接用 `.mean()` 按 prompt token 归约;非 PROMPTS_ONLY 保持原逻辑不变。
- **数值口径**:该指标在采集配置下是 **raw logprob 空间**的 prompt 段均值绝对差。prompt 段是大量低概率 token,很小的概率差会被放大成几个 nats 的 log 差;而标准训练的 response 段是高概率 token,log 差小。**两个口径数量级不同属预期,不要把两者当作应相等**。要对比就同口径对比(同空间 + 同样本段)。
- **预防**:① 加诊断 key 进 loss 时,先确认它进了 `get_batch` 的 keys 白名单(model.py 训练前向那个列表);② 看到"某归约器返回恒 0",先查它是否有 PROMPTS_ONLY/全 0 分母早退 stub;③ E2E 指标在裁剪模式下要么算 prompt 段对齐值、要么显式标"没算",禁止用"返回 0"冒充一致。

### P13. 重建脚本时漏掉基线里的"不报错就不想带"参数
- **症状**(2026-09-15):`NotImplementedError: Rule-based RM type is not specified`(rm_hub)。
- **根因**:重建采集脚本时漏了基线里的 `--rm-type deepscaler`。deepscaler 是**本地 rule-based RM**,不需要外部服务,但参数必须带。
- **修复**:从基线脚本逐段复制完整 RM 参数(不改 RM 语义)。
- **预防**:任何基于基线重建的脚本,跑之前逐段 `diff` 基线,特别留意 RM / 并行大小 / dispatcher 这类"觉得没用就省掉"的段。

### P14. `ray job submit -- <cmd>` 会把含空格的参数拆开
- **症状**(2026-09-15):vLLM 收到截断的 `{"dump_config_path":`;additional-config JSON 解析失败。
- **根因**:`ray job submit` 把 entrypoint 参数拼接成字符串、在 driver 端重新 shell 解析;JSON 含空格时在空格处被拆成多个 argv、双引号被剥(铁证:argv 截断在空格处)。
- **修复**:要过 `ray job submit` 的含空格 JSON 一律写成**无空格 compact 形式** `'{"dump_config_path":"/abs/path"}'`。
- **预防**:传给 ray job 的参数先看它是否需要空格;启动失败先查日志里 argv 是否被截断(截断即此坑)。

### P15. 容器内慢命令 + docker exec 默认超时/输出换行
- **症状**(2026-09-15):`docker exec` 超时 exit 143;`source /usr/local/Ascend/nnal/atb/set_env.sh` 无输出假死;监控脚本 `[: integer expression expected`。
- **根因**:docker exec 批次继承 Bash 默认 120s timeout,而容器内 set_env 单是 source 就要 ~21 秒,job 启动更久;宿主对容器内进程无 kill 权限,exec 只能等超时。docker exec 输出带换行,`[ $x ...]` 判数值直接炸。
- **修复**:sleep 与 docker exec 拆开、给长 timeout(≥300s);长任务用 `docker exec -d` 分离;监控轮询对 docker exec 输出先 `tr -d '\n\r'` 再提取数值(`sed -n 's/.*TD=\([0-9]*\).*/\1/p'`)。
- **补充(2026-09-17)**:`bash -lc 'cat > /tmp/x.py << "EOF" ... EOF; python3 /tmp/x.py'` 这种方式在 docker exec 里,heredoc 内容若含**单引号 f-string**(如 `f"[{'EQ' if ...}]"`)会被外层单引号提前截断/转义炸掉(输出变成被执行的命令)。修复:一律**宿主 Write 脚本 → `docker cp` 进容器 → 容器内 `python3 /tmp/x.py`**,别在 `bash -lc` 里嵌长 heredoc。
- **预防**:容器内"source 环境脚本"先实测耗时再定 timeout;所有 docker exec 批次默认给长 timeout 或后台分离;跨容器传脚本用 docker cp,不在命令行嵌 heredoc。

### P22. ray start Session name 不匹配 → 必须清 /tmp/ray_* 残留 Redis
- **症状**(2026-09-20 打桩 round2 首启):`ray start` 报 `Session name 'session_2026-09-1X_...' does not match persisted value b'session_2026-09-1X_...'`(两次值不同)。
- **根因**:上一轮采集的 ray head 用了同一个 `--temp-dir`(常带 `VIME_TAG` 后缀),Redis/日志残留写在 `/tmp/ray_<tag>/`;新一轮 `ray start` 读到旧 Session name,与新值不匹配直接拒。只 pkill 进程不够——**持久化数据还在**。
- **修复**:清理要连 `/tmp/ray_*` 一起删:`pkill -9 -f '<temp-dir 关键字>' && rm -rf /tmp/ray_<tag>`,再 `unset RAY_ADDRESS RAY_REDIS_ADDRESS` 后重启。
- **预防**:每轮采集/打桩脚本开头固定"清理本集群残留"段(pkill 旧 temp-dir + rm -rf 该 temp-dir),`RAY_TMPDIR` 每轮换唯一 tag;遇到 Session name mismatch 先 `ls /tmp/ray_*` 确认残留再清。

### P23. "No available agent to submit job" → 残留 ray 子进程占 GCS/dashboard 端口
- **症状**(2026-09-20 打桩 round3 两次失败):`ray job submit` 报 `No available agent to submit job` / `Failed to submit job`,dashboard 端口有响应但提交不进去。
- **根因**:上一轮 ray head 的**子进程没被上轮 pkill pattern 命中**,一直占着 GCS(6382)/dashboard(8267)端口。常见漏网进程 cmdline:`ray.util.client.server`、`ray/dashboard/dashboard.py`、`ray/core/src/ray/gcs`、`ray/core/src/ray/raylet`、`ray/_private`、`ray/autoscaler`、`ray/job`,以及 `redis-server`——它们**不以 `ray` 开头**(路径含 `/ray/` 但首字段是别的),只 pkill `^ray`/`raylet` 等窄 pattern 打不中。
- **修复**:清理 pattern 扩到含 `/ray/` 的路径段:`pkill -9 -f 'ray/util.client.server'`、`pkill -9 -f 'ray/dashboard'`、`pkill -9 -f 'ray/core/src'`、`pkill -9 -f 'ray/_private'`、`pkill -9 -f 'redis-server'` 等(全部用 `[r]ay` 正则防自匹配);配合 P6 全量扫持 davinci 句柄进程兜底,再验端口释放(`ss -ltnp | grep <port>` 空)。
- **预防**:清残留时别只按"进程名首字段"猜,凡 cmdline 含 `/ray/` 的路径段都纳入 pattern;清完必须验证 GCS/dashboard 端口被释放;脚本里把"本轮用到的 ray pattern 全集"沉淀下来,新轮次直接复用(本次教训:pattern 是逐步踩出来的,第一次只列了 `ray/core/src`、`ray-dashboard`,漏了 `ray.util.client.server` 等)。

### P24. ray 控制面 socket 路径超 107 字节 → MetricsHead 启动失败
- **症状**(2026-09-20 打桩 round1 首次起就崩):vLLM/MetricsHead 相关启动报 `AF_UNIX socket path length exceeds the maximum socket path length`(或等价)后失败。
- **根因**:ray 的 dashboard/agent socket 落在 temp-dir 下,`<temp-dir>/<job>/<长 id>/...` 拼出来的 **AF_UNIX socket 绝对路径超过 107 字节**(Linux `sun_path` 上限)。tag/路径越长越容易触发——如 `VIME_TAG` 加 `_roundN` 后缀、temp-dir 里带长 job 名,路径就顶线。
- **修复**:缩短运行时临时路径——`VIME_TAG` 用短名(不带长后缀),`RAY_TMPDIR`/`--temp-dir` 放短路径(`/tmp/ray_<短tag>`),必要时缩短 job 名;已顶线就换更短 tag 重试。
- **预防**:tag/路径长度是硬约束,不是建议——计划 tag 时先估 `len(temp_dir)+len(job) ≈ <107`;每次加后缀(roundN 之类)都要复查总长,别等启动崩了再回退。

## 分析环节(consistency-dump-analysis,不在本 skill)

本 skill 只负责采集;分析定位、ULP 量化、放大链、结论与报告由 **consistency-dump-analysis** skill 承担(相关分析坑已收敛到该 skill 的 references)。

## 已单独记录的教训

- **设备分配(09-10,最重的一次)**:改动前先确认卡号来源,先做原生对照,不得凭记忆二次映射卡号。具体事故(某框架 vLLM 基址取自 ray 物理卡号后被二次取模越界)见 `references/vime.md`;已沉淀为 SKILL.md 阶段 3 通用警示,不重复。
