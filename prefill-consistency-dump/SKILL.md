---
name: prefill-consistency-dump
description: 为训推一致性(prefill 训推一致)**采集**两侧可比的 dump 数据,并按分析侧的打桩指令做**数据覆盖加载(load)二次采集**以隔离差异。完整流程:预检清卡 → 授权 → 识别现场 → 输入对齐(PROMPTS_ONLY)→ 接入采集 → 采集验证 → 打桩二次采集(与 consistency-dump-analysis 循环)→ 恢复代码。产出"两侧可比的 prefill dump"与"打桩隔离后的下游 dump";分析差异点、首差异定位、打桩指令、所有差异点整体结论由 **consistency-dump-analysis** skill 承担。当用户提到训推一致、训推一致性、prefill 比对、msprobe dump/采集、精度比对数据采集、打桩二次采集、要求先清卡验环境再采数据时,使用本 skill;分析已采 dump(dump 已采好帮我分析/首差异定位/对齐报告)转 consistency-dump-analysis。
---

# Prefill 训推一致性数据采集

为训推一致性比对准备两侧可比的 dump 数据。核心思路:训练输入是 prompt+response,推理 prefill 输入只有 prompt,必须先把训练侧输入对齐成 prompt-only,两侧各用 msprobe 采集 prefill 前向,再用 request_id 把两侧 dump 钉在一起,产出"两侧可比的 prefill dump";**分析差异点、定位首差异、输出打桩指令、汇总所有差异点整体结论由 `consistency-dump-analysis` skill 承担(本 skill 只采集 + 按指令打桩再采,不分析)**。**闭环:本 skill 产 dump → analysis skill 分析出差异(首差异/后续独立差异)→ 本 skill 按打桩指令用 msprobe `load` 覆盖差异边界输入、隔离前序误差 → 再产 dump → 再分析……直到找出所有差异点。** **对齐的关键约束:训练端 padding 必须去掉,两侧输入 shape 对齐到真实 prompt 长度(如 190 vs 256 须去 padding 统一到 190),否则 dump 形状天然不等、逐 token 比对无从谈起(见 pitfalls P18)。**

## 本 skill 做什么 / 不做什么

- **做**:预检清卡、输入对齐、接入采集、request_id 关联、采集验证、**打桩二次采集(按 analysis 侧打桩指令,用 msprobe `load` 覆盖边界输入再产 dump)**、恢复代码。产出是"两侧可比的 prefill dump"与"打桩隔离后的下游 dump"。
- **不做**:修复差异并验证修复;分析差异点、定位首差异、输出打桩指令、汇总所有差异点整体结论(那是 **consistency-dump-analysis** 的职责,本 skill 只采集 + 按指令打桩)。
- **纪律**:采集与打桩需要临时修改训练/推理代码/config,这是预期内的必须动作;但**采集完成后必须恢复代码,不留任何诊断残留**。恢复是本流程的一部分,不是可选项。

## 执行顺序

```
0 锁定目标 → 1 预检清卡 → 2 授权 → 3 识别现场
  → 4 输入对齐 → 5 接入采集 → 6 采集验证
  → 7 打桩二次采集(与分析侧循环:拿打桩指令 → load 覆盖边界 → 再产 dump)
  → 8 恢复代码
→ 采集完成:dump 交接给 consistency-dump-analysis(差异点枚举/打桩指令/所有差异点整体结论)
```

按顺序走,但 1 和 2 可以合并:先把环境看一遍,再一次性把要问的都问清。阶段 6→7→6 是**与 analysis skill 的循环**:第 0 轮 dump 交接后,analysis 给出首差异与打桩指令,回本 skill 做打桩再采(阶段 7),产出 round N+1 dump 再交接;循环到 analysis 判定"所有独立差异点已找到"为止。恢复代码(阶段 8)只在全部轮次结束后执行一次。

## 0 锁定目标

动手前把下面信息记入 `task_card.json`(模板见 `references/task_card.json`):

- 模型
- 训练后端(Megatron / FSDP / 其他)
- 推理引擎及版本(vLLM / SGLang / 其他)
- 硬件(Ascend NPU / GPU / 其他)
- 证据工具(msprobe PrecisionDebugger 或等价)
- 采集范围(确定性单样本 prefill)
- 停止条件(采到两侧可比的 prefill dump)

**通用触发输入模板**(框架/后端/硬件无关,默认走完整流程;用户给的信息越全,后续阶段越少反问):

```text
训练框架及版本 / 训练后端(FSDP/Megatron/DeepSpeed/自研)
推理引擎及版本(vLLM/SGLang/Transformers/自研) / 硬件后端(Ascend/CUDA/ROCm)
执行环境(SSH目标/容器/conda或venv激活) / 工作目录(绝对路径) / 基线命令 / 日志位置
已知环境约束:占卡残留服务 / PID namespace 隔离 / 宿主无 root kill 权限 / 端口占用等
模型与样本:权重路径 / 固定样本 ID / prompt 来源
证据工具:msprobe PrecisionDebugger(不确定就先检查并提出适配方案)
采集范围:确定性单样本 prefill(默认) / 停止条件:采到两侧可比的 prefill dump(默认)
目标 E2E 指标(选填,如 rollout_probs_diff_mean == 0)
授权与纪律:一次问清、采完恢复代码、全程中文
```

用户只给一段话、未按模板填写时,按上面字段主动补齐提问;信息缺失较多时先确认再动手。框架识别按阶段 3 通用方法论现场完成。

只记事实,不知道的字段留空或标 `unknown`,不要编默认值。这份卡片是后续所有动作的依据。

## 1 预检清卡

目的:确保目标卡空闲、残留进程不干扰采集、环境可用。**清卡只动本任务相关进程**,不做宽泛清理。

1. **查卡占用**:运行 `npu-smi info`(Ascend)或等价命令,确认目标卡的占用状态和占用者。
2. **识别残留**:找出占卡的 vLLM / ray / 训练服务。注意 PID namespace 隔离:宿主上看到的进程列表与容器内不同,杀进程必须进对应容器操作;宿主通常也没有 root kill 权限。
3. **停残留**:用精确 PID/PGID 或容器内正常退出方式停,禁止宽泛 `pkill` 或无界 `/tmp` glob。
4. **验环境**:确认容器 / conda / venv 激活方式;确认 msprobe 已安装(`pip show mindstudio-probe`),**并写清它在哪个容器 / 哪个 conda env 里**——一台机器常有多个容器/多个 env,查错环境会得出相反结论。训练侧和推理侧的环境要分开确认;SGLang ≥0.5.11 的原生集成在 `ImportError` 时**静默跳过**,msprobe 装错环境会导致 dump 静默为空、不报错。确认训练、推理两侧环境可启动。
5. **确认资产**:模型权重路径存在、基线命令可执行、日志位置可写、产物根目录已规划。
6. **记录**:清卡前后的卡状态、停掉的进程,写进 task_card。

## 2 授权

把本流程要做的所有动作**一次性问清**,记录后流程中不再打断用户。至少确认:

- `AUTH_STOP_RUNNING_SERVICES` — 允许停掉当前占卡的残留服务
- `AUTH_EDIT_TARGET_CODE` — 允许临时修改训练/推理代码(接入采集,采集后恢复)
- `AUTH_RUN_DUMP_JOB` — 允许启停采集作业
- `AUTH_CLEAN_APPROVED_DIRS` — 允许删除指定的产物/临时目录
- `AUTH_RESTORE_CODE` — 允许采集完成后恢复代码

同时确认:SSH 目标、环境激活方式、工作目录、命令、日志位置、产物根目录。凭据绝不写入 skill、工作区或报告。

## 3 识别现场(框架无关)

采集的接入位置强依赖训练框架版本和后端。**先现场识别,再动手;references 只是线索,不是坐标**。本 skill 不预设特定框架,框架识别按本阶段方法论现场完成。

1. **识别框架形态**(先不查文档):
   - 训练框架及版本(verl v0.x / slime / 其他框架)
   - 训练后端(FSDP / Megatron / DeepSpeed / 自研…)
   - 推理引擎(vLLM / SGLang / 原生 / **其他**)——决定原生 msprobe 集成是否存在
   - 架构形态:**显式 micro_batch 循环**(FSDP 型,包循环体)vs **连续前向**(Megatron 型,插 forward 内部)
2. **查 references 作线索**(有就翻,没有/过时不阻塞):对应框架有适配文档(`verl.md` / `slime.md` / `vime.md`)可快速拿到"可能"的裁剪/接入/关联位置;**新框架、或文档与现场冲突时,直接按 `references/new-framework.md` 的通用方法现场搞定**。
3. **现场验证三件事**(必须,无论 reference 有没有、对不对):
   - PROMPTS_ONLY 裁剪位置:`grep compute_log_prob / get_data_iterator / rollout_data`,确认符号真实存在,顺数据流找到真正构造 log_prob 输入的那一处
   - 接入位置:找到 log_prob 主前向,判断包循环体还是插 forward 内部;确认主前向是否被缓存跳过(类似 `can_reuse_log_probs_in_loss` 的条件)——采不到数据先怀疑它
   - request_id 关联:找推理调度日志与训练 rollout 数据的对应字段;没有现成日志就在接入处自己打
4. **reference 与现场冲突时**:以现场为准,把偏差记入改动清单,采完修正 reference 或标注 outdated。**reference 是活的文档,不是不可变依据**。
5. **判断官方采集姿势适用性**:模型/框架是否在官方 `*_consistency_preprocess_dump.md` 采集姿势支持范围内(verl + fsdp/megatron + 官方名单模型,名单以官方文档为准,如 Qwen 系列;**名单外模型不在官方支持范围**)?记入 task_card:**范围内** → 采集姿势跟官方对应文档;**范围外**(或框架封装后 dump 结构与 verl 原生不符)→ 按本 skill 通用方法采集。能否用官方 `--consistent_check` 自动比对属分析侧判断,由 consistency-dump-analysis 处理。

> **references 是线索,不是坐标**:适配文档的文件路径/函数名基于某次真实核对写成,可能已过时或**错误**。真实排查发现过:`get_prompts_only_batch`、`dispatch_logger.py`、`_ensure_debugger` 在部分仓库版本里**不存在、需新增**;`get_data_iterator` 签名与 msprobe 文档示例不同。改任何文件前先 `grep` 确认符号存在;不存在就按"新增"处理并记入改动清单,别照抄行号硬改。完整通用方法论见 `references/new-framework.md`。

> **设备分配与可见卡:改动前先确认卡号来源,先做原生对照**。采集时若引擎无法启动/拿不到设备,**应先排查自身改动**,不宜直接断言框架"不支持非 0 基址/不支持分卡"。真实教训:某框架的 vLLM 基址取自 ray placement group,本身即为物理卡号,原生支持各卡位;曾出现将物理卡号二次取模的"物理映射 patch",导致越界 → `visible_devices=''` → EngineCore `NPU device is not available` 崩溃,反复 7 次才定位。排查三步:确认子进程实际拿到的可见卡变量(如 `ASCEND_RT_VISIBLE_DEVICES`)→ 沿启动链定位断点 → 回滚改动做原生对照。框架具体机制见 `references/vime.md` 及对应框架适配文档。

## 4 输入对齐

目的:让训练 prefill 与推理 prefill 在数学上等价,且逐卡逐样本可对应。以下约束**框架无关**,按阶段 3 现场确认的并行/数据流落实(适配文档仅作线索):

1. **输入对齐**:设 `PROMPTS_ONLY=1`,按适配文档在指定位置裁剪掉 response,只保留 prompt。
2. **batch 约束**:`mini_batch_num=1`(训练 batch 不拆分)、`gac=1`(不梯度累积)、`balance_batch=False`(保证同一请求在两侧落在同一张卡)。
3. **无 pad(必须做,否则输入 shape 对不上)**:`use_remove_padding=True`(verl);slime 用 `--data-pad-size-multiplier 1`,TP>1 时两侧按同一 pad_size 对齐。**训练端 padding 必须去掉**:padding 只在训练侧(Megatron 系按对齐长度 pad),推理侧是真实 prompt 长度。裁剪成 prompt-only 后,训练侧也要按真实 prompt token 数截取(log_probs 取 `[:, :-1]` 前先按真实长度切),保证两侧 dump 的**输入 shape 都是真实 prompt 长度、逐 token 1:1**。采完验证:两侧同一边界输入 shape 必须相等且等于真实 prompt 长度——若出现训练侧 [256,...] vs 推理侧 [190,...](一边 pad 后形状、一边真实长度),就是 padding 没去净,回这里修掉重采,别把带 padding 的 dump 交接给分析环节(P18)。
4. **eager 与确定性**:`enforce_eager=True`、`TORCHDYNAMO_DISABLE=1`、SGLang `--sglang-disable-cuda-graph`(msprobe 只在 eager 下能钩住算子);`seed_all()` 固定随机种子,关闭 shuffle / 随机采样。
5. **并行切分一致**:训练与推理的 TP/PP/CP/DP 必须一致,dump 才能按 rank 一一对应。
6. **排除干扰**:训练前若有验证/预热前向(verl 的 `val_before_train`),关掉,避免其 generate 污染 dump。

> **PROMPTS_ONLY 裁剪副作用**:裁剪副本**只喂 log_prob 前向的 iterator,别替换整个 rollout_data**(整体替换会让 advantages/统计按 response_lengths=0 split 崩);`loss_masks` 裁 `m[:0]` 再靠 pad 展开;adv/correction 路径让 `rollout_corr_helper` 在 `PROMPTS_ONLY=1` 时直接 `return batch, {}` 早退,并同步改 loss/metrics 兜底,**训练照常走完**(细节见 `references/pitfalls.md` P1-P3、P6b)。

## 5 接入采集

> **接入补丁与返回体结构(两次真实翻车,先读再改)**:① 补丁脚本是"改代码的代码"——old/new 字符串定界符避开目标代码里的引号(用 `'''`),应用前先 `py_compile` 自检(P11);② 推理引擎返回字段别照抄文档:`prompt_logprobs` 在 vLLM-ascend 是 **top-k 嵌套 dict**(`{token_id: {logprob, rank, decoded_token}}`),解析前先打一行 `[DUMP_DIAG]` 确认结构再写解析(P12)。细节见 `references/pitfalls.md` P11-P12。

目的:在训练侧和推理侧各包住目标前向,用 msprobe PrecisionDebugger 采集。接入会改目标代码,这是预期内的,但**每一步改动都要记录**。

1. **训练侧接入(官方做法:直接改代码,不写独立 hook 文件,也不用框架预留的 hook 入口)**:官方 `*_consistency_preprocess_dump.md` 的改法是在训练 worker/actor 源码里实例化 `PrecisionDebugger`、包住 `compute_log_prob`,裁剪和 loss 处理也直接改训练源码——**没有"单独写一个 hook.py"的形态**。统一按官方做法直接改源码落地;不另写 hook 模块,也不走框架预留的 hook 入口(即使框架提供了 hook 入口,该写法也偏离官方形态;框架特定入口见对应适配文档)。具体:
   - **初始化**(worker `__init__` / 训练类构造):`DUMP_ON=1` 时 `from msprobe.pytorch import PrecisionDebugger, seed_all; seed_all(mode=True)` 并实例化 `PrecisionDebugger(task='tensor', level='L0', step=[0], dump_path=...)`;`DUMP_ON=0` 时为 `None`(env-gated,正常训练零影响)。只 actor 引擎建 debugger,ref 引擎(`forward_only`)跳过。
   - **包主前向**(`compute_log_prob`,log_prob 主前向):`debugger.start(model)` → 前向 → `debugger.stop()` → `debugger.step()`。FSDP 有显式 micro_batch 循环,包循环体;Megatron 没有循环,插在 `forward_step`/`forward_only` 内部(per-micro-batch 粒度)。
   - 同时把 micro_batch 的 `request_id` 写入训练侧日志(verl 系 `update_actor_log.jsonl`;其他框架按现场确认的日志/字段)。
2. **PROMPTS_ONLY 裁剪 + 同步改 loss/metrics(官方做法,训练照常走完)**:`PROMPTS_ONLY=1` 把训练输入裁成 prompt-only(使训练 prefill 与推理 prefill 对齐),**但裁剪不能导致训练崩**——必须按官方 `*_consistency_preprocess_dump.md` 同步改三处,让裁剪后的输入能正常走完 forward→loss→backward→step:
   - **裁剪**(`compute_log_prob` 里,对齐官方 `megatron_actor.py`):有 `responses` 时按 `response_length` 截掉 `input_ids/attention_mask/position_ids` 的 response 段,并 `pop("responses"/"rollout_log_probs"/"response_mask")`;log_probs 提取分支改为 prompt-only(`log_probs[:, :-1]`)。
   - **改 loss**(对齐官方 `loss_func` 分支):`response_length==0` 时不再按 response 段切 logits,改用 prompt 段;`response_mask=None` 时用全 True prompt mask 兜底;官方还改 `rollout_corr_helper` 让 `PROMPTS_ONLY=1` 时直接 `return batch, {}` 跳过 correction。
   - **改 metrics**(对齐官方 `metrics.py`):`responses`/`rollout_log_probs`/`old_log_probs` 不在 batch 时 `calculate_debug_metrics` **主动返回 `{rollout_probs_diff_valid:0, _max/_mean/_std:0.0, ...}`,不崩**。
   - **训练照常走完**:loss → backward → step 全跑,不提前结束循环。官方明确:**该模式下 loss 和梯度不代表正常训练结果**;采集完取消 `DUMP_ON`/`PROMPTS_ONLY` 恢复正常训练配置。**不要在采集开关下跳过训练**——正确做法是让裁剪后的 loss 也能算,而不是绕开它(见 `references/pitfalls.md` P6b)。
3. **prefill logprob_diff(采集侧产出的数值指标——训练侧裁剪后 prefill logprob vs 推理侧 prompt 段 logprob,训练循环内直接算出并打印到日志)**:在训练循环里找到两侧 logprob 碰头的位置(actor 前向重算的 `log_probs` 与 rollout 带回的推理侧 prompt 段 logprob 同时可见处,思路同 response 段 probs_diff 的插入点),**直接算 diff 并输出到训练日志**,不做训推结束后的 dump 离线分析:
   - **数据来源**:训练侧 PROMPTS_ONLY 裁剪 prefill 前向的输出 logits → `log_softmax` 得 `lp_train`(取 prompt 段 `log_probs[:, :-1]` 去末位);推理侧 **prompt 段** logprob `lp_infer`(来自 rollout 返回——需推理侧配置返回 prompt logprob,vLLM sampling `prompt_logprobs` / verl `calculate_log_probs` / 后端等价;**拿不到 `lp_infer` 就算不了,别拿 response 段 `rollout_log_probs` 顶替**)。
   - **对齐**:按真实 prompt token 对齐(去 padding、训练侧去末位),两侧覆盖同一组 (position, next_token) 对。
   - **算 diff**:逐 token `|exp(lp_train) − exp(lp_infer)|`(先还原概率再相减)+ `_mean/_max/_std` 聚合(只算有效 token 位)。
   - **打印到训练日志**:带统一关键词(如 `prefill_logprob_diff_mean`),保证 grep 到;env-gated(默认关)。这就是"训推两侧对齐输入后,prefill 输出的 logprob 差异",**训练进行中直接产出**。通用步骤见 `references/prefill-logprob-diff.md`。
4. **官方 response 段 E2E 指标口径(`rollout_probs_diff` / `train_rollout_logprob_abs_diff`,与上一条是不同层面,别混)**:官方 `metrics.py` 在 `responses`/`rollout_log_probs`/`old_log_probs` 不在 batch 时,`calculate_debug_metrics` **主动返回 `{rollout_probs_diff_valid:0, _max/_mean/_std:0.0, pearson_corr:0.0}`,不崩**。该指标锚定 **response 段**(vLLM 生成时记的 `rollout_log_probs` vs actor 重算),**PROMPTS_ONLY 裁剪下天然算不出——日志里 valid=0/mean=0 是"没算",不是"一致"**。要实测必须单独补跑一步 `PROMPTS_ONLY=0` 的标准训练 step,从标准训练日志抓数值;没抓到标 `evidence_missing`,不编。verl 原生开关 `+trainer.enable_token_level_prob_diff=True`(前提 `calculate_log_probs=True`);其他框架按 `references/probs-diff-monitoring.md` 通用方法现场写码。
5. **推理侧**:先确认引擎原生 msprobe 集成是否存在——**vLLM-ascend / SGLang ≥0.5.11 都原生集成,不用改码**,传 config 即启用(vLLM-ascend: `--vllm-enforce-eager` + `--vllm-additional-config '{"dump_config_path": ...}'`;SGLang: `--sglang-msprobe-dump-config`);若需 request_id 调度日志才在 `model_runner` 注入 DispatchLogger(verl 系 `dispatch_log.jsonl`,见 `references/verl.md`);**引擎无原生集成(非 vLLM/SGLang)需手写推理侧接入,见 `references/new-framework.md` §2**。
6. **request_id 贯穿**:按适配文档确认 request_id 能从推理引擎一路流到训练侧 micro_batch 日志。
7. **msprobe 配置**:两侧各一份 config(模板见 `references/msprobe-config.md`),`task=statistics`(轻量)或 `tensor`(完整),`level=L0/mix`,两侧 `dump_path` 分开。**要算 prefill logprob_diff,两侧 dump 必须采到 final 输出边界(logits)——确认采集级别覆盖 output_layer/lm_head 边界**。
8. **记录改动清单**:用 `git diff` 快照或备份,登记每一处改动(文件、位置、内容、恢复方式)。这是第 7 步恢复的依据。

## 6 采集验证

1. 在基线命令上追加采集开关(`DUMP_ON=1`、`PROMPTS_ONLY=1`,推理侧 `dump_config_path`;**具体开关名按阶段 3 现场确认**,不以特定框架符号为准)跑一次。
2. **验证产物**:两侧 dump 路径下出现 `step_N/rank_M/dump.json`(或等价),非空。
3. **验证关联**:在推理侧调度日志中挑 phase=prefill 且单请求的 step,拿 `request_id` 去训练侧日志找对应 step/rank。注意 vllm ≥0.14 会给 request_id 追加 8 位 hex 后缀,匹配前要剥掉。
4. **验证可比性**:两侧对应样本的 token 序列一致(长度/内容),**输入 shape 相等且等于真实 prompt 长度(去 padding 后)**。若训练侧 [256,...] vs 推理侧 [190,...],即训练端 padding 没去净,回阶段 4 修掉重采,别硬比对(P18)。
5. **记录**:采集到的 step/rank、两侧 dump 路径、关联关系,写进 task_card。
6. **验证采集侧数值指标产出**:
   - **prefill logprob_diff(训练循环内直接算出并打印到日志)**:训练侧裁剪 prefill logprob vs 推理侧 prompt 段 logprob 的 diff,在训练循环碰头处直接计算并输出到训练日志(带统一关键词,如 `prefill_logprob_diff_mean`)。**验证训练日志里确实出现了该输出且非空、数值合理**(不能光看没报错);其"是否训推一致"的解读由 consistency-dump-analysis 承担。通用步骤见 `references/prefill-logprob-diff.md`。
   - **逐边界比对 / 首差异定位**:不在本 skill 内,由 **consistency-dump-analysis** 消费已落盘的 dump 完成。
   - **官方 response 段 E2E(`rollout_probs_diff`),定位清楚**:PROMPTS_ONLY 采集日志里 valid=0/mean=0 是"没算"不是"一致",**不要拿 0 值当"训推一致"证据**;要实测须单独补跑 `PROMPTS_ONLY=0` 标准训练 step 抓数值,没抓到标 `evidence_missing`。前提:推理侧确实返回了 `rollout_log_probs`(缺字段时函数静默返回空不报错,光看"没报错"不算通过)。
7. **交接 round 0**:第 0 轮 dump 验证通过后,把 dump 路径、step/rank、request_id 关联方式、采集配置、task_card 交接给 **consistency-dump-analysis**,等它返回首差异结论与**打桩指令**(阶段 7 的输入)。此时**不要**急着恢复代码——打桩轮次还要用同一套接入。

> **采集踩坑**:训练侧有效 dump 在 **step0**(compute_log_prob 主前向),step1 未 flush 是空的;只有 `can_reuse_log_probs_in_loss=False` 时主前向才执行,需 `--get-mismatch-metrics` 强制。docker exec 清理用 `[r]ay` 正则防自匹配、长任务用 `-d` 分离;每次采集前先确认目标卡 HBM 回基线(残留 VLLM EngineCore 会让二次采集 OOM)。细节见 `references/pitfalls.md` P4-P6。
>
> **启动失败先对坑,别从头 debug**:采集作业跑不起来,先按顺序查——① 基于基线重建的脚本逐段 diff 基线,**漏 `--rm-type` 这类"不报错就不带"的参数会 `Rule-based RM type is not specified`**(P13);② 含空格 JSON 参数过 `ray job submit` 会被重新 shell 解析截断,改成无空格 compact 形式(P14);③ 容器内慢命令(docker exec 超时 143 / set_env source 要 21s / 监控输出换行炸 `[ ]`)用长 timeout + `-d` 分离 + `tr -d '\n\r'`(P15)。都是已踩过的坑,复现症状直接翻 `references/pitfalls.md` P13-P15。

## 7 打桩二次采集(数据覆盖加载,与分析侧循环)

拿到 consistency-dump-analysis 的差异结论后,若下游还存在待确认的独立差异,按它的**打桩指令**做隔离再采,迭代找出**所有独立差异点**。原理:用先前采集的模块级输入 tensor 覆盖目标模型该模块的实际输入(msprobe `load`,在 `forward_pre_hook` 注入),把差异边界"强行对齐",隔离前序累计误差,再继续 dump 下游——覆盖后下游若仍有**非舍入级**差异,就是下一个独立差异点,继续打桩,直到下游全 EXACT / 只剩余舍入级。官方能力与全部约束见 `references/stubbing.md`。

> **前提(msprobe 必须有 load 能力)**:容器 msprobe 需为含 `module_load/tensor_loader.py` 的版本(26.1 正式版**没有**,需上游 master 源码安装;验证:`python3 -c "import msprobe, importlib.util; print(importlib.util.find_spec('msprobe.pytorch.dump.module_load.tensor_loader'))"`)。

1. **接打桩指令**:analysis 侧输出 `stub_instruction.json`(round N),含:`target_side`(被覆盖一侧 train/infer)、`source_side`(提供数据的一侧)、`isolate_boundary`(隔离边界语义名)、`modules[]`(每条:目标模型条目名 + 源 .pt 文件 + **变换规则**)、`dump_after_load`。这是本阶段的唯一输入。
2. **检查源 dump**:必须是 `level=L0` 或 `mix` 采集(load 只读模块级 tensor);源 dump 与目标运行的 step/rank 自动对齐(也可用 `load.step/rank` 显式指定)。
3. **改造源数据(训推两侧算子名不同,必须做)**:load 要求源 dump 与目标模型"模块结构相同、命名一致";训推两侧命名/结构天然不同(如训练独立 Q 投影算子 ↔ 推理融合 QKV 投影输出的 q 段)。按打桩指令的变换规则,把源 `.pt` **重命名成目标模型的条目名**、对齐 shape/dtype(切片/reshape/切分),产物放新目录(如 `stub_source_roundN/`),`load.path` 指向它。
4. **写 load 配置**:在目标侧 config 顶层加 `load` 段:
   ```json
   {
       "task": "tensor",
       "dump_path": "/<本轮的 dump 根目录>",
       "level": "L0",
       "rank": [], "step": [],
       "load": {
           "path": "/<改造后源dump根>",
           "modules": ["Module.<目标模块路径>.<类名>.forward.0"],
           "step": [], "rank": [],
           "dump_after_load": true
       }
   }
   ```
   条目名 = dump 侧 `Module.{dotted_path}.{ClassName}.forward.{N}`(取 `.input.{i}.pt` 之前的部分),**必须以 `Module.` 开头、以 `.forward.{N}` 结尾**,且能被目标模型 `named_modules()` 命中。
5. **重跑采集**:沿用阶段 5 的采集开关/接入不变,只改 config(要覆盖哪侧就改哪侧 config 并重跑该侧;只覆盖不 dump 的验证可设 `dump_after_load=false`,此时观察下游指标/loss 恢复即可)。注意:load 覆盖后,被覆盖模块及其**直接上游**的 backward 数据在 dump 中可能缺失——只影响 dump 采集,不影响模型实际反向结果。
6. **验证 load 生效**:比对覆盖边界两侧输入/输出 cosine=1.0(工具自动加载才生效);`debugger.stop()` 无"未命中条目"warning;`load.path` 缺失/`modules` 空/条目格式非法会启动报错。shape/dtype 不匹配只打 warning 仍替换,forward 可能报错——改造源数据时务必对齐。
7. **交接下一轮**:把 round N+1 的 dump 交接给 analysis 侧,**交接内容 = 新 dump 路径(dump_path)+ round 号 + 目标侧 + 被覆盖条目清单**。由它比对下游,决定继续打桩(还有独立差异)还是结束(下游全 EXACT / 只剩余舍入级)。

> **打桩方向**:要隔离/验证**训练侧**前序差异 → 用**推理侧** dump 覆盖训练侧模块输入(训练侧打桩);反之亦然。选"改动成本低、语义映射最清晰"的一侧。目的都是让两侧从覆盖边界起输入一致,从而判断下游是否还有独立差异。

## 8 恢复代码

全部采集/打桩轮次结束(analysis 侧已给出"所有独立差异点整体结论",或用户叫停)后恢复,无论结果如何都要恢复:

1. 按改动清单逐一还原(反向 patch / `git checkout` / 恢复备份)。
2. 验证:`git diff` 干净,或与改动前备份完全一致。
3. 确认无诊断残留:生产脚本和代码里没有 `DUMP_ON`、`PROMPTS_ONLY`、`dump_config_path`、debugger 调用、**`load` 配置段/改造后的源 dump 目录**等任何采集痕迹。

恢复是采集的组成部分,跳过恢复等于没做完。**打桩轮次之间不要恢复**——阶段 6→7→6 循环共用同一套接入。

## 采集完成 → 交接

每轮采集/打桩后,把两侧 dump(路径、step/rank、request_id 关联方式、采集配置、task_card)交接给 **consistency-dump-analysis** skill:第 0 轮做语义映射、逐边界精确比对、首差异定位与定性,输出**打桩指令**;后续轮次做差异点隔离验证,直到输出**所有独立差异点整体结论**。采集侧只负责:产 dump、按打桩指令做 load 二次采集、全部轮次结束后恢复代码。**本 skill 不承担任何分析/报告职责**。

## 纪律(借自 true-on-policy-align,精简版)

- **现场验证优先于 reference**:任何代码改动以现场 grep / 数据流验证为准,references 仅作线索;reference 有错要能识别,以现场为准并采完修正。本 skill 不预设特定框架。
- **证据决定采集完成**:没有 `dump.json`、关联不上,就不算采集完成。
- **只记事实**:不知道的标 `unknown` / `evidence_missing`,不填默认值。
- **不泄露**:原始 tensor / dump / prompt / 凭据不进采集产物与交接文档。
- **最小 case 不是目标**:采集阶段做最小化是为了对齐,不代表最终范围。

## Reference 文件

| 文件 | 内容 | 何时读 |
|---|---|---|
| `references/verl.md` | verl 全部版本线索卡(SPMD ≤0.6 / Async v0.7–0.8 / V1 v0.9):先识别版本架构,再看对应节 | 识别到 verl(任何版本) |
| `references/slime.md` | slime v0.2.2(Megatron + SGLang)线索卡:训推分卡、padding 约束、SGLang 原生/侵入接入 | 识别到 slime |
| `references/vime.md` | vime(Megatron-Bridge 训练 + vLLM-ascend 推理)适配;vLLM 启动链路、设备分配(base_gpu_id=ray 物理卡号,勿二次映射)、引擎起不来排查、训练侧接入(官方直接改代码,不写 hook) | 识别到 vime |
| `references/msprobe-config.md` | PrecisionDebugger 接口 + config 模板 + dump.json 字段 | 配置或验证时 |
| `references/stubbing.md` | **打桩二次采集(数据覆盖加载,msprobe `load`)执行手册**:官方 load 文档要点、训推场景命名改造(重命名 .pt/对齐 shape/dtype)、config 写法、验证与坑 | 阶段 7(打桩二次采集)前必读 |
| `references/templates/stub_instruction.example.json` | 打桩指令 `stub_instruction.json` 示例模板(训推融合 vs 独立算子场景:融合 QKV→独立 Q 切片;算子名/路径/维度仅为占位符,以现场语义映射为准) | 阶段 7 接打桩指令时对照 |
| `references/prefill-logprob-diff.md` | **prefill logprob_diff(采集侧产出的数值指标,训练循环内直接算出打印日志)**:训练侧裁剪 prefill logprob vs 推理侧 prompt 段 logprob 的 diff——训练循环碰头处 `log_softmax` → 按 token 对齐(去 padding/末位)→ 逐 token 概率差 + mean/max/std 聚合 → 打印训练日志(解读归 consistency-dump-analysis) | 阶段 5 训练循环内接入计算、阶段 6 验证日志输出 |
| `references/probs-diff-monitoring.md` | 训推 response 段 E2E 指标(`rollout_probs_diff` / `train_rollout_logprob_abs_diff`):含义、verl 原生开关、各框架落点、通用落地方法;PROMPTS_ONLY 裁剪下 metrics 主动返回 0 值(valid=0/mean=0,是"没算"不是"一致"),要实测需 PROMPTS_ONLY=0 标准训练 | 阶段 5 训练侧接入、阶段 6 验证 E2E |
| `references/new-framework.md` | **新框架/未知框架适配通用方法论**:现场识别清单、三件事通用找法、如何验证 reference 是否正确(自我纠错) | **各框架的阶段 3**(含新框架、文档过时/错误的场景) |
| `references/pitfalls.md` | 实战失败教训汇总(PROMPTS_ONLY 裁剪副作用、msprobe step 语义、docker exec/残留进程、补丁脚本嵌套引号、prompt_logprobs 嵌套结构、漏 rm-type、ray job 拆 JSON、TE/F.silu 接入坑、孤儿 EngineCore 清理等) | 流程前通读一次;阶段 5/6 遇到对应症状时对照 |
| `references/task_card.json` | 目标/授权/环境记录模板 | 阶段 0 |

**职责边界**:本 skill 只负责采集与打桩——采集到"两侧可比的 prefill dump",按 analysis 侧打桩指令做 load 二次采集产"打桩隔离后的下游 dump";**分析差异点、首差异定位、输出打桩指令、所有独立差异点整体结论由 `consistency-dump-analysis` skill 承担**(其 references 含 analysis.md / semantic-mapping.md / report-source.md / stubbing-loop.md 与 probe_dump.py / analyze_dump_boundaries.py / generate_alignment_report.py 脚本)。每轮采集后把 dump 路径与 task_card 交接给它,等它的打桩指令;全部轮次结束后恢复代码。
