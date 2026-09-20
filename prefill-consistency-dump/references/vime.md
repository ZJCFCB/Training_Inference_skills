# vime 框架适配线索卡(Megatron-Bridge 训练 + vLLM-ascend 推理)

vime 是基于 verl 封装的训推框架:训练侧是 Megatron-Bridge(`/root/vime/vime/backends/megatron_utils/actor.py`),推理侧是 vLLM-ascend(**由 vime rollout 自动拉起**,不是单独起服务)。actor(训练)+ rollout(vLLM 推理)按卡位分角色。基于 2026-09-10 GLM-4.7 真实排查写成。

> **地图不是坐标**:路径/函数名/签名基于某次真实核对,可能随版本不同。改任何文件前先在本机仓库 `grep` 确认符号存在,差异就按真实签名适配并记入改动清单。

## 设备分配机制(最高优先级,踩坑重灾区)

vime 的 vLLM 设备分配**原生就是正确的,支持各卡位**(生产 0-3 / 分卡 4,5 / 6,7 均真实验证引擎+rollout 正常)。不要凭函数名表象断言"非 0 基址缺陷",更不要在 `_compute_server_args`/设备分配处加"映射 patch"。

机制链(启动 vLLM 引擎前,ray 已把卡分给各 actor):

```text
rollout.py  pg, reordered_bundle_indices, reordered_gpu_ids = self.pg   # placement group
rollout.py  base_gpu_id = int(reordered_gpu_ids[gpu_index])            # ★ 真实物理卡号
            → base_gpu_id 显式传给 VLLMEngine(即 RolloutRayActor)
vllm_engine.py  _compute_server_args: visible_devices = ",".join(str(base + i) for i in range(local_num_gpus))
                build_vllm_subprocess_env: env["ASCEND_RT_VISIBLE_DEVICES"] = server_args["visible_devices"]
```

**`base_gpu_id` 来自 ray placement group 的 `reordered_gpu_ids`,本身就是物理卡号**;原生 `visible_devices = base_gpu_id + i` 直接就是物理卡号。`get_base_gpu_id` 那套"索引取模"只用于默认布局,运行时 base 由 ray 显式注入,别混淆。

**真实事故(2026-09-10)**:误把已是物理卡号的 `base_gpu_id` 当"可见索引",从 `ASCEND_RT_VISIBLE_DEVICES="4,5"` 二次取 `_vis_ids[base + i]`,base=5 越界 → 空列表 → `visible_devices=''` → vLLM 子进程 `ASCEND_RT_VISIBLE_DEVICES` 为空 → EngineCore 崩,反复 7 次才定位。同一脚本回滚成原生后,4,5 和 6,7 **都能正常启动**。

**"引擎起不来"三步排查**(按此顺序,别急着改代码):

1. 看 vLLM 子进程**实际拿到**的 `ASCEND_RT_VISIBLE_DEVICES`(启动日志里 `visible_devices=''` / 容器内 `ps e -p <pid>`)。为空 = 设备传递链断了。
2. 沿 `_compute_server_args → build_vllm_subprocess_env` 确认 `visible_devices` 怎么变成空的。
3. **回滚自己的改动 → 原生跑一次对照**(同一脚本只改卡号)。原生能起,就是你的 patch 引入了 bug,不是框架/卡的缺陷。

崩溃点:启动链路 `VLLMEngine → _compute_server_args → build_vllm_cmd_and_env → build_vllm_subprocess_env → launch_server_process → _exec_vllm_cmd`。vime `utils/common.py` 的 `is_npu()` 在 `ASCEND_RT_VISIBLE_DEVICES` 为空时**要么 True 要么 raise**,所以报错是 `RuntimeError: torch_npu detected, but NPU device is not available or visible.`,不是"没走 NPU 分支"。

## 三件事的定位(现场 grep 确认符号)

| 事项 | 位置(线索) | 现场验证 |
|---|---|---|
| PROMPTS_ONLY 裁剪 | 训练侧 `actor.py` `train_actor`,在 `get_data_iterator(rollout_data)` 之前,若 `DUMP_ON=1` 且 `PROMPTS_ONLY=1`,构造 prompt-only 副本:tokens 截到 prompt_length(`total_length - response_length`,data.py:139),response_lengths 置 0,loss_masks 置空,total_lengths 改 prompt_length | 裁剪副本只喂 log_prob 前向,不替换整个 rollout_data;采完 `git checkout actor.py` |
| 训练侧接入 | 主前向是 `actor.py` `compute_log_prob(store_prefix="")` → `model.py` `forward_only`。**官方做法:直接改源码,不写独立 hook 文件,也不走 vime 预留的 hook 入口(`--custom-megatron-before-log-prob-hook-path`,勿用)**——actor 构造时按 `DUMP_ON=1` 实例化 `PrecisionDebugger`(env-gated,否则 `None`),`compute_log_prob` 主前向处 `start(model=...) → 前向 → stop() → step()`,只采 actor 主前向(ref 引擎 `forward_only` 跳过);裁剪 + loss/metrics 处理也直接改源码,对齐官方 `*_consistency_preprocess_dump.md` | 改动记入改动清单,采完恢复 |
| 推理侧 msprobe | vLLM-ascend 0.22.1 `model_runner_v1.py` 原生集成 PrecisionDebugger,不用改码,传 config 即启用:`--vllm-enforce-eager`(★ cudagraph_mode==NONE 才走 PrecisionDebugger,必须带)+ `--vllm-additional-config '{"dump_config_path": "<session>/msprobe_infer_config.json"}'` | config 示例 `{"task":"tensor","dump_path":"<session>/dumps/infer","rank":[],"step":[0],"level":"mix"}`(两侧统一 level=mix) |
| request_id 关联 | 按通用方法:`references/new-framework.md` §2③ | vllm ≥0.14 的 8 位 hex 后缀剥除 |

## 训练侧指标与碰头点(vime 特有,2026-09-15 现场确认)

- **response 段 E2E 指标**:vime 里叫 `train_rollout_logprob_abs_diff`(`loss.py`,定义 `(old_log_probs − rollout_log_probs).abs()`),锚定 response 段;PROMPTS_ONLY 裁剪下 response 段被裁,官方 metrics 缺字段主动返回 0 值(valid=0/mean=0 是"没算"不是"一致",见 `references/probs-diff-monitoring.md`)。
- **PROMPTS_ONLY 下使其可算(两处,缺一不可)**:
  1. `model.py` 训练前向 `get_batch` 的 keys 列表**必须加 `"prompt_logprobs"`**——否则 `policy_loss_function` 的 `batch.get("prompt_logprobs")` 恒为 None,推理侧 prompt 段 logprob 进不了 loss。
  2. `loss.py` 的 `train_rollout_logprob_abs_diff` 分支:PROMPTS_ONLY 下将 `old_log_probs`(train prompt 段)与 `batch["prompt_logprobs"]`(推理侧同一 prompt 段)对齐,`torch.cat` 后算 `(old_log_probs − rollout_log_probs).abs().mean()`——**绕过** `sum_of_sample_mean`(`get_sum_of_sample_mean` 在 `response_lengths` 全 0 时是常量-0 stub,任何传入值都返回 0),直接用 `.mean()` 按 prompt token 归约;非 PROMPTS_ONLY 保持原逻辑。
  - 数值口径:raw logprob 空间的 prompt 段均值绝对差。prompt 段是大量低概率 token,很小的概率差会被放大成几个 nats 的 log 差;与标准训练 response 段(高概率 token,log 差小)数量级不同属预期,须同口径对比。
- **prefill logprob_diff 碰头点**:vime 在 `log_rollout_data` / `compute_advantages_and_returns` 附近(此时 actor 主前向已把 `log_probs` update 进 rollout_data;推理侧 prompt 段 logprob 在 train_data 的 `prompt_logprobs`)。**2026-09-17 实测实现**:`actor.py` 新增 `_print_prefill_logprob_diff(rollout_id, rollout_data)`,数据源 `rollout_data["lp_train_prompt"]`(训练侧 log_softmax 后的 prompt 段)+ `rollout_data["prompt_logprobs"]`(vLLM prompt_logprobs 解析),逐 token `|exp(lp_train)-exp(lp_infer)|` + mean/max/std 打印。`lp_train_prompt` 在 `model.py` 主前向 `forward_only` 里按 `log_softmax` 取 `[:, :-1]` 存入 rollout_data(DUMP_ON gate)。通用方法见 `references/prefill-logprob-diff.md`。
- **`rollout_log_probs` 写入条件**:`samples[0].rollout_log_probs is not None` 时才写入 train_data(`ray/rollout.py:732`);未返回则 probs_diff 函数静默返回空 dict 不报错——须验证训练日志确实出现输出,不能只看未报错。
- **response 段 mask 字段**:vime 用 `loss_masks`(非 verl 的 `response_mask`)。
- **框架预留 hook 入口**:`--custom-megatron-before-log-prob-hook-path` **不要使用**——官方形态是直接改源码(`actor.py` 实例化 PrecisionDebugger 并包住 `compute_log_prob`),hook 入口属偏离官方形态的写法。

## 启动脚本约束(分卡模式)

- 卡位由 `VIME_DEVICES` + `VIME_NUM_NPUS` 定义(生成 ray 的 NPU 资源),`ASCEND_RT_VISIBLE_DEVICES` 与 `VIME_DEVICES` 保持一致;`RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`。
- ray head:`ray start --head ... --num-gpus=0 --resources='{"NPU": N}'`;作业经 `ray job submit` 跑 `train.py`。
- **分卡模式**(actor 卡 + rollout 卡分开):`--actor-num-gpus-per-node 1 --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1`(每 engine 单卡即 TP=1)。生产全量模式是 actor 2 + rollout 2、`--rollout-num-gpus-per-engine 2`。
- 确定性单样本:不传 `--rollout-shuffle`;`--num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 1 --rollout-max-response-len 4 --rollout-temperature 0.0 --global-batch-size 1`;关 eval(MTP 关、不传 `--eval-interval` 或 `--skip-eval-before-train`);`--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-* 1`。
- 无 pad:`--data-pad-size-multiplier 1`(或 vime 等价),TP=1 时无需填充,天然对齐。

## 已知坑(训练侧,与引擎无关)

采集验证 job 即使引擎/设备全正常,也可能在 actor 训练反向报 `found NaN in local grad norm for bucket #0`——这是**训练侧问题**(如 5 层截断权重 + 单样本确定性配置),不代表引擎或采集失败。判断时把"引擎起来/rollout 成功"与"训练反向 NaN"分开看。

> **运行环境不预设容器形态**:vime 的部署环境(容器/裸机、PID namespace、容器名、docker exec 方式)因机器而异,不属于框架本身。清卡/杀残留/确认可见卡等环境动作一律按 SKILL 阶段 1 预检清卡现场确认,勿照抄本文档或记忆里的具体容器名。

## Megatron-Bridge 特有坑(采集接入)

### TE 硬依赖假阻塞(lora_layers.py:20)——先可选化再谈"不支持"
- **症状**:孤立环境(无 transformer_engine)里任何 `import megatron.bridge` 直接崩 `ModuleNotFoundError: transformer_engine`;连锁假象是 `AutoBridge._validate_config` 报 `'Glm4MoeLiteForCausalLM' is not yet supported`(注册语句因导入期崩溃从未执行)。
- **根因**:`peft/lora_layers.py:20` 顶层**无条件** `import transformer_engine.pytorch as te`;TE 只服务 LoRA,但 `import megatron.bridge` 必经,TE 缺失即全链崩。
- **修复**:TE 可选化(不碰 LoRA 场景数值):
  1. 顶层 `import transformer_engine.pytorch as te` → `try/except ModuleNotFoundError`(`te=None, HAVE_TE=False`);
  2. `class TELinearAdapter(te.Linear):` → `class TELinearAdapter(te.Linear if HAVE_TE else nn.Module):`(类定义期唯一 TE 依赖);
  3. 文件顶部加 `from __future__ import annotations`——**Python 3.12 无 future 时类体内方法返回注解在类定义期求值**,`-> tuple[te.ops.Sequential, ...]` 在 te=None 时即崩。
- **预防**:① 遇 "not yet supported" 先查导入链是否有硬依赖在**导入期**炸(注册没执行到),别只盯着 AutoBridge 校验;② 孤立环境无 TE ≠ 作业进程无 TE——modelopt 警告("module 'transformer_engine.pytorch' has no attribute 'RMSNorm'")说明作业里 TE 模块存在但缺 RMSNorm,注入源不明,别在这上面耗;③ 这类改动是**采集态补丁,采完用 bridge_backup 还原**。

### msprobe 替换 F.silu 的引用缓存连锁(三层,时序敏感)
- **症状**:msprobe(level=mix)提前全局替换 `torch.nn.functional.silu` 后,Megatron 模型构造报错:
  1. 第一层 `mlp.py:198`:`self.activation_func == F.silu` 引用相等失败 → `Only support fusion of gelu and swiglu`(debugger 晚于模型构造时);
  2. 第三层 `transformer_config.py:1438`:`bias_activation_fusion` 校验 `self.activation_func not in [F.gelu, F.silu, quick_gelu]` 失败(debugger 提前到模块导入期后仍崩)。
- **根因**:Megatron-Bridge `model_bridge.py:322` `ACTIVATION_MAPPING = {"silu": F.silu, ...}` 在**模块导入期**缓存原生 silu 引用;debugger 提前到模块导入期后,`model_provider → import megatron.bridge` 链在 debugger 实例化**之前**跑完,缓存仍是原生引用;模型构造时 `hf_to_megatron_activation("silu")` 返回缓存的**原生 silu**,而校验对照的动态 `F.silu` 已是**包装函数** → 引用不匹配 → 崩。
- **修复**(`model_bridge.py` 两处):
  1. `hf_to_megatron_activation`:`hidden_act in ("silu","gelu")` 时**调用期动态访问 `F.silu`/`F.gelu`**(此时即包装函数,数值与原生一致),其余走映射表;
  2. `megatron_to_hf_activation` 反向兜底:`activation_func is F.silu` → `"silu"`、`is F.gelu` → `"gelu"`(否则包装函数对不上缓存原生引用会 raise)。
- **通用教训**:msprobe 替换 op 后,凡"模块导入期缓存原生 F.silu 引用、替换后与动态 F.silu 做**引用相等/成员判断**"的代码都会崩。搜索特征:`not in [F.gelu, F.silu, quick_gelu]` 校验、`ACTIVATION_MAPPING = {"silu": F.silu}` 缓存、`self.activation_func == F.silu`。修复一律改**调用期动态求值**,不是改 msprobe 排除 silu(排除了就采不到 silu 边界)。
- **预防**:接入 msprobe 前先把"谁缓存了 F.silu 原生引用"扫一遍(`grep -rn "F.silu"` 模型/转换代码);改了 debugger 实例化时机后,重新扫一遍调用链,时序一变连锁就变。

### 采集环境变量
| 变量 | 作用 |
|---|---|
| `VIME_TRAIN_DUMP_CONFIG` | 训练侧 msprobe config 路径(debugger 模块导入期读取) |
| `DUMP_ON=1` | 训练侧开启采集(模块导入期实例化 debugger,env-gated) |
| `PROMPTS_ONLY=1` | 裁剪 prompt-only |
| `--data-pad-size-multiplier 1` | 去训练端 padding(190 对齐) |

## 环境变量汇总

| 变量 | 作用 |
|---|---|
| `DUMP_ON=1` / `PROMPTS_ONLY=1` | 开启裁剪 + 采集(采集代码读) |
| `VIME_TRAIN_MSPROBE_DUMP_PATH` | 训练侧 dump 根目录(采集 env-gate) |
| `ASCEND_RT_VISIBLE_DEVICES` / `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` | 卡位 + 禁止 ray 覆盖可见卡 |
| `VLLM_ADD_CONFIG`(或 `--vllm-additional-config`) | 推理侧 msprobe config(dump_config_path) |

## 结果目录与比对

```text
{session}/dumps/train/step0/rank{M}/{dump.json,...}    # 训练侧(路径来自 VIME_TRAIN_MSPROBE_DUMP_PATH)
{session}/dumps/infer/step0/rank{M}/{dump.json,...}    # 推理侧(来自 dump_config_path)
```

两侧 dump 落盘后,语义映射、逐边界比对、首差异定位、ULP 量化、放大链、排除假设、结论与报告由 **consistency-dump-analysis** skill 承担(本 skill 只采集不分析)。

