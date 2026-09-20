---
name: consistency-dump-analysis
description: 通用训推一致性(prefill 训推一致)的 dump 数据分析 skill。分析**已采集**的训推两侧 dump 数据:探查 dump 结构 → 按语义建立边界映射(不要求算子同名)→ 逐边界精确门禁比对(same_shape/same_dtype/no_nonfinite/torch.equal/max_abs_diff)→ 沿语义执行顺序定位首差异 → ULP 量化区分舍入级与逻辑级差异 → 追放大链 → **差异点隔离验证(打桩迭代:输出打桩指令给采集侧,等它用 msprobe load 覆盖边界再采后比下游,枚举所有独立差异点)** → 排除根因假设 → **输出所有独立差异点整体结论** → 产出四格式对齐报告(JSON/TXT/MD/HTML)。与具体训练/推理框架、硬件解耦,适用于 msprobe PrecisionDebugger 或等价工具采集的任何 dump。当用户提到训推一致性分析、分析训推差异、dump 数据分析、dump 已经采好了帮我分析、首差异定位、所有差异点、差异点隔离、打桩指令、对齐报告、训推不一致结论、diff dump、比较两侧 tensor 时,使用本 skill。只要用户在做训推一致性分析的解读,都调用本 skill;采集与打桩再采由 prefill-consistency-dump 负责,本 skill 只消费已落盘的 dump、输出打桩指令驱动再采。
---

# 训推一致性 Dump 数据分析

拿到两侧已采集的 dump(training prefill forward vs inference prefill forward,同一步/同一请求),离线比对出训推不一致的首差异在哪、根因是什么、放大到多大,并**通过打桩迭代枚举所有独立差异点**,产出机器可读结论与四格式报告。**核心约束:比对必须按语义边界,不按算子名**——两侧算子命名差异极大(训练侧独立的 Q 投影算子 ↔ 推理侧融合 QKV 投影输出的 q 段切片是常态),强行同名比对等于没比。本 skill 是**分析 + 差异点隔离**,只读 dump、不改训练/推理代码;**需要再采数据时输出"打桩指令"交给采集侧 `prefill-consistency-dump` 执行(msprobe `load` 覆盖边界再采),本 skill 拿到下一轮 dump 继续比下游,直到枚举出所有独立差异点**。

## 本 skill 做什么 / 不做什么

- **做**:读 dump(含元数据与 tensor 文件)、探查两侧算子轨迹、建立语义边界映射、逐边界精确门禁比对、定位首差异、ULP 量化、追放大链、**输出打桩指令(差异点隔离)驱动采集侧再采、枚举所有独立差异点**、排除假设、输出整体结论、产出四格式对齐报告。
- **不做**:采集/打桩重跑 dump(那是 `prefill-consistency-dump`,按本 skill 的打桩指令用 msprobe `load` 覆盖边界再采)、修复差异并验证、接入训练循环算指标。本 skill 消费**已落盘**的两侧 dump,只读。
- **纪律**:dump 只读;原始 tensor / prompt / 凭据不进报告、不进仓库,报告只留聚合数值与边界统计;证据不足标 `evidence_missing`,绝不编数值。**本 skill 可以"要求再采一轮"(输出打桩指令),但自己不碰采集/代码。**

## 执行顺序

```
0 锁定范围 → 1 探查 dump 结构 → 2 语义边界映射 → 3 逐边界精确比对
  → 4 首差异定位与定性 → 5 差异点隔离验证(打桩迭代)
  → 6 排除假设 → 7 所有独立差异点整体结论 → 8 产出报告
```

按顺序走,1 和 2 可以合并(探查时顺手记下两侧算子命名规律)。阶段 5 是本 skill 与 `prefill-consistency-dump` 的**循环接缝**:本 skill 输出打桩指令 → 采集侧 load 覆盖边界再采 → 本 skill 比下游 → 有独立差异就再输出下一轮指令,直到下游全 EXACT / 只剩余舍入级。首轮可以没有阶段 5(下游全 EXACT 直接到结论);有差异才进打桩迭代。

## 0 锁定范围

动手前先写清(记入 `task_card.json`,模板见 `references/task_card.json`):

- 两侧 dump 路径(training 侧与 inference 侧,各自 `dump_tensor_data` 目录)
- 模型 / 训练后端 / 推理引擎及版本 / 硬件
- 采集配置:step、rank、request_id 关联方式(若能拿到)、prompt token 长度、并行切分(TP/PP/EP)
- 目标:只需首差异?还是要完整报告?(默认完整报告)
- 已知陷阱:训练侧 tensor 常带前导/中间 1 维 batch 维、两侧输入是否有 padding、推理侧是否有逐 position 而非全量 tensor

只记事实,不知道的标 `unknown`,不编默认值。

## 1 探查 dump 结构

用 `scripts/probe_dump.py`(通用探查脚本,不绑定架构)快速摸清两侧 dump:

1. **读元数据**:两侧 `dump.json`(`task/level/framework` 等)、`construct.json`(算子轨迹,dict 的 key 顺序即执行顺序)、`stack.json`。
2. **列算子轨迹**:两侧 `construct.json` 的 key 列表——每条 key 是一串以 `.` 分隔的算子路径(常为 `前缀.模块路径.算子类名.forward.<序号>`),训练侧与推理侧的**模块路径前缀、算子类名、融合方式都可能不一致**;对比两侧前缀规律、算子类名差异、融合算子(fused 名称常含多个子算子的语义)。
3. **列 tensor 文件**:`dump_tensor_data/*.pt` 文件名 = 算子路径 + `forward.{idx}.output.{n}.pt`(输出)/ `.input.{n}.pt`(输入)/ `.parameters.weight.pt`(权重);统计每侧文件数、每个算子的 input/output/weight 分布。
4. **随机抽查 shape/dtype**:任挑几个同名或同语义算子两侧的 tensor,记 shape/dtype 差异规律(前导 batch 维、中间广播维、逐 position vs 全量)。

**探查结论记入分析记录**:两侧算子命名规律(前缀替换规则、融合算子清单)、需要切片对齐的边界候选。这是阶段 2 建映射的原材料。

## 2 语义边界映射

核心方法论:**按语义对齐,不按名字**。语义边界 = 数学上同一位置的张量(如「GEMM 前输入归一化输出」「Q 下投影 GEMM 输出」「attention 输出(reshape 前/后)」「残差加和输出」「最终 logits」)。完整方法见 `references/semantic-mapping.md`。

1. **划语义边界清单**,沿数据流:embedding → 各层输入归一化 → Q/K/V 下投影 → (norm →) 上投影 → RoPE → attention → o_proj → 残差 → MLP(fc1/act/fc2 或 MoE)→ 最终归一化 → logits/lm_head。
2. **逐边界对齐两侧算子**:训练侧某个算子的输出,数学上等于推理侧哪个算子的输入/输出?靠 construct 轨迹顺序 + 数据流 + shape 校验对齐。融合算子用**切片/拆分**对齐(如推理侧融合 QKV 投影输出 `[seq, q_kv总维度]`,按权重行序切成 q 段与 kv 段,分别对训练侧独立的 Q/KV 投影);切片前先做相关性扫描确认布局(见 references/semantic-mapping.md §布局确认)。
3. **权重边界单独对齐**:两侧权重(embedding / qkv / o_proj / lm_head)同样按语义切好对齐,用于阶段 6 排除权重不同步。
4. **映射不了的边界标 `evidence_missing`,不猜**:shape 对不上、语义找不到对应、结构差异过大(MoE 内部路由/permute 两侧算子完全不同)的,记下来但不硬比。

产出:**语义映射表**(边界名 / 训练侧算子 glob / 推理侧算子 glob / 变换规则(切片、squeeze、reshape)/ 映射依据)。这正是 `scripts/analyze_dump_boundaries.py` 的 `--mapping` 输入。

> 本 skill 不带任何预设映射:每个新 dump 都按本阶段方法论,以两侧实际算子名/shape **现场建映射**,写进 `--mapping` JSON。方法与判定指纹见 `references/analysis.md` 与 `references/semantic-mapping.md`。

## 3 逐边界精确比对

用 `scripts/analyze_dump_boundaries.py --train <训练侧 dump_tensor_data> --infer <推理侧 dump_tensor_data>` 对每个边界跑精确门禁:

- `same_shape` / `same_dtype` / `no_nonfinite` / `torch.equal` / `max_abs_diff=0`

沿 `execution_order`(语义执行顺序)输出每条边界的证据,直到第一个不等的边界。用法与 `--mapping` 格式:

```bash
python3 scripts/analyze_dump_boundaries.py \
  --train <train>/step0/rank0/dump_tensor_data \
  --infer <infer>/step0/rank0/dump_tensor_data \
  --mapping semantic_mapping.json \
  --output boundary_results.json
```

`--mapping` 传自建语义映射 JSON(list of BOUNDARY,格式见脚本头注释),**新架构直接写 JSON,不必改脚本**。比对只读 dump。

结果分类:每边界 ∈ {`EXACT`, `NOT_EXACT`, `SHAPE_DIFF`, `DTYPE_DIFF`, `NONFINITE`, `MISSING`}(脚本已输出);加上语义层面映射不上的标 `evidence_missing`。

## 4 首差异定位与定性

首差异 = **沿语义执行顺序第一个非 EXACT 的边界**。脚本已按 execution_order 排序并给出 `first_divergence`。

关键纪律:**首差异必须是最早的那一个**——下游所有非零差异都是它放大的结果。若比对发现某边界是首个 NEQ,但更靠前的边界被跳过/漏比,先补全前序边界再定论。核对方式:首差异的**所有前序边界必须全部 EXACT**(输入已精确对齐、前序计算无差异),否则继续前推。

首差异往往是极小值(如融合 vs 独立 GEMM 的 ±1~2 ULP),必须量化定性它是**舍入级**还是**逻辑级**:

1. **ULP 量化**:对首差异边界,算 `diff / bf16(f32)ULP(value)` 的倍数分布。典型指纹:
   - **舍入级**(融合 vs 独立 GEMM / 不同累加顺序):差异极稀疏(占比 <1‰)、倍数集中于 ±1~2 ULP、position 稀疏分布、无随步长大差。→ 属预期浮点差异,非逻辑错误。
   - **逻辑级**(配置/权重/内核错误):差异密集或倍数远超 ULP、形态有规律(整行/整列/整 position)。
2. **放大链**:记录首差异量级 → 沿下游边界逐层 max_abs 变化(首差异常是 1e-4~1e-2 量级,经 norm/attention/MLP 逐层放大可到 1e0~1e2 量级),证明"首差异很小但被逐层放大到最终指标非零"。放大链是结论的证据主线。

3. **定性决定是否进打桩**:首差异为**逻辑级** → 下游所有差异可能只是它放大、也可能有第二独立差异,进阶段 5 打桩隔离;首差异为**舍入级**(ULP 指纹)→ 若下游也只剩舍入级可不再打桩,直接记舍入级差异点(已知 gap);若下游仍存在逻辑级差异,仍需打桩隔离舍入级首差异后确认下游独立差异。

## 5 差异点隔离验证(打桩迭代,输出打桩指令)

要回答"下游非零差异是首差异的放大,还是存在第二个独立差异?",用**打桩(数据覆盖加载)**把已确认差异边界"强行对齐"后比下游。完整方法论、打桩指令 schema 与判定指纹见 `references/stubbing-loop.md`。

1. **判定是否需下一轮**:
   - 下游所有边界 EXACT → 已隔离差异点是唯一独立差异,无下一轮。
   - 下游存在逻辑级差异(非 ULP 指纹)→ 可能为下一独立差异,进打桩。
   - 下游仅剩余舍入级(ULP 指纹)→ 记入舍入级差异点清单,可结束。
   - 下游存在 evidence_missing 未比边界 → 先补映射或标注,不因"没比"误判"无差异"。
2. **输出打桩指令** `stub_instruction.json`(round N,交给 `prefill-consistency-dump`;模板见 `references/templates/stub_instruction.example.json`):
   - `source_side` / `target_side`:覆盖方向(要隔离哪侧前序误差,就用另一侧数据覆盖它)
   - `isolate_boundary`:本轮隔离边界(语义名 + 两侧算子)
   - `modules[]`:每条 = 目标模型条目名(`Module.{dotted_path}.{ClassName}.forward.{N}`)+ 源 `.pt` 文件 + **变换规则**(重命名/切片/reshape/dtype 对齐,来自阶段 2 语义映射)
   - `dump_after_load`:覆盖后继续 dump 下游
   - `expected`:覆盖后该边界两侧输入应一致;若下游仍逻辑级差异 → 确认下一独立差异
3. **下一轮分析(采集侧返回 round N+1 dump 后)**:先验证**隔离边界本身**两侧 bitwise 一致(load 生效证据,对应采集侧阶段 7 步骤 6;不一致说明打桩没生效,先别比下游),再比对被隔离边界之后的下游:
   - 全 EXACT → 确认此前的差异都是已隔离差异点的放大,无新独立差异,结束。
   - 仍逻辑级差异 → 确认第 N+1 个**独立差异点**,记录其边界/定性/证据 → 输出下一轮打桩指令(隔离新边界)。
   - 残留舍入级 → 记入舍入级差异点清单。
4. **停止条件**:下游全 EXACT / 只剩舍入级 / 剩余边界 evidence_missing(标注后停止)。所有独立差异点枚举完,进阶段 7 出整体结论。

## 6 排除假设

把可能的根因逐一列出并排除,每条记录排除依据。通用假设清单(按序):

1. **权重不同步**:权重边界(embedding/qkv/o_proj/lm_head)是否 `torch.equal` 精确一致?是 → 排除。
2. **输入未对齐 / padding**:首差异前序输入边界是否 EXACT?训练侧是否有 padding(带 batch 维与推理侧 shape 不一致)?是未去净 → 修正映射或标注,不回退。
3. **配置差异**:两侧注意力/MLP 结构(独立 vs 融合、RMSNorm 融合位置、MoE 路由)是否语义等价?结构不同但数学等价 → 不算根因。
4. **并行切分不一致**:两侧 TP/PP/EP 是否一致?一致 → 排除。
5. **fused 内核 / 累加顺序**:首差异形态符合 ULP 指纹 → 定性为浮点舍入,非错误。
6. **随机性**:采样/seed 是否确定性?单样本确定性配置 → 排除。

排除假设的证据和首差异证据同等重要——结论里必须写清「为什么不是权重/配置/输入问题」。

## 7 所有独立差异点整体结论

1. **边界级分类**:每个边界 `EXACT` / `NOT_EXACT` / `evidence_missing`(含 SHAPE_DIFF/DTYPE_DIFF/NONFINITE/MISSING 归入 NOT_EXACT 或不比)。
2. **差异点清单(整体结论的核心)**:枚举**所有独立差异点**(经阶段 5 打桩隔离验证),每个记:
   - 序号 + 隔离轮次(第几轮打桩后确认它是独立差异:覆盖其前序差异边界后下游仍逻辑级差异)
   - 边界位置(语义名 + 两侧算子)
   - 定性:舍入级(ULP 指纹,非错误,已知 gap)/ 逻辑级(配置/权重/内核)
   - 放大链(首差异 → 最终指标)
   - 根因假设 + 排除依据(阶段 6)
   - 是否可解
   附:舍入级差异点清单、evidence_missing 清单(未比边界)。
3. **整体级四分类**(映射到报告):全部边界 EXACT → `exact`;存在逻辑级独立差异 → `nonzero`;未执行 → `not_run`;证据不足 → `evidence_missing`。全部差异点仅舍入级 → 训推一致(允许浮点差异,须写明每个舍入级差异点)。
4. **结论口径**:训推不一致结论必须指向「**全部独立差异点(位置/定性/根因/证据链/是否可解)**」,不能只报首差异或最终指标非零。差异点定性为舍入级(ULP 指纹)时,明确写「非逻辑/配置/权重错误,属浮点累加差异」。
5. **数值指标口径**(如有 prefill logprob diff / E2E 指标,只引用训练日志实测):
   - prefill 段 logprob diff(训练循环直接算的 prefill logprob diff,日志实测 `mean/max/std`):`mean≈0` 是训推一致,非 0 结合逐边界比对定位;训练日志没有就 `evidence_missing`,不编。计算口径见 `references/prefill-logprob-diff.md`。
   - response 段 E2E(如 `rollout_probs_diff`)在 prompt-only 裁剪下无实测(valid=0 是「没算」不是「一致」);要数值须跑含 response 的标准训练实测,否则 `evidence_missing`。
   - 两指标不同层面,不互相冒充。

## 8 产出报告

产出两类:**分析结论记录**(每次必做)与**四格式对齐报告**(评估/交付必做)。

### 8.1 分析结论记录(必须)

- 两侧 dump 路径 / step / rank / request_id 关联(含每轮 round N 的 dump 路径)
- 语义映射表(或映射文件路径)
- 比对汇总:边界总数、EXACT/NOT_EXACT/evidence_missing 计数
- 打桩迭代记录:每轮打桩指令 + 采集侧返回的下游结果(覆盖后全 EXACT / 仍逻辑级差异 / 只剩舍入级)
- **所有独立差异点清单**(每个:序号、隔离轮次、边界位置、定性、放大链、根因假设、是否可解)
- 舍入级差异点清单(已知 gap)、evidence_missing 清单
- 排除假设清单
- 整体结论 + 未解决问题

### 8.2 四格式对齐报告(建议;评估/交付必做)

从单一事实源 `report_source.json` 用 `scripts/generate_alignment_report.py` 渲染四种格式:

```bash
# 首次: 从内置中文模板初始化 report_source.json(可跳过,直接手写也行)
python3 scripts/generate_alignment_report.py --init-source report_source.json
# 编辑 report_source.json 填入本案例事实后,生成四种格式:
python3 scripts/generate_alignment_report.py --input report_source.json --output-dir reports/
# 产出: reports/alignment_report.json / alignment_summary.txt / alignment_report.md / alignment_report.html
```

**report_source.json 要点**(完整 schema 与字段说明见 `references/report-source.md` 与 `references/templates/alignment_report_source.zh-CN.json`):

- `results`:`native/conditional_prefill_status` 与 `e2e_metric_status` 取 `exact|nonzero|not_run|evidence_missing`;**nonzero/zero 必须带数值 `e2e_metric_value`;没抓到实测就 `evidence_missing` + `value=null`**。
- `differences[]`:`status ∈ {excluded_known_gap, pending, resolved}`;**每个独立差异点一条**;逻辑级独立差异在整体 nonzero 时标 `excluded_known_gap`(已确认未解决差异),不伪装成 resolved;舍入级差异点可在 differences 或 limitations 中注明「浮点级,非错误」(属训推一致允许范围)。
- `verification_axes`:`tensor_prefill` 必须与 `native_prefill_status` 一致;`weight_sync` 记 `exact`(权重 bitwise 一致的证据)。
- `limitations[]`:如实记未覆盖边界、E2E 数值缺失、诊断模型范围。
- `provenance` / `comparison` / `artifacts`:记版本、prompt 对齐口径(长度、切片、因果位移)、产物清单。

> **环境提示(先读)**:三个脚本都不能在"裸宿主 python3"上直接跑——`probe_dump.py`/`analyze_dump_boundaries.py` 依赖 torch,`generate_alignment_report.py` 依赖 Python 3.12+(f-string 特性)。宿主默认 `python3` 若是 3.9 且无 torch,就在容器(有 torch + 3.12)里执行,产物拷回。
> **报告生成器的模板定位**:`generate_alignment_report.py` 从"脚本所在目录的上一级 + `references/templates/alignment_report_source.zh-CN.json`"读模板,**单独拷一个 .py 出来跑会报 `cannot read report source`**。容器内跑就保持相对结构(把整个 skill 目录拷进去,SKILL.md 里的 `scripts/...` 相对命令即可原位执行);`--init-source` 是标准起点,也可以从你自己历史产出的 `report_source.json` 复制改事实。

## 纪律

- **架构无关是设计约束**:主体方法论不绑定任何框架/模型/硬件;不带任何预设映射,一切以现场 dump 为准。
- **现场验证优先于 reference**:任何映射/比对以两侧 dump 的实际算子名、shape、dtype 为准;reference 只是线索,冲突时以现场为准并修正 reference。
- **只记事实**:不知道的标 `unknown` / `evidence_missing`,不填默认值、不编数值。
- **不泄露**:原始 tensor / dump / prompt / 凭据不进报告、不进仓库。
- **只读**:不改 dump、不写回采集侧代码。
- **可要求再采,不自采**:需要隔离验证时输出打桩指令给 `prefill-consistency-dump`,等它用 msprobe `load` 覆盖边界再采后比下游;本 skill 本身不碰采集/代码,也不在"没比到"的边界上假装"无差异"。

## Reference 文件

| 文件 | 内容 | 何时读 |
|---|---|---|
| `references/dump-structure.md` | msprobe/等价 dump 通用结构:dump.json / construct.json / stack.json / dump_tensor_data 字段与命名规则、两侧 shape/dtype 差异规律 | 阶段 1 探查前 |
| `references/semantic-mapping.md` | **语义边界映射方法论(架构无关)**:边界清单、融合算子切片对齐、权重边界、布局相关性扫描、evidence_missing 判定 | 阶段 2 |
| `references/analysis.md` | 逐边界精确门禁、首差异定位、ULP 量化、放大链、排除假设、分类结论的详细方法与判定指纹 | 阶段 3-4、6 |
| `references/stubbing-loop.md` | **差异点隔离验证(打桩迭代)方法论 + 打桩指令 JSON schema**:独立差异点 vs 放大差异判定、每轮打桩指令格式、迭代停止条件、整体结论格式 | 阶段 5(差异点隔离验证) |
| `references/prefill-logprob-diff.md` | prefill logprob diff(训练循环直接算的数值指标)的通用计算口径与解读陷阱 | 阶段 7 填数值指标时 |
| `references/report-source.md` | `report_source.json` 字段说明与填法(结合模板) | 阶段 8 填报告源时 |
| `references/task_card.json` | 目标/范围记录模板 | 阶段 0 |
| `references/templates/stub_instruction.example.json` | 打桩指令 `stub_instruction.json` 示例模板(训推融合 vs 独立算子场景:融合 QKV→独立 Q 切片;算子名/路径/维度仅为占位符,以现场语义映射为准) | 阶段 5 输出打桩指令时 |
| `references/templates/alignment_report_source.zh-CN.json` | `report_source.json` 中文模板与字段 schema | 阶段 8 初始化/填报告源时 |
| `scripts/probe_dump.py` | 通用 dump 探查脚本:读元数据、列两侧算子轨迹、列 tensor 文件、抽查 shape/dtype | 阶段 1 |
| `scripts/analyze_dump_boundaries.py` | 逐边界精确门禁比对脚本(`--mapping` 自建语义映射),输出 JSON + 首差异 | 阶段 3 |
| `scripts/generate_alignment_report.py` | 四格式对齐报告生成(JSON/TXT/Markdown/HTML),来自 true-on-policy-align | 阶段 8 |
