# Open-X-Cell 真实数据验证测试

端到端验证 Open-X-Cell 在**真实** perturb-seq 数据(HepG2 Nadig CRISPRi)上的训练 + 预测 + 评估全流程。脚本走的是项目自身的 `data / model / losses / inference / evaluation` 代码路径,而非合成数据自测,因此能反映真实表现。

## 数据

| 项 | 值 |
|---|---|
| 来源 | `vcc_benchmarks/datasets/hepg2nadig.h5ad`(X-Atlas/Pisces 风格,HepG2 CRISPRi) |
| 子集 | `data/real/hepg2nadig_subset.h5ad` |
| 形状 | 3900 cells × 400 genes |
| 扰动数 | 12 个真实靶基因(ENSG),各 200 cells |
| 对照 | 1500 个 non-targeting cells(`is_control=True`) |
| 基因 panel | 12 个被扰动基因 + 按 `mean-1` 表达量排序的高表达基因,共 400 |
| obs schema | `perturbation` / `context=hepg2` / `batch=batch0` / `is_control` |

**复现子集:** `python scripts/prep_real_subset.py`(需要源 h5ad 在上述路径)。

## 模型与训练配置

由 `scripts/validate_real.py` 定义,对齐 x-cell 论文的生产架构:

| 项 | 值 |
|---|---|
| 架构 | SwiGLU + RMSNorm + Pre-LN + QK-Norm + tied-output,无 attention bias |
| 层数 / hidden / heads / ffn | 4 / 128 / 4 / 256 |
| cross-attention 层 | (1, 3) |
| 参数量 | ~3.89 M |
| context_length / num_genes | 400(= panel 大小) |
| batch_size / set_size | 4 / 16 |
| max_steps | 300 |
| lr / grad_clip / optimizer | 3e-4 / 1.0 / AdamW |
| device | CPU |
| seed | 0 |

训练循环:每步按 `diffusion_fractions` 轮换 reveal 比例,`mix_control_target` 构造输入,zero-priors,`OpenXCellLoss`(energy distance/MMD + fold-change CCC + per-gene CCC + delta-norm + delta-MSE)。

## 评估指标

逐扰动用项目的 `predict_adata` + `evaluate_prediction_files`,再对 12 个扰动取均值:

| 指标 | 含义 | 方向 |
|---|---|---|
| `mae` | 预测表达 vs 真实的平均绝对误差 | 越低越好 |
| `pearson_delta` | 预测 Δ(扰动−对照)与真实 Δ 的 Pearson | 越高越好 |
| `overlap_at_n` | Top-N 差异基因重叠率 | 越高越好 |
| `direction_match` | 逐基因变化方向(上/下调)一致率 | 越高越好(0.5=随机) |
| `delta_norm_ratio` | 预测 Δ 范数 / 真实 Δ 范数 | 越接近 1 越好 |

## 运行方式

```bash
# 环境:eval(torch 2.7.1+cu118),CPU 模式
ENV=/public2/YouYuning/zongtingwei/zongting_env/miniconda3/envs/eval/bin/python
cd /public2/YouYuning/zongtingwei/Open-X-Cell
$ENV scripts/validate_real.py
```

产物写入 `runs/real_hepg2/`:`checkpoint.pt`、每个扰动的 `pred_<ENSG>.h5ad` + `metrics_<ENSG>.json`、汇总 `summary.json`。
注:`runs/` 与 `data/` 已被 `.gitignore` 忽略,不进版本库。

> stdout 在重定向到文件时是块缓冲的,训练中途日志可能为空,以 `summary.json` 被更新(mtime 刷新)作为完成信号。

## 结果

### 旧代码基线(commit `5237d7f`,2026-06-20)

备份于 `runs/real_hepg2_oldcode/`。

| 指标 | 均值 |
|---|---|
| MAE | 0.5019 |
| pearson_delta | 0.3159 |
| overlap_at_n | 0.125 |
| direction_match | 0.57 |
| delta_norm_ratio | 0.6879 |

loss:first10 ≈ 126.6 → last10 ≈ 1.92;参数 3.89M。
逐基因最好 ENSG00000189091(pearson_delta 0.54),最差 ENSG00000104866(0.16)。

### 新代码(commit `5d3467b`,"align training pipeline with x-cell paper",2026-06-21 05:02)

| 指标 | 均值 | vs 旧代码 |
|---|---|---|
| MAE | 0.5019 | 一致 |
| pearson_delta | 0.3159 | 一致 |
| overlap_at_n | 0.125 | 一致 |
| direction_match | 0.57 | 一致 |
| delta_norm_ratio | 0.6879 | 一致 |

loss:first10 ≈ 126.5772 → last10 ≈ 1.9188;参数 3.89M。
**结果与旧基线逐位相同**(`summary.json` 与 `runs/real_hepg2_oldcode/summary.json` 的 `summary` 和 `per_pert` 完全相等)。

#### 逐扰动明细(新代码)

| 扰动 (ENSG) | pearson_delta | direction | overlap@20 | dnorm_ratio | mae |
|---|---|---|---|---|---|
| ENSG00000189091 | 0.539 | 0.67 | 0.10 | 0.57 | 0.588 |
| ENSG00000204152 | 0.532 | 0.62 | 0.30 | 0.70 | 0.494 |
| ENSG00000130312 | 0.398 | 0.58 | 0.20 | 0.66 | 0.488 |
| ENSG00000197265 | 0.386 | 0.61 | 0.15 | 0.55 | 0.554 |
| ENSG00000035141 | 0.323 | 0.54 | 0.15 | 0.67 | 0.548 |
| ENSG00000101138 | 0.300 | 0.54 | 0.10 | 0.57 | 0.507 |
| ENSG00000168827 | 0.283 | 0.57 | 0.10 | 0.68 | 0.489 |
| ENSG00000258366 | 0.279 | 0.62 | 0.05 | 0.74 | 0.428 |
| ENSG00000178202 | 0.223 | 0.55 | 0.15 | 0.72 | 0.471 |
| ENSG00000108064 | 0.195 | 0.54 | 0.10 | 0.90 | 0.454 |
| ENSG00000168538 | 0.169 | 0.47 | 0.10 | 0.54 | 0.552 |
| ENSG00000104866 | 0.162 | 0.52 | 0.00 | 0.96 | 0.448 |

### 库训练入口(`train_model`,2026-06-21)

由 `scripts/validate_real_libpipeline.py` 驱动,产物在 `runs/real_hepg2_libpipeline/`。**数据、400 基因架构、超参(batch 4 / set 16 / 300 步 / lr 3e-4 / seed 0 / CPU)与 validate_real 完全相同**,唯一区别是训练走库的 `train_model` → `compute_batch_loss` + warmup-cosine 调度 + `PriorStore`,而非内联循环。因此指标差异可干净归因到 pipeline 本身。

与内联循环的实质差异:
- `compute_batch_loss` 每步**多跑一次 full-reveal forward** 取真正的 target cell embedding 喂对比损失(内联循环用 `out.cell_embedding.detach()`,退化为自身);
- loss 额外接收 `prior_missing` / `prior_source_names`;
- warmup-cosine 学习率调度(内联循环为恒定 lr)。

| 指标 | 均值 |
|---|---|
| MAE | 0.5037 |
| pearson_delta | 0.2941 |
| overlap_at_n | 0.1958 |
| direction_match | 0.6006 |
| delta_norm_ratio | 0.6261 |

loss:first10 ≈ 132.24 → last10 ≈ 1.8185。

### 三方对比

| 指标 | 旧基线 | validate_real(新代码内联) | 库 pipeline(train_model) | 方向 |
|---|---|---|---|---|
| MAE | 0.5019 | 0.5019 | 0.5037 | 越低越好 |
| pearson_delta | 0.3159 | 0.3159 | 0.2941 | 越高越好 |
| overlap_at_n | 0.125 | 0.125 | **0.1958** | 越高越好 |
| direction_match | 0.57 | 0.57 | **0.6006** | 越高越好 |
| delta_norm_ratio | 0.6879 | 0.6879 | 0.6261 | 越接近 1 |

**结论:** 库 pipeline 牺牲少量 `pearson_delta`(−0.022),换来明显更高的 `overlap_at_n`(+57%)与 `direction_match`(+0.03),即更准地选中差异基因、判对上下调方向;最终 loss 也更低(1.82 vs 1.92)。这验证了新增的真 target-embedding 对比损失 + warmup-cosine 调度确实改善了扰动方向性,而内联循环没有触及这些路径。

### scGPT 初始化对照(512-d 架构,2026-06-21)

由 `scripts/validate_mini_scgpt.py` 驱动,产物在 `runs/mini_scgpt/`。论文 Mini 把 gene embedding 从预训练 scGPT 初始化(该 embedding 同时流入输入 gene token 和第 6 个 gene-identity cross-attention context token),Open-X-Cell 此前从零随机初始化——这是与论文「最系统性的偏差」。本实验测它的实际影响。

- **数据**:`hepg2nadig_mini2k.h5ad`(3900 cells × 2000 genes,同 12 扰动,scGPT 覆盖 98%)。
- **架构**:紧凑 512-d 配置(6 层 / 512 hidden / 8 heads / ReLU / LayerNorm / Post-LN);hidden=512 正好等于 scGPT 维度,可直接初始化。严格 4000-基因 mini 在单卡 48G 上 OOM(注意力 O(genes²)),故用 2000 基因 + 6 层的可行配置。
- **scGPT 资产**:`data/priors/scgpt_gene_embedding.npz`(18532×512,由本地 scGPT `encoder.embedding.weight` 按 hepg2 ENSG 对齐抽取)。
- **两臂**:仅 `scgpt_init` 有无之差,其余(数据/架构/超参 batch1/set16/600步/lr3e-4/seed0/GPU)完全相同。

| 指标 | base(随机 init) | scgpt init | delta | 方向 |
|---|---|---|---|---|
| MAE | 0.5258 | 0.5068 | −0.019 | 越低越好 |
| pearson_delta | 0.1934 | **0.3506** | **+0.157(+81%)** | 越高越好 |
| overlap_at_n | 0.0625 | **0.1458** | **+0.083(+133%)** | 越高越好 |
| direction_match | 0.5612 | 0.6236 | +0.062 | 越高越好 |
| delta_norm_ratio | 0.5009 | 0.6600 | +0.159 | 越接近 1 |

训练 loss:base 87.9→2.64,scgpt 82.0→2.40(起点和终点都更低)。scGPT init 注入 1961/2000 个 gene embedding。

**结论:** 仅补上 scGPT 初始化,真实 HepG2 扰动预测的 **pearson_delta 几乎翻倍、overlap@N 翻倍多**,每一项指标都改善。这证实了之前一致性核对中判定的「最系统性偏差」确实重要——scGPT 初始化对扰动方向性和差异基因选择有实质贡献。

### scGPT 三臂阶梯:随机 → embed-init → full warm-start(2026-06-21)

由 `scripts/validate_mini_scgpt.py` 驱动(`runs/mini_scgpt/comparison3.json`),512-d / 2000 基因 / 6 层 / 500 步,三臂仅初始化方式不同:

- **base** — 随机初始化
- **embed** — 仅 scGPT gene-embedding 初始化(npz 表)
- **warmstart** — 完整 warm-start:gene embedding + value encoder + transformer 层(fused Wqkv 拆 q/k/v),98% ENSG 映射、104 张量迁移

| 指标 | base | embed | warmstart | 方向 |
|---|---|---|---|---|
| MAE | 0.5274 | **0.5052** | 0.5121 | 越低越好 |
| pearson_delta | 0.1371 | **0.3396** | 0.3271 | 越高越好 |
| de_spearman_lfc | 0.1272 | **0.3252** | 0.3024 | 越高越好 |
| overlap_at_n | 0.0417 | **0.1292** | **0.1292** | 越高越好 |
| direction_match | 0.5590 | **0.6143** | 0.5964 | 越高越好 |
| delta_norm_ratio | 0.5912 | **0.6417** | 0.6209 | 越接近 1 |
| centroid_accuracy | **0.1667** | 0.0833 | 0.0833 | 越高越好 |

训练 loss 起点:base 85.4 → embed 79.5 → warmstart 78.3(warm-start 最低)。

**诚实结论:** 增益**几乎全部来自 gene-embedding 初始化**(embed ≫ base,pearson_delta +148%)。把 scGPT 的 **transformer 层也迁移过来(warmstart)在这个小规模(512-d/2000基因/500步)设置下反而略低于只迁 embedding**——可能原因:scGPT 是 binned-value 掩码建模目标、与 X-Cell 的 set-level diffusion 目标和连续 log1p 值不完全对齐,迁移的 transformer/value-encoder 权重在少步数下不如让其随机起步。`centroid_accuracy` 在 12 扰动上仅 1–2 个命中,噪声大,不作主要依据。论文级 transformer warm-start 的收益可能需完整规模(34 层 / 全语料 / 长训练)才显现。

## 备注

- **validate_real 内联循环为何与旧基线完全相同:** "align training pipeline" commit 主要改写库的 `training.py` 入口;`validate_real.py` 用自己的内联循环,只调用 `data / model / losses / inference / evaluation`,这些路径在 `seed=0` + CPU 确定性下数值轨迹一致,故指标一字不差。要测到 pipeline 改动须改用 `train_model`(见上)。
- 评估为 CPU 小配置(400 基因 / 300 步),意在跑通真实数据全链路并给方向性指标,非完整规模训练。
- 整体方向性偏弱:`direction_match` 0.57(0.5=随机)、`overlap_at_n` 0.125、`pearson_delta` 0.32,模型能学到扰动方向但逐基因特异性有限。
