# Open-X-Cell 论文一致性与功能说明

本文记录 Open-X-Cell（X-Cell 论文的独立复现）相对论文的实现状态:**已实现并对齐**、
**近似实现**、**有意不做(超出单机/clean-room 范围)**,以及各能力的 CLI 用法。

> 核对基准 = X-Cell 论文(`X-Cell.pdf`)+ 官方 MODEL_CARD/docs。官方 `src/xcell/model.py`
> 是 `NotImplementedError` 空壳,不可作参考实现。

---

## 1. 已实现并对齐论文

| 维度 | 说明 |
|---|---|
| 架构(mini/ultra) | 层数/hidden/heads、ReLU↔SwiGLU、LayerNorm↔RMSNorm、Pre/Post-LN、QK-Norm、tied-output + PaLM 1/√d、cross-attn 间隔 |
| 扩散 | 训练 reveal `{.25,.5,.75,1}` 每步随机采样;推理 4 步累积 coarse-to-fine |
| 损失 | 论文 Eq.(5) 七项 + L1 输出正则(`LossConfig.l1_output`,Mini=1.0/Ultra=0) |
| cross-attn prior | 6-token context = 5 外部 prior + gene-identity(stop-grad);各 prior 维度对齐 |
| 归一化 | log1p CP10k(`--normalize cp10k-log1p`) |
| **scGPT 初始化** | gene-embedding 从预训练 scGPT 初始化(见 §3) |
| **scGPT warm-start** | gene embedding + value encoder + transformer 层迁移(见 §3) |
| **两阶段课程** | Phase 1 = top high-effect 子集(全损失);Phase 2 = 全语料(fresh optimizer,丢 per-gene CCC)(见 §4) |
| **确定性 set 构造** | permute→chunk→丢弃不完整 set,NTC 不足时整 set replicate(论文 A.10.1) |
| **推理 set 大小** | 默认 64(`--n-cells`),非旧版硬编码 8 |
| **context-aware 推理** | 按目标 context/batch 选 NTC + 多 set ensemble(见 §5) |
| **TTA** | MMD-only NTC 自监督 + freeze cross-attn(见 §6) |
| **cell-eval 指标** | DE Spearman LFC(逐扰动)+ Centroid Accuracy(跨扰动)(见 §7) |

---

## 2. 近似实现(功能在,但非论文精确做法)

| 维度 | 现状 | 论文做法 |
|---|---|---|
| high-effect 选择 | pseudobulk L2 delta 范数排序 | FDR<0.05 二分类器筛 + LASSO 显著特征数排序 |
| scGPT unmapped fallback | `<pad>` token 嵌入 | 论文称 `<mask>`;scGPT-human vocab 无 `<mask>` gene token(mask 是数值 −1),故用 `<pad>` |
| MMD | 自实现 energy distance(与 geomloss 语义等价) | geomloss energy distance |

---

## 3. scGPT 初始化 / warm-start

scGPT 在论文中既是 gene-embedding 的初始化来源,又通过 gene-identity token 进入 cross-attn。

- **仅 embedding 初始化**(轻量,需一份 `.npz`,含 `genes`(ENSG/symbol)+ `embeddings`(N×hidden)两个数组):
  ```bash
  open-x-cell train --adata data.h5ad --output run --model-size mini \
    --scgpt-init scgpt_gene_embedding.npz
  ```
- **完整 warm-start**(gene embedding + value encoder + transformer 层;读 `vocab.json`+`best_model.pt`,
  不需 scgpt 包):
  ```bash
  open-x-cell train --adata data.h5ad --output run --model-size mini \
    --scgpt-warm-start /path/scGPT_human --gene-id-map ens2sym.pt \
    --scgpt-min-overlap 0.95 [--scgpt-strict-overlap]
  ```
  - 多级映射:symbol 直查 → Ensembl→symbol(via `--gene-id-map`)→ vocab;unmapped → `<pad>`。
  - overlap 检查:低于 `--scgpt-min-overlap` 时 warn,加 `--scgpt-strict-overlap` 改为 fail。
  - transformer 迁移:fused `Wqkv` 拆 q/k/v;仅当架构 scGPT 兼容(relu/layernorm/post-LN/no-qk-norm/ffn==hidden)时迁移,否则只迁 embedding。

**实测(512-d/2000基因/500步,真实 HepG2,见 `real-data-validation.md`):** 增益主要来自
gene-embedding 初始化(pearson_delta 0.137→0.340);transformer 层 warm-start 在此小规模下无额外收益。

---

## 4. 两阶段课程(`--curriculum`）

```bash
open-x-cell train --adata data.h5ad --output run --model-size ultra --curriculum \
  --phase1-steps 9000 --phase2-steps 30000 --top-frac 0.05 --phase2-lr 1e-4
```
- Phase 1(high-effect 初始化):top `--top-frac` 高效应扰动,**全损失**。
- Phase 2(全语料):fresh optimizer + 独立 `--phase2-lr`/`--phase2-warmup`,**丢弃 per-gene CCC**。

---

## 5. context-aware 推理 + ensemble

```bash
open-x-cell predict --checkpoint run/checkpoint.pt --adata data.h5ad \
  --perturbation BRCA1 --output pred.h5ad --n-cells 64 \
  --context hepg2 --batch batch0 --n-sets 4
```
- NTC 三级 fallback:(context,batch) → context → 全体。
- `--n-sets>1`:多个 control set 预测后拼接(平摊采样噪声)。

---

## 6. TTA(`predict --tta`）

```bash
open-x-cell predict ... --tta --tta-steps 50 --tta-lr 1e-4
```
- 在目标 NTC 上做 MMD-only 自监督适配,cross-attn 冻结,内存内适配后预测。
- 程序化接口 `tta.adapt_checkpoint_tta` 可存 adapted checkpoint。
- **未做**:LR/set-count sweep、validation-loss 选最优、checkpoint override(可后续补)。

---

## 7. 评估

```bash
# 单文件
open-x-cell evaluate --truth truth.h5ad --pred pred.h5ad --output m.json
# 多文件汇总 + 跨扰动 Centroid Accuracy
open-x-cell evaluate-batch --truth truth.h5ad --pred p1.h5ad p2.h5ad ... --output sum.json
```
指标:MAE、pearson_delta、**DE Spearman LFC**、overlap@N、direction_match、delta_norm_ratio、
**Centroid Accuracy**(≥2 扰动时)。**未做**:Radar-by-Rank(可视化)、完整 cell-eval 包兼容。

---

## 8. 有意不做(超出单机 / clean-room 范围)

| 维度 | 原因 |
|---|---|
| FSDP2 / activation checkpoint / bf16 / torch.compile | 多卡大规模才有意义;训练为单进程 PyTorch loop |
| protein-coding gene 过滤 / 官方 19k vocab 构建 / ESM key 对齐 | 需 GRCh38 蛋白编码基因表与官方 vocab 构建管线 |
| 外部 prior 生产管线(ESM/STRING/GenePT/DepMap/JUMP 下载·清洗·对齐) | 需各 release 数据源;当前为通用表加载器(CSV/TSV/Parquet/NPZ) |
| 跨数据集 vocab remap tensor 注入 collator | warm-start 已做 embedding 级 remap;collator 级未进训练数据流 |
| scaling law 实验(L(N)~N^−0.32) | 需训练 83M–3.1B 多档模型 |
| 预训练 scGPT/Xaira 权重、生产 remap 表 | 不随仓库分发;自备 scGPT checkpoint 目录 |

---

## 9. 验证

完整验证(旧基线 / 库 pipeline / scGPT 三臂阶梯,真实 HepG2)见
[`real-data-validation.md`](real-data-validation.md)。测试套件:`pytest tests/`(41 passed)。
