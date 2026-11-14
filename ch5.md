# 第5章 显存与重计算：ZeRO、激活重计算、混合精度与 offload

**一句话**：显存不够，就分片（`ZeRO`）、重算（`Activation Checkpointing`）、降精度（`混合精度` / `FP8`）、卸载（`offload`）。

**为什么需要它**：`16Φ` 里大部分是冗余副本（优化器状态、梯度、重复参数），必须削减，才能把模型放进有限显存。

**核心结构**：`ZeRO-1` / `ZeRO-2` / `ZeRO-3` 逐级把优化器状态 → 梯度 → 参数分片；`Activation Checkpointing` 用计算换激活显存；`混合精度` 与 `FP8` 降数值位宽；`ZeRO-Infinity` 把状态卸载到 CPU 内存与 NVMe。

**关键设计**：`ZeRO-3` 前向/反向时按层用 `All-Gather` 动态取回当前层参数、用完即弃；`Activation Checkpointing` 用约 20–30% 额外前向换约 70% 显存；`FlashAttention` 把注意力显存从 O(N²) 降到 O(N)。

**拆解**：
### ZeRO 三阶段各分什么

`ZeRO-1` 只分片优化器状态；`ZeRO-2` 再加梯度分片（反向后 `Reduce-Scatter`）；`ZeRO-3` 再加参数分片（按需 `All-Gather`）。越往后越省显存，通信也越多——万亿参数模型下 `ZeRO-3` 单卡显存可降约 8 倍。

### 激活重计算

只存少量检查点，反向需要时从最近保存点重算。代价是约 20–30% 额外前向计算，换来激活显存降约 70%。`PyTorch` 经 `torch.utils.checkpoint.checkpoint` 提供。

### 混合精度与 FP8

`混合精度`（AMP）用 `FP16` / `BF16` 做前反向、`FP32` 保存主权重与梯度累积，配 `loss scaling` 保数值稳定。`FP8` 训练在超大规模上验证，多数 GEMM 走 `FP8`、关键组件保 `BF16` / `FP32`，相对损失误差 < 0.25%。

### offload 与 ZeRO-Infinity

当单卡与 `NVMe` 都不够，可把优化器状态/梯度卸载到 CPU 内存甚至 `NVMe`（`ZeRO-Infinity`），用更慢的存储换更大的模型上限，代价是主机端搬运的延迟。

**今天怎么做**：先上 `ZeRO-2`；层数深、显存紧再升 `ZeRO-3`；长序列必开 `FlashAttention`。不要一上来 `ZeRO-3`，通信开销会吞掉收益。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `ZeRO-3` 比 `ZeRO-2` 更快——不，它每步多一次 `All-Gather` 取参数，通信更多，只是更省显存。选阶段是显存与通信的取舍。

**记住这一句**：这些手段拆的是 `16Φ` 里的冗余副本，不是计算本身。

</div>
