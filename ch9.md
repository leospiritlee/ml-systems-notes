# 第9章 连续批处理与 PagedAttention：vLLM 的 KV cache 管理

**一句话**：`vLLM` 用迭代级调度（`连续批处理`）和 `PagedAttention`，把 `KV cache` 碎片从 60–80% 压到 4% 以下。

**为什么需要它**：朴素 contiguous 分配要为最坏情况预留连续显存，因序列长度不可预测而浪费大量 `GPU` 显存，直接限制并发。

**核心结构**：`连续批处理` 把调度单位从「请求」降到「token 步」；`PagedAttention` 把 `KV cache` 切成固定大小 `block`，序列的逻辑块经 `block table` 映射到非连续的物理块。

**关键设计**：物理块按需分配、完成即回收；相同前缀的序列通过 `前缀缓存` 与 `Copy-on-Write` 共享块；`Chunked Prefill` 把长 prompt 切片与 `decode` 交错，避免独占 SM 造成 ITL 抖动。

**拆解**：
### 连续批处理怎么调度

每次 forward 后，已完成的序列立即释放并退出，排队请求立刻补入活跃批次。调度单位从请求边界下沉到 token 步，batch 组成每步都变，GPU 利用率不再被最长请求拖住。

### PagedAttention 怎么分页

借鉴 `虚拟内存` 与 `分页`：每序列的 `KV cache` 切成固定大小 `block`（通常 16 token），逻辑块经 `block table` 映射到非连续 `物理块`。块不必连续，按需分配。

### 碎片为何降到 4%

朴素分配浪费在内部碎片（预留未用）与外部碎片（缺连续块）。`PagedAttention` 只在序列最后一个 `block` 留少量浪费，实践中低于 4%，显存利用率接近 100%，从而能 batch 更多序列。

### 前缀缓存与 Copy-on-Write

并行采样、beam search 或多请求共享系统提示时，逻辑块指向同一 `物理块`，引用计数管理；分支产生分歧时才复制——即 `Copy-on-Write`。这让复杂采样的显存开销最高降 55%、吞吐最高提升 2.2 倍。

**今天怎么做**：默认开 `连续批处理` + `前缀缓存`；长上下文开 `Chunked Prefill` 避免 ITL 抖动；`block` 大小通常保持 16 不用动。监控 `KV cache` 利用率而非只看模型权重占用。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `PagedAttention` 减小了每 token 的 KV 大小——它改善的是分配与复用，不减小单条 `KV cache`；与之对应的 `FlashAttention` 才是优化注意力计算本身，两者可叠加。

**记住这一句**：`PagedAttention` 借鉴 OS 虚拟内存分页，`KV cache` 是推理显存的第一大户。

</div>
