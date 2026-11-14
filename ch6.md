# 第6章 集合通信：all-reduce / all-gather / reduce-scatter / broadcast 与拓扑

**一句话**：`all-reduce` / `All-Gather` / `Reduce-Scatter` / `broadcast` 是多卡协作的几种基本动作，由 `NCCL` 按拓扑实现。

**为什么需要它**：梯度同步（`all-reduce`）、`FSDP` 取回参数（`All-Gather`）、`ZeRO` 存梯度（`Reduce-Scatter`）、MoE 分发（`All-to-All`）都靠这些原语；它们决定了通信墙的高度。

**核心结构**：`NCCL`（NVIDIA Collective Communications Library）节点内走 `NVLink` / `NVSwitch`，跨节点走 `GPUDirect RDMA` over `InfiniBand` / `RoCE`，无高速链路回落 `PCIe`；把 rank 排成 `ring` 或 `tree`，使每条链路接近带宽。

**关键设计**：`ring` 大消息按带宽取胜、`tree` 小消息按延迟取胜（`α+βN` 模型下约 256 KiB 以下 tree 胜）；`SHARP` 在交换机 ASIC 内归约使 `all-reduce` 数据量约减半；`ncclGroupStart/End` 把多个小集合通信融合成一次 launch。

**拆解**：
### 四种原语各做什么

`Broadcast` 把一卡缓冲发全卡；`Reduce` 归约留一卡；`all-reduce` 结果留全卡；`All-Gather` 各卡缓冲拼接、每卡得完整；`Reduce-Scatter` 归约并分片、每卡留一片。`all-reduce` 精确等于 `Reduce-Scatter` + `All-Gather`。

### ring 与 tree 的取舍

`Ring all-reduce` 每卡 `2(N-1)/N·D` 字节，大消息时接近带宽最优；`tree` 用 ~2·log₂(P) 跳，小消息延迟更低。NCCL 按消息大小与拓扑每调用择优，可用 NCCL_DEBUG=INFO 看实际选择。

### SHARP 与 GPUDirect RDMA

`SHARP` 在 Quantum `InfiniBand` 交换机里做归约，数据量约减半（每节点发一次、收一次答案，而非绕 ring 推完整体积）。`GPUDirect RDMA` 让数据直接过网卡进出 `GPU` 显存，不经主机内存。

### 集合通信是 CUDA kernel

`NCCL` 集合通信作为 `CUDA kernel` 入队到 stream，可与计算重叠；它占 SM 与 HBM 带宽，不是免费的背景操作。约 30 个 `NCCL` 环境变量在通信吞吐与计算资源间找平衡。

**今天怎么做**：让集合通信与计算重叠：`DDP` 靠分桶、`FSDP` 靠预取 `All-Gather`、TP 靠把通信融进 GEMM。在 profiler 上看 compute 与 `NCCL` kernel 交错的锯齿状时间线，确认重叠生效；若呈先全 backward、再全 all-reduce 的两段式，说明没重叠。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `all-reduce` 是个原子黑盒——它精确等于 `Reduce-Scatter` 接 `All-Gather`，最快实现正是利用这个分解，让每字节移动次数最小。

**记住这一句**：`NCCL` 集合通信是占 SM 的 `CUDA kernel`，把通信藏进计算才是性能分水岭。

</div>
