# 资源限制与运维

## 可配置项 `ServiceConfig`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_nodes` | 100000 | 单作业最大节点数，超过在开始前拒绝 |
| `max_elements` | 100000 | 单作业最大单元数 |
| `max_jobs` | 256 | 同时存在（未到终态）作业上限 |
| `max_memory_bytes` | 512 MiB | 组装系统的内存上限估算 |
| `job_timeout_seconds` | 60.0 | 单作业默认墙钟时限，`0` 不限 |
| `equilibrium_tolerance` | 1e-6 | 默认平衡残差阈值（可被作业覆盖） |
| `residual_force_scale` | 1.0 | 残差绝对力下限的尺度因子 |
| `worker_threads` | 4 | 并发求解线程数 |
| `storage_dir` | `.fem_jobs` | 检查点/结果目录 |
| `checkpoint_interval` | 4096 | 每多少单元落一次组装检查点 |

## 规模与内存的事前拒绝

提交时（任何作业被创建或落盘之前）先执行：

1. 节点/单元数量与上限比较；
2. 自由度 `ndof = 2·nn`，估算
   `memory ≈ 8·ndof² + 256·ndof`（全局稠密矩阵 + 缩减工作矩阵与
   索引/向量），超过 `max_memory_bytes` 即拒绝。

超限返回 `invalid_node`（携带 `ndof`、`limit_bytes`）或
`job_limit_exceeded`（携带 `limit`、`requested`、`active`）。拒绝请求
**没有任何副作用**：不创建作业、不写文件、不入队。

> 注：当前使用稠密矩阵求解，规模上限相对保守；内存估计偏上界以确保真正
> 分配前即拒绝超限作业。

## 无界缓存与残留状态防护

* 作业到达终态后释放并发槽位；`active_slots` 由 future 完成回调严格归还。
* `completed`：删除可能很大的 `<job>.K.npy`，仅保留结果快照。
* `failed`/`cancelled`/`timeout`：删除 `<job>.K.npy`，结果字段恒为空。
* 所有写入均为 `*.tmp → fsync → rename`，任何路径下都不残留 `*.tmp`。
* 显式 `DELETE /jobs/{id}` 才会清理作业历史；在此之前终态作业可查询。

## 批量提交

`POST /jobs/batch` 为原子语义：先对整批模型逐个完整校验并统计是否会
超出 `max_jobs`，任一不合法或会超限则整批拒绝（422/413），不创建任何
作业。已完成的相同模型在批内直接复用其结果。

## 存储布局（`storage_dir`）

```
<job_id>.json        # 作业快照（原子写入）
<job_id>.model.json  # 原始模型载荷，用于重启续算
<job_id>.K.npy       # 组装检查点（仅未完成运行期间存在）
```

损坏的 `.json` 在整体恢复时跳过（绝不当作结果），直接 `load` 则报
`checkpoint_error`。

## 运行

```bash
uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

健康检查：`GET /health`。服务生命周期内自动恢复 `storage_dir` 中的作业。
