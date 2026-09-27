# FEA Static Analysis Service

本地结构有限元静力分析服务。输入节点、单元、材料、边界条件与集中载荷，
输出节点位移、单元应力与支座反力。支持批量/并发求解、作业取消与超时、
检查点恢复、确定性结果查询以及资源限额约束。

## 快速开始

```bash
uv sync
uv run uvicorn main:app --port 8000
```

运行测试：

```bash
uv run pytest
```

## 分析模型

- 单元类型：`Q4` 四节点双线性四边形等参元（2x2 高斯积分）
- 分析类型：`plane_stress`（默认）或 `plane_strain`
- 每个节点 2 个平动自由度（x, y），单元节点可顺时针或逆时针给出，
  但必须是严格凸四边形
- 单元应力在高斯点平均意义下取单元中心（ξ=η=0）处的值，
  并输出 von Mises 应力（平面应变含 σzz 贡献）

## 输入格式

`POST /jobs` 请求体：

```json
{
  "analysis_type": "plane_stress",
  "nodes": [{"id": 1, "x": 0.0, "y": 0.0}, ...],
  "elements": [
    {"id": 1, "type": "Q4", "nodes": [1, 2, 3, 4],
     "material_id": 1, "thickness": 0.1}
  ],
  "materials": [{"id": 1, "E": 2.0e11, "nu": 0.3}],
  "displacement_bcs": [{"node_id": 1, "direction": "x", "value": 0.0}],
  "point_loads": [{"node_id": 3, "fx": 0.0, "fy": -1000.0}],
  "options": {"equilibrium_tol": 1e-6}
}
```

字段约束：

| 字段 | 约束 |
|---|---|
| `nodes[].id` | 整数，全局唯一 |
| `elements[].nodes` | 恰好 4 个已定义节点，不得重复，几何上严格凸 |
| `elements[].thickness` | > 0 |
| `materials[].E` | > 0 |
| `materials[].nu` | (-1, 0.5) 开区间 |
| `displacement_bcs[].direction` | `"x"` 或 `"y"`；同一 (node, direction) 只允许一条 |
| `options.equilibrium_tol` | > 0，整体力平衡最大残差阈值，默认 1e-6 |

## 求解语义

1. **校验**：提交时同步校验。出现以下情况一律拒绝（HTTP 422，
   `error.code = "validation_error"`），绝不静默修正：
   退化单元（重复节点 / 零面积 / 共线 / 自交）、悬空节点、
   未定义的材料或节点引用、重复 id、冲突边界条件、非法材料参数。
2. **装配与求解**：自由度按节点 id 升序排列，指定位移边界条件采用
   自由/约束自由度精确分块求解（`K_ff u_f = F_f - K_fc u_c`）。
   刚度矩阵奇异（刚体模态、机构）时作业失败，
   `error.code = "singular_matrix"`。
3. **力平衡**：求解后校验自由自由度上的整体残差 `max|K·u − F|`，
   超过 `options.equilibrium_tol` 时作业失败，
   `error.code = "equilibrium_violation"`。支座反力由 `K·u − F`
   在约束自由度上的取值给出，与外加载荷整体平衡。
4. **确定性**：装配顺序完全由排序后的 id 决定，稠密 float64 求解；
   相同模型与参数重复求解得到逐位一致的结果（跨作业、跨进程重启）。
   每个作业记录 `model_hash`（规范化模型 JSON 的 SHA-256），
   结果与该哈希绑定，可用于版本核对。
5. **结果可见性**：只有 `completed` 状态的作业会返回 `result`；
   失败/取消的作业的任何中间输出都会被清除，不会留下可被误认为
   有效结果的部分输出。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/jobs` | 提交求解作业，202 返回作业句柄 |
| `GET` | `/jobs` | 列出全部作业 |
| `GET` | `/jobs/{job_id}` | 查询状态与结果 |
| `POST` | `/jobs/{job_id}/cancel` | 取消 pending/running 作业 |
| `GET` | `/health` | 存活探针 |

作业状态机：`pending → running → completed | failed | cancelled`。

错误响应统一为 `{"error": {"code", "message", "details"}}`：

| code | HTTP | 含义 |
|---|---|---|
| `validation_error` | 422 | 输入结构校验失败 |
| `limit_exceeded` | 413 | 超过规模/并发限额，开始前拒绝 |
| `job_not_found` | 400 | 作业不存在 |
| `job_state_error` | 400 | 非法状态转换（如取消已完成作业） |
| `singular_matrix` | 200（作业内 error 字段） | 刚度矩阵奇异 |
| `equilibrium_violation` | 200（作业内 error 字段） | 力平衡残差超限 |
| `job_timeout` | 200（作业内 error 字段） | 作业超时 |

## 并发、取消与恢复

- 作业在线程池中并发执行（默认 4 worker），每个作业有独立的取消事件
  与截止时间，任一作业的取消/超时/失败不影响其他作业。
- 终态写入是原子条件更新（compare-and-set），取消与完成竞争时
  状态不会回跳。
- 检查点协议：`validated → solved → completed`。进程重启后：
  - 停在 `solved` 检查点的作业直接落库完成，**不重复求解**；
  - 更早状态的作业从求解阶段重新执行；
  - `completed` 作业保持可查询，结果不丢失、不重算。
- 持久化使用 SQLite（默认 `fea_jobs.db`，可用 `FEA_DB_PATH` 指定）。

## 资源限制

通过环境变量配置（超限作业在开始前被拒绝，HTTP 413）：

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `FEA_MAX_NODES` | 50000 | 单作业最大节点数 |
| `FEA_MAX_ELEMENTS` | 50000 | 单作业最大单元数 |
| `FEA_MAX_DOFS` | 100000 | 单作业最大自由度数 |
| `FEA_MAX_CONCURRENT_JOBS` | 4 | 并发 worker 数 |
| `FEA_MAX_QUEUED_JOBS` | 64 | 最大排队作业数 |
| `FEA_JOB_TIMEOUT_SECONDS` | 300 | 单作业超时（秒） |
| `FEA_DB_PATH` | `fea_jobs.db` | 持久化数据库路径 |

内存约束：进程内只跟踪非终态作业（数量 ≤ 并发数 + 队列上限），
终态作业结果全部落盘 SQLite，查询时按需读取，不产生无界缓存或
残留状态。

## 项目结构

```
fea/
  models.py      # 输入/输出 Pydantic 模型
  errors.py      # 可区分错误码
  validation.py  # 结构校验（拒绝而非修正）
  solver.py      # Q4 平面应力/应变求解器
  storage.py     # SQLite 持久化与检查点
  jobs.py        # 并发作业管理（取消/超时/恢复）
  limits.py      # 资源限额配置
main.py          # FastAPI 端点
tests/           # 校验、求解、并发、取消、恢复、确定性、API 测试
```
