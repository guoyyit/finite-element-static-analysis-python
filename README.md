# 本地结构有限元静力分析服务

一个基于 FastAPI + NumPy 的本地结构有限元静力分析（linear static FEM）
服务。输入节点、单元、材料、位移边界条件与节点集中载荷，输出节点位移、
单元应力与支座反力；支持批量求解、并发作业、取消/超时/失败隔离、检查点
重启恢复，以及版本绑定的确定性结果查询。

## 能力

* **单元**：4 节点双线性平面应力 `CPS4` 与平面应变 `CPE4` 四边形
  （2×2 高斯积分、形心应力、von Mises）。
* **严格校验，绝不静默修正**：退化单元、悬空节点、未定义材料、重复 id、
  非有限数值、约束缺失等分别返回可区分错误码。
* **约束与载荷**：位移边界条件（含非零指定位移）与集中载荷；缩减系统
  严格满足约束，并校验整体力平衡残差不超过可配置阈值。
* **并发隔离**：有界线程池并发求解；任一作业取消、超时或失败不影响其他
  作业，也不会留下可被误认为有效结果的部分输出。
* **检查点恢复**：原子写入的作业快照 + 可续算的组装检查点；进程重启后
  未完成作业从最近检查点继续，已完成作业保持可查询、不重复求解。
* **确定性**：输入规范化哈希与求解器/输入版本绑定；相同模型与参数跨运行
  逐位复现，并自动复用已完成结果。
* **资源约束**：节点/单元数量、活动作业数与内存估算在开始前即拒绝超限
  作业；终态作业清理大矩阵，无无界缓存或残留临时文件。

## 快速开始

```bash
uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8000
# 健康检查
curl -s localhost:8000/health
```

也可在代码中直接使用：

```python
from fem.config import ServiceConfig
from fem.manager import JobManager
from fem.models import (ModelInput, NodeInput, ElementInput,
                        MaterialInput, BoundaryConditionInput, NodalLoadInput)

model = ModelInput(
    nodes=[NodeInput(id=1, x=0, y=0), NodeInput(id=2, x=1, y=0),
           NodeInput(id=3, x=1, y=1), NodeInput(id=4, x=0, y=1)],
    elements=[ElementInput(id=1, type="CPS4",
                           nodes=[1, 2, 3, 4], material="s")],
    materials=[MaterialInput(name="s", youngs_modulus=1.0,
                             poisson_ratio=0.0, thickness=1.0)],
    boundary_conditions=[BoundaryConditionInput(node=1, dof=0),
                         BoundaryConditionInput(node=1, dof=1),
                         BoundaryConditionInput(node=2, dof=1),
                         BoundaryConditionInput(node=4, dof=1)],
    loads=[NodalLoadInput(node=3, dof=0, value=1.0)])

mgr = JobManager(ServiceConfig(storage_dir=".fem_jobs"))
mgr.start()
info = mgr.submit(model)
# ... 轮询 mgr.get(info.job_id) 直到 completed ...
result = mgr.get_result(info.job_id)
```

## 文档

* [输入格式与错误码](docs/input_format.md)
* [求解语义、平衡、确定性与恢复](docs/solver_semantics.md)
* [HTTP API](docs/api.md)
* [资源限制与运维](docs/resource_limits.md)

## 目录结构

```
fem/
  errors.py       可区分错误码
  config.py       资源限制、阈值与内存估算
  models.py       输入/输出模型、状态枚举、求解器版本
  validation.py   严格校验、归一化、版本绑定哈希
  mesh.py         Q4 单元（退化检测、本构、刚度、应力）
  solver.py       可恢复组装 + 静力求解 + 平衡校验
  checkpoint.py   原子检查点存储
  job.py          线程安全作业状态与取消事件
  manager.py      并发、批量、恢复、去重与资源管控
  api.py          FastAPI 接口
tests/            自动化测试（校验/约束/并发/取消/重启/确定性）
```

## 测试

```bash
uv run pytest
```

覆盖：输入校验与各可区分错误、约束处理与分片检验、并发作业隔离、取消与
失败隔离、批量原子性、作业上限、（模拟）进程重启恢复、从组装检查点续算、
跨运行浮点确定性与版本绑定。
