# 输入格式

服务通过 HTTP JSON 接收静力分析模型。所有字段使用国际单位制（长度、力、
Pa），单位一致性由调用方负责。

## 顶层模型 `ModelInput`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `version` | int | 否 | 输入 schema 版本，当前为 `1` |
| `nodes` | `NodeInput[]` | 是 | 节点定义，id 必须唯一 |
| `elements` | `ElementInput[]` | 是 | 单元定义，id 必须唯一 |
| `materials` | `MaterialInput[]` | 是 | 命名字典式材料 |
| `boundary_conditions` | `BoundaryConditionInput[]` | 是 | 位移约束；至少一条，否则刚体运动使矩阵奇异 |
| `loads` | `NodalLoadInput[]` | 是 | 节点集中载荷，可为空数组 |
| `options` | `SolveOptions` | 否 | 单作业求解选项 |

### NodeInput

```json
{ "id": 1, "x": 0.0, "y": 0.0 }
```

坐标必须为有限实数；重复 id、NaN/Inf 一律拒绝。

### ElementInput

```json
{ "id": 1, "type": "CPS4", "nodes": [1, 2, 3, 4], "material": "steel" }
```

* `type`：单元类型。当前支持
  * `CPS4`：4 节点双线性平面应力四边形单元（按 `thickness` 厚度）；
  * `CPE4`：4 节点双线性平面应变四边形单元（单位厚度切片）。
* `nodes`：恰好 4 个**互不相同**的已定义节点 id，按逆时针顺序给出。
* `material`：必须引用 `materials` 中已定义的材料名。

服务**绝不静默修正**单元。以下情况分别返回可区分的错误：
* 角点重合、零/负面积、折叠或自交（蝴蝶结）单元 → `degenerate_element`；
* 引用未定义节点 → `invalid_element`；
* 引用未定义材料 → `undefined_material`；
* 已定义但不属于任何单元的"悬空节点" → `dangling_node`。

### MaterialInput

```json
{ "name": "steel", "youngs_modulus": 2.1e11,
  "poisson_ratio": 0.3, "thickness": 0.01 }
```

* `youngs_modulus` > 0；
* `-1 < poisson_ratio < 0.5`；
* `thickness` > 0（平面应力参与积分，平面应变忽略其缩放）。

### BoundaryConditionInput（位移边界条件）

```json
{ "node": 1, "dof": 0, "value": 0.0 }
```

* `dof`：`0` = x 向位移 `ux`，`1` = y 向位移 `uy`；
* `value`：指定位移，缺省为 0；
* 同一 `(node, dof)` 不允许重复约束；
* 模型必须约束掉全部刚体位移，否则返回 `unconstrained_model`（预先）
  或求解期 `singular_matrix`。

### NodalLoadInput（集中载荷）

```json
{ "node": 3, "dof": 0, "value": 1000.0 }
```

同一 `(node, dof)` 的多个载荷在归一化时求和。载荷必须为有限数。

### SolveOptions（可选）

```json
{ "equilibrium_tolerance": 1e-6, "timeout_seconds": 30.0 }
```

* `equilibrium_tolerance`：覆盖服务默认的平衡残差阈值；
* `timeout_seconds`：覆盖单作业墙钟时限；`0` 表示不超时。

## 错误响应

所有错误返回统一信封，HTTP 状态码与机器可读 `code`：

```json
{ "error": { "code": "dangling_node",
             "message": "1 node(s) are not connected to any element",
             "details": { "nodes": [99], "count": 1 } } }
```

| code | HTTP | 含义 |
|------|------|------|
| `invalid_node` | 422 | 节点缺失/重复/非有限/超规模或内存 |
| `invalid_element` | 422 | 单元类型不支持、节点未定义、id 重复等 |
| `degenerate_element` | 422 | 退化（零/负面积、折叠、重合角点） |
| `dangling_node` | 422 | 悬空节点 |
| `undefined_material` | 422 | 材料未定义或材料参数非法 |
| `invalid_boundary_condition` | 422 | 约束节点不存在/重复/非有限 |
| `invalid_load` | 422 | 载荷节点不存在或非有限 |
| `unconstrained_model` | 422 | 无任何约束 |
| `singular_matrix` | 500 | 刚度矩阵奇异（机构/约束不足） |
| `equilibrium_not_satisfied` | 500 | 最大力残差超阈值 |
| `job_not_found` | 404 | 作业不存在 |
| `job_invalid_state` | 409 | 状态不允许该操作（如对失败作业取结果） |
| `job_cancelled` | （作业状态） | 已取消 |
| `job_timeout` | （作业状态） | 超时终止 |
| `job_limit_exceeded` | 413 | 作业数/规模超限 |
| `invalid_request_schema` | 422 | JSON 未通过 schema 校验 |
