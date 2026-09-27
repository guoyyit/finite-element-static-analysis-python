# HTTP API

基址：`http://127.0.0.1:8000`（启动见 `resource_limits.md`）。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查与求解器版本 |
| POST | `/jobs` | 提交单个作业，`201` 返回 JobInfo |
| POST | `/jobs/batch` | 原子批量提交 `{ "models": [...] }` |
| GET | `/jobs` | 列出全部作业 |
| GET | `/jobs/{id}` | 查询作业状态/进度/错误/结果 |
| GET | `/jobs/{id}/result` | 仅取结果；未完成/失败返回 409 |
| POST | `/jobs/{id}/cancel` | 请求取消（协作式） |
| DELETE | `/jobs/{id}` | 删除终态作业及其全部磁盘文件 |

作业状态：`pending` → `running` → `completed`；或 `failed` /
`cancelled` / `timeout`（终态）。

## 示例：提交并轮询

```bash
curl -sX POST localhost:8000/jobs -H 'content-type: application/json' -d '{
  "nodes": [
    {"id":1,"x":0,"y":0},{"id":2,"x":1,"y":0},
    {"id":3,"x":1,"y":1},{"id":4,"x":0,"y":1}],
  "elements": [
    {"id":1,"type":"CPS4","nodes":[1,2,3,4],"material":"s"}],
  "materials": [
    {"name":"s","youngs_modulus":1.0,"poisson_ratio":0.0,"thickness":1.0}],
  "boundary_conditions": [
    {"node":1,"dof":0},{"node":1,"dof":1},
    {"node":2,"dof":1},{"node":4,"dof":1}],
  "loads": [{"node":3,"dof":0,"value":1.0}]
}'
# -> {"job":{"job_id":"...","state":"pending", ...}}

curl -s localhost:8000/jobs/<job_id>/result
```

相同模型与参数重复提交会返回同一个已完成作业（基于版本绑定的模型哈希
去重），不重复求解。
