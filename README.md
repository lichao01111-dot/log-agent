# Log Diagnosis Agent

一个基于证据、显式状态机和端口适配器架构的日志诊断 Agent。

当前已实现纯领域内核、应用端口、异步命令执行器、Fake 诊断闭环、ScopePolicy + 类型化 QueryPlan + Policy Gate 安全查询管线、严格 JSON 领域知识快照、有界知识投影、provider-neutral 结构化推理 Adapter，以及严格 incident Loader + deterministic Fake eval harness。Fake 模型与五类评估场景已跑通，但尚未连接真实 LLM 或 Splunk；生产日志在 Sanitizer/Redactor 和字段分类完成前禁止外发。

## 开发基线

- Python 3.12+
- 运行时零第三方依赖
- pytest 单元测试
- ruff 静态检查

## 运行检查

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
ruff format --check .
```

当前本地验证环境为 Python 3.14；CI 会在 Python 3.12、3.13 和 3.14 上执行同一套检查。

教学文档唯一入口见 [`doc/README.md`](doc/README.md)，设计背景见 [`doc/日志诊断Agent-设计文档.md`](doc/日志诊断Agent-设计文档.md)。章节如下：

- [`doc/01-领域内核与状态机.md`](doc/01-领域内核与状态机.md)
- [`doc/02-端口与适配器.md`](doc/02-端口与适配器.md)
- [`doc/03-命令执行器与Fake闭环.md`](doc/03-命令执行器与Fake闭环.md)
- [`doc/04-安全查询管线.md`](doc/04-安全查询管线.md)
- [`doc/05-上下文与记忆设计.md`](doc/05-上下文与记忆设计.md)
- [`doc/06-领域知识配置.md`](doc/06-领域知识配置.md)
- [`doc/07-知识投影与结构化推理.md`](doc/07-知识投影与结构化推理.md)
- [`doc/08-真实适配器与脱敏边界.md`](doc/08-真实适配器与脱敏边界.md)
- [`doc/09-评估与回归.md`](doc/09-评估与回归.md)
- [`doc/10-AgentLoop装配与安全入口.md`](doc/10-AgentLoop装配与安全入口.md)
- [`doc/11-可观测性与迭代.md`](doc/11-可观测性与迭代.md)
- [`doc/Capstone-迁移与扩展.md`](doc/Capstone-迁移与扩展.md)
