# 变更日志

## v0.4.0 — P3 自进化Agent循环 (2026-05-27)

### 核心目标
实现自进化Agent循环：“对局 → 分析 → 优化 → 再对局”。

---

### 变更文件清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `agent/evolution_agent.py` | 新增 | EvolutionAgent 自进化分析Agent |
| `evaluation/evolution_tracker.py` | 新增 | 进化效果追踪器 |
| `evaluation/__init__.py` | 修改 | 新增 EvolutionTracker 导出 |
| `memory/experience.py` | **增强** | 策略标签 + 智能检索 + 经验上限 + 清空功能 |
| `engine/game_engine.py` | 修改 | 新增 `_run_evolution_analysis` 自动触发进化分析 |
| `api/server.py` | 修改 | 新增排行榜/进化报告/经验查询API端点 |
| `tests/test_evolution.py` | 新增 | 自进化模块测试（283行） |
| `tests/test_experience.py` | 修改 | 新增策略标签/上限/检索/清空测试 |
| `tests/test_api.py` | 修改 | 新增排行榜/进化/经验API测试 |

---

### 详细变更

#### 1. EvolutionAgent (`agent/evolution_agent.py`)

- 分析多个对局经验，提炼进化建议
- 输出：趋势分析、反复错误、有效策略、进化计划、Prompt修改建议
- `batch_analyze()` — 批量分析所有角色
- 支持评测指标数据输入

#### 2. EvolutionTracker (`evaluation/evolution_tracker.py`)

- JSON文件持久化追踪进化历史
- `record_evolution()` — 记录进化前后指标
- `get_evolution_report()` — 获取单角色进化报告（含平均改进幅度）
- `get_all_reports()` — 获取所有角色报告
- `get_version_comparison()` — 各版本对比

#### 3. 经验存储增强 (`memory/experience.py`)

- **策略标签提取**：`_extract_strategy_tags()` 自动从经验中抽取策略标签
- **智能检索**：`get_experience_prompt(situation=...)` 优先匹配相关经验
- **经验上限**：每角色最多保留 MAX_EXPERIENCES=20 条
- **新增功能**：`get_all_role_experiences()` 和 `clear_experiences()`

#### 4. 引擎集成 (`engine/game_engine.py`)

- 新增 `_run_evolution_analysis()` 方法
- 游戏结束时自动触发：当角色积累 ≥ 3 局经验时记录到 EvolutionTracker

#### 5. 新增 API 端点 (`api/server.py`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/leaderboard` | GET | Agent能力排行榜 |
| `/evolution/report` | GET | 进化报告（支持 ?role_type=xxx） |
| `/evolution/version-comparison/{role_type}` | GET | 版本对比 |
| `/evolution/analyze` | POST | 手动触发进化分析 |
| `/experiences/{role_type}` | GET | 角色历史经验查询 |

---

## v0.3.0 — P1 跑聪明 + P2 评测体系 (2026-05-27)

### P1 并发决策与异常保护

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `engine/game_engine.py` | 修改 | 投票并发 + 发言/投票超时保护 |

- **投票并发**：`_vote()` 改用 `asyncio.gather` 并发收集所有玩家投票，大幅减少等待时间
- **发言超时保护**：`_public_speeches()` 每个发言加 `asyncio.wait_for(timeout=30)` 保护
- **异常回退**：新增 `_safe_vote()` 方法，超时/异常时返回空投票而非崩溃
- **评测集成**：游戏结束时自动运行评测并记录排行榜

### P2 评测体系 + 复盘归因 + Leaderboard

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `evaluation/__init__.py` | 新增 | 评测模块入口 |
| `evaluation/evaluator.py` | 新增 | GameEvaluator + Leaderboard |
| `evaluation/review_agent.py` | 新增 | ReviewAgent 复盘归因 |
| `tests/test_evaluation.py` | 新增 | 评测模块测试（286行） |

**GameEvaluator（三维评测）：**
- 结果评测：存活天数、贡献分、胜负加权
- 过程评测：发言质量、投票准确率、信息利用效率
- 对抗评测：欺骗能力（狼人）、识别能力（好人）、悍跳成功率

**Leaderboard：**
- JSON文件持久化存储
- 按模型统计胜率、对局质量、角色平均得分
- 支持按角色筛选最高分玩家

**ReviewAgent：**
- LLM驱动的复盘分析
- 关键转折点识别、胜负归因、MVP/背锅评选
- 各角色策略建议

---

## v0.2.0 — P0 跑通主流程 (2026-05-27)

### 核心目标
先跑通主流程、跑聪明。全面优化 Prompt 指令模板，新增平票PK、遗言机制、猎人智能决策。

---

### 变更文件清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `agent/player_agent.py` | **重写** | Prompt全面优化 + 新增5个方法 |
| `engine/game_engine.py` | **重大修改** | 猎人智能决策 + 平票PK + 遗言机制 |
| `tests/test_player_agent.py` | 修改 | 适配新版Prompt断言 |
| `tests/test_engine.py` | 修改 | 新增7个测试类覆盖新功能 |
| `tests/test_api.py` | 修改 | mock新增的遗言和猎人开枪方法 |
| `docs/optimization.md` | **重写** | 业务优先的P0-P3四阶段优化路线 |
| `docs/architecture.md` | 新增 | 系统架构文档（含Mermaid图） |
| `docs/introduction.md` | 新增 | 项目介绍文档 |
| `docs/README.md` | 新增 | 文档中心索引 |
| `docs/CHANGELOG.md` | 新增 | 本变更日志 |

---

### 详细变更

#### 1. `agent/player_agent.py` — Prompt 全面优化（最高优先级）

**新增内容：**
- `ROLE_STRATEGIES` — 5个角色（狼人/预言家/女巫/猎人/村民）的专属策略指导常量
- `_get_night_action_guide()` — 按角色返回夜间行动指南
- `_build_dialogue_summary()` — 对话历史压缩为可读摘要（取最近20条）
- `decide_last_words()` — 遗言决策方法
- `decide_hunter_shot()` — 猎人开枪决策方法
- `_build_last_words_prompt()` — 遗言提示模板
- `_build_hunter_shot_prompt()` — 猎人开枪提示模板

**重写内容：**
- `_build_instructions()` — 加入角色专属策略、信息隔离原则、结构化输出要求
- `_build_night_prompt()` — 加入角色专属夜间行动指南
- `_build_speech_prompt()` — 加入对话历史摘要
- `_build_vote_prompt()` — 加入投票建议和分析要求
- `_parse_json_output()` — 增强JSON解析（支持代码块提取、嵌套JSON、多层兜底）

**优化要点：**
- 系统提示词从通用描述改为角色专属策略指导
- 新增信息隔离原则（善良阵营不知他人身份、狼人只看队友）
- 对话历史压缩避免上下文溢出
- JSON解析从简单正则升级为4层兜底策略

#### 2. `engine/game_engine.py` — 猎人智能决策 + 平票PK + 遗言

**猎人智能开枪（`_handle_hunter_death`）：**
- ~~硬编码带走第一个存活玩家~~ → AI代理决策开枪目标
- 保留回退机制：AI决策失败时自动选择第一个存活玩家

**平票PK（新增 `_tie_breaker_pk`）：**
- 平票玩家进行PK发言（调用 `decide_speech`）
- 非平票玩家重新投票（调用 `decide_vote`）
- 仍平票则随机淘汰一个

**遗言机制（新增 `_eliminate_player`）：**
- 统一处决入口，含遗言收集
- 投票出局：调用 `decide_last_words` → 记录遗言 → 触发猎人开枪
- 夜间死亡：`_announce_night_deaths` 中也支持遗言

**新增方法：**
- `_eliminate_player(player_id, cause)` — 统一处决+遗言
- `_tie_breaker_pk(tied_player_ids)` — 平票PK流程

#### 3. 测试文件更新

**`tests/test_player_agent.py`：**
- 适配新版Prompt的断言关键字
- `test_build_night_prompt` 改为使用 `role_type="werewolf"`
- `test_cautious_style_prompt` / `test_bold_style_prompt` / `test_random_style_prompt` / `test_balanced_style_prompt` 全部添加 `role_type` 参数

**`tests/test_engine.py`：**
- 现有mock增加 `decide_last_words` 和 `decide_hunter_shot`
- 新增7个测试类：
  - `TestHunterSmartDecision` — 猎人AI决策 + 失败回退
  - `TestLastWords` — 投票遗言 + 夜间遗言
  - `TestTieBreakerPK` — 平票PK淘汰 + 无人投票PK
  - `TestDialogueSummary` — 空对话/发言摘要/截断
  - `TestRoleStrategies` — 角色策略完整性 + 指令包含
  - `TestNightActionGuide` — 狼人/预言家/女巫夜间指南

**`tests/test_api.py`：**
- `mock_llm` fixture 新增 `decide_last_words` 和 `decide_hunter_shot` 的mock

---

### 测试结果

全部 299 个测试用例 PASSED，无失败。

---

## v0.1.0 — 初始版本

- 基础游戏引擎（状态机驱动，8阶段循环）
- 5个角色（狼人/预言家/女巫/猎人/村民）
- FastAPI Server + Vue 3 前端观战台
- 基础AI代理（通用Prompt，无角色策略）
- 经验记忆系统（JSON文件存储）
- 游戏总结代理
