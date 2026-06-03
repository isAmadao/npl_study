# AI狼人杀 — 环境配置

## 开发环境

| 组件 | 版本要求 | 当前状态 |
|------|---------|---------|
| OS | Windows 11 | ✅ |
| Python | 3.12 (conda py312) | ✅ (Windows侧) |
| Node.js | v22+ | ✅ v22.22.0 |
| npm | 10+ | ✅ 10.9.4 |
| DeepSeek API Key | — | ⚠️ 需配置 |

## 后端环境配置

### 1. 激活 conda 环境（Windows终端）

```powershell
conda activate py312
```

### 2. 安装Python依赖

```powershell
cd E:\Code\badou2\韩鹏\week16\Werewolf-Game\backend
pip install -r requirements.txt
```

### 3. 配置DeepSeek API Key

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

获取API Key：访问 https://platform.deepseek.com 注册并创建API Key。

## 前端环境配置

```powershell
cd E:\Code\badou2\韩鹏\week16\Werewolf-Game\frontend
npm install
```

## 启动方式

### Phase 1-2：终端模式

```powershell
conda activate py312
cd E:\Code\badou2\韩鹏\week16\Werewolf-Game\backend
python -m werewolf.main
```

### Phase 3-4：完整服务模式

终端1（后端）：
```powershell
conda activate py312
cd E:\Code\badou2\韩鹏\week16\Werewolf-Game\backend
python -m werewolf.server
```

终端2（前端）：
```powershell
cd E:\Code\badou2\韩鹏\week16\Werewolf-Game\frontend
npm run dev
```

浏览器访问：`http://localhost:5173`

## 依赖清单 (requirements.txt)

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
websockets>=12.0
pydantic>=2.0.0
openai>=1.0.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

## 故障排除

| 问题 | 解决方法 |
|------|---------|
| DeepSeek API返回401 | 检查 `.env` 中API Key是否正确 |
| pip安装失败 | 确认conda py312已激活 |
| 前端构建失败 | `rm -rf node_modules && npm install` |
