# AI Solver 使用指南

## 🚀 快速开始

### 方案1：本地测试服务器（推荐）
```bash
# 启动本地测试服务器
node local-server.js

# 服务器将在 http://localhost:3002 运行
# 访问 https://smiler488.github.io/app/solver
# 点击 "Local Server" 按钮
```

### 方案2：Vercel 生产环境
- 最新部署：`https://smiler488github-3q35x1gqq-smiler488s-projects.vercel.app/api/solve`
- 环境变量已配置：HUNYUAN_SECRET_ID, HUNYUAN_SECRET_KEY, HUNYUAN_VERSION

## 📱 功能特性

### 三种输入模式
1. **📷 摄像头拍照**：拍摄数学题、文档等进行AI分析
2. **🖥️ 屏幕截图**：截取屏幕内容进行分析
3. **💬 文本提问**：直接输入问题获得AI回答

### 连接选项
- **Local Server**：本地测试服务器（稳定）
- **Latest Vercel**：最新Vercel部署
- **CORS Proxy**：通过代理访问（备用）

## 🔧 技术架构

### 前端（GitHub Pages）
- 网址：`https://smiler488.github.io/app/solver`
- 框架：Docusaurus + React
- 功能：摄像头、截图、文本输入

### 后端API
1. **本地服务器**：`http://localhost:3002/api/solve`
   - 模拟AI响应，用于测试界面功能
   - 启动：`node local-server.js`

2. **Vercel无服务器函数**：`/api/solve.js`
   - 真实的腾讯云混元AI集成
   - 环境变量：HUNYUAN_SECRET_ID, HUNYUAN_SECRET_KEY, HUNYUAN_VERSION

## 📝 API接口

### POST /api/solve

**文本提问：**
```json
{
  "question": "What is 2+2?",
  "model": "hunyuan-lite"
}
```

**图片分析：**
```json
{
  "imageBase64": "base64编码的图片数据",
  "question": "Please analyze this image",
  "model": "hunyuan-vision"
}
```

**响应格式：**
```json
{
  "success": true,
  "response": "AI的回答内容",
  "model": "使用的模型",
  "timestamp": "2025-09-20T06:50:27.751Z",
  "mock": false
}
```

## 🚀 部署流程

### 更新前端
```bash
git add .
git commit -m "更新描述"
git push origin master
npm run deploy
```

### 更新API
```bash
git add .
git commit -m "更新API"
git push origin master
# Vercel会自动重新部署
```

## 🔍 故障排除

### 1. "Error: Failed to fetch"
- 尝试不同的连接选项
- 使用本地服务器：`node local-server.js`
- 检查网络连接和防火墙设置

### 2. 404错误
- 确认Vercel环境变量已配置
- 检查API函数是否正确部署
- 使用CORS代理作为备选

### 3. 摄像头无法访问
- 确保浏览器允许摄像头权限
- 使用HTTPS访问（GitHub Pages自动提供）
- 尝试不同浏览器

## 📊 测试命令

```bash
# 测试本地服务器
curl -X POST "http://localhost:3002/api/solve" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is 2+2?","model":"hunyuan-lite"}'

# 测试Vercel API
curl -X POST "https://smiler488github-3q35x1gqq-smiler488s-projects.vercel.app/api/solve" \
  -H "Content-Type: application/json" \
  -d '{"question":"test","model":"hunyuan-lite"}'

# 健康检查
curl "http://localhost:3002/api/health"
```

## 🎯 使用建议

1. **开发测试**：使用本地服务器，快速验证界面功能
2. **生产使用**：配置真实API密钥，使用Vercel部署
3. **网络问题**：使用CORS代理选项绕过限制
4. **功能测试**：三种输入模式都要测试确保正常工作

## 📱 移动端支持

- 响应式设计，支持手机和平板
- 摄像头自动选择后置摄像头
- 触摸友好的界面设计
- 支持移动端截图功能