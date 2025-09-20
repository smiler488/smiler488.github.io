const express = require('express');
const cors = require('cors');
const crypto = require('crypto');

const app = express();
const port = 3002;

// 启用 CORS
app.use(cors({
  origin: ['http://localhost:3000', 'https://smiler488.github.io'],
  credentials: true
}));

app.use(express.json({ limit: '10mb' }));

// 模拟 AI 响应的函数
function generateMockResponse(input) {
  if (input.imageBase64) {
    return {
      success: true,
      response: "I can see an image has been uploaded. This is a mock response since we don't have real AI API credentials configured. The image appears to contain mathematical or technical content that would normally be analyzed by the Hunyuan AI model.",
      model: input.model || 'hunyuan-vision',
      timestamp: new Date().toISOString(),
      mock: true
    };
  } else if (input.question) {
    return {
      success: true,
      response: `This is a mock response to your question: "${input.question}". In a real deployment, this would be processed by the Hunyuan AI model and provide a detailed answer. Please configure the HUNYUAN_SECRET_ID and HUNYUAN_SECRET_KEY environment variables for actual AI functionality.`,
      model: input.model || 'hunyuan-lite',
      timestamp: new Date().toISOString(),
      mock: true
    };
  } else {
    return {
      success: false,
      error: "No valid input provided",
      mock: true
    };
  }
}

// API 端点
app.post('/api/solve', (req, res) => {
  console.log('Received request:', {
    hasImage: !!req.body.imageBase64,
    hasQuestion: !!req.body.question,
    model: req.body.model
  });

  try {
    const response = generateMockResponse(req.body);
    res.json(response);
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      mock: true
    });
  }
});

// 健康检查端点
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    message: 'Local AI Solver API is running',
    timestamp: new Date().toISOString()
  });
});

app.listen(port, () => {
  console.log(`🚀 Local AI Solver API running at http://localhost:${port}`);
  console.log(`📝 Test endpoint: http://localhost:${port}/api/health`);
  console.log(`🔧 This is a mock server for testing. Configure real API keys for production.`);
});