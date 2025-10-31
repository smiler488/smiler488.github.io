const express = require('express');
const cors = require('cors');
const crypto = require('crypto');

const app = express();
const port = 3001;

// 启用 CORS
app.use(cors({
  origin: ['http://localhost:3000', 'https://smiler488.github.io'],
  credentials: true
}));

app.use(express.json({ limit: '10mb' }));

// 模拟 AI 响应的函数
function formatMockContent(input) {
  if (input.imageBase64 || input.imageUrl) {
    return `Mock Vision Analysis\n\nModel: ${input.model || 'hunyuan-vision'}\nSummary: Received an image payload (base64 length ${input.imageBase64?.length || 0}).\n\nIn a real deployment this would be analyzed by the Hunyuan vision model.`;
  }

  if (input.question) {
    return `Mock Text Analysis\n\nModel: ${input.model || 'hunyuan-lite'}\nQuestion: ${input.question}\n\nThis is a placeholder response from the local mock server. Deploy the real /api/solve proxy to get actual answers.`;
  }

  return 'No valid input provided.';
}

function generateMockResponse(input) {
  const content = formatMockContent(input);
  return {
    id: `mock-${crypto.randomBytes(6).toString('hex')}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: input.model || 'hunyuan-vision',
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content,
        },
        finish_reason: 'stop',
      },
    ],
    usage: {
      prompt_tokens: 64,
      completion_tokens: 128,
      total_tokens: 192,
    },
    mock: true,
  };
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
