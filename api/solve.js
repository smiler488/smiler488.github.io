/**
 * Serverless proxy for Tencent Hunyuan OpenAI-compatible API (/v1/chat/completions).
 * Accepts JSON bodies in the shape: { imageBase64, imageUrl, question, model, temperature, top_p, max_tokens }
 * Converts inputs into the official "messages" format and forwards them with the provided API key.
 *
 * Required env vars:
 *   - HUNYUAN_API_KEY: API key string (starts with sk-)
 * Optional env vars:
 *   - HUNYUAN_BASE_URL: defaults to https://api.hunyuan.cloud.tencent.com
 *   - ALLOW_ORIGIN: CORS allow list (defaults to '*')
 */

const DEFAULT_BASE_URL = process.env.HUNYUAN_BASE_URL || 'https://api.hunyuan.cloud.tencent.com';
const CHAT_COMPLETIONS_PATH = '/v1/chat/completions';

function setCors(req, res) {
  const allowOrigin = process.env.ALLOW_ORIGIN || '*';
  const reqAllowHeaders = req.headers['access-control-request-headers'];
  res.setHeader('Access-Control-Allow-Origin', allowOrigin);
  res.setHeader('Vary', 'Origin');
  res.setHeader('Access-Control-Allow-Methods', 'POST,OPTIONS');
  res.setHeader(
    'Access-Control-Allow-Headers',
    reqAllowHeaders || 'Content-Type, Authorization, Accept, X-Requested-With, X-CSRF-Token'
  );
  res.setHeader('Access-Control-Max-Age', '600');
}

function normalizeBase64(imageBase64) {
  if (!imageBase64) return null;
  if (imageBase64.startsWith('data:')) {
    return imageBase64;
  }
  return `data:image/jpeg;base64,${imageBase64}`;
}

async function parseJsonBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.setEncoding('utf8');
    req.on('data', (chunk) => (data += chunk));
    req.on('end', () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch (err) {
        reject(err);
      }
    });
    req.on('error', reject);
  });
}

function buildMessages({ question, imageBase64, imageUrl }) {
  const contentBlocks = [];
  const trimmedQuestion = (question || '').trim();
  if (trimmedQuestion) {
    contentBlocks.push({
      type: 'text',
      text: trimmedQuestion,
    });
  }

  const normalizedBase64 = normalizeBase64(imageBase64);
  const normalizedImageUrl = (imageUrl || '').trim();

  if (normalizedImageUrl) {
    contentBlocks.push({
      type: 'image_url',
      image_url: {
        url: normalizedImageUrl,
      },
    });
  } else if (normalizedBase64) {
    contentBlocks.push({
      type: 'image_url',
      image_url: {
        url: normalizedBase64,
      },
    });
  }

  if (contentBlocks.length === 0) {
    return [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: '你好',
          },
        ],
      },
    ];
  }

  return [
    {
      role: 'user',
      content: contentBlocks,
    },
  ];
}

module.exports = async function handler(req, res) {
  setCors(req, res);

  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST,OPTIONS');
    res.status(405).send('Method Not Allowed');
    return;
  }

  const apiKey = process.env.HUNYUAN_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: 'Server not configured: missing HUNYUAN_API_KEY' });
    return;
  }

  let body;
  try {
    body = await parseJsonBody(req);
  } catch (err) {
    res.status(400).json({ error: 'Invalid JSON body', detail: String(err) });
    return;
  }

  const {
    imageBase64,
    imageUrl,
    question,
    model = 'hunyuan-vision',
    temperature,
    top_p,
    max_tokens,
    stream = false,
  } = body || {};

  const messages = buildMessages({ question, imageBase64, imageUrl });

  const payload = {
    model,
    messages,
    stream: Boolean(stream),
  };

  if (typeof temperature === 'number') {
    payload.temperature = temperature;
  }
  if (typeof top_p === 'number') {
    payload.top_p = top_p;
  }
  if (typeof max_tokens === 'number') {
    payload.max_tokens = max_tokens;
  }

  const baseUrl = DEFAULT_BASE_URL.replace(/\/$/, '');
  const url = `${baseUrl}${CHAT_COMPLETIONS_PATH}`;

  try {
    const upstreamResp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(payload),
    });

    const text = await upstreamResp.text();
    const contentType = upstreamResp.headers.get('content-type') || 'application/json; charset=utf-8';

    res.status(upstreamResp.status);
    res.setHeader('Content-Type', contentType);
    res.send(text);
  } catch (err) {
    console.error('Hunyuan proxy error:', err);
    res.status(502).json({ error: 'Upstream call failed', detail: String(err) });
  }
};
