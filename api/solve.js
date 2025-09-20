/**
 * Serverless proxy for Tencent Cloud Hunyuan "ImageQuestion" with TC3-HMAC-SHA256
 * - Accepts JSON from client: { imageBase64, imageUrl, question, model }
 * - Builds Messages schema and signs request to Hunyuan API
 * - CORS is controlled by ALLOW_ORIGIN
 *
 * Required ENV:
 * - HUNYUAN_SECRET_ID
 * - HUNYUAN_SECRET_KEY
 * - HUNYUAN_VERSION  (e.g. "2024-09-01")  // check official doc for the correct version
 * Optional ENV:
 * - HUNYUAN_REGION   (e.g. "ap-guangzhou")
 * - ALLOW_ORIGIN     (front-end origin, e.g. "https://smiler488.github.io")
 *
 * Endpoint/Service:
 *   Host: hunyuan.tencentcloudapi.com
 *   Service: "hunyuan"
 *   Action: "ImageQuestion"
 *
 * NOTE:
 * - imageUrl is preferred if your image is hosted; otherwise use imageBase64 (without "data:image/jpeg;base64,")
 */

const crypto = require('crypto');

const HOST = 'hunyuan.tencentcloudapi.com';
const SERVICE = 'hunyuan';
const ACTION = 'ChatCompletions';

function setCors(req, res) {
  const allowOrigin = process.env.ALLOW_ORIGIN || '*';
  const reqAllowHeaders = req.headers['access-control-request-headers'];
  res.setHeader('Access-Control-Allow-Origin', allowOrigin);
  res.setHeader('Vary', 'Origin');
  res.setHeader('Access-Control-Allow-Methods', 'POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', reqAllowHeaders || 'Content-Type, Authorization, Accept, X-Requested-With, X-CSRF-Token');
  res.setHeader('Access-Control-Max-Age', '600');
}

function hmacSHA256(key, msg, encoding) {
  return crypto.createHmac('sha256', key).update(msg).digest(encoding || undefined);
}
function sha256(msg, encoding) {
  return crypto.createHash('sha256').update(msg).digest(encoding || undefined);
}

/**
 * Build TC3 Authorization header
 */
function buildTC3Auth({ secretId, secretKey, service, host, action, version, region, timestamp, payload }) {
  const date = new Date(timestamp * 1000).toISOString().slice(0, 10); // YYYY-MM-DD

  // Step 1: Canonical Request
  const httpRequestMethod = 'POST';
  const canonicalUri = '/';
  const canonicalQueryString = '';
  const canonicalHeaders = `content-type:application/json\nhost:${host}\n`;
  const signedHeaders = 'content-type;host';
  const hashedRequestPayload = sha256(JSON.stringify(payload), 'hex');
  const canonicalRequest = [
    httpRequestMethod,
    canonicalUri,
    canonicalQueryString,
    canonicalHeaders,
    signedHeaders,
    hashedRequestPayload,
  ].join('\n');

  // Step 2: String To Sign
  const algorithm = 'TC3-HMAC-SHA256';
  const credentialScope = `${date}/${service}/tc3_request`;
  const hashedCanonicalRequest = sha256(canonicalRequest, 'hex');
  const stringToSign = [algorithm, String(timestamp), credentialScope, hashedCanonicalRequest].join('\n');

  // Step 3: Signature
  const secretDate = hmacSHA256(`TC3${secretKey}`, date);
  const secretService = hmacSHA256(secretDate, service);
  const secretSigning = hmacSHA256(secretService, 'tc3_request');
  const signature = hmacSHA256(secretSigning, stringToSign, 'hex');

  const authorization = `${algorithm} Credential=${secretId}/${credentialScope}, SignedHeaders=${signedHeaders}, Signature=${signature}`;
  return authorization;
}

function parseJsonBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.setEncoding('utf8');
    req.on('data', (chunk) => (data += chunk));
    req.on('end', () => {
      try {
        const json = data ? JSON.parse(data) : {};
        resolve(json);
      } catch (e) {
        reject(e);
      }
    });
    req.on('error', reject);
  });
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

  const secretId = process.env.HUNYUAN_SECRET_ID;
  const secretKey = process.env.HUNYUAN_SECRET_KEY;
  const version = process.env.HUNYUAN_VERSION || '2023-09-01'; // 使用你提供的版本
  const region = process.env.HUNYUAN_REGION || '';

  if (!secretId || !secretKey) {
    res.status(500).json({ error: 'Server not configured: missing HUNYUAN_SECRET_ID / HUNYUAN_SECRET_KEY' });
    return;
  }

  try {
    // Expecting JSON: { imageBase64, imageUrl, question, model }
    const { imageBase64, imageUrl, question, model } = await parseJsonBody(req);

    const isVisionModel = model && model.includes('vision');
    
    if (isVisionModel && !imageUrl && !imageBase64) {
      res.status(400).json({ error: 'Vision model requires imageUrl or imageBase64' });
      return;
    }

    // Build Hunyuan ChatCompletions request payload (按照官方SDK格式)
    let payload;
    
    if (isVisionModel && (imageUrl || imageBase64)) {
      // Vision model - 使用 Contents 数组格式
      const contents = [
        {
          Type: 'text',
          Text: question || '请分析这张图片',
        }
      ];
      
      if (imageUrl) {
        contents.push({
          Type: 'image_url',
          ImageUrl: { Url: imageUrl },
        });
      } else if (imageBase64) {
        contents.push({
          Type: 'image_url', 
          ImageUrl: { Url: `data:image/jpeg;base64,${imageBase64}` },
        });
      }

      payload = {
        Model: model || 'hunyuan-vision',
        Messages: [
          {
            Role: 'user',
            Contents: contents,  // 使用 Contents 而不是 Content
          },
        ],
        Stream: false,
      };
    } else {
      // Text model - 纯文本格式
      payload = {
        Model: model || 'hunyuan-lite',
        Messages: [
          {
            Role: 'user',
            Content: question || '你好',
          },
        ],
        Stream: false,
      };
    }

    const timestamp = Math.floor(Date.now() / 1000);
    const authorization = buildTC3Auth({
      secretId,
      secretKey,
      service: SERVICE,
      host: HOST,
      action: ACTION,
      version,
      region,
      timestamp,
      payload,
    });

    const headers = {
      'Content-Type': 'application/json',
      Host: HOST,
      'X-TC-Action': ACTION,
      'X-TC-Version': version,
      'X-TC-Timestamp': String(timestamp),
    };
    if (region) headers['X-TC-Region'] = region;
    headers['Authorization'] = authorization;

    const upstreamResp = await fetch(`https://${HOST}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    const text = await upstreamResp.text();
    res.status(upstreamResp.status);
    // Try to set appropriate content type
    const ct = upstreamResp.headers.get('content-type') || 'application/json; charset=utf-8';
    res.setHeader('Content-Type', ct);
    res.send(text);
  } catch (err) {
    console.error('Proxy error:', err);
    res.status(502).json({ error: 'Upstream call failed', detail: String(err) });
  }
}