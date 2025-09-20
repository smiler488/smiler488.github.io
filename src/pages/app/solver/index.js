import React, { useEffect, useRef, useState } from 'react';

function useCamera() {
  const videoRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stream = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setReady(true);
        }
      } catch (e) {
        setError(e?.message || String(e));
      }
    })();

    return () => {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return { videoRef, ready, error };
}

async function captureCompressedJpeg(video, maxSide = 1280, quality = 0.85) {
  const w = video.videoWidth || 1080;
  const h = video.videoHeight || 1440;
  const scale = Math.min(1, maxSide / Math.max(w, h));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return await new Promise((resolve) => canvas.toBlob((b) => resolve(b), 'image/jpeg', quality));
}

async function postJson(url, json) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(json),
  });
}

export default function SolverAppPage() {
  const { videoRef, ready, error } = useCamera();

  // 状态管理
  const [mode, setMode] = useState('proxy'); // 'proxy' 或 'direct'
  const [proxyUrl, setProxyUrl] = useState(() => {
    // 自动检测环境并设置默认API地址
    if (typeof window !== 'undefined') {
      const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
      if (isLocalhost) {
        return 'http://localhost:3002/api/solve'; // 本地开发
      } else {
        // 生产环境默认为空，需要用户手动配置
        return '';
      }
    }
    return ''; // 服务端渲染时的默认值
  });
  const [secretId, setSecretId] = useState('');
  const [secretKey, setSecretKey] = useState('');
  const [question, setQuestion] = useState('Please analyze the problem in the image and provide a step-by-step solution.');
  const [model, setModel] = useState('hunyuan-vision');
  const [respText, setRespText] = useState('');
  const [busy, setBusy] = useState(false);
  const [lastSizeKB, setLastSizeKB] = useState(null);
  const [captureMode, setCaptureMode] = useState('camera'); // 'camera', 'screenshot', 'text'
  const [textInput, setTextInput] = useState(''); // 文本输入

  // 通用的发送到AI的函数
  async function sendToAI(payload) {
    if (mode === 'direct') {
      if (!secretId || !secretKey) throw new Error('Please enter SecretId and SecretKey');
      setRespText('Direct mode is not implemented yet, please use proxy mode');
      return;
    }

    if (!proxyUrl || !proxyUrl.trim()) {
      throw new Error('Please provide a valid proxy URL in the settings below');
    }

    try {
      const response = await postJson(proxyUrl, payload);

      if (!response.ok) {
        const t = await response.text().catch(() => '');
        throw new Error(`Request failed ${response.status}: ${t}`);
      }
      const data = await response.json().catch(async () => ({ raw: await response.text() }));
      setRespText(JSON.stringify(data, null, 2));
    } catch (e) {
      if (e.name === 'TypeError' && e.message.includes('fetch')) {
        throw new Error(`Network error: Cannot connect to ${proxyUrl}. Please check the proxy URL or your network connection.`);
      }
      throw e;
    }
  }

  // 发送图片到AI
  async function sendImageToAI(base64) {
    const payload = { imageBase64: base64, question, model };
    await sendToAI(payload);
  }

  // 发送文本到AI
  async function sendTextToAI(text) {
    const payload = { question: text, model: 'hunyuan-lite' };
    await sendToAI(payload);
  }

  // 处理文本提问
  async function handleTextQuestion() {
    if (!textInput.trim()) {
      setRespText('Please enter your question');
      return;
    }

    setRespText('');
    setBusy(true);
    try {
      await sendTextToAI(textInput.trim());
    } catch (e) {
      setRespText('Error: ' + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  }

  // 截图功能
  async function handleScreenshot() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      setRespText('Error: The browser does not support screen capture');
      return;
    }

    setRespText('');
    setBusy(true);
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { mediaSource: 'screen' },
        audio: false
      });

      const video = document.createElement('video');
      video.srcObject = stream;
      video.play();

      await new Promise((resolve) => {
        video.onloadedmetadata = resolve;
      });

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      stream.getTracks().forEach(track => track.stop());

      const blob = await new Promise((resolve) => 
        canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.85)
      );
      
      setLastSizeKB(Math.round(blob.size / 1024));

      const dataUrl = await new Promise((res) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.readAsDataURL(blob);
      });
      const base64 = String(dataUrl).split(',')[1];

      await sendImageToAI(base64);

    } catch (e) {
      if (e.name === 'NotAllowedError') {
        setRespText('Error: The user canceled screen sharing permission');
      } else {
        setRespText('Error: ' + (e?.message || String(e)));
      }
    } finally {
      setBusy(false);
    }
  }

  // 拍照功能
  async function handleShoot() {
    if (!videoRef.current) return;
    setRespText('');
    setBusy(true);
    try {
      const blob = await captureCompressedJpeg(videoRef.current);
      setLastSizeKB(Math.round(blob.size / 1024));

      const dataUrl = await new Promise((res) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.readAsDataURL(blob);
      });
      const base64 = String(dataUrl).split(',')[1];

      await sendImageToAI(base64);
    } catch (e) {
      setRespText('Error: ' + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ maxWidth: 920, margin: '0 auto', padding: '24px' }}>
      <h1>AI Solver (Hunyuan)</h1>
      <p>Supports camera capture, screen capture, and text questions. For security, the page does not accept any keys.</p>

      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          {/* Input Mode Selection */}
          <fieldset style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12, marginBottom: 16 }}>
            <legend>Input Mode</legend>
            <label style={{ display: 'block', marginBottom: 8 }}>
              <input 
                type="radio" 
                value="camera" 
                checked={captureMode === 'camera'} 
                onChange={e => setCaptureMode(e.target.value)} 
              />
              <span style={{ marginLeft: 8 }}> Camera Capture</span>
            </label>
            <label style={{ display: 'block', marginBottom: 8 }}>
              <input 
                type="radio" 
                value="screenshot" 
                checked={captureMode === 'screenshot'} 
                onChange={e => setCaptureMode(e.target.value)} 
              />
              <span style={{ marginLeft: 8 }}> Screen Capture</span>
            </label>
            <label style={{ display: 'block' }}>
              <input 
                type="radio" 
                value="text" 
                checked={captureMode === 'text'} 
                onChange={e => setCaptureMode(e.target.value)} 
              />
              <span style={{ marginLeft: 8 }}> Text Question</span>
            </label>
          </fieldset>

          {/* 摄像头预览 */}
          {captureMode === 'camera' && (
            <div>
              <video ref={videoRef} autoPlay playsInline style={{ width: 320, background: '#000', borderRadius: 8 }} />
              <div style={{ marginTop: 8, color: '#666' }}>
                {ready ? 'Camera ready' : error ? `Camera error: ${error}` : 'Requesting camera permission…'}
              </div>
              <button onClick={handleShoot} disabled={!ready || busy} style={{ marginTop: 12, padding: '8px 16px', fontSize: 14 }}>
                {busy ? 'Processing…' : ' Capture and Solve'}
              </button>
            </div>
          )}

          {/* 截图模式 */}
          {captureMode === 'screenshot' && (
            <div>
              <div style={{ width: 320, height: 240, background: '#f5f5f5', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', border: '2px dashed #ccc' }}>
                <div style={{ textAlign: 'center', color: '#666' }}>
                  <div style={{ fontSize: 48, marginBottom: 8 }}> </div>
                  <div>Click the button below to start screenshot</div>
                </div>
              </div>
              <div style={{ marginTop: 8, color: '#666' }}>
                Screenshot mode: the entire screen will be captured
              </div>
              <button onClick={handleScreenshot} disabled={busy} style={{ marginTop: 12, padding: '8px 16px', fontSize: 14 }}>
                {busy ? 'Processing…' : ' Capture Screen and Solve'}
              </button>
            </div>
          )}

          {/* 文本提问模式 */}
          {captureMode === 'text' && (
            <div>
              <div style={{ width: 320, minHeight: 240, background: '#f9f9f9', borderRadius: 8, padding: 16, border: '1px solid #ddd' }}>
                <div style={{ marginBottom: 12, color: '#666', fontSize: 14 }}> Text Question Mode</div>
                <textarea 
                  value={textInput}
                  onChange={e => setTextInput(e.target.value)}
                  placeholder="Please enter your question, for example:&#10;• Explain the basic principles of quantum mechanics&#10;• Write a Python sorting algorithm&#10;• What is 1+1?"
                  style={{ 
                    width: '100%', 
                    height: 160, 
                    border: '1px solid #ccc', 
                    borderRadius: 4, 
                    padding: 8, 
                    fontSize: 14,
                    resize: 'vertical',
                    fontFamily: 'inherit'
                  }}
                />
              </div>
              <div style={{ marginTop: 8, color: '#666' }}>
                Enter your question directly and the AI will provide a detailed answer
              </div>
              <button 
                onClick={handleTextQuestion} 
                disabled={busy || !textInput.trim()} 
                style={{ marginTop: 12, padding: '8px 16px', fontSize: 14 }}
              >
                {busy ? 'Thinking…' : ' Ask and Get Answer'}
              </button>
            </div>
          )}

          {lastSizeKB != null && (
            <div style={{ marginTop: 6, fontSize: 12, color: '#999' }}>Last image size: {lastSizeKB} KB</div>
          )}
        </div>

        <div style={{ flex: 1, minWidth: 280 }}>
          <fieldset style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12, marginBottom: 16 }}>
            <legend>Connection Mode</legend>
            <label>
              <input type="radio" value="proxy" checked={mode === 'proxy'} onChange={e => setMode(e.target.value)} />
              Proxy Mode (Recommended)
            </label>
            <label style={{ marginLeft: 16 }}>
              <input type="radio" value="direct" checked={mode === 'direct'} onChange={e => setMode(e.target.value)} />
              Direct Mode (Testing)
            </label>

          </fieldset>

          {mode === 'proxy' ? (
            <fieldset style={{ border: '1px solid #d7e3ff', borderRadius: 8, padding: 12, marginBottom: 16, background: '#f7faff' }}>
              <legend>Proxy Settings</legend>
              <label>Proxy URL<br />
                <input value={proxyUrl} onChange={e => setProxyUrl(e.target.value)} placeholder="Enter your API endpoint or select from options below" style={{ width: '100%' }} />
              </label>
              
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>Quick Options:</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  <button 
                    type="button"
                    onClick={() => setProxyUrl('http://localhost:3002/api/solve')}
                    style={{ padding: '4px 8px', fontSize: 11, background: '#e8f4fd', border: '1px solid #b3d9ff', borderRadius: 4, cursor: 'pointer' }}
                  >
                    Local Dev
                  </button>
                  <button 
                    type="button"
                    onClick={() => setProxyUrl('https://api.allorigins.win/raw?url=' + encodeURIComponent('https://smiler488-github-io.vercel.app/api/solve'))}
                    style={{ padding: '4px 8px', fontSize: 11, background: '#fff2e8', border: '1px solid #ffcc99', borderRadius: 4, cursor: 'pointer' }}
                  >
                    CORS Proxy (AllOrigins)
                  </button>
                  <button 
                    type="button"
                    onClick={() => setProxyUrl('https://cors-anywhere.herokuapp.com/https://smiler488-github-io.vercel.app/api/solve')}
                    style={{ padding: '4px 8px', fontSize: 11, background: '#f0f8e8', border: '1px solid #b3d9b3', borderRadius: 4, cursor: 'pointer' }}
                  >
                    CORS Proxy (Heroku)
                  </button>
                  <button 
                    type="button"
                    onClick={() => setProxyUrl('http://localhost:3002/api/solve')}
                    style={{ padding: '4px 8px', fontSize: 11, background: '#e8f4fd', border: '1px solid #b3d9ff', borderRadius: 4, cursor: 'pointer' }}
                  >
                    Local Server
                  </button>
                  <button 
                    type="button"
                    onClick={() => setProxyUrl('https://smiler488github-3q35x1gqq-smiler488s-projects.vercel.app/api/solve')}
                    style={{ padding: '4px 8px', fontSize: 11, background: '#e8f5e8', border: '1px solid #4caf50', borderRadius: 4, cursor: 'pointer' }}
                  >
                    Latest Vercel
                  </button>
                  <button 
                    type="button"
                    onClick={() => setProxyUrl('https://smiler488githubio.vercel.app/api/solve')}
                    style={{ padding: '4px 8px', fontSize: 11, background: '#f8f0ff', border: '1px solid #d9b3ff', borderRadius: 4, cursor: 'pointer' }}
                  >
                    Old Vercel
                  </button>
                </div>
              </div>
              
              <div style={{ color: '#345', fontSize: 12, marginTop: 12 }}>
                <strong>Tips:</strong><br />
                • Local Dev: For development with local server<br />
                • CORS Proxy: Bypass CORS restrictions for GitHub Pages<br />
                • Direct Vercel: Try direct access (may have CORS issues)<br />
                • Configure in .env.local: HUNYUAN_SECRET_ID, HUNYUAN_SECRET_KEY, HUNYUAN_VERSION
              </div>
            </fieldset>
          ) : (
            <fieldset style={{ border: '1px solid #ffe7d7', borderRadius: 8, padding: 12, marginBottom: 16, background: '#fff7f0' }}>
              <legend>Direct Settings (Testing only)</legend>
              <label>Secret ID<br />
                <input value={secretId} onChange={e => setSecretId(e.target.value)} placeholder="AKIDxxxxx" style={{ width: '100%' }} />
              </label>
              <label style={{ display: 'block', marginTop: 8 }}>Secret Key<br />
                <input type="password" value={secretKey} onChange={e => setSecretKey(e.target.value)} placeholder="Secret Key" style={{ width: '100%' }} />
              </label>
              <div style={{ color: '#d63031', fontSize: 12, marginTop: 8 }}>
                 Direct mode exposes the key, for local testing only!
              </div>
            </fieldset>
          )}

          <fieldset style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12 }}>
            <legend>Request Parameters</legend>
            <label>Model<br />
              <input value={model} onChange={e => setModel(e.target.value)} style={{ width: '100%' }} />
            </label>
            <label style={{ display: 'block', marginTop: 8 }}>Question (for image mode)<br />
              <textarea value={question} onChange={e => setQuestion(e.target.value)} rows={3} style={{ width: '100%' }} />
            </label>
          </fieldset>
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <h3>Response</h3>
        <pre style={{ whiteSpace: 'pre-wrap', background: '#111', color: '#0f0', padding: 12, borderRadius: 8, maxHeight: 360, overflow: 'auto' }}>{respText}</pre>
      </div>
    </div>
  );
}