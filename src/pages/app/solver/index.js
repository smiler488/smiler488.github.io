
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
  // Mock mode: allow using a fake upstream when url starts with "mock://"
  if (typeof url === 'string' && url.startsWith('mock://')) {
    const now = new Date().toISOString();
    const mockText = (() => {
      const q = json?.question || 'No question provided';
      const model = json?.model || 'hunyuan-lite';
      const hasImage = !!json?.imageBase64 || !!json?.imageUrl;
      const header = hasImage ? 'Mock Vision Analysis' : 'Mock Text Analysis';
      return `${header} (model: ${model})\n\nUser question:\n${q}\n\nThis is a mocked response for demo purposes. Replace proxy URL with your real API when ready.`;
    })();

    const body = {
      Response: {
        RequestId: 'mock-' + Math.random().toString(36).slice(2),
        Choices: [
          {
            Message: {
              Content: mockText,
            },
          },
        ],
        Usage: {
          PromptTokens: 128,
          CompletionTokens: 256,
          TotalTokens: 384,
        },
        Timestamp: now,
      },
    };

    return {
      ok: true,
      status: 200,
      json: async () => body,
      text: async () => JSON.stringify(body),
      headers: new Map([['content-type', 'application/json']]),
    };
  }

  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(json),
  });
}

export default function SolverAppPage() {
  const { videoRef, ready, error } = useCamera();

  // State management
  const [apiUrl, setApiUrl] = useState('0000000000');
  // Preset prompt list
  const promptPresets = [
    { 
      id: 'default', 
      name: 'Default Analysis', 
      prompt: 'Please analyze the problem in the image and provide a step-by-step solution.',
      description: 'General analysis for most problems'
    },
    { 
      id: 'math', 
      name: 'Math Problem Solver', 
      prompt: 'Please analyze the mathematical problem in the image and provide detailed solution steps and the final answer. If there are multiple approaches, show the most elegant method.',
      description: 'For math equations, geometry problems, etc.'
    },
    { 
      id: 'lab_safety', 
      name: 'Laboratory Safety Officer', 
      prompt: 'As a laboratory safety officer, please analyze the safety concerns in this image. Identify potential hazards, safety violations, or improper procedures. Provide specific recommendations for correcting these issues according to standard laboratory safety protocols. If PPE (Personal Protective Equipment) is visible, evaluate if it\'s appropriate and properly worn. For chemical or biological hazards, include proper handling and disposal procedures.',
      description: 'Safety assessment for laboratory environments'
    },
    { 
      id: 'plant', 
      name: 'Plant Identification', 
      prompt: 'Please identify the plant in the image. Provide its scientific name, common names, growth habits, native regions, and any special uses (medicinal, ornamental, etc.). For common plants, include care tips.',
      description: 'Identify plant species and related information'
    },
    { 
      id: 'code', 
      name: 'Code Analysis', 
      prompt: 'Please analyze the code in the image. Explain its functionality, identify potential issues or bugs, and suggest optimizations. If there are obvious errors, provide corrected code.',
      description: 'Analyze code issues and provide fixes'
    },
    { 
      id: 'translation', 
      name: 'Academic Translation', 
      prompt: 'Please translate the academic text in the image to English, maintaining accuracy of technical terms. The translation should be fluent and natural while preserving the academic style.',
      description: 'Professional translation of academic literature'
    },
    { 
      id: 'physics', 
      name: 'Physics Problem Solver', 
      prompt: 'Please analyze the physics problem in the image. Explain the relevant physics concepts and principles, then provide detailed solution steps and the final answer. For formula derivations, clearly show each step.',
      description: 'Detailed analysis of physics problems'
    },
    { 
      id: 'chemistry', 
      name: 'Chemistry Problem Solver', 
      prompt: 'Please analyze the chemistry problem in the image. Explain the relevant chemical concepts and principles, provide chemical equations, balanced equations or reaction mechanisms, then give detailed solution steps and the final answer.',
      description: 'Detailed analysis of chemistry problems'
    },
    { 
      id: 'english', 
      name: 'English Learning', 
      prompt: 'Please analyze the English text in the image. Explain its grammatical structure, key vocabulary, and expressions. If it\'s an exercise, provide the correct answers with detailed explanations. For difficult vocabulary, provide definitions and example sentences.',
      description: 'English learning and exercise analysis'
    },
    { 
      id: 'ocr', 
      name: 'Text Extraction', 
      prompt: 'Please extract all text content from the image, maintaining the original paragraph structure and format. If there are tables, try to preserve the table structure. The extracted text should be accurate.',
      description: 'Extract text content from images'
    },
    { 
      id: 'custom', 
      name: 'Custom', 
      prompt: '',
      description: 'Custom prompt'
    }
  ];

  const [selectedPreset, setSelectedPreset] = useState('default');
  const [question, setQuestion] = useState(promptPresets[0].prompt);
  const [model, setModel] = useState('hunyuan-vision');
  const [respText, setRespText] = useState('');
  const [busy, setBusy] = useState(false);
  const [lastSizeKB, setLastSizeKB] = useState(null);
  const [captureMode, setCaptureMode] = useState('camera'); // 'camera', 'screenshot', 'text'
  const [textInput, setTextInput] = useState(''); // Text input
  const [screenshotData, setScreenshotData] = useState(null); // Screenshot data
  const [selectionBox, setSelectionBox] = useState(null); // Selection box
  const [isSelecting, setIsSelecting] = useState(false); // Whether selecting

  // 通用的发送到AI的函数
  async function sendToAI(payload) {
    // If API URL looks invalid (e.g., default zeros or missing http), use local mock to keep UI functional
    const useMock = !apiUrl || !apiUrl.trim() || !/^https?:\/\//.test(apiUrl);

    try {
      const response = await postJson(useMock ? 'mock://ai-solver' : apiUrl, payload);

      if (!response.ok) {
        const t = await response.text().catch(() => '');
        throw new Error(`Request failed ${response.status}: ${t}`);
      }
      const data = await response.json().catch(async () => ({ raw: await response.text() }));
      
      // Format the response for better readability
      if (data.Response && data.Response.Choices && data.Response.Choices.length > 0) {
        const aiMessage = data.Response.Choices[0].Message?.Content || '';
        const usage = data.Response.Usage || {};
        
        // Create a formatted display with the most important information
        const formattedResponse = `${aiMessage}\n\n---\nTokens: ${usage.TotalTokens || 'N/A'} (Prompt: ${usage.PromptTokens || 'N/A'}, Completion: ${usage.CompletionTokens || 'N/A'})`;
        setRespText(formattedResponse);
        
        // Also store the full response data in case it's needed
        window.lastFullResponse = data;
      } else {
        // Fallback to showing the full JSON if we can't extract the message
        setRespText(JSON.stringify(data, null, 2));
      }
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

  // Send text to AI
  async function sendTextToAI(text) {
    // Check if it's a special command
    if (text.startsWith('/preset ')) {
      const presetName = text.substring(8).trim().toLowerCase();
      const preset = promptPresets.find(p => p.id.toLowerCase() === presetName || p.name.toLowerCase() === presetName);
      
      if (preset) {
        setSelectedPreset(preset.id);
        setQuestion(preset.prompt);
        setRespText(`Switched to preset: ${preset.name}\n\nPreset prompt: ${preset.prompt}`);
        return;
      } else {
        setRespText(`Preset "${presetName}" not found. Available presets: ${promptPresets.map(p => p.name).join(', ')}`);
        return;
      }
    }
    
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

  // 截图功能 - 第一步：捕获整个屏幕
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

      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      
      // 保存截图数据，进入选择模式
      setScreenshotData({
        dataUrl,
        width: canvas.width,
        height: canvas.height,
        canvas
      });
      
      // 初始化选择框（默认选择中间区域）
      const defaultSize = Math.min(canvas.width, canvas.height) * 0.5;
      setSelectionBox({
        x: (canvas.width - defaultSize) / 2,
        y: (canvas.height - defaultSize) / 2,
        width: defaultSize,
        height: defaultSize
      });
      
      setIsSelecting(true);

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

  // 确认选择区域并发送给AI
  async function handleConfirmSelection() {
    if (!screenshotData || !selectionBox) return;
    
    setBusy(true);
    try {
      // 创建新的canvas来裁剪选中区域
      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = selectionBox.width;
      cropCanvas.height = selectionBox.height;
      const cropCtx = cropCanvas.getContext('2d');
      
      // 从原始canvas裁剪选中区域
      cropCtx.drawImage(
        screenshotData.canvas,
        selectionBox.x, selectionBox.y, selectionBox.width, selectionBox.height,
        0, 0, selectionBox.width, selectionBox.height
      );
      
      const blob = await new Promise((resolve) => 
        cropCanvas.toBlob((b) => resolve(b), 'image/jpeg', 0.85)
      );
      
      setLastSizeKB(Math.round(blob.size / 1024));

      const dataUrl = await new Promise((res) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.readAsDataURL(blob);
      });
      const base64 = String(dataUrl).split(',')[1];

      await sendImageToAI(base64);
      
      // 清理状态
      setScreenshotData(null);
      setSelectionBox(null);
      setIsSelecting(false);

    } catch (e) {
      setRespText('Error: ' + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  }

  // 取消选择
  function handleCancelSelection() {
    setScreenshotData(null);
    setSelectionBox(null);
    setIsSelecting(false);
  }

  // 处理选择框拖拽 - 完全重写以提高稳定性
  function handleSelectionDrag(e, type) {
    if (!screenshotData || !selectionBox) return;
    
    // 阻止默认行为和冒泡，防止干扰
    e.preventDefault();
    e.stopPropagation();
    
    // 获取图像容器的位置和尺寸信息
    const containerRect = e.currentTarget.getBoundingClientRect();
    
    // 计算图像的实际尺寸与显示尺寸的比例
    const scaleX = screenshotData.width / containerRect.width;
    const scaleY = screenshotData.height / containerRect.height;
    
    // 保存初始状态，避免在拖动过程中使用可能变化的状态
    const initialBox = JSON.parse(JSON.stringify(selectionBox));
    const startX = e.clientX;
    const startY = e.clientY;
    
    // 创建一个引用，用于存储最新的选择框状态
    // 这有助于避免React状态更新延迟导致的跳动
    const currentBoxRef = { ...initialBox };
    
    const handleMouseMove = (moveEvent) => {
      // 阻止默认行为，防止选择文本等
      moveEvent.preventDefault();
      
      // 计算鼠标移动的像素距离
      const deltaPixelX = moveEvent.clientX - startX;
      const deltaPixelY = moveEvent.clientY - startY;
      
      // 将像素距离转换为图像坐标系中的距离
      const deltaX = deltaPixelX * scaleX;
      const deltaY = deltaPixelY * scaleY;
      
      // 根据操作类型计算新的选择框
      let newBox;
      
      if (type === 'move') {
        // 移动操作 - 更新位置但保持大小不变
        const newX = Math.max(0, Math.min(screenshotData.width - initialBox.width, initialBox.x + deltaX));
        const newY = Math.max(0, Math.min(screenshotData.height - initialBox.height, initialBox.y + deltaY));
        newBox = { 
          ...initialBox, 
          x: newX, 
          y: newY 
        };
      } else if (type === 'resize') {
        // 调整大小操作 - 更新宽度和高度
        // 确保最小尺寸并且不超出图像边界
        const newWidth = Math.max(50, Math.min(screenshotData.width - initialBox.x, initialBox.width + deltaX));
        const newHeight = Math.max(50, Math.min(screenshotData.height - initialBox.y, initialBox.height + deltaY));
        newBox = { 
          ...initialBox, 
          width: newWidth, 
          height: newHeight 
        };
      }
      
      // 更新引用中的当前状态
      Object.assign(currentBoxRef, newBox);
      
      // 使用函数式更新确保我们总是基于最新状态进行更新
      setSelectionBox(newBox);
    };
    
    const handleMouseUp = () => {
      // 清理事件监听器
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mouseleave', handleMouseUp);
    };
    
    // 添加事件监听器到document而不是组件
    // 这样即使鼠标移出组件区域也能继续跟踪
    document.addEventListener('mousemove', handleMouseMove, { passive: false });
    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mouseleave', handleMouseUp);
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
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1>AI Solver (Hunyuan)</h1>
        <a 
          href="/docs/ai-solver-tutorial" 
          style={{
            padding: "8px 16px",
            backgroundColor: "#000000",
            color: "#ffffff",
            textDecoration: "none",
            borderRadius: "10px",
            fontSize: "14px",
            fontWeight: "500",
            border: "1px solid #000000",
            transition: "all 0.2s ease"
          }}
          onMouseOver={(e) => {
            e.target.style.backgroundColor = "#333333";
            e.target.style.borderColor = "#333333";
            e.target.style.transform = "translateY(-1px)";
          }}
          onMouseOut={(e) => {
            e.target.style.backgroundColor = "#000000";
            e.target.style.borderColor = "#000000";
            e.target.style.transform = "translateY(0)";
          }}
        >
           Tutorial
        </a>
      </div>
      <p>Supports camera capture, screen capture, and text questions. For security, the page does not accept any keys.</p>
      <p style={{ fontSize: 14, color: '#666' }}>Tip: In text mode, type <code>/preset name</code> to quickly switch presets, e.g. <code>/preset Math Problem Solver</code></p>

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

          {/* Camera Preview */}
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

          {/* Screenshot Mode */}
          {captureMode === 'screenshot' && (
            <div>
              {!isSelecting ? (
                // 初始状态 - 显示截图按钮
                <>
                  <div style={{ width: 320, height: 240, background: '#f5f5f5', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', border: '2px dashed #ccc' }}>
                    <div style={{ textAlign: 'center', color: '#666' }}>
                      <div style={{ fontSize: 48, marginBottom: 8 }}> </div>
                      <div>Click the button below to start screenshot</div>
                    </div>
                  </div>
                  <div style={{ marginTop: 8, color: '#666' }}>
                    Screenshot mode: You can select a specific area after capturing
                  </div>
                  <button onClick={handleScreenshot} disabled={busy} style={{ marginTop: 12, padding: '8px 16px', fontSize: 14 }}>
                    {busy ? 'Processing…' : ' Capture Screen'}
                  </button>
                </>
              ) : (
                // 选择状态 - 显示截图和选择框
                <>
                  <div style={{ position: 'relative', width: 320, height: 240, border: '2px solid #000000', borderRadius: 8, overflow: 'hidden' }}>
                    <img 
                      src={screenshotData?.dataUrl} 
                      alt="Screenshot" 
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      onMouseDown={(e) => handleSelectionDrag(e, 'move')}
                    />
                    {selectionBox && screenshotData && (
                      <div
                        style={{
                          position: 'absolute',
                          left: `${(selectionBox.x / screenshotData.width) * 100}%`,
                          top: `${(selectionBox.y / screenshotData.height) * 100}%`,
                          width: `${(selectionBox.width / screenshotData.width) * 100}%`,
                          height: `${(selectionBox.height / screenshotData.height) * 100}%`,
                          border: '2px solid #333333',
                          backgroundColor: 'rgba(0, 0, 0, 0.06)',
                          cursor: 'move',
                          boxSizing: 'border-box'
                        }}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          handleSelectionDrag(e, 'move');
                        }}
                      >
                        {/* 右下角调整大小的手柄 */}
                        <div
                          style={{
                            position: 'absolute',
                            right: -4,
                            bottom: -4,
                            width: 8,
                            height: 8,
                            backgroundColor: '#333333',
                            cursor: 'se-resize',
                            borderRadius: '50%'
                          }}
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            handleSelectionDrag(e, 'resize');
                          }}
                        />
                      </div>
                    )}
                  </div>
                  <div style={{ marginTop: 8, color: '#666', fontSize: 12 }}>
                     Selection box shows selected area. Drag to move, drag corner to resize.
                  </div>
                  <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
                    <button 
                      onClick={handleConfirmSelection} 
                      disabled={busy} 
                      style={{ padding: '8px 16px', fontSize: 14, backgroundColor: '#6c6c70', color: 'white', border: '1px solid #6c6c70', borderRadius: 4, cursor: 'pointer' }}
>
                      {busy ? 'Processing…' : 'Analyze Selected Area'}
                    </button>
                    <button 
                      onClick={handleCancelSelection} 
                      disabled={busy}
                      style={{ padding: '8px 16px', fontSize: 14, backgroundColor: '#6c6c70', color: 'white', border: '1px solid #6c6c70', borderRadius: 4, cursor: 'not-allowed' }}
                    >
                       Cancel
                    </button>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Text Question Mode */}
          {captureMode === 'text' && (
            <div>
              <div style={{ width: 320, minHeight: 240, background: '#f9f9f9', borderRadius: 8, padding: 16, border: '1px solid #ddd' }}>
                <div style={{ marginBottom: 12, color: '#666', fontSize: 14 }}> Text Question Mode</div>
                <textarea 
                  value={textInput}
                  onChange={e => setTextInput(e.target.value)}
                  placeholder="Please enter your question, for example:&#10;• Explain the basic principles of quantum mechanics&#10;• Write a Python sorting algorithm&#10;• Type /preset to see available presets"
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
          <fieldset style={{ border: '1px solid #e5e5ea', borderRadius: 8, padding: 12, marginBottom: 16, background: '#f5f5f7' }}>
            <legend>API Settings</legend>
            <label>API URL<br />
              <input value={apiUrl} onChange={e => setApiUrl(e.target.value)} placeholder="https://your-api.example.com/api/solve" style={{ width: '100%' }} />
            </label>
            <div style={{ color: '#345', fontSize: 12, marginTop: 8 }}>
              Default shows zeros. Enter your API endpoint to enable real calls.
            </div>
          </fieldset>

          <fieldset style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12 }}>
            <legend>Request Parameters</legend>
            <label>Model<br />
              <input value={model} onChange={e => setModel(e.target.value)} style={{ width: '100%' }} />
            </label>
            
            {/* Prompt预设选择 */}
            <div style={{ marginTop: 8 }}>
              <label>PromptPreset<br />
                <select 
                  value={selectedPreset} 
                  onChange={(e) => {
                    setSelectedPreset(e.target.value);
                    const preset = promptPresets.find(p => p.id === e.target.value);
                    if (preset && preset.id !== 'custom') {
                      setQuestion(preset.prompt);
                    }
                  }}
                  style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid #ccc' }}
                >
                  {promptPresets.map(preset => (
                    <option key={preset.id} value={preset.id}>{preset.name}</option>
                  ))}
                </select>
              </label>
              <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
                {promptPresets.find(p => p.id === selectedPreset)?.description || ''}
              </div>
            </div>
            
            <label style={{ display: 'block', marginTop: 8 }}>Question (for image mode)<br />
              <textarea 
                value={question} 
                onChange={e => {
                  setQuestion(e.target.value);
                  if (selectedPreset !== 'custom') {
                    setSelectedPreset('custom');
                  }
                }} 
                rows={3} 
                style={{ width: '100%' }} 
              />
            </label>
          </fieldset>
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <h3>Response</h3>
        <div style={{ 
          whiteSpace: 'pre-wrap', 
          background: '#f8f9fa', 
          color: '#212529', 
          padding: 16, 
          borderRadius: 8, 
          maxHeight: 400, 
          overflow: 'auto',
          border: '1px solid #dee2e6',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
          fontSize: '16px',
          lineHeight: 1.5
        }}>
          {respText}
        </div>
        <div style={{ marginTop: 8, textAlign: 'right' }}>
          <button 
            onClick={() => {
              if (window.lastFullResponse) {
                setRespText(JSON.stringify(window.lastFullResponse, null, 2));
              }
            }}
            style={{ 
              padding: '4px 8px', 
              fontSize: 12, 
              background: 'transparent', 
              border: '1px solid #ccc', 
              borderRadius: 4, 
              cursor: 'pointer',
              color: '#666'
            }}
          >
            Show Raw JSON
          </button>
        </div>
      </div>
    </div>
  );
}