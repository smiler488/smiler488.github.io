export const HARDCODED_API_ENDPOINT = 'https://api.hunyuan.cloud.tencent.com/v1/chat/completions';
export const HARDCODED_API_KEY = 'sk-JdwAvFcfyW5ngP2i3cpeB43QrR92gjnRcNzKkMfpcEVu8hlE';

export function computeDefaultApiEndpoint() {
  return HARDCODED_API_ENDPOINT;
}

export function buildHunyuanPayload(question, model) {
  const trimmed = (question || '').trim();
  return {
    model: model || 'hunyuan-lite',
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text:
              trimmed ||
              'You are a helpful assistant that generates ECharts option JSON based on CSV summaries.',
          },
        ],
      },
    ],
    stream: false,
  };
}

export function extractAssistantText(data) {
  if (Array.isArray(data?.choices) && data.choices.length > 0) {
    return data.choices[0].message?.content || '';
  }
  if (data?.Response && Array.isArray(data.Response.Choices) && data.Response.Choices.length > 0) {
    return data.Response.Choices[0].Message?.Content || '';
  }
  return typeof data === 'string' ? data : JSON.stringify(data, null, 2);
}

export async function postJson(url, json, extraHeaders = {}) {
  if (typeof url === 'string' && url.startsWith('mock://')) {
    const tag = url.slice('mock://'.length);
    if (tag === 'ai-data-visualizer') {
      const mockBody = {
        summary:
          'Sample analysis: sales grew steadily across quarters, with Q4 showing the highest revenue. Seasonal dips occur in Q2 due to inventory resets.',
        insights: [
          'Strong upward trend overall; expect ~12% YoY growth.',
          'Product beta overtook alpha in Q3 and maintained the lead.',
          'Recommend stacked area chart to communicate contribution per product.',
        ],
        chart_option: {
          title: { text: 'Mock Revenue Correlation Heatmap', left: 'center' },
          tooltip: { position: 'top' },
          grid: { left: '5%', right: '5%', top: '12%', bottom: '18%' },
          xAxis: { type: 'category', data: ['North', 'South', 'East', 'West'], splitArea: { show: true } },
          yAxis: { type: 'category', data: ['Q1', 'Q2', 'Q3', 'Q4'], splitArea: { show: true } },
          visualMap: { min: 60, max: 210, calculable: true, orient: 'horizontal', left: 'center', bottom: '4%' },
          series: [
            { name: 'Revenue', type: 'heatmap', label: { show: true }, data: [[0,0,120],[1,0,135],[2,0,150],[3,0,110],[0,1,98],[1,1,142],[2,1,160],[3,1,100],[0,2,160],[1,2,175],[2,2,182],[3,2,138],[0,3,190],[1,3,205],[2,3,210],[3,3,168]] }
          ],
        },
      };
      const payload = { choices: [{ message: { content: JSON.stringify(mockBody, null, 2) } }] };
      return { ok: true, status: 200, json: async () => payload, text: async () => JSON.stringify(payload), headers: new Map([['content-type', 'application/json']]) };
    }
    if (tag === 'journal-selector') {
      const mockBody = {
        overview: {
          abstract_summary: json?.question?.slice(0, 120) || 'N/A',
          alignment_summary: 'Mocked response (replace API URL for real results).',
        },
        journals: [
          {
            journal_name: 'Mock Journal of AI',
            publisher: 'Mock Press',
            discipline_scope: 'AI & Data Science',
            impact_factor_2024: '8.5',
            jcr_quartile: 'Q1',
            acceptance_rate: '18%',
            initial_review_weeks: 6,
            oa_type: 'Hybrid OA',
            apc_usd: 2400,
            submission_advice: 'Strong alignment with ML/DS topics.',
            warning_status: '-',
          },
        ],
        notes: 'This is a mock response generated locally.',
      };
      const payload = { choices: [{ message: { content: JSON.stringify(mockBody, null, 2) } }] };
      return { ok: true, status: 200, json: async () => payload, text: async () => JSON.stringify(payload), headers: new Map([['content-type', 'application/json']]) };
    }
    if (tag === 'ai-solver') {
      const now = new Date().toISOString();
      const q = json?.question || 'No question provided';
      const model = json?.model || 'hunyuan-lite';
      const hasImage = !!json?.imageBase64 || !!json?.imageUrl;
      const header = hasImage ? 'Mock Vision Analysis' : 'Mock Text Analysis';
      const content = `${header} (model: ${model})\n\nUser question:\n${q}\n\nThis is a mocked response for demo purposes. Replace proxy URL with your real API when ready.`;
      const payload = { Response: { RequestId: 'mock-' + Math.random().toString(36).slice(2), Choices: [{ Message: { Content: content } }], Usage: { PromptTokens: 128, CompletionTokens: 256, TotalTokens: 384 }, Timestamp: now } };
      return { ok: true, status: 200, json: async () => payload, text: async () => JSON.stringify(payload), headers: new Map([['content-type', 'application/json']]) };
    }
  }

  return fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', ...extraHeaders }, body: JSON.stringify(json) });
}

