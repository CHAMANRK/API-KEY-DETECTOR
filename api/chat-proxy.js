// api/chat-proxy.js
// Universal Chat Proxy — all AI services ek jagah se
// Fixed: body parsing + auth header issue

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  // ✅ FIX: body string ho to parse karo
  let body = req.body;
  if (typeof body === 'string') {
    try { body = JSON.parse(body); } catch (e) {
      return res.status(400).json({ error: 'Invalid JSON body' });
    }
  }
  if (!body || typeof body !== 'object') {
    return res.status(400).json({ error: 'Empty or invalid body' });
  }

  const { service, apiKey, model, messages } = body;

  // ✅ FIX: missing field check with detail
  if (!service) return res.status(400).json({ error: 'Missing: service' });
  if (!apiKey)  return res.status(400).json({ error: 'Missing: apiKey — key was not passed to proxy' });
  if (!model)   return res.status(400).json({ error: 'Missing: model' });
  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'Missing or empty: messages' });
  }

  try {
    switch (service) {
      case 'openai':     return await handleOpenAI(apiKey, model, messages, res);
      case 'groq':       return await handleGroq(apiKey, model, messages, res);
      case 'anthropic':  return await handleAnthropic(apiKey, model, messages, res);
      case 'gemini':     return await handleGemini(apiKey, model, messages, res);
      case 'openrouter': return await handleOpenRouter(apiKey, model, messages, res);
      case 'nvidia':     return await handleNvidia(apiKey, model, messages, res);
      case 'mistral':    return await handleMistral(apiKey, model, messages, res);
      default:           return res.status(400).json({ error: 'Unknown service: ' + service });
    }
  } catch (e) {
    return res.status(500).json({ error: 'Proxy error: ' + e.message });
  }
}

// ── OpenAI ────────────────────────────────────────────────────────────
async function handleOpenAI(apiKey, model, messages, res) {
  const r = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer ' + apiKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ model, messages, max_tokens: 512 })
  });
  const data = await r.json();
  if (!r.ok) return res.status(r.status).json({ error: data.error?.message || 'OpenAI error' });
  return res.json({
    content: data.choices?.[0]?.message?.content || '',
    tokens: data.usage?.total_tokens || null
  });
}

// ── Groq ──────────────────────────────────────────────────────────────
async function handleGroq(apiKey, model, messages, res) {
  const r = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer ' + apiKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ model, messages, max_tokens: 512 })
  });
  const data = await r.json();
  if (!r.ok) return res.status(r.status).json({ error: data.error?.message || 'Groq error' });
  return res.json({
    content: data.choices?.[0]?.message?.content || '',
    tokens: data.usage?.total_tokens || null
  });
}

// ── Anthropic ─────────────────────────────────────────────────────────
async function handleAnthropic(apiKey, model, messages, res) {
  const systemMsg = messages.find(m => m.role === 'system');
  const chatMsgs  = messages.filter(m => m.role !== 'system');

  const bodyObj = { model, max_tokens: 512, messages: chatMsgs };
  if (systemMsg) bodyObj.system = systemMsg.content;

  const r = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(bodyObj)
  });
  const data = await r.json();
  if (!r.ok) return res.status(r.status).json({ error: data.error?.message || 'Anthropic error' });
  return res.json({
    content: data.content?.[0]?.text || '',
    tokens: data.usage ? (data.usage.input_tokens + data.usage.output_tokens) : null
  });
}

// ── Gemini ────────────────────────────────────────────────────────────
async function handleGemini(apiKey, model, messages, res) {
  const contents = messages
    .filter(m => m.role !== 'system')
    .map(m => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }]
    }));

  const r = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contents, generationConfig: { maxOutputTokens: 512 } })
    }
  );
  const data = await r.json();
  if (!r.ok) return res.status(r.status).json({ error: data.error?.message || 'Gemini error' });
  const text   = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
  const tokens = data.usageMetadata
    ? (data.usageMetadata.promptTokenCount || 0) + (data.usageMetadata.candidatesTokenCount || 0)
    : null;
  return res.json({ content: text, tokens });
}

// ── OpenRouter ────────────────────────────────────────────────────────
async function handleOpenRouter(apiKey, model, messages, res) {
  const r = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer ' + apiKey,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://api-key-detective.vercel.app',
      'X-Title': 'API Key Detective'
    },
    body: JSON.stringify({ model, messages, max_tokens: 512 })
  });
  const data = await r.json();
  if (!r.ok) return res.status(r.status).json({ error: data.error?.message || 'OpenRouter error' });
  return res.json({
    content: data.choices?.[0]?.message?.content || '',
    tokens: data.usage?.total_tokens || null
  });
}

// ── NVIDIA NIM ────────────────────────────────────────────────────────
async function handleNvidia(apiKey, model, messages, res) {
  const r = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer ' + apiKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ model, messages, max_tokens: 512 })
  });
  const data = await r.json();
  if (!r.ok) return res.status(r.status).json({ error: data.error?.message || 'NVIDIA error' });
  return res.json({
    content: data.choices?.[0]?.message?.content || '',
    tokens: data.usage?.total_tokens || null
  });
}

// ── Mistral ───────────────────────────────────────────────────────────
async function handleMistral(apiKey, model, messages, res) {
  const r = await fetch('https://api.mistral.ai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer ' + apiKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ model, messages, max_tokens: 512 })
  });
  const data = await r.json();
  if (!r.ok) return res.status(r.status).json({ error: data.error?.message || 'Mistral error' });
  return res.json({
    content: data.choices?.[0]?.message?.content || '',
    tokens: data.usage?.total_tokens || null
  });
}
