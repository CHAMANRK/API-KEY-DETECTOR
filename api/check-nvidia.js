// api/check-nvidia.js
// Vercel Serverless Function — NVIDIA NIM CORS Proxy

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { apiKey } = req.body;

  if (!apiKey) {
    return res.status(400).json({ error: 'apiKey is required' });
  }

  try {
    const response = await fetch('https://integrate.api.nvidia.com/v1/models', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    // Forward exact status from NVIDIA
    return res.status(response.status).json(data);

  } catch (error) {
    return res.status(500).json({ error: 'Proxy error: ' + error.message });
  }
}
