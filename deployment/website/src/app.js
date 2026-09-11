/**
 * ChainGuard - Frontend Dashboard Controller
 * REST API Client, Animated Canvas Graph Engine, & Threat Intelligence UI
 */

const API_BASE_URL = window.CHAIN_GUARD_API_URL || 
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://localhost:8080' 
    : 'https://chainguard-model-server.onrender.com');

// Global Animation State
let animFrameId = null;
let animPhase = 0;

// Check REST API connection status on load
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  initCanvasGraph();
});

// Tab Switcher
function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

  const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(btn => 
    btn.getAttribute('onclick') && btn.getAttribute('onclick').includes(tabName)
  );
  if (activeBtn) activeBtn.classList.add('active');

  const targetTab = document.getElementById(`tab-${tabName}`);
  if (targetTab) targetTab.classList.add('active');

  // Trigger resize and re-render canvas when switching tabs
  setTimeout(() => {
    resizeCanvas();
  }, 60);
}

// API Health Check
async function checkApiHealth() {
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');

  try {
    const response = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (response.ok) {
      const data = await response.json();
      statusDot.style.backgroundColor = 'var(--accent-green)';
      statusDot.style.boxShadow = '0 0 14px var(--accent-green)';
      statusText.textContent = `REST API: ${data.status.toUpperCase()} (Port 8080)`;
    } else {
      throw new Error('API degraded');
    }
  } catch (err) {
    statusDot.style.backgroundColor = 'var(--text-secondary)';
    statusDot.style.boxShadow = 'none';
    statusText.textContent = 'REST API: Demo Evaluation Engine';
  }
}

// Cryptocurrency Address Format Validator
function isValidCryptoAddress(address, blockchain) {
  if (!address || typeof address !== 'string') return false;
  const addr = address.trim();
  if (addr.length < 24 || addr.length > 105 || addr.includes(' ')) return false;

  const chain = (blockchain || 'BTC').toUpperCase();
  if (chain === 'BTC') {
    return /^(1|3)[a-km-zA-HJ-NP-Z1-9]{25,34}$|^bc1[a-zA-Z0-9]{8,87}$/.test(addr);
  } else if (chain === 'ETH' || chain === 'USDT') {
    return /^0x[a-fA-F0-9]{40}$|^T[a-zA-Z0-9]{33}$/.test(addr);
  } else if (chain === 'SOL') {
    return /^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(addr);
  } else if (chain === 'LTC') {
    return /^(L|M)[a-km-zA-HJ-NP-Z1-9]{26,34}$|^ltc1[a-zA-Z0-9]{8,87}$/.test(addr);
  } else if (chain === 'XMR') {
    return /^[48][0-9a-zA-Z]{94}$/.test(addr);
  } else if (chain === 'XRP') {
    return /^r[0-9a-zA-Z]{24,34}$/.test(addr);
  }
  return /^[a-zA-Z0-9]{25,95}$/.test(addr);
}

function showValidationError(message) {
  const badgeContainer = document.getElementById('riskBadgeContainer');
  if (badgeContainer) {
    badgeContainer.innerHTML = `<span class="risk-badge" style="background: rgba(255, 68, 68, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red);">INVALID ADDRESS</span>`;
  }
  document.getElementById('probabilityVal').textContent = '0.00%';
  const probBar = document.getElementById('probabilityBar');
  if (probBar) {
    probBar.style.width = '0%';
    probBar.style.background = 'var(--accent-red)';
  }
  document.getElementById('familyVal').textContent = 'Validation Error: Invalid Input';
  document.getElementById('familyVal').style.color = 'var(--accent-red)';
  document.getElementById('familyConfidence').textContent = 'No Threat Calculations Executed';
  document.getElementById('riskScoreVal').textContent = '0.00';
  document.getElementById('evidenceList').innerText = `• ERROR: ${message}\n• Action Required: Please enter a valid cryptocurrency wallet address format for the selected blockchain.`;
}

// Preset Sample Loader
function loadSampleAddress() {
  document.getElementById('addressInput').value = '13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94';
  document.getElementById('incomeInput').value = '2500000000';
  document.getElementById('loopInput').value = '4';
  document.getElementById('countInput').value = '42';
  document.getElementById('chainSelect').value = 'BTC';
  document.getElementById('modelSelect').value = 'xgboost';
  
  // Submit scan automatically
  handleScanSubmit(new Event('submit'));
}

// Scan Submission Handler
async function handleScanSubmit(event) {
  if (event) event.preventDefault();

  const address = (document.getElementById('addressInput').value || '').trim();
  const blockchain = document.getElementById('chainSelect').value || 'BTC';
  const targetModel = document.getElementById('modelSelect').value || 'xgboost';
  const income = parseFloat(document.getElementById('incomeInput').value) || 100000000;
  const loop = parseInt(document.getElementById('loopInput').value) || 0;
  const count = parseInt(document.getElementById('countInput').value) || 5;

  if (!isValidCryptoAddress(address, blockchain)) {
    showValidationError(`"${address || 'Input'}" is not a valid ${blockchain} wallet address format.`);
    return;
  }

  const payload = {
    address: address,
    blockchain: blockchain,
    target_model: targetModel,
    income: income,
    loop: loop,
    count: count,
    length: 10,
    weight: 0.5,
    neighbors: 8
  };

  try {
    const response = await fetch(`${API_BASE_URL}/predict/risk`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (response.ok) {
      const result = await response.json();
      if (result.status === 'error') {
        showValidationError(result.message);
      } else {
        updateScanUI(result);
      }
    } else {
      const errData = await response.json().catch(() => ({}));
      if (response.status === 400 && errData.message) {
        showValidationError(errData.message);
        return;
      }
      throw new Error('Fallback to local evaluation engine');
    }
  } catch (err) {
    // Offline simulation mode for instant client-side testing
    const simulated = simulateThreatAssessment(payload);
    if (simulated.status === 'error') {
      showValidationError(simulated.message);
    } else {
      updateScanUI(simulated);
    }
  }
}

// Simulated Engine for Client-Side Demo
function simulateThreatAssessment(payload) {
  const addressStr = (payload.address || '').trim();
  if (!isValidCryptoAddress(addressStr, payload.blockchain)) {
    return {
      status: 'error',
      message: `Invalid cryptocurrency wallet address format for ${payload.blockchain}.`
    };
  }

  const isKnownRansomware = ['13AM4', '111K8', '12t9Y', '132F2', '14E15'].some(sub => addressStr.includes(sub));

  const incomeFactor = Math.min(1.0, payload.income / 2e9);
  const loopFactor = Math.min(1.0, payload.loop / 10.0);
  const countFactor = Math.min(1.0, payload.count / 100.0);

  let prob = Math.min(0.99, Math.max(0.025, (loopFactor * 0.35) + (incomeFactor * 0.2) + (countFactor * 0.15)));
  if (isKnownRansomware) prob = Math.max(prob, 0.948);

  const isRansomware = prob >= 0.5;
  let modelName = 'XGBoost Dual-Head Classifier';
  let family = isKnownRansomware ? 'Locky Ransomware' : (isRansomware ? 'Suspicious High-Risk Obfuscation Cluster' : 'Licit Wallet Address');

  if (payload.target_model === 'isolation_forest') {
    modelName = 'Isolation Forest Anomaly Engine';
    family = isRansomware ? 'Behavioral Drift Outlier' : 'Normal Behavioral Profile';
  } else if (payload.target_model === 'graphsage') {
    modelName = 'PyTorch GraphSAGE GNN';
    family = isRansomware ? 'Illicit Graph Node' : 'Licit Transaction Node';
  }

  let riskLevel = 'MINIMAL';
  if (prob >= 0.75) riskLevel = 'HIGH';
  else if (prob >= 0.45) riskLevel = 'MEDIUM';
  else if (prob >= 0.20) riskLevel = 'LOW';

  const evidence = [
    `Target Feed: ${payload.blockchain} | Engine: ${modelName}`,
    `Ransomware probability score: ${(prob * 100).toFixed(1)}% (${riskLevel} RISK)`,
    `Evaluated inputs: Income=${payload.income.toLocaleString()} Sats, Loop Count=${payload.loop}, Tx Count=${payload.count}`
  ];

  if (payload.loop > 2) {
    evidence.push(`Obfuscation Pattern: Peel-chain recursion across ${payload.loop} linear hops`);
  }
  if (payload.count > 30) {
    evidence.push(`Dispersion Pattern: High-velocity fan-out consolidation (${payload.count} transactions)`);
  }
  if (isRansomware) {
    evidence.push(`Forensic Match: Signature associated with ${family}`);
  } else {
    evidence.push('Clean Status: No threat signatures or anomalous graph drifts identified');
  }

  return {
    status: 'success',
    risk_assessment: {
      address: payload.address,
      composite_risk_score: prob,
      risk_level: riskLevel,
      target_model_used: modelName,
      blockchain_target: payload.blockchain,
      ransomware_prediction: {
        is_ransomware: isRansomware,
        probability: prob,
        family: family,
        confidence: isRansomware ? 0.962 : 0.998
      },
      evidence_chain: evidence,
      peel_chain_detected: Boolean(payload.loop > 2),
      fan_out_detected: Boolean(payload.count > 30),
      count: payload.count
    }
  };
}

function getExplorerUrl(chain, address) {
  const addr = encodeURIComponent((address || '').trim());
  switch ((chain || 'BTC').toUpperCase()) {
    case 'BTC':
      return addr ? `https://mempool.space/address/${addr}` : 'https://mempool.space';
    case 'USDT':
    case 'ETH':
      return addr ? `https://etherscan.io/address/${addr}` : 'https://etherscan.io';
    case 'LTC':
      return addr ? `https://litecoinspace.org/address/${addr}` : 'https://litecoinspace.org';
    case 'SOL':
      return addr ? `https://solscan.io/account/${addr}` : 'https://solscan.io';
    case 'XMR':
      return addr ? `https://xmrchain.net/search?value=${addr}` : 'https://xmrchain.net';
    case 'XRP':
      return addr ? `https://xrpscan.com/account/${addr}` : 'https://xrpscan.com';
    default:
      return addr ? `https://mempool.space/address/${addr}` : 'https://mempool.space';
  }
}

// UI Updater
function updateScanUI(result) {
  const risk = result.risk_assessment;
  const prob = (risk.ransomware_prediction.probability * 100).toFixed(1);
  const isHigh = risk.risk_level === 'HIGH' || risk.risk_level === 'MEDIUM';

  // Explorer Link Button Update
  const explorerBtn = document.getElementById('explorerBtn');
  if (explorerBtn) {
    const chainTarget = risk.blockchain_target || document.getElementById('chainSelect').value || 'BTC';
    explorerBtn.href = getExplorerUrl(chainTarget, risk.address);
    explorerBtn.title = `Inspect ${risk.address || 'Target'} on ${chainTarget} Explorer`;
  }

  // Probability
  document.getElementById('probabilityVal').textContent = `${prob}%`;
  const probBar = document.getElementById('probabilityBar');
  probBar.style.width = `${prob}%`;
  probBar.style.background = isHigh ? 'var(--accent-red)' : 'var(--accent-green)';

  // Family & Confidence
  document.getElementById('familyVal').textContent = risk.ransomware_prediction.family;
  document.getElementById('familyVal').style.color = isHigh ? 'var(--accent-red)' : 'var(--text-primary)';
  document.getElementById('familyConfidence').textContent = `Confidence: ${(risk.ransomware_prediction.confidence * 100).toFixed(1)}%`;

  // Risk Score
  const scoreVal = document.getElementById('riskScoreVal');
  scoreVal.textContent = risk.composite_risk_score.toFixed(2);
  scoreVal.style.color = isHigh ? 'var(--accent-red)' : 'var(--accent-green)';

  // Badge
  const badgeContainer = document.getElementById('riskBadgeContainer');
  badgeContainer.innerHTML = `<span class="risk-badge risk-${risk.risk_level.toLowerCase()}">${risk.risk_level} RISK</span>`;

  // Evidence List
  const evidenceList = document.getElementById('evidenceList');
  evidenceList.innerHTML = risk.evidence_chain.map(item => `• ${item}`).join('\n');

  // Full Report Container update
  document.getElementById('reportTierVal').textContent = risk.risk_level;
  document.getElementById('reportTierVal').style.color = isHigh ? 'var(--accent-red)' : 'var(--accent-green)';
  document.getElementById('fullReportContainer').textContent = JSON.stringify(result, null, 2);

  // Dynamic Graph Visualizer Update
  let patternType = 'peel';
  if (risk.fan_out_detected || (risk.count && risk.count > 30)) {
    patternType = 'fanout';
  } else if (!risk.peel_chain_detected && risk.risk_level === 'LOW') {
    patternType = 'fanin';
  }

  renderGraphPattern(patternType, risk.address, risk.risk_level, risk.ransomware_prediction.family);
}

// Canvas Graph Visualizer State & Loop Engine
let currentGraphAddress = '13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94';
let currentRiskLevel = 'HIGH';
let currentFamily = 'Locky Ransomware';
let currentPatternType = 'peel';

function initCanvasGraph() {
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  window.addEventListener('load', resizeCanvas);
  setTimeout(resizeCanvas, 200);
  setTimeout(resizeCanvas, 800);

  startAnimationLoop();
}

function startAnimationLoop() {
  if (animFrameId) cancelAnimationFrame(animFrameId);
  function loop() {
    animPhase = (animPhase + 0.008) % 1.0;
    renderGraphPattern(currentPatternType);
    animFrameId = requestAnimationFrame(loop);
  }
  animFrameId = requestAnimationFrame(loop);
}

function resizeCanvas() {
  const canvases = [
    document.getElementById('scannerGraphCanvas'),
    document.getElementById('graphCanvas')
  ];

  canvases.forEach(canvas => {
    if (!canvas) return;
    const parent = canvas.parentElement;
    let w = parent ? parent.clientWidth : 0;
    if (w <= 0) {
      w = parent?.parentElement ? parent.parentElement.clientWidth : 800;
    }
    canvas.width = Math.max(w, 400);
    canvas.height = canvas.id === 'scannerGraphCanvas' ? 260 : 380;
  });

  renderGraphPattern(currentPatternType);
}

function renderGraphPattern(type = 'peel', address = null, riskLevel = null, family = null) {
  if (address) currentGraphAddress = address;
  if (riskLevel) currentRiskLevel = riskLevel;
  if (family) currentFamily = family;
  if (type) currentPatternType = type;

  const canvases = [
    document.getElementById('scannerGraphCanvas'),
    document.getElementById('graphCanvas')
  ];

  canvases.forEach(canvas => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = canvas.width;
    let height = canvas.height;

    if (width <= 0 || height <= 0) {
      const parent = canvas.parentElement;
      let w = parent ? parent.clientWidth : 0;
      if (w <= 0) w = 800;
      canvas.width = Math.max(w, 400);
      canvas.height = canvas.id === 'scannerGraphCanvas' ? 260 : 380;
      width = canvas.width;
      height = canvas.height;
    }

    // Ultra-Dark Cyber Slate background fill
    ctx.fillStyle = '#03050a';
    ctx.fillRect(0, 0, width, height);

    // Subtle grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const centerY = height / 2;
    const addrStr = currentGraphAddress || 'Target Node';
    const displayAddr = addrStr.length > 14 
      ? `${addrStr.substring(0, 6)}...${addrStr.substring(addrStr.length - 6)}` 
      : addrStr;

    const isHighRisk = currentRiskLevel === 'HIGH' || currentRiskLevel === 'MEDIUM';
    const sinkColor = isHighRisk ? '#ff0055' : '#00f5a0';
    const sinkLabel = isHighRisk ? `Sink (${currentFamily})` : 'Clean Wallet Output';

    let nodes = [];
    let edges = [];

    if (currentPatternType === 'peel') {
      nodes = [
        { x: width * 0.12, y: centerY, label: `Origin: ${displayAddr}`, color: '#ffffff' },
        { x: width * 0.32, y: centerY - 35, label: 'Peel Hop 1', color: '#94a3b8' },
        { x: width * 0.32, y: centerY + 45, label: 'Change UTXO', color: '#00f5a0' },
        { x: width * 0.52, y: centerY - 35, label: 'Peel Hop 2', color: '#94a3b8' },
        { x: width * 0.52, y: centerY + 45, label: 'Mixing Pool', color: '#64748b' },
        { x: width * 0.72, y: centerY - 35, label: 'Peel Hop 3', color: '#94a3b8' },
        { x: width * 0.88, y: centerY, label: sinkLabel, color: sinkColor },
      ];
      edges = [[0, 1], [0, 2], [1, 3], [1, 4], [3, 5], [5, 6]];
    } else if (currentPatternType === 'fanout') {
      nodes = [
        { x: width * 0.15, y: centerY, label: `Source: ${displayAddr}`, color: '#ffffff' },
        { x: width * 0.60, y: centerY - 90, label: 'Dispersion Output 1', color: '#94a3b8' },
        { x: width * 0.60, y: centerY - 30, label: 'Dispersion Output 2', color: '#94a3b8' },
        { x: width * 0.60, y: centerY + 30, label: 'Dispersion Output 3', color: '#94a3b8' },
        { x: width * 0.60, y: centerY + 90, label: sinkLabel, color: sinkColor },
      ];
      edges = [[0, 1], [0, 2], [0, 3], [0, 4]];
    } else {
      nodes = [
        { x: width * 0.18, y: centerY - 75, label: 'Input Address A', color: '#00f5a0' },
        { x: width * 0.18, y: centerY, label: 'Input Address B', color: '#00f5a0' },
        { x: width * 0.18, y: centerY + 75, label: 'Input Address C', color: '#00f5a0' },
        { x: width * 0.75, y: centerY, label: `Target: ${displayAddr}`, color: sinkColor },
      ];
      edges = [[0, 3], [1, 3], [2, 3]];
    }

    drawGraph(ctx, nodes, edges);
  });
}

function drawGraph(ctx, nodes, edges) {
  // Draw static edge connecting lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
  ctx.lineWidth = 2;
  edges.forEach(([from, to]) => {
    ctx.beginPath();
    ctx.moveTo(nodes[from].x, nodes[from].y);
    ctx.lineTo(nodes[to].x, nodes[to].y);
    ctx.stroke();
  });

  // Draw animated flowing energy particles along edges
  edges.forEach(([from, to], index) => {
    const startNode = nodes[from];
    const endNode = nodes[to];
    
    // Offset phase per edge
    const phase = (animPhase + index * 0.25) % 1.0;
    const pX = startNode.x + (endNode.x - startNode.x) * phase;
    const pY = startNode.y + (endNode.y - startNode.y) * phase;

    ctx.beginPath();
    ctx.arc(pX, pY, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = '#ffffff';
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
  });

  // Draw nodes with animated sonar aura rings
  const time = Date.now() / 300;
  nodes.forEach(node => {
    // Dynamic sonar ring pulse
    const ringRadius = 16 + Math.sin(time + node.x) * 4;
    ctx.beginPath();
    ctx.arc(node.x, node.y, ringRadius + 4, 0, Math.PI * 2);
    ctx.strokeStyle = node.color;
    ctx.lineWidth = 1.2;
    ctx.globalAlpha = 0.5;
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Node Main Body
    ctx.beginPath();
    ctx.arc(node.x, node.y, 16, 0, Math.PI * 2);
    ctx.fillStyle = node.color;
    ctx.shadowColor = node.color;
    ctx.shadowBlur = 18;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Node outer border ring
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Node Text Label
    ctx.fillStyle = '#ffffff';
    ctx.font = '700 11px Outfit, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(node.label, node.x, node.y + 30);
  });
}

function generateReportJSON() {
  const content = document.getElementById('fullReportContainer').textContent;
  const blob = new Blob([content], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'ChainGuard_Forensic_Evidence_Report.json';
  a.click();
}
