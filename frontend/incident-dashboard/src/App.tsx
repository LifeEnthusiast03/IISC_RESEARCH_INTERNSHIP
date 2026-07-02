import './App.css'
import { useWebSocket } from './hooks/useWebSocket'
import { WsStatusBar }   from './components/WsStatusBar'
import { WsMessageFeed } from './components/WsMessageFeed'

// ── Mock data (replace with real WebSocket feed in later stages) ──────────
const STATS = [
  { label: 'Flows Analysed',    value: '1,188,543', unit: 'total',      color: 'var(--col-cyan)',  icon: '⚡' },
  { label: 'Threats Detected',  value: '0',          unit: 'active',     color: 'var(--col-red)',   icon: '🛡' },
  { label: 'Autoencoder Loss',  value: '—',          unit: 'val MSE',    color: 'var(--col-amber)', icon: '🧠' },
  { label: 'Threshold',         value: '—',          unit: 'p95 error',  color: 'var(--col-green)', icon: '📊' },
]

const PIPELINE_STAGES = [
  {
    step: '01',
    title: 'Data Ingestion',
    desc: 'CICIDS2017 benign flows preprocessed → MinMax-scaled 115-feature vectors',
    status: 'complete',
    file: 'preprocess_benign.py',
  },
  {
    step: '02',
    title: 'Autoencoder Training',
    desc: '115→64→32→16→32→64→115 FC network trained on benign traffic only',
    status: 'pending',
    file: 'train_autoencoder.py',
  },
  {
    step: '03',
    title: 'Threshold Calibration',
    desc: '95th-percentile reconstruction error on X_val_benign → anomaly threshold',
    status: 'pending',
    file: 'models/threshold.json',
  },
  {
    step: '04',
    title: 'DQN Agent Training',
    desc: 'Deep Q-Network learns optimal remediation policy in Gymnasium environment',
    status: 'pending',
    file: 'train_dqn.py',
  },
  {
    step: '05',
    title: 'Live Inference',
    desc: 'FastAPI → Autoencoder → DQN → PostgreSQL → WebSocket → This dashboard',
    status: 'pending',
    file: 'backend/inference.py',
  },
]

const RECENT_ALERTS = [
  { time: '--:--:--', src: '0.0.0.0',        type: 'System',   action: 'Awaiting model',  severity: 'info'     },
  { time: '--:--:--', src: '0.0.0.0',        type: 'System',   action: 'Awaiting model',  severity: 'info'     },
  { time: '--:--:--', src: '0.0.0.0',        type: 'System',   action: 'Awaiting model',  severity: 'info'     },
]

// ── Status dot component ────────────────────────────────────────────────────
function StatusDot({ status }: { status: string }) {
  if (status === 'complete') {
    return (
      <span
        className="inline-block w-2 h-2 rounded-full"
        style={{ background: 'var(--col-green)', boxShadow: '0 0 6px var(--col-green)' }}
      />
    )
  }
  if (status === 'running') {
    return (
      <span className="relative inline-flex">
        <span
          className="pulse-ring absolute inline-block w-2 h-2 rounded-full opacity-75"
          style={{ background: 'var(--col-cyan)' }}
        />
        <span
          className="relative inline-block w-2 h-2 rounded-full"
          style={{ background: 'var(--col-cyan)' }}
        />
      </span>
    )
  }
  return (
    <span
      className="inline-block w-2 h-2 rounded-full"
      style={{ background: 'rgba(148,163,184,0.3)', border: '1px solid rgba(148,163,184,0.4)' }}
    />
  )
}

// ── Severity badge ──────────────────────────────────────────────────────────
function Badge({ severity }: { severity: string }) {
  const cls: Record<string, string> = {
    critical: 'badge badge-critical',
    warning:  'badge badge-warning',
    info:     'badge badge-info',
    safe:     'badge badge-safe',
  }
  return <span className={cls[severity] ?? 'badge badge-info'}>{severity}</span>
}

// ── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const { status, messages, clearMessages } = useWebSocket()

  return (
    <div className="relative min-h-screen" style={{ background: 'var(--col-bg)' }}>

      {/* -- WebSocket status bar (fixed top-right) ----------------------- */}
      <div className="fixed top-4 right-4 z-50 w-auto max-w-sm">
        <WsStatusBar status={status} messageCount={messages.length} />
      </div>

      {/* Ambient background */}
      <div className="grid-bg" />
      <div
        className="orb"
        style={{
          width: 600, height: 600,
          top: -200, left: -150,
          background: 'radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%)',
        }}
      />
      <div
        className="orb"
        style={{
          width: 500, height: 500,
          bottom: -100, right: -100,
          background: 'radial-gradient(circle, rgba(248,113,113,0.05) 0%, transparent 70%)',
        }}
      />

      {/* ── NAV BAR ──────────────────────────────────────────────────── */}
      <header
        className="relative z-10 flex items-center justify-between px-6 py-4 border-b"
        style={{ borderColor: 'var(--col-border)', background: 'rgba(13,17,23,0.8)', backdropFilter: 'blur(12px)' }}
      >
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div
            className="flex items-center justify-center w-8 h-8 rounded-lg text-sm font-bold"
            style={{ background: 'var(--col-cyan-dim)', border: '1px solid rgba(56,189,248,0.3)', color: 'var(--col-cyan)' }}
          >
            TS
          </div>
          <div>
            <p className="text-sm font-semibold leading-none" style={{ color: 'var(--col-text-hi)' }}>
              ThreatSentinel
            </p>
            <p className="text-xs mt-0.5" style={{ color: 'var(--col-text)', fontFamily: 'var(--font-mono)' }}>
              IDS Dashboard v0.1
            </p>
          </div>
        </div>

        {/* Live status pill */}
        <div
          className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium"
          style={{
            background: 'rgba(248,113,113,0.08)',
            border: '1px solid rgba(248,113,113,0.2)',
            color: 'var(--col-red)',
            fontFamily: 'var(--font-mono)',
          }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: 'var(--col-red)' }}
          />
          MODEL NOT TRAINED
        </div>

        {/* Right actions */}
        <nav className="flex items-center gap-4">
          {['Dashboard', 'Incidents', 'Analytics', 'Settings'].map((item) => (
            <button
              key={item}
              className="text-sm transition-colors duration-150 hover:text-white cursor-not-allowed"
              style={{ color: 'var(--col-text)' }}
              disabled
            >
              {item}
            </button>
          ))}
        </nav>
      </header>

      {/* ── MAIN CONTENT ─────────────────────────────────────────────── */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8 space-y-8">

        {/* ── Hero title ───────────────────────────────────────────── */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <span
              className="mono text-xs font-medium px-2 py-0.5 rounded"
              style={{ background: 'var(--col-cyan-dim)', color: 'var(--col-cyan)' }}
            >
              STAGE 1 IN PROGRESS
            </span>
            <span className="mono text-xs" style={{ color: 'var(--col-text)' }}>
              // Real-Time Incident Tracking & Autonomous Threat Remediation
            </span>
          </div>
          <h1
            className="text-3xl font-bold tracking-tight"
            style={{ color: 'var(--col-text-hi)' }}
          >
            Incident Command Center
          </h1>
          <p className="mt-1 text-sm" style={{ color: 'var(--col-text)' }}>
            Two-stage pipeline · Autoencoder anomaly detection + DQN autonomous remediation · CICIDS2017
          </p>
        </div>

        {/* ── Stat cards ───────────────────────────────────────────── */}
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          {STATS.map((s) => (
            <div
              key={s.label}
              className="stat-card rounded-xl p-4 border"
              style={{
                background: 'var(--col-surface)',
                borderColor: 'var(--col-border)',
              }}
            >
              <div className="flex items-start justify-between">
                <span className="text-lg">{s.icon}</span>
                <span
                  className="mono text-xs px-1.5 py-0.5 rounded"
                  style={{ background: 'rgba(255,255,255,0.04)', color: 'var(--col-text)' }}
                >
                  {s.unit}
                </span>
              </div>
              <p
                className="mt-3 text-2xl font-bold tracking-tight mono"
                style={{ color: s.color }}
              >
                {s.value}
              </p>
              <p className="mt-1 text-xs" style={{ color: 'var(--col-text)' }}>
                {s.label}
              </p>
            </div>
          ))}
        </div>

        {/* ── Two-column: Pipeline + Alerts ────────────────────────── */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">

          {/* Pipeline stages */}
          <div
            className="rounded-xl border p-5"
            style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold" style={{ color: 'var(--col-text-hi)' }}>
                Training Pipeline
              </h2>
              <span className="badge badge-info">2 / 5 complete</span>
            </div>
            <div className="space-y-3">
              {PIPELINE_STAGES.map((stage) => (
                <div
                  key={stage.step}
                  className="flex items-start gap-3 p-3 rounded-lg transition-colors duration-150"
                  style={{
                    background: stage.status === 'complete'
                      ? 'var(--col-green-dim)'
                      : 'rgba(255,255,255,0.02)',
                    border: '1px solid',
                    borderColor: stage.status === 'complete'
                      ? 'rgba(52,211,153,0.2)'
                      : 'var(--col-border)',
                  }}
                >
                  <StatusDot status={stage.status} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="mono text-xs" style={{ color: 'var(--col-text)' }}>
                        {stage.step}
                      </span>
                      <span className="text-sm font-medium" style={{ color: 'var(--col-text-hi)' }}>
                        {stage.title}
                      </span>
                    </div>
                    <p className="mt-0.5 text-xs leading-relaxed" style={{ color: 'var(--col-text)' }}>
                      {stage.desc}
                    </p>
                    <p className="mt-1 mono text-xs" style={{ color: 'rgba(148,163,184,0.5)' }}>
                      → {stage.file}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Alert feed */}
          <div
            className="rounded-xl border p-5"
            style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold" style={{ color: 'var(--col-text-hi)' }}>
                Live Threat Feed
              </h2>
              <div className="flex items-center gap-2">
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ background: 'rgba(148,163,184,0.3)' }}
                />
                <span className="mono text-xs" style={{ color: 'var(--col-text)' }}>
                  OFFLINE
                </span>
              </div>
            </div>

            {/* Table header */}
            <div
              className="grid gap-2 px-2 py-1.5 rounded-lg mb-2 mono text-xs font-semibold"
              style={{
                background: 'rgba(255,255,255,0.03)',
                color: 'var(--col-text-hi)',
                gridTemplateColumns: '80px 2fr 1.5fr 1.5fr 80px',
              }}
            >
              <span>TIME</span>
              <span>CONNECTION</span>
              <span>ATTACK</span>
              <span>ACTION</span>
              <span>SEV</span>
            </div>

            {/* Rows */}
            <div className="space-y-1 overflow-y-auto" style={{ maxHeight: '400px', paddingRight: '4px' }}>
              {messages.slice().reverse().map((msg) => {
                const action = msg.data?.dqn_action || msg.data?.action || '-';
                const severity = action === 'Block IP' ? 'critical' : action === 'Isolate' ? 'warning' : 'info';
                
                return (
                <div
                  key={msg.id}
                  className="grid gap-2 px-2 py-2 rounded-lg text-xs transition-colors items-center"
                  style={{
                    background: 'rgba(255,255,255,0.01)',
                    border: '1px solid var(--col-border)',
                    gridTemplateColumns: '80px 2fr 1.5fr 1.5fr 80px',
                    color: 'var(--col-text)',
                  }}
                >
                  <span className="mono">{msg.receivedAt.toLocaleTimeString([], { hour12: false })}</span>
                  <span className="mono truncate leading-tight" title={`${msg.data?.source_ip} -> ${msg.data?.dest_ip}`}>
                    {msg.data?.source_ip || 'System'}
                    {msg.data?.dest_ip && <span className="opacity-40 block text-[10px]">&rarr; {msg.data.dest_ip}</span>}
                  </span>
                  <span className="truncate font-medium" title={msg.data?.attack_type || msg.data?.message} style={{ color: 'var(--col-amber)' }}>
                    {msg.data?.attack_type || msg.data?.message || 'Info'}
                  </span>
                  <span className="truncate leading-tight" title={`Action: ${action}`}>
                    {action}
                    {msg.data?.recon_error !== undefined && <span className="text-[10px] opacity-50 block">Err: {msg.data.recon_error}</span>}
                  </span>
                  <Badge severity={severity} />
                </div>
              )})}
              {messages.length === 0 && (
                <div className="text-center p-4 text-xs" style={{ color: 'var(--col-text)' }}>
                  Listening for threats...
                </div>
              )}
            </div>

            {/* Waiting banner */}
            <div
              className="mt-4 rounded-lg p-4 text-center"
              style={{
                background: 'rgba(56,189,248,0.04)',
                border: '1px dashed rgba(56,189,248,0.2)',
              }}
            >
              <p className="mono text-xs" style={{ color: 'var(--col-cyan)' }}>
                {'>'} Awaiting autoencoder training
                <span className="blink ml-0.5">_</span>
              </p>
              <p className="text-xs mt-1" style={{ color: 'var(--col-text)' }}>
                Run <code className="mono px-1" style={{ background: 'rgba(255,255,255,0.06)', borderRadius: 3 }}>
                  python training/train_autoencoder.py
                </code> to enable live threat detection.
              </p>
            </div>

            {/* WebSocket status */}
            <div className="glow-divider my-4" />
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ background: 'rgba(148,163,184,0.3)' }}
                />
                <span className="mono text-xs" style={{ color: 'var(--col-text)' }}>
                  ws://localhost:8000/ws
                </span>
              </div>
              <span className="mono text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>
                NOT CONNECTED
              </span>
            </div>
          </div>
        </div>

        {/* ── Architecture quick-ref ────────────────────────────────── */}
        <div
          className="rounded-xl border p-5"
          style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
        >
          <h2 className="text-sm font-semibold mb-4" style={{ color: 'var(--col-text-hi)' }}>
            System Architecture
          </h2>
          <div className="flex flex-wrap items-center gap-2 mono text-xs">
            {[
              { label: 'Network Flow',    color: 'var(--col-text)' },
              { label: '→',              color: 'var(--col-border)', arrow: true },
              { label: 'Autoencoder',     color: 'var(--col-cyan)'  },
              { label: '→',              color: 'var(--col-border)', arrow: true },
              { label: 'Recon Error',     color: 'var(--col-text)'  },
              { label: '> threshold?',    color: 'var(--col-amber)' },
              { label: '→',              color: 'var(--col-border)', arrow: true },
              { label: 'DQN Agent',       color: 'var(--col-green)' },
              { label: '→',              color: 'var(--col-border)', arrow: true },
              { label: 'Block / Isolate / Monitor', color: 'var(--col-red)' },
              { label: '→',              color: 'var(--col-border)', arrow: true },
              { label: 'PostgreSQL',      color: 'var(--col-text)'  },
              { label: '→',              color: 'var(--col-border)', arrow: true },
              { label: 'WebSocket',       color: 'var(--col-cyan)'  },
              { label: '→',              color: 'var(--col-border)', arrow: true },
              { label: 'This Dashboard',  color: 'var(--col-text-hi)', bold: true },
            ].map((item, i) =>
              item.arrow ? (
                <span key={i} style={{ color: 'rgba(148,163,184,0.3)' }}>{item.label}</span>
              ) : (
                <span
                  key={i}
                  className="px-2 py-1 rounded"
                  style={{
                    color: item.color,
                    background: 'rgba(255,255,255,0.04)',
                    fontWeight: item.bold ? 600 : 400,
                  }}
                >
                  {item.label}
                </span>
              )
            )}
          </div>
        </div>

            </main>

      {/* ── FOOTER ───────────────────────────────────────────────────── */}
      <footer
        className="relative z-10 mt-8 px-6 py-4 border-t flex items-center justify-between"
        style={{ borderColor: 'var(--col-border)' }}
      >
        <p className="mono text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>
          ThreatSentinel IDS · IISc Research Internship · 2026
        </p>
        <p className="mono text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>
          PyTorch 2.6.0+cu124 · React 19 · Vite 8 · Tailwind v4
        </p>
      </footer>

    </div>
  )
}
