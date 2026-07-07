/**
 * src/pages/HomePage.tsx
 * ──────────────────────
 * About / landing page for ThreatSentinel.
 * Static content with one live API call (GET /health) for the system
 * status chip. No WebSocket needed here — layout is informational.
 *
 * Sections:
 *  1. Hero — project name, stage badge, live health chip
 *  2. Two-Stage Pipeline cards — Autoencoder → DQN
 *  3. Key project metric StatCards
 *  4. Target performance table
 *  5. Tech stack pills
 *  6. Navigation cards to the other 3 pages
 */

import { Link } from 'react-router-dom'
import { Brain, Zap, Shield, BarChart3, Terminal, List, Activity, Database, Cpu, Network } from 'lucide-react'
import { StatCard } from '../components/StatCard'
import { StatusPill } from '../components/StatusPill'
import { useHealth } from '../hooks/useHealth'

// ── Performance target rows ───────────────────────────────────────────────────

const PERF_TARGETS = [
  { metric: 'Recall',    target: '> 90%',  color: 'var(--col-green)' },
  { metric: 'Precision', target: '> 85%',  color: 'var(--col-cyan)'  },
  { metric: 'F1 Score',  target: '> 0.87', color: 'var(--col-amber)' },
  { metric: 'ROC-AUC',   target: '> 0.95', color: 'var(--col-cyan)'  },
]

// ── Tech stack tags ───────────────────────────────────────────────────────────

const TECH_STACK = [
  'PyTorch 2.6', 'FastAPI', 'React 19', 'PostgreSQL', 'Gymnasium',
  'CICIDS2017', 'Tailwind v4', 'Scikit-learn', 'NumPy', 'Pandas', 'Vite 8',
]

// ── Navigation cards ──────────────────────────────────────────────────────────

const NAV_CARDS = [
  {
    to: '/terminal',
    icon: Terminal,
    title: 'Live Terminal',
    desc: 'Real-time WebSocket stream of anomaly detections and DQN actions as they happen.',
    color: 'var(--col-cyan)',
    border: 'rgba(56,189,248,0.2)',
    bg: 'rgba(56,189,248,0.04)',
  },
  {
    to: '/incidents',
    icon: List,
    title: 'Incident Log',
    desc: 'Paginated table of all flagged network flows with full feature vectors and action history.',
    color: 'var(--col-red)',
    border: 'rgba(248,113,113,0.2)',
    bg: 'rgba(248,113,113,0.04)',
  },
  {
    to: '/analytics',
    icon: BarChart3,
    title: 'Analytics',
    desc: 'Attack type distribution, reconstruction error trends, and DQN action breakdowns.',
    color: 'var(--col-amber)',
    border: 'rgba(251,191,36,0.2)',
    bg: 'rgba(251,191,36,0.04)',
  },
]

// ── Health chip ───────────────────────────────────────────────────────────────

function SystemHealthChip() {
  const { data, isLoading } = useHealth()

  if (isLoading) {
    return <StatusPill variant="slate" pulse>CHECKING…</StatusPill>
  }
  if (!data) {
    return <StatusPill variant="slate">OFFLINE</StatusPill>
  }

  const allOk = data.db_connected && data.autoencoder_loaded && data.dqn_loaded
  if (allOk) return <StatusPill variant="green" pulse>SYSTEM ONLINE</StatusPill>

  const parts = [
    !data.db_connected       && 'DB DOWN',
    !data.autoencoder_loaded && 'AE NOT LOADED',
    !data.dqn_loaded         && 'DQN NOT LOADED',
  ].filter(Boolean).join(' · ')

  return <StatusPill variant="amber" pulse>DEGRADED — {parts}</StatusPill>
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function HomePage() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-10 space-y-12">

      {/* ── Hero ──────────────────────────────────────────────────────── */}
      <section className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <span
            className="mono text-xs font-medium px-2 py-0.5 rounded"
            style={{ background: 'var(--col-cyan-dim)', color: 'var(--col-cyan)' }}
          >
            STAGE 1 IN PROGRESS
          </span>
          <span className="mono text-xs" style={{ color: 'var(--col-text)' }}>
            // Real-Time Incident Tracking &amp; Autonomous Threat Remediation
          </span>
          <SystemHealthChip />
        </div>

        <h1 className="text-4xl font-bold tracking-tight" style={{ color: 'var(--col-text-hi)' }}>
          ThreatSentinel
        </h1>
        <p className="text-base max-w-2xl leading-relaxed" style={{ color: 'var(--col-text)' }}>
          Traditional cyber defense relies on human analysts who take hours to respond to alerts. 
          This project builds an autonomous security system capable of millisecond response times. 
          An unsupervised Autoencoder catches zero-day attacks by detecting deviations from normal traffic, 
          and a Deep Q-Network (DQN) agent selects the optimal containment action autonomously.
        </p>
      </section>

      {/* ── Project Overview ────────────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-5">
          <span className="mono text-xs font-semibold tracking-widest" style={{ color: 'var(--col-text)' }}>
            PROJECT OVERVIEW
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--col-border)' }} />
        </div>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="rounded-xl border p-5" style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}>
            <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--col-text-hi)' }}>The Problem</h3>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--col-text)' }}>
              Human response times are the core bottleneck in cybersecurity. By the time analysts review alerts, malware can encrypt entire databases. Signature-based systems also completely miss novel "zero-day" attacks.
            </p>
          </div>
          <div className="rounded-xl border p-5" style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}>
            <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--col-text-hi)' }}>Anomaly Detection</h3>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--col-text)' }}>
              By training an Autoencoder strictly on benign CICIDS2017 traffic, the model learns the "fingerprint" of normal activity. Any attack, known or unknown, yields a high reconstruction error, instantly flagging it.
            </p>
          </div>
          <div className="rounded-xl border p-5" style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}>
            <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--col-text-hi)' }}>Autonomous Remediation</h3>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--col-text)' }}>
              Once flagged, the Deep Q-Network agent takes over. Trained in a custom Gymnasium environment, it evaluates the anomaly score and attack embedding to instantly execute containment actions autonomously.
            </p>
          </div>
        </div>
      </section>

      {/* ── Two-Stage Pipeline ─────────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-5">
          <span className="mono text-xs font-semibold tracking-widest" style={{ color: 'var(--col-text)' }}>
            PIPELINE ARCHITECTURE
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--col-border)' }} />
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr,auto,1fr]">

          {/* Stage 1 — Autoencoder */}
          <div
            className="rounded-xl border p-6 space-y-4"
            style={{ background: 'var(--col-surface)', borderColor: 'rgba(56,189,248,0.2)' }}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center"
                style={{ background: 'var(--col-cyan-dim)', border: '1px solid rgba(56,189,248,0.3)' }}
              >
                <Brain size={20} color="var(--col-cyan)" />
              </div>
              <div>
                <p className="mono text-xs" style={{ color: 'var(--col-text)' }}>STAGE 1</p>
                <h2 className="text-sm font-semibold" style={{ color: 'var(--col-text-hi)' }}>
                  Autoencoder — Anomaly Detection
                </h2>
              </div>
            </div>

            <div
              className="rounded-lg p-4 mono text-xs leading-loose"
              style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)', color: 'var(--col-cyan)' }}
            >
              <div>Input  [115] ──→ [64] ──→ [32] ──→ [16]</div>
              <div style={{ color: 'var(--col-text)' }}>               Bottleneck (latent)</div>
              <div>Output [115] ←── [64] ←── [32] ←── [16]</div>
            </div>

            <ul className="space-y-1.5">
              {[
                'Trained exclusively on benign CICIDS2017 flows',
                '115-feature MinMax-scaled input vectors',
                '95th-percentile reconstruction error → threshold',
                'MSE > threshold ⟹ anomaly flag raised',
              ].map(t => (
                <li key={t} className="flex items-start gap-2 text-xs" style={{ color: 'var(--col-text)' }}>
                  <span style={{ color: 'var(--col-cyan)', marginTop: 1 }}>›</span>
                  {t}
                </li>
              ))}
            </ul>
          </div>

          {/* Arrow connector */}
          <div className="hidden lg:flex flex-col items-center justify-center gap-1 px-2">
            <div className="w-px flex-1" style={{ background: 'var(--col-border)' }} />
            <div className="mono text-xs px-2" style={{ color: 'var(--col-amber)' }}>
              anomaly?
            </div>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M5 12h14M13 6l6 6-6 6" stroke="var(--col-border)" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <div className="w-px flex-1" style={{ background: 'var(--col-border)' }} />
          </div>

          {/* Stage 2 — DQN */}
          <div
            className="rounded-xl border p-6 space-y-4"
            style={{ background: 'var(--col-surface)', borderColor: 'rgba(52,211,153,0.2)' }}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center"
                style={{ background: 'var(--col-green-dim)', border: '1px solid rgba(52,211,153,0.3)' }}
              >
                <Zap size={20} color="var(--col-green)" />
              </div>
              <div>
                <p className="mono text-xs" style={{ color: 'var(--col-text)' }}>STAGE 2</p>
                <h2 className="text-sm font-semibold" style={{ color: 'var(--col-text-hi)' }}>
                  DQN Agent — Autonomous Remediation
                </h2>
              </div>
            </div>

            <div className="space-y-1.5">
              <p className="mono text-xs" style={{ color: 'var(--col-text)' }}>ACTION SPACE (5 actions):</p>
              {[
                { label: 'block_ip',           color: 'var(--col-red)'   },
                { label: 'revoke_credentials', color: 'var(--col-amber)' },
                { label: 'isolate_server',     color: 'var(--col-amber)' },
                { label: 'kill_process',       color: 'var(--col-red)'   },
                { label: 'monitor',            color: 'var(--col-cyan)'  },
              ].map(({ label, color }) => (
                <div
                  key={label}
                  className="mono text-xs px-3 py-1.5 rounded flex items-center gap-2"
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)', color }}
                >
                  <span style={{ color: 'rgba(148,163,184,0.4)' }}>›</span> {label}
                </div>
              ))}
            </div>

            <ul className="space-y-1.5">
              {[
                'Trained in custom Gymnasium environment',
                'State: reconstruction error + attack category embedding',
                'Reward: penalises FP, rewards correct remediation',
              ].map(t => (
                <li key={t} className="flex items-start gap-2 text-xs" style={{ color: 'var(--col-text)' }}>
                  <span style={{ color: 'var(--col-green)', marginTop: 1 }}>›</span>
                  {t}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* ── Key Metrics ───────────────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-5">
          <span className="mono text-xs font-semibold tracking-widest" style={{ color: 'var(--col-text)' }}>
            DATASET &amp; MODEL PARAMETERS
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--col-border)' }} />
        </div>

        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatCard icon={<Database size={18} color="var(--col-cyan)" />}    label="Benign Flows"   value="1.3M"  subLabel="Autoencoder Train Data"  accentColor="var(--col-cyan)"  />
          <StatCard icon={<Shield size={18} color="var(--col-red)" />}       label="Attack Flows"   value="1.4M"   subLabel="14 attack types"    accentColor="var(--col-red)"   />
          <StatCard icon={<Network size={18} color="var(--col-amber)" />}    label="Input Features" value="115"    subLabel="MinMax scaled" accentColor="var(--col-amber)" />
          <StatCard icon={<Cpu size={18} color="var(--col-green)" />}        label="Model Size"     value="~200 KB" subLabel="Extremely lightweight"  accentColor="var(--col-green)" />
        </div>
      </section>

      {/* ── Data Preprocessing ────────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-5">
          <span className="mono text-xs font-semibold tracking-widest" style={{ color: 'var(--col-text)' }}>
            DATA ENGINEERING WORKFLOW
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--col-border)' }} />
        </div>
        
        <div className="rounded-xl border p-6" style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div>
              <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--col-text-hi)' }}>CICIDS2017 Data Cleaning</h3>
              <ul className="space-y-2 mono text-xs" style={{ color: 'var(--col-text)' }}>
                <li className="flex items-start gap-2"><span style={{ color: 'var(--col-cyan)' }}>1.</span> Fixed invisible whitespace in column headers (e.g. ' Label')</li>
                <li className="flex items-start gap-2"><span style={{ color: 'var(--col-cyan)' }}>2.</span> Removed infinity values caused by CICFlowMeter division-by-zero</li>
                <li className="flex items-start gap-2"><span style={{ color: 'var(--col-cyan)' }}>3.</span> Deduplicated ~6% of identical network flows</li>
                <li className="flex items-start gap-2"><span style={{ color: 'var(--col-cyan)' }}>4.</span> Converted 64-bit floats to float32 to halve memory usage</li>
                <li className="flex items-start gap-2"><span style={{ color: 'var(--col-cyan)' }}>5.</span> Split into strictly isolated benign (train) and attack (test) pipelines</li>
              </ul>
            </div>
            <div>
              <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--col-text-hi)' }}>Strict Validation Strategy</h3>
              <ul className="space-y-2 mono text-xs" style={{ color: 'var(--col-text)' }}>
                <li className="flex items-start gap-2"><span style={{ color: 'var(--col-green)' }}>›</span> <b>No Data Leakage:</b> MinMaxScaler fit ONLY on benign training split</li>
                <li className="flex items-start gap-2"><span style={{ color: 'var(--col-green)' }}>›</span> <b>Autoencoder Training:</b> 70/15/15 split on benign data only</li>
                <li className="flex items-start gap-2"><span style={{ color: 'var(--col-green)' }}>›</span> <b>DQN Environment:</b> Uses separate attack data pipeline to simulate threat states</li>
                <li className="flex items-start gap-2"><span style={{ color: 'var(--col-green)' }}>›</span> <b>Cross-Dataset Test:</b> UNSW-NB15 dataset reserved for zero-day generalization</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ── Performance Targets ───────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-5">
          <span className="mono text-xs font-semibold tracking-widest" style={{ color: 'var(--col-text)' }}>
            TARGET PERFORMANCE THRESHOLDS
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--col-border)' }} />
        </div>

        <div
          className="rounded-xl border overflow-hidden"
          style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
        >
          {/* Table header */}
          <div
            className="grid grid-cols-2 px-5 py-2.5 mono text-xs font-semibold"
            style={{ background: 'rgba(255,255,255,0.03)', color: 'var(--col-text-hi)' }}
          >
            <span>METRIC</span>
            <span>TARGET</span>
          </div>
          {PERF_TARGETS.map(({ metric, target, color }, i) => (
            <div
              key={metric}
              className="grid grid-cols-2 px-5 py-3 mono text-xs items-center"
              style={{
                borderTop: i === 0 ? 'none' : '1px solid var(--col-border)',
                color: 'var(--col-text)',
              }}
            >
              <span>{metric}</span>
              <span className="font-bold" style={{ color }}>{target}</span>
            </div>
          ))}
        </div>

        <p className="mt-3 mono text-xs text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>
          Pending offline training · Awaiting autoencoder.pt, dqn_agent.pt, scaler.pkl, and threshold.json
        </p>
      </section>

      {/* ── Tech Stack ────────────────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-5">
          <span className="mono text-xs font-semibold tracking-widest" style={{ color: 'var(--col-text)' }}>
            TECHNOLOGY STACK
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--col-border)' }} />
        </div>
        <div className="flex flex-wrap gap-2">
          {TECH_STACK.map(tech => (
            <span
              key={tech}
              className="mono text-xs px-3 py-1.5 rounded-full border"
              style={{
                background: 'rgba(255,255,255,0.03)',
                borderColor: 'var(--col-border)',
                color: 'var(--col-text)',
              }}
            >
              {tech}
            </span>
          ))}
        </div>
      </section>

      {/* ── Navigation Cards ──────────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-5">
          <span className="mono text-xs font-semibold tracking-widest" style={{ color: 'var(--col-text)' }}>
            EXPLORE THE DASHBOARD
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--col-border)' }} />
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {NAV_CARDS.map(({ to, icon: Icon, title, desc, color, border, bg }) => (
            <Link
              key={to}
              to={to}
              className="group block rounded-xl border p-5 transition-all duration-200 hover:-translate-y-1 focus-visible:outline-2 focus-visible:outline-offset-2"
              style={{
                background: 'var(--col-surface)',
                borderColor: 'var(--col-border)',
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.background = bg
                ;(e.currentTarget as HTMLElement).style.borderColor = border
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background = 'var(--col-surface)'
                ;(e.currentTarget as HTMLElement).style.borderColor = 'var(--col-border)'
              }}
            >
              <div className="flex items-center gap-3 mb-3">
                <div
                  className="w-9 h-9 rounded-lg flex items-center justify-center transition-all"
                  style={{ background: `${color}18`, border: `1px solid ${color}40` }}
                >
                  <Icon size={18} color={color} />
                </div>
                <span className="text-sm font-semibold" style={{ color: 'var(--col-text-hi)' }}>{title}</span>
                <Activity size={12} className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity" color={color} />
              </div>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--col-text)' }}>{desc}</p>
              <div className="mt-3 mono text-xs" style={{ color }}>
                Open {title} →
              </div>
            </Link>
          ))}
        </div>
      </section>

    </div>
  )
}
