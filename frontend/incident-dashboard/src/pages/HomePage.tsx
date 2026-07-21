/**
 * src/pages/HomePage.tsx
 * ──────────────────────
 * About / landing page for ThreatSentinel.
 * Static content with one live API call (GET /health) for the system
 * status chip. No WebSocket needed here — layout is informational.
 *
 * Sections:
 *  1. Hero — project name, stage badge, live health chip
 *  2. Four-Stage Pipeline cards — Stage 1A → 1B → 2 → 3 + Stage 4 LLM layer
 *  3. Key project metric StatCards
 *  4. Target performance table (with CURRENT column)
 *  5. Tech stack pills
 *  6. Navigation cards to the other 3 pages
 *
 * Facts sourced from README.md (last updated 8 July 2026):
 *  - Stage 1A: Autoencoder v2, 24-dim bottleneck (115→80→48→24→48→80→115), ~90 KB
 *  - Stage 1B: Hybrid XGBoost, 7-class, ~2.5 MB
 *  - Stage 2: Attack-Type NN, 115→128→64→11, excludes Heartbleed + Web_SQL_Injection
 *  - Stage 3: DQN, state_dim→128→64→5, action-match 99.22%, ~95 KB
 *  - Stage 4: openai-agents SDK, ManagerAgent GPT-4o, 13 specialist GPT-4o-mini
 *  - Combined pipeline TPR: 0.7497 (AE alone: 0.7121, +3.76 pp from hybrid)
 *  - DQN action-match accuracy: 0.9922 (99.22%)
 */

import { Link } from 'react-router-dom'
import {
  Brain, Zap, Shield, BarChart3, Terminal, List, Activity,
  Database, Cpu, Network, Bot, GitMerge,
} from 'lucide-react'
import { StatCard } from '../components/StatCard'
import { StatusPill } from '../components/StatusPill'
import { useHealth } from '../hooks/useHealth'

// ── Performance target + current rows ────────────────────────────────────────
// Targets from README project charter (15 June 2026 entry).
// Current values from README combined pipeline evaluation (23 June + 24 June entries).

const PERF_TARGETS = [
  {
    metric: 'Recall (TPR)',
    target: '> 90%',
    current: '74.97%',
    currentNote: 'combined pipeline',
    metColor: false,
    color: 'var(--col-green)',
  },
  {
    metric: 'Precision',
    target: '> 85%',
    current: '90.50%',
    currentNote: 'at p92.5 threshold',
    metColor: true,
    color: 'var(--col-cyan)',
  },
  {
    metric: 'F1 Score',
    target: '> 0.87',
    current: '0.8217',
    currentNote: 'AE v2 alone',
    metColor: false,
    color: 'var(--col-amber)',
  },
  {
    metric: 'DQN Action-Match',
    target: '> 95%',
    current: '99.22%',
    currentNote: 'post web-attack merge',
    metColor: true,
    color: 'var(--col-green)',
  },
]

// ── Tech stack tags ───────────────────────────────────────────────────────────

const TECH_STACK = [
  'PyTorch 2.6', 'FastAPI', 'React 19', 'PostgreSQL', 'Gymnasium',
  'CICIDS2017', 'Tailwind v4', 'Scikit-learn', 'XGBoost',
  'NumPy', 'Pandas', 'Vite 8', 'openai-agents SDK', 'GPT-4o', 'GPT-4o-mini',
]

// ── Navigation cards ──────────────────────────────────────────────────────────

const NAV_CARDS = [
  {
    to: '/terminal',
    icon: Terminal,
    title: 'Live Terminal',
    desc: 'Real-time WebSocket stream of anomaly detections, DQN actions, and Stage 4 LLM agent reasoning as they happen.',
    color: 'var(--col-cyan)',
    border: 'rgba(56,189,248,0.2)',
    bg: 'rgba(56,189,248,0.04)',
  },
  {
    to: '/incidents',
    icon: List,
    title: 'Incident Log',
    desc: 'Paginated table of all flagged network flows with attack type, DQN action taken, reconstruction error, and full feature vectors.',
    color: 'var(--col-red)',
    border: 'rgba(248,113,113,0.2)',
    bg: 'rgba(248,113,113,0.04)',
  },
  {
    to: '/analytics',
    icon: BarChart3,
    title: 'Analytics',
    desc: 'Attack-type distribution breakdown, reconstruction error trends, DQN action frequency, and Stage 4 agent outcome statistics.',
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

// ── Shared pipeline card shell ────────────────────────────────────────────────

interface StageCardProps {
  stageLabel: string
  title: string
  icon: React.ReactNode
  borderColor: string
  accentColor: string
  children: React.ReactNode
}

function StageCard({ stageLabel, title, icon, borderColor, children }: StageCardProps) {
  return (
    <div
      className="rounded-xl border p-6 space-y-4"
      style={{ background: 'var(--col-surface)', borderColor }}
    >
      <div className="flex items-center gap-3">
        {icon}
        <div>
          <p className="mono text-xs" style={{ color: 'var(--col-text)' }}>{stageLabel}</p>
          <h2 className="text-sm font-semibold" style={{ color: 'var(--col-text-hi)' }}>{title}</h2>
        </div>
      </div>
      {children}
    </div>
  )
}

function BulletList({ items, color }: { items: string[]; color: string }) {
  return (
    <ul className="space-y-1.5">
      {items.map(t => (
        <li key={t} className="flex items-start gap-2 text-xs" style={{ color: 'var(--col-text)' }}>
          <span style={{ color, marginTop: 1 }}>›</span>
          {t}
        </li>
      ))}
    </ul>
  )
}

function ArrowConnector({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-1 py-1 lg:py-0 lg:px-1">
      <div className="w-px h-4 lg:flex-1" style={{ background: 'var(--col-border)' }} />
      <div className="mono text-[10px] px-1 text-center" style={{ color: 'var(--col-amber)' }}>
        {label}
      </div>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="rotate-90 lg:rotate-0">
        <path d="M5 12h14M13 6l6 6-6 6" stroke="var(--col-border)" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
      <div className="w-px h-4 lg:flex-1" style={{ background: 'var(--col-border)' }} />
    </div>
  )
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
            style={{ background: 'var(--col-green-dim)', color: 'var(--col-green)' }}
          >
            ALL 4 STAGES INTEGRATED
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
          ThreatSentinel replaces that delay with a fully autonomous 4-stage pipeline: a 24-dim
          Autoencoder and 7-class Hybrid XGBoost Classifier detect anomalies in network flows;
          an Attack-Type Neural Network classifies the threat across 11 CICIDS2017 attack types;
          a Deep Q-Network selects the optimal containment action; and a Stage 4 multi-agent LLM
          layer (ManagerAgent + 13 GPT-4o-mini specialists) delivers human-readable incident
          reasoning over WebSocket to the live dashboard.
        </p>

        {/* Pipeline status strip */}
        <div
          className="inline-flex flex-wrap items-center gap-3 px-4 py-2.5 rounded-lg border mono text-xs"
          style={{
            background: 'rgba(52,211,153,0.05)',
            borderColor: 'rgba(52,211,153,0.2)',
            color: 'var(--col-text)',
          }}
        >
          <span style={{ color: 'var(--col-green)' }}>✓</span>
          All 4 stages trained &amp; integrated
          <span className="opacity-40">·</span>
          Combined pipeline TPR: <span className="font-semibold" style={{ color: 'var(--col-cyan)' }}>0.7497</span>
          <span className="opacity-40">·</span>
          DQN action-match: <span className="font-semibold" style={{ color: 'var(--col-green)' }}>99.22%</span>
          <span className="opacity-40">·</span>
          <span style={{ color: 'rgba(248,113,113,0.8)' }}>Recall gap vs &gt;90% target: open item</span>
        </div>
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
            <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--col-text-hi)' }}>Detection Layer</h3>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--col-text)' }}>
              A 24-dim Autoencoder catches unknown attacks via reconstruction error. A 7-class Hybrid XGBoost Classifier recovers five attack types that statistically resemble benign traffic and evade the autoencoder. An 11-class Attack-Type NN then provides precise classification.
            </p>
          </div>
          <div className="rounded-xl border p-5" style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}>
            <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--col-text-hi)' }}>Remediation Layer</h3>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--col-text)' }}>
              A Deep Q-Network evaluates the full pipeline state and selects from 5 containment actions autonomously. A Stage 4 orchestration layer routes each incident to one of 13 GPT-4o-mini specialist agents that execute domain-specific response tools and broadcast AI reasoning to the dashboard.
            </p>
          </div>
        </div>
      </section>

      {/* ── Four-Stage Pipeline ──────────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-5">
          <span className="mono text-xs font-semibold tracking-widest" style={{ color: 'var(--col-text)' }}>
            PIPELINE ARCHITECTURE
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--col-border)' }} />
        </div>

        {/* Row 1 — Stage 1A + 1B */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr,auto,1fr] mb-4">

          {/* Stage 1A — Autoencoder */}
          <StageCard
            stageLabel="STAGE 1A"
            title="Autoencoder — Anomaly Detection"
            icon={
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
                style={{ background: 'var(--col-cyan-dim)', border: '1px solid rgba(56,189,248,0.3)' }}
              >
                <Brain size={20} color="var(--col-cyan)" />
              </div>
            }
            borderColor="rgba(56,189,248,0.2)"
            accentColor="var(--col-cyan)"
          >
            <div
              className="rounded-lg p-4 mono text-xs leading-loose"
              style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)', color: 'var(--col-cyan)' }}
            >
              <div>Input  [115] ──→ [80] ──→ [48] ──→ [24]</div>
              <div style={{ color: 'var(--col-text)' }}>               Bottleneck (24-dim latent)</div>
              <div>Output [115] ←── [80] ←── [48] ←── [24]</div>
            </div>
            <BulletList color="var(--col-cyan)" items={[
              'v2: 24-dim bottleneck (upgraded from 16-dim v1)',
              'Trained exclusively on benign CICIDS2017 flows',
              '115-feature MinMax-scaled input vectors',
              'p92.5-percentile reconstruction error → threshold',
              'MSE > threshold ⟹ anomaly flag raised',
            ]} />
          </StageCard>

          <ArrowConnector label="evades AE?" />

          {/* Stage 1B — Hybrid XGBoost */}
          <StageCard
            stageLabel="STAGE 1B"
            title="Hybrid XGBoost Classifier"
            icon={
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
                style={{ background: 'rgba(251,191,36,0.1)', border: '1px solid rgba(251,191,36,0.3)' }}
              >
                <GitMerge size={20} color="var(--col-amber)" />
              </div>
            }
            borderColor="rgba(251,191,36,0.2)"
            accentColor="var(--col-amber)"
          >
            <div
              className="rounded-lg p-4 mono text-xs leading-loose"
              style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)', color: 'var(--col-amber)' }}
            >
              <div>7 classes: Benign + 6 attack types</div>
              <div style={{ color: 'var(--col-text)' }}>XGBoost · balanced class weights</div>
              <div style={{ color: 'var(--col-text)' }}>~2.5 MB · hybrid_classifier.pkl</div>
            </div>
            <BulletList color="var(--col-amber)" items={[
              'Catches 5 attacks that evade the autoencoder',
              'FTP-Patator, Botnet_ARES, SSH-Patator, Web_Brute_Force, Web_XSS — statistically benign-like flows',
              'DoS_Hulk added as 7th class (143K missed sub-variant flows)',
              'Combined AE + Hybrid TPR: 0.7497 (+3.76 pp vs AE alone)',
            ]} />
          </StageCard>
        </div>

        <ArrowConnector label="anomaly confirmed →" />

        {/* Row 2 — Stage 2 + Stage 3 */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr,auto,1fr] mt-4">

          {/* Stage 2 — Attack-Type NN */}
          <StageCard
            stageLabel="STAGE 2"
            title="Attack-Type Neural Network"
            icon={
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
                style={{ background: 'rgba(248,113,113,0.1)', border: '1px solid rgba(248,113,113,0.3)' }}
              >
                <Network size={20} color="var(--col-red)" />
              </div>
            }
            borderColor="rgba(248,113,113,0.2)"
            accentColor="var(--col-red)"
          >
            <div
              className="rounded-lg p-4 mono text-xs leading-loose"
              style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)', color: 'var(--col-red)' }}
            >
              <div>Input [115] ──→ [128] ──→ [64] ──→ [11]</div>
              <div style={{ color: 'var(--col-text)' }}>Feedforward · softmax output</div>
              <div style={{ color: 'var(--col-text)' }}>attack_type_nn.pt</div>
            </div>
            <BulletList color="var(--col-red)" items={[
              '11-class multi-class classifier across all CICIDS2017 attack types',
              'Heartbleed (n=12) & Web_SQL_Injection (n=24) excluded — insufficient samples',
              'Outputs calibrated softmax probability vector + confidence score',
              'DoS_Hulk → FTP-Patator confusion confirmed; DQN compensates at 99.97% accuracy',
              'Web_Brute_Force / Web_XSS merged to Revoke Credentials action downstream',
            ]} />
          </StageCard>

          <ArrowConnector label="attack type →" />

          {/* Stage 3 — DQN Agent */}
          <StageCard
            stageLabel="STAGE 3"
            title="DQN Agent — Autonomous Remediation"
            icon={
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
                style={{ background: 'var(--col-green-dim)', border: '1px solid rgba(52,211,153,0.3)' }}
              >
                <Zap size={20} color="var(--col-green)" />
              </div>
            }
            borderColor="rgba(52,211,153,0.2)"
            accentColor="var(--col-green)"
          >
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
            <BulletList color="var(--col-green)" items={[
              'Trained in custom Gymnasium environment (state_dim → 128 → 64 → 5)',
              'State: [115 raw features | AE recon error | attack-type NN softmax probs | confidence]',
              'Reward: +10 threat neutralised, −3 false alarm, −5 service disrupted',
              'Action-match accuracy: 99.22% (post web-attack action merge)',
              'Naturally robust to upstream NN errors (DQN uses raw features as fallback)',
            ]} />
          </StageCard>
        </div>

        <ArrowConnector label="incident →" />

        {/* Stage 4 — Multi-Agent LLM Orchestration */}
        <StageCard
          stageLabel="STAGE 4"
          title="Multi-Agent LLM Orchestration"
          icon={
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
              style={{ background: 'rgba(148,163,184,0.08)', border: '1px solid rgba(148,163,184,0.2)' }}
            >
              <Bot size={20} color="var(--col-text)" />
            </div>
          }
          borderColor="rgba(148,163,184,0.15)"
          accentColor="var(--col-text)"
        >
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div>
              <div
                className="rounded-lg p-4 mono text-xs leading-loose mb-3"
                style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)', color: 'var(--col-text)' }}
              >
                <div style={{ color: 'var(--col-cyan)' }}>ManagerAgent (GPT-4o)</div>
                <div>Routes by attack_type ──→</div>
                <div>13 Specialist Agents (GPT-4o-mini)</div>
                <div style={{ color: 'rgba(148,163,184,0.5)' }}>openai-agents SDK · tool_choice="required"</div>
              </div>
              <BulletList color="var(--col-text)" items={[
                'Built on openai-agents SDK, fires as asyncio background task',
                'ManagerAgent (GPT-4o) routes by attack_type to the matching specialist',
                '13 specialist agents: DoSHulk, DDoSLOIT, PortScan, FTP-Patator, SSH-Patator, DoSGoldenEye, DoSSlowHTTPTest, DoSSlowloris, BotnetARES, WebBruteForce, WebXSS, WebSQLi, Heartbleed',
                'tool_choice="required": LLM must invoke domain tools before responding',
              ]} />
            </div>
            <div>
              <p className="mono text-xs font-semibold mb-2" style={{ color: 'var(--col-text)' }}>SPECIALIST DOMAIN TOOLS:</p>
              {[
                'block_ip_address',
                'isolate_server',
                'revoke_credentials',
                'kill_process',
                'rotate_tls',
                'broadcast_alert_to_client → WebSocket',
              ].map(t => (
                <div
                  key={t}
                  className="mono text-xs px-3 py-1.5 rounded flex items-center gap-2 mb-1"
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)', color: 'var(--col-text)' }}
                >
                  <span style={{ color: 'rgba(148,163,184,0.4)' }}>›</span> {t}
                </div>
              ))}
              <p className="mono text-xs mt-3 leading-relaxed" style={{ color: 'rgba(148,163,184,0.5)' }}>
                Advisory &amp; logging scope only — containment decision remains with the DQN (Stage 3). Tool bodies execute locally.
              </p>
            </div>
          </div>
        </StageCard>
      </section>

      {/* ── Key Metrics ───────────────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-5">
          <span className="mono text-xs font-semibold tracking-widest" style={{ color: 'var(--col-text)' }}>
            DATASET &amp; MODEL PARAMETERS
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--col-border)' }} />
        </div>

        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4 mb-4">
          <StatCard icon={<Database size={18} color="var(--col-cyan)" />}    label="Benign Flows"   value="1.3M"    subLabel="Autoencoder Train Data"  accentColor="var(--col-cyan)"  />
          <StatCard icon={<Shield size={18} color="var(--col-red)" />}       label="Attack Flows"   value="600K+"   subLabel="13 attack types · 115 features"    accentColor="var(--col-red)"   />
          <StatCard icon={<Network size={18} color="var(--col-amber)" />}    label="Input Features" value="115"     subLabel="MinMax scaled to [0, 1]" accentColor="var(--col-amber)" />
          <StatCard icon={<Cpu size={18} color="var(--col-green)" />}        label="Model Footprint" value="~2.7 MB" subLabel="4 models combined"  accentColor="var(--col-green)" />
        </div>

        {/* Per-model breakdown */}
        <div
          className="rounded-xl border overflow-hidden"
          style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
        >
          <div
            className="grid px-5 py-2.5 mono text-xs font-semibold"
            style={{
              gridTemplateColumns: '1fr 1fr 1fr 2fr',
              background: 'rgba(255,255,255,0.03)',
              color: 'var(--col-text-hi)',
            }}
          >
            <span>MODEL FILE</span>
            <span>STAGE</span>
            <span>SIZE</span>
            <span>ARCHITECTURE</span>
          </div>
          {[
            { file: 'autoencoder.pt',        stage: '1A', size: '~90 KB',  arch: '115→80→48→24→48→80→115 · PyTorch' },
            { file: 'hybrid_classifier.pkl', stage: '1B', size: '~2.5 MB', arch: 'XGBoost · 7-class · balanced weights' },
            { file: 'attack_type_nn.pt',     stage: '2',  size: '—',       arch: '115→128→64→11 · PyTorch · softmax' },
            { file: 'dqn_agent.pt',          stage: '3',  size: '~95 KB',  arch: 'state_dim→128→64→5 · PyTorch' },
            { file: 'scaler.pkl',            stage: '—',  size: '~7 KB',   arch: 'MinMaxScaler · 115 features' },
          ].map(({ file, stage, size, arch }, i) => (
            <div
              key={file}
              className="grid px-5 py-2.5 mono text-xs items-center"
              style={{
                gridTemplateColumns: '1fr 1fr 1fr 2fr',
                borderTop: i === 0 ? 'none' : '1px solid var(--col-border)',
                color: 'var(--col-text)',
              }}
            >
              <span style={{ color: 'var(--col-cyan)' }}>{file}</span>
              <span>{stage}</span>
              <span style={{ color: size === '—' ? 'rgba(148,163,184,0.35)' : 'var(--col-text)' }}>
                {size}
                {size === '—' && (
                  <span
                    className="ml-1 text-[10px]"
                    title="File size not explicitly documented in README — see models/attack_type_nn.pt directly"
                    style={{ color: 'var(--col-amber)', cursor: 'help' }}
                  >
                    ⚠
                  </span>
                )}
              </span>
              <span style={{ color: 'rgba(148,163,184,0.6)' }}>{arch}</span>
            </div>
          ))}
        </div>
        <p className="mt-2 mono text-xs" style={{ color: 'rgba(148,163,184,0.35)' }}>
          ⚠ attack_type_nn.pt file size not explicitly documented in README — check models/ directory directly.
          Combined footprint ~2.7 MB dominated by hybrid_classifier.pkl (~2.5 MB).
        </p>
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
                <li className="flex items-start gap-2"><span style={{ color: 'var(--col-green)' }}>›</span> <b>Hybrid Classifier:</b> Trained only on AE-missed flows (prevents double-count)</li>
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
            className="grid grid-cols-4 px-5 py-2.5 mono text-xs font-semibold"
            style={{ background: 'rgba(255,255,255,0.03)', color: 'var(--col-text-hi)' }}
          >
            <span>METRIC</span>
            <span>TARGET</span>
            <span>CURRENT</span>
            <span>NOTE</span>
          </div>
          {PERF_TARGETS.map(({ metric, target, current, currentNote, metColor, color }, i) => (
            <div
              key={metric}
              className="grid grid-cols-4 px-5 py-3 mono text-xs items-center"
              style={{
                borderTop: i === 0 ? 'none' : '1px solid var(--col-border)',
                color: 'var(--col-text)',
              }}
            >
              <span>{metric}</span>
              <span className="font-bold" style={{ color }}>{target}</span>
              <span
                className="font-bold"
                style={{ color: metColor ? 'var(--col-green)' : 'var(--col-red)' }}
              >
                {current}
                {!metColor && (
                  <span className="ml-1 font-normal text-[10px]" style={{ color: 'rgba(248,113,113,0.7)' }}>
                    ↓ below target
                  </span>
                )}
              </span>
              <span style={{ color: 'rgba(148,163,184,0.5)' }}>{currentNote}</span>
            </div>
          ))}
        </div>

        <p className="mt-3 mono text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>
          Recall/TPR gap (0.7497 vs &gt;0.90 target) is an open item documented in README — the DoS_Hulk low-signal sub-variant remains the primary contributor to false negatives.
          All 4 model artefacts trained and integrated. Precision figure is at p92.5 threshold on CICIDS2017 val set.
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
