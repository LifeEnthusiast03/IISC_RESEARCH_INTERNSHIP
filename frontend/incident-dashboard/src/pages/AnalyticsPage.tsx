/**
 * src/pages/AnalyticsPage.tsx
 * ───────────────────────────
 * Analytics dashboard derived from the incidents dataset.
 * Charts are computed client-side from useIncidentsAnalytics (100 items).
 *
 * Charts:
 *  1. Pie/donut — attack type distribution
 *  2. Area chart — reconstruction error over time
 *  3. Bar chart  — incident count by DQN action taken
 *
 * Performance metrics row is shown as "PENDING EVALUATION" pills since
 * no /metrics endpoint is confirmed in the backend.
 */

import { useMemo } from 'react'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  BarChart, Bar,
} from 'recharts'
import { format } from 'date-fns'
import { TrendingUp, PieChart as PieIcon, BarChart2, Target, Zap } from 'lucide-react'
import { useIncidentsAnalytics } from '../hooks/useIncidents'
import ATTACK_LABEL_MAP from '../lib/attack_type_label_map.json'

// ── Palette ───────────────────────────────────────────────────────────────────

const CHART_COLORS = [
  '#38bdf8', '#f87171', '#fbbf24', '#34d399', '#a78bfa',
  '#f472b6', '#60a5fa', '#fb923c', '#4ade80', '#e879f9',
]

// ── Custom tooltip ────────────────────────────────────────────────────────────

function DarkTooltip({ active, payload, label }: {
  active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: string
}) {
  if (!active || !payload?.length) return null
  return (
    <div
      className="rounded-lg border px-3 py-2 mono text-xs"
      style={{ background: '#0d1117', borderColor: 'rgba(56,189,248,0.2)', color: '#e2e8f0' }}
    >
      {label && <p className="mb-1 opacity-50">{label}</p>}
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' && p.value < 1 ? p.value.toExponential(3) : p.value}
        </p>
      ))}
    </div>
  )
}

// ── Evaluation metric card ────────────────────────────────────────────────────
// All static values sourced from README.md (last updated 8 July 2026).
// README reference lines noted inline.

interface EvalMetricProps {
  label: string
  value: string
  subValue?: string
  target: string
  gap: string
  gapMet: boolean
  source: 'readme' | 'not-computed'
  note?: string
}

function EvalMetricCard({ label, value, subValue, target, gap, gapMet, source, note }: EvalMetricProps) {
  return (
    <div
      className="rounded-xl border p-4 space-y-2"
      style={{ background: 'var(--col-surface)', borderColor: gapMet ? 'rgba(52,211,153,0.2)' : 'rgba(248,113,113,0.15)' }}
    >
      <p className="mono text-xs" style={{ color: 'var(--col-text)' }}>{label}</p>
      {source === 'not-computed' ? (
        <>
          <p className="mono text-lg font-bold" style={{ color: 'rgba(148,163,184,0.4)' }}>—</p>
          <p className="mono text-[10px] leading-relaxed" style={{ color: 'rgba(148,163,184,0.4)' }}>
            Not yet computed on live traffic
          </p>
        </>
      ) : (
        <>
          <p
            className="mono text-xl font-bold leading-none"
            style={{ color: gapMet ? 'var(--col-green)' : 'var(--col-red)' }}
          >
            {value}
          </p>
          {subValue && (
            <p className="mono text-[10px]" style={{ color: 'var(--col-text)' }}>{subValue}</p>
          )}
          <div
            className="mono text-[10px] px-1.5 py-0.5 rounded inline-block"
            style={{
              background: gapMet ? 'rgba(52,211,153,0.08)' : 'rgba(248,113,113,0.08)',
              color: gapMet ? 'var(--col-green)' : 'var(--col-red)',
            }}
          >
            vs target {target} · {gap}
          </div>
          {note && (
            <p className="mono text-[10px]" style={{ color: 'rgba(148,163,184,0.4)' }}>{note}</p>
          )}
        </>
      )}
    </div>
  )
}

// ── Chart card wrapper ────────────────────────────────────────────────────────

function ChartCard({
  icon: Icon, title, eyebrow, children,
}: {
  icon: React.ElementType; title: string; eyebrow: string; children: React.ReactNode
}) {
  return (
    <div
      className="rounded-xl border p-5 space-y-4"
      style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
    >
      <div className="flex items-center gap-2">
        <Icon size={15} color="var(--col-cyan)" />
        <div>
          <p className="mono text-xs" style={{ color: 'var(--col-text)' }}>{eyebrow}</p>
          <h3 className="text-sm font-semibold" style={{ color: 'var(--col-text-hi)' }}>{title}</h3>
        </div>
      </div>
      {children}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const { data, isLoading, isError } = useIncidentsAnalytics()

  // ── Derived chart data ────────────────────────────────────────────────────

  // Attack type distribution
  const attackTypePie = useMemo(() => {
    if (!data) return []
    const counts: Record<string, number> = {}
    for (const item of data.items) {
      // Resolve numeric ID → human name; fall back to raw value or 'Benign'
      const raw = item.attack_type_predicted
      const label = raw != null
        ? (ATTACK_LABEL_MAP as Record<string, string>)[String(raw)] ?? String(raw)
        : 'Benign'
      counts[label] = (counts[label] ?? 0) + 1
    }
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => ({ name, value }))
  }, [data])

  // Reconstruction error over time (last 50)
  const reconTimeSeries = useMemo(() => {
    if (!data) return []
    return [...data.items]
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .slice(-50)
      .map(item => ({
        time: format(new Date(item.timestamp), 'HH:mm:ss'),
        error: item.reconstruction_error,
        anomaly: item.is_anomaly ? item.reconstruction_error : null,
      }))
  }, [data])

  // DQN action distribution
  const actionBar = useMemo(() => {
    if (!data) return []
    const counts: Record<string, number> = {}
    for (const item of data.items) {
      const key = item.dqn_action ?? 'none'
      counts[key] = (counts[key] ?? 0) + 1
    }
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([name, count]) => ({ name, count }))
  }, [data])

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">

      {/* ── Header ──────────────────────────────────────────────────── */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <span className="mono text-xs" style={{ color: 'var(--col-cyan)' }}>
            ANALYTICS // Derived from Incident Dataset
          </span>
        </div>
        <h1 className="text-2xl font-bold tracking-tight" style={{ color: 'var(--col-text-hi)' }}>
          Analytics
        </h1>
        <p className="text-sm mt-1" style={{ color: 'var(--col-text)' }}>
          Attack distributions, anomaly score trends, and remediation action breakdown.
          {data && (
            <span className="mono ml-2 opacity-60">
              (from latest {data.items.length} incidents)
            </span>
          )}
        </p>
      </div>

      {/* ── Performance Metrics ─────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-4">
          <span className="mono text-xs font-semibold tracking-widest" style={{ color: 'var(--col-text)' }}>
            MODEL PERFORMANCE METRICS
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--col-border)' }} />
          <span
            className="mono text-[10px] px-2 py-0.5 rounded"
            style={{ background: 'rgba(56,189,248,0.08)', color: 'var(--col-cyan)' }}
          >
            README · offline CICIDS2017 test set
          </span>
        </div>

        {/* Primary cards — large headline metrics */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 mb-4">
          {/* Combined Detection TPR — README line 1038 */}
          <div
            className="rounded-xl border p-5"
            style={{ background: 'var(--col-surface)', borderColor: 'rgba(248,113,113,0.2)' }}
          >
            <div className="flex items-center gap-2 mb-1">
              <Target size={14} color="var(--col-red)" />
              <p className="mono text-xs" style={{ color: 'var(--col-text)' }}>COMBINED DETECTION TPR (Recall)</p>
            </div>
            <p className="mono text-3xl font-bold" style={{ color: 'var(--col-red)' }}>74.97%</p>
            <p className="mono text-xs mt-1" style={{ color: 'var(--col-text)' }}>
              AE v2 alone: 71.21% · +3.76 pp from Stage 1B Hybrid XGBoost
            </p>
            <div
              className="mono text-[10px] mt-2 px-1.5 py-0.5 rounded inline-block"
              style={{ background: 'rgba(248,113,113,0.08)', color: 'var(--col-red)' }}
            >
              vs target &gt;90% · gap: 15.03 pp — primary open item
            </div>
            <p className="mono text-[10px] mt-1" style={{ color: 'rgba(148,163,184,0.35)' }}>
              DoS_Hulk low-signal sub-variant is dominant source of missed flows
            </p>
          </div>

          {/* DQN Action-Match — README line 1118 */}
          <div
            className="rounded-xl border p-5"
            style={{ background: 'var(--col-surface)', borderColor: 'rgba(52,211,153,0.2)' }}
          >
            <div className="flex items-center gap-2 mb-1">
              <Zap size={14} color="var(--col-green)" />
              <p className="mono text-xs" style={{ color: 'var(--col-text)' }}>DQN ACTION-MATCH ACCURACY</p>
            </div>
            <p className="mono text-3xl font-bold" style={{ color: 'var(--col-green)' }}>99.22%</p>
            <p className="mono text-xs mt-1" style={{ color: 'var(--col-text)' }}>
              Post web-attack action merge · avg reward: 9.8869 / 10
            </p>
            <div
              className="mono text-[10px] mt-2 px-1.5 py-0.5 rounded inline-block"
              style={{ background: 'rgba(52,211,153,0.08)', color: 'var(--col-green)' }}
            >
              vs target &gt;95% · ✓ exceeds target by 4.22 pp
            </div>
            <p className="mono text-[10px] mt-1" style={{ color: 'rgba(148,163,184,0.35)' }}>
              99.97% on DoS_Hulk→FTP-Patator confused rows (DQN compensates via raw features)
            </p>
          </div>
        </div>

        {/* Secondary cards — AE-level P/F1, and gaps */}
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          {/* AE v2 Precision — README line 882 */}
          <EvalMetricCard
            label="Precision (AE v2 at p92.5)"
            value="90.50%"
            target=">85%"
            gap="✓ +5.50 pp"
            gapMet={true}
            source="readme"
            note="On CICIDS2017 val set · FPR 7.5%"
          />
          {/* AE v2 F1 — README line 887 */}
          <EvalMetricCard
            label="F1 Score (AE v2 at p92.5)"
            value="0.8217"
            target=">0.87"
            gap="↓ gap 0.048"
            gapMet={false}
            source="readme"
            note="AE v2 alone; combined pipeline F1 not yet re-evaluated"
          />
          {/* ROC-AUC — not in README or any persisted output file */}
          <EvalMetricCard
            label="ROC-AUC"
            value="—"
            target=">0.95"
            gap="—"
            gapMet={false}
            source="not-computed"
            note="evaluate_combined_pipeline.py prints this but output not persisted"
          />
          {/* DQN avg reward for context */}
          <EvalMetricCard
            label="DQN Avg Reward / Episode"
            value="9.8869"
            subValue="max possible: 10.0"
            target=">9.5"
            gap="✓ +0.39"
            gapMet={true}
            source="readme"
            note="Post merge, 100K episodes · README line 1119"
          />
        </div>

        <p className="mt-2 mono text-[10px]" style={{ color: 'rgba(148,163,184,0.35)' }}>
          All values from offline CICIDS2017 evaluation (README.md, last updated 8 July 2026). Live traffic metrics not yet aggregated.
          ROC-AUC and combined F1 require re-running evaluate_combined_pipeline.py with output persistence.
        </p>
      </section>

      {/* ── Loading / Error ──────────────────────────────────────────── */}
      {isLoading && (
        <div
          className="rounded-xl border p-10 text-center mono text-sm"
          style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)', color: 'var(--col-cyan)' }}
        >
          Loading incident data<span className="blink ml-0.5">_</span>
        </div>
      )}

      {isError && (
        <div
          className="rounded-xl border p-6"
          style={{ background: 'rgba(248,113,113,0.06)', borderColor: 'rgba(248,113,113,0.2)' }}
        >
          <p className="mono text-xs" style={{ color: 'var(--col-red)' }}>
            Failed to load incident data. Is the backend running?
          </p>
        </div>
      )}

      {/* ── No data state ────────────────────────────────────────────── */}
      {!isLoading && !isError && data?.items.length === 0 && (
        <div
          className="rounded-xl border p-10 text-center"
          style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
        >
          <p className="mono text-sm" style={{ color: 'var(--col-cyan)' }}>
            {'>'} No incidents recorded yet<span className="blink ml-0.5">_</span>
          </p>
          <p className="text-xs mt-2" style={{ color: 'var(--col-text)' }}>
            Start sending flows via POST /predict or the simulator to populate charts.
          </p>
        </div>
      )}

      {/* ── Charts (only shown when there is data) ───────────────────── */}
      {data && data.items.length > 0 && (
        <>
          {/* Row 1: Pie + Area */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">

            {/* Attack type distribution */}
            <ChartCard icon={PieIcon} eyebrow="ATTACK DISTRIBUTION" title="Attack Type Breakdown">
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={attackTypePie}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {attackTypePie.map((_, i) => (
                      <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} stroke="transparent" />
                    ))}
                  </Pie>
                  <Tooltip content={<DarkTooltip />} />
                  <Legend
                    formatter={(value) => (
                      <span style={{ color: 'var(--col-text)', fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                        {value}
                      </span>
                    )}
                  />
                </PieChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Reconstruction error over time */}
            <ChartCard icon={TrendingUp} eyebrow="ANOMALY SCORE TREND" title="Reconstruction Error (last 50 events)">
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={reconTimeSeries} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <defs>
                    <linearGradient id="errorGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#38bdf8" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#38bdf8" stopOpacity={0}    />
                    </linearGradient>
                    <linearGradient id="anomalyGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#f87171" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#f87171" stopOpacity={0}   />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
                  <XAxis
                    dataKey="time"
                    tick={{ fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'rgba(148,163,184,0.5)' }}
                    interval="preserveStartEnd"
                    stroke="rgba(255,255,255,0.1)"
                  />
                  <YAxis
                    tick={{ fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'rgba(148,163,184,0.5)' }}
                    stroke="rgba(255,255,255,0.1)"
                    tickFormatter={v => v.toExponential(1)}
                  />
                  <Tooltip content={<DarkTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="error"
                    name="Recon Error"
                    stroke="#38bdf8"
                    fill="url(#errorGrad)"
                    strokeWidth={1.5}
                    dot={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="anomaly"
                    name="Anomaly"
                    stroke="#f87171"
                    fill="url(#anomalyGrad)"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* Row 2: DQN Action Bar */}
          <ChartCard icon={BarChart2} eyebrow="REMEDIATION ACTIONS" title="Incidents per DQN Action">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={actionBar} margin={{ top: 5, right: 20, left: 0, bottom: 55 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'rgba(148,163,184,0.6)' }}
                  stroke="rgba(255,255,255,0.1)"
                  angle={-30}
                  textAnchor="end"
                  height={55}
                  interval={0}
                />
                <YAxis
                  tick={{ fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'rgba(148,163,184,0.5)' }}
                  stroke="rgba(255,255,255,0.1)"
                  allowDecimals={false}
                />
                <Tooltip content={<DarkTooltip />} />
                <Bar dataKey="count" name="Incidents" radius={[4, 4, 0, 0]}>
                  {actionBar.map((_, i) => (
                    <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Summary stats row */}
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            {[
              { label: 'Total Incidents',       value: data.total,                                                       color: 'var(--col-cyan)'  },
              { label: 'Anomalous Flows',        value: data.items.filter(i => i.is_anomaly).length,                     color: 'var(--col-red)'   },
              { label: 'Unique Attack Types',    value: new Set(data.items.map(i => i.attack_type_predicted)).size,       color: 'var(--col-amber)' },
              { label: 'Actions Taken',          value: data.items.filter(i => i.dqn_action && i.dqn_action !== 'none').length, color: 'var(--col-green)' },
            ].map(({ label, value, color }) => (
              <div
                key={label}
                className="rounded-xl border p-4"
                style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
              >
                <p className="mono text-2xl font-bold" style={{ color }}>{value}</p>
                <p className="text-xs mt-1" style={{ color: 'var(--col-text)' }}>{label}</p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
