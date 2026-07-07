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
import { TrendingUp, PieChart as PieIcon, BarChart2 } from 'lucide-react'
import { useIncidentsAnalytics } from '../hooks/useIncidents'
import { StatusPill } from '../components/StatusPill'

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

// ── Pending metric card ───────────────────────────────────────────────────────

function PendingMetric({ label }: { label: string }) {
  return (
    <div
      className="rounded-xl border p-4 space-y-2"
      style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
    >
      <p className="mono text-xs" style={{ color: 'var(--col-text)' }}>{label}</p>
      <StatusPill variant="amber">PENDING EVALUATION</StatusPill>
      <p className="mono text-xs opacity-40" style={{ color: 'var(--col-text)' }}>
        Run model evaluation to populate
      </p>
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
      const key = item.attack_type_predicted ?? 'Benign'
      counts[key] = (counts[key] ?? 0) + 1
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

      {/* ── Performance Metrics (pending) ─────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-4">
          <span className="mono text-xs font-semibold tracking-widest" style={{ color: 'var(--col-text)' }}>
            MODEL PERFORMANCE METRICS
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--col-border)' }} />
        </div>
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          {['Recall', 'Precision', 'F1 Score', 'ROC-AUC'].map(m => (
            <PendingMetric key={m} label={m} />
          ))}
        </div>
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
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={actionBar} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'rgba(148,163,184,0.6)' }}
                  stroke="rgba(255,255,255,0.1)"
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
