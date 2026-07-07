/**
 * src/pages/IncidentsPage.tsx
 * ───────────────────────────
 * Paginated incident log table — fetches from GET /incidents via React Query.
 * Features: filter by attack type, search by IP, sortable columns,
 * expandable row detail, skeleton loading, error, and empty states.
 */

import { useState, useMemo } from 'react'
import { format } from 'date-fns'
import { ChevronLeft, ChevronRight, ChevronDown, ChevronUp, AlertTriangle, Search, Filter } from 'lucide-react'
import { useIncidents } from '../hooks/useIncidents'
import { StatusPill } from '../components/StatusPill'
import type { IncidentOut } from '../lib/api'
import ATTACK_TYPE_MAP from '../lib/attack_type_label_map.json'

function getAttackLabel(typeId: string | null | undefined): string | null {
  if (!typeId) return null
  return (ATTACK_TYPE_MAP as Record<string, string>)[typeId] || typeId
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function severityFromAction(action: string | null): { variant: 'red' | 'amber' | 'cyan' | 'slate'; label: string } {
  if (!action) return { variant: 'slate', label: 'INFO' }
  const a = action.toLowerCase()
  if (a.includes('block') || a.includes('kill') || a.includes('isolate')) return { variant: 'red',   label: 'CRITICAL' }
  if (a.includes('revoke'))                                                return { variant: 'amber', label: 'HIGH'     }
  if (a.includes('monitor'))                                               return { variant: 'cyan',  label: 'LOW'      }
  return { variant: 'slate', label: 'INFO' }
}

function formatReconError(err: number): string {
  return err.toExponential(3)
}

// ── Skeleton row ──────────────────────────────────────────────────────────────

function SkeletonRow() {
  return (
    <tr>
      {Array.from({ length: 7 }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div
            className="h-3 rounded animate-pulse"
            style={{
              background: 'rgba(255,255,255,0.06)',
              width: `${60 + Math.random() * 40}%`,
            }}
          />
        </td>
      ))}
    </tr>
  )
}

// ── Expanded row detail ───────────────────────────────────────────────────────

function ExpandedDetail({ incident }: { incident: IncidentOut }) {
  return (
    <tr>
      <td
        colSpan={7}
        className="px-6 py-4"
        style={{ background: 'rgba(56,189,248,0.03)', borderTop: '1px solid rgba(56,189,248,0.1)' }}
      >
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Left: incident metadata */}
          <div className="space-y-2">
            <p className="mono text-xs font-semibold" style={{ color: 'var(--col-text-hi)' }}>
              INCIDENT #{incident.id} — DETAIL
            </p>
            {[
              ['Timestamp',        format(new Date(incident.timestamp), 'yyyy-MM-dd HH:mm:ss.SSS')],
              ['Source IP',        `${incident.source_ip}:${incident.src_port ?? '?'}`],
              ['Dest IP',          `${incident.dest_ip}:${incident.dst_port ?? '?'}`],
              ['Is Anomaly',       incident.is_anomaly ? 'YES' : 'NO'],
              ['Recon Error',      incident.reconstruction_error.toFixed(6)],
              ['Attack Type',      getAttackLabel(incident.attack_type_predicted) ?? '—'],
              ['DQN Action',       incident.dqn_action ?? '—'],
              ['Action Status',    incident.action_status],
              ['Created At',       format(new Date(incident.created_at), 'yyyy-MM-dd HH:mm:ss')],
            ].map(([label, value]) => (
              <div key={label} className="flex gap-3 text-xs">
                <span className="mono shrink-0 w-28 opacity-50" style={{ color: 'var(--col-text)' }}>{label}</span>
                <span className="mono" style={{ color: 'var(--col-text-hi)' }}>{value}</span>
              </div>
            ))}
          </div>

          {/* Right: raw features */}
          {incident.raw_features && (
            <div>
              <p className="mono text-xs font-semibold mb-2" style={{ color: 'var(--col-text-hi)' }}>
                RAW FEATURE VECTOR
              </p>
              <pre
                className="mono text-xs overflow-auto rounded-lg p-3 max-h-52"
                style={{
                  background: 'rgba(0,0,0,0.4)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  color: 'var(--col-cyan)',
                }}
              >
                {JSON.stringify(incident.raw_features, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </td>
    </tr>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

const PAGE_SIZE = 20

export default function IncidentsPage() {
  const [page, setPage] = useState(1)
  const [expandedId, setExpandedId] = useState<number | null>(null)
  const [attackFilter, setAttackFilter] = useState('')
  const [ipSearch, setIpSearch] = useState('')

  const { data, isLoading, isError, error } = useIncidents(page, PAGE_SIZE)

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0

  // Unique attack types for filter dropdown (from current page)
  const attackTypes = useMemo(() => {
    if (!data) return []
    const types = [...new Set(data.items.map(i => getAttackLabel(i.attack_type_predicted)).filter(Boolean))]
    return types as string[]
  }, [data])

  // Client-side filter (on current page)
  const filteredItems = useMemo(() => {
    if (!data) return []
    return data.items.filter(item => {
      const mappedLabel = getAttackLabel(item.attack_type_predicted)
      const matchType = !attackFilter || mappedLabel === attackFilter
      const matchIp = !ipSearch || item.source_ip.includes(ipSearch) || item.dest_ip.includes(ipSearch)
      return matchType && matchIp
    })
  }, [data, attackFilter, ipSearch])

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">

      {/* ── Header ──────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="mono text-xs" style={{ color: 'var(--col-cyan)' }}>
              INCIDENTS // Paginated Anomaly Log
            </span>
            {data && (
              <StatusPill variant="cyan">{data.total} total</StatusPill>
            )}
          </div>
          <h1 className="text-2xl font-bold tracking-tight" style={{ color: 'var(--col-text-hi)' }}>
            Incident Log
          </h1>
          <p className="text-sm mt-1" style={{ color: 'var(--col-text)' }}>
            All flagged network flows — newest first
          </p>
        </div>
      </div>

      {/* ── Filters ─────────────────────────────────────────────────── */}
      <div className="flex flex-wrap gap-3">
        {/* IP search */}
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-lg border flex-1 min-w-48"
          style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
        >
          <Search size={13} color="var(--col-text)" />
          <input
            type="text"
            placeholder="Search by IP address…"
            value={ipSearch}
            onChange={e => setIpSearch(e.target.value)}
            className="flex-1 bg-transparent mono text-xs outline-none placeholder:opacity-40"
            style={{ color: 'var(--col-text-hi)' }}
          />
        </div>

        {/* Attack type filter */}
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-lg border"
          style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
        >
          <Filter size={13} color="var(--col-text)" />
          <select
            value={attackFilter}
            onChange={e => { setAttackFilter(e.target.value); setPage(1) }}
            className="bg-transparent mono text-xs outline-none"
            style={{ color: 'var(--col-text-hi)' }}
          >
            <option value="">All attack types</option>
            {attackTypes.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        {(attackFilter || ipSearch) && (
          <button
            onClick={() => { setAttackFilter(''); setIpSearch('') }}
            className="px-3 py-2 rounded-lg border mono text-xs transition-colors"
            style={{
              background: 'rgba(248,113,113,0.08)',
              borderColor: 'rgba(248,113,113,0.2)',
              color: 'var(--col-red)',
            }}
          >
            Clear filters
          </button>
        )}
      </div>

      {/* ── Error state ──────────────────────────────────────────────── */}
      {isError && (
        <div
          className="flex items-center gap-3 p-4 rounded-xl border"
          style={{ background: 'rgba(248,113,113,0.06)', borderColor: 'rgba(248,113,113,0.2)' }}
        >
          <AlertTriangle size={16} color="var(--col-red)" />
          <div>
            <p className="mono text-xs font-semibold" style={{ color: 'var(--col-red)' }}>
              FETCH ERROR — backend unreachable
            </p>
            <p className="mono text-xs mt-0.5 opacity-70" style={{ color: 'var(--col-red)' }}>
              {(error as Error)?.message ?? 'Unknown error'}
            </p>
          </div>
        </div>
      )}

      {/* ── Table ───────────────────────────────────────────────────── */}
      <div
        className="rounded-xl border overflow-hidden"
        style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
      >
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr style={{ background: 'rgba(255,255,255,0.03)', borderBottom: '1px solid var(--col-border)' }}>
                {['TIME', 'SOURCE → DEST', 'ATTACK TYPE', 'RECON ERROR', 'DQN ACTION', 'STATUS', 'SEV'].map(col => (
                  <th
                    key={col}
                    className="px-4 py-3 text-left mono text-xs font-semibold"
                    style={{ color: 'var(--col-text-hi)' }}
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {/* Loading skeletons */}
              {isLoading && Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} />)}

              {/* Empty state */}
              {!isLoading && !isError && filteredItems.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-16 text-center">
                    <p className="mono text-sm" style={{ color: 'var(--col-cyan)' }}>
                      {'>'} No incidents found
                      <span className="blink ml-0.5">_</span>
                    </p>
                    <p className="text-xs mt-2" style={{ color: 'var(--col-text)' }}>
                      {attackFilter || ipSearch
                        ? 'Try adjusting your filters.'
                        : "The model hasn't flagged any anomalies yet."}
                    </p>
                  </td>
                </tr>
              )}

              {/* Data rows */}
              {filteredItems.map(incident => {
                const { variant, label } = severityFromAction(incident.dqn_action)
                const isExpanded = expandedId === incident.id

                return (
                  <>
                    <tr
                      key={incident.id}
                      onClick={() => setExpandedId(isExpanded ? null : incident.id)}
                      className="cursor-pointer transition-colors duration-100"
                      tabIndex={0}
                      onKeyDown={e => e.key === 'Enter' && setExpandedId(isExpanded ? null : incident.id)}
                      style={{ borderTop: '1px solid var(--col-border)' }}
                      onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)'}
                      onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = 'transparent'}
                    >
                      {/* Time */}
                      <td className="px-4 py-3 mono text-xs" style={{ color: 'var(--col-text)' }}>
                        {format(new Date(incident.timestamp), 'HH:mm:ss')}
                        <div className="text-[10px] opacity-40">
                          {format(new Date(incident.timestamp), 'MM-dd')}
                        </div>
                      </td>

                      {/* Source → Dest */}
                      <td className="px-4 py-3 mono text-xs max-w-[140px]">
                        <div className="truncate" style={{ color: 'var(--col-text-hi)' }} title={incident.source_ip}>
                          {incident.source_ip}
                        </div>
                        <div className="truncate opacity-40 text-[10px]" title={incident.dest_ip}>
                          → {incident.dest_ip}
                        </div>
                      </td>

                      {/* Attack type */}
                      <td className="px-4 py-3 text-xs" style={{ color: 'var(--col-amber)' }}>
                        {getAttackLabel(incident.attack_type_predicted) ?? <span className="opacity-30">Benign</span>}
                      </td>

                      {/* Recon error */}
                      <td className="px-4 py-3 mono text-xs" style={{ color: 'var(--col-cyan)' }}>
                        {formatReconError(incident.reconstruction_error)}
                      </td>

                      {/* DQN Action */}
                      <td className="px-4 py-3 text-xs" style={{ color: 'var(--col-text-hi)' }}>
                        <span className="mono">{incident.dqn_action ?? '—'}</span>
                      </td>

                      {/* Status */}
                      <td className="px-4 py-3">
                        <span
                          className="mono text-[10px] px-1.5 py-0.5 rounded"
                          style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--col-text)' }}
                        >
                          {incident.action_status}
                        </span>
                      </td>

                      {/* Severity + expand icon */}
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-1.5">
                          <StatusPill variant={variant} dot={false}>{label}</StatusPill>
                          {isExpanded
                            ? <ChevronUp size={12} color="var(--col-text)" />
                            : <ChevronDown size={12} color="var(--col-text)" />
                          }
                        </div>
                      </td>
                    </tr>

                    {/* Expanded detail */}
                    {isExpanded && <ExpandedDetail incident={incident} />}
                  </>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Pagination ───────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <p className="mono text-xs" style={{ color: 'var(--col-text)' }}>
          {data
            ? `Page ${data.page} of ${totalPages} · ${data.total} incidents total`
            : 'Loading…'
          }
        </p>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page <= 1 || isLoading}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg border mono text-xs transition-all disabled:opacity-30"
            style={{
              background: 'var(--col-surface)',
              borderColor: 'var(--col-border)',
              color: 'var(--col-text)',
            }}
          >
            <ChevronLeft size={13} /> Prev
          </button>

          <div className="flex items-center gap-1">
            {Array.from({ length: Math.min(totalPages, 7) }).map((_, i) => {
              const p = i + 1
              return (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  className="w-7 h-7 rounded mono text-xs transition-all"
                  style={{
                    background: p === page ? 'var(--col-cyan-dim)' : 'transparent',
                    border: `1px solid ${p === page ? 'rgba(56,189,248,0.3)' : 'transparent'}`,
                    color: p === page ? 'var(--col-cyan)' : 'var(--col-text)',
                  }}
                >
                  {p}
                </button>
              )
            })}
          </div>

          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages || isLoading}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg border mono text-xs transition-all disabled:opacity-30"
            style={{
              background: 'var(--col-surface)',
              borderColor: 'var(--col-border)',
              color: 'var(--col-text)',
            }}
          >
            Next <ChevronRight size={13} />
          </button>
        </div>
      </div>

    </div>
  )
}
