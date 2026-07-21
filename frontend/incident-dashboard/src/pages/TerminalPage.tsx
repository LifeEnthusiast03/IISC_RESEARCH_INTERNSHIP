/**
 * src/pages/TerminalPage.tsx
 * ──────────────────────────
 * Full-height terminal emulator fed by the shared WebSocket connection.
 * Reads from WsContext — does NOT open a second socket.
 *
 * Features (original):
 * - Auto-scroll to bottom on new messages (unless paused)
 * - Pause / Resume toggle
 * - Clear button
 * - Color-coded lines by message type/severity
 * - Blinking cursor at bottom
 *
 * Phase 2 additions:
 * - Sticky filter bar: level-chip toggles + text search (Feature 4)
 * - Clickable Incident #ID → per-incident log filter (Feature 5)
 * - Auto-scroll toggle with manual-scroll detection (Feature 6)
 *
 * Phase 4 additions:
 * - Live mini-stats strip: anomaly/action counts + connection status (Feature 9)
 * - Max-width 1100px content constraint (Feature 10)
 */

import { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import {
  Terminal, Wifi, WifiOff, Trash2, PauseCircle, PlayCircle,
  Lock, Unlock, Search, X,
} from 'lucide-react'
import { useWsContext } from '../contexts/WsContext'
import { TerminalLine } from '../components/TerminalLine'
import { StatusPill } from '../components/StatusPill'
import { WS_BASE_URL } from '../lib/api'
import type { WsStatus } from '../hooks/useWebSocket'
import type { LogLevel } from '../lib/logFormatter'

import { format } from 'date-fns'

const WS_URL = `${WS_BASE_URL}/ws/connect`

// ── Helpers ───────────────────────────────────────────────────────────────────

function wsVariant(status: WsStatus) {
  if (status === 'connected')  return 'green'  as const
  if (status === 'connecting') return 'amber'  as const
  if (status === 'error')      return 'red'    as const
  return 'slate' as const
}

// ── Feature 4 — Chip definitions ─────────────────────────────────────────────
// Each chip controls one or more LogLevel values (see level-mapping note in plan)

interface ChipDef {
  label: string
  levels: LogLevel[]
  activeColor: string
  activeBg: string
  activeBorder: string
}

const CHIPS: ChipDef[] = [
  {
    label: 'INFO',
    levels: ['info', 'connected'],
    activeColor: 'var(--col-cyan)',
    activeBg: 'rgba(56,189,248,0.12)',
    activeBorder: 'rgba(56,189,248,0.35)',
  },
  {
    label: 'ACTION',
    levels: ['action'],
    activeColor: 'var(--col-amber)',
    activeBg: 'rgba(251,191,36,0.12)',
    activeBorder: 'rgba(251,191,36,0.35)',
  },
  {
    label: 'ANOMALY',
    levels: ['anomaly', 'warning', 'error'],
    activeColor: 'var(--col-red)',
    activeBg: 'rgba(248,113,113,0.12)',
    activeBorder: 'rgba(248,113,113,0.35)',
  },
  {
    label: 'SYSTEM',
    levels: ['system'],
    activeColor: 'var(--col-text)',
    activeBg: 'rgba(148,163,184,0.08)',
    activeBorder: 'rgba(148,163,184,0.25)',
  },
]

// All levels covered by the chips — used as the default "all on" set
const ALL_LEVELS = new Set<LogLevel>(CHIPS.flatMap(c => c.levels))

// ── Page ──────────────────────────────────────────────────────────────────────

export default function TerminalPage() {
  const { status, messages, clearMessages } = useWsContext()

  // ── Legacy pause (kept for UX continuity) ────────────────────────────────
  const [paused, setPaused] = useState(false)

  // ── Feature 6 — Auto-scroll ───────────────────────────────────────────────
  const [autoScroll, setAutoScroll] = useState(true)

  // ── Feature 4 — Filter state ──────────────────────────────────────────────
  const [activeFilters, setActiveFilters] = useState<Set<LogLevel>>(
    new Set(ALL_LEVELS),
  )
  const [searchText, setSearchText] = useState('')

  // ── Feature 5 — Per-incident filter ──────────────────────────────────────
  const [incidentFilter, setIncidentFilter] = useState<string | null>(null)

  // ── Refs ──────────────────────────────────────────────────────────────────
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // ── Feature 6 — Detect manual scroll-up → disable auto-scroll ────────────
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onScroll = () => {
      const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
      if (distFromBottom > 60) {
        setAutoScroll(false)
      }
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => el.removeEventListener('scroll', onScroll)
  }, [])

  // ── Auto-scroll effect ────────────────────────────────────────────────────
  useEffect(() => {
    if (!paused && autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, paused, autoScroll])

  // ── Feature 4+5 — Filtered visible messages ───────────────────────────────
  const visibleMessages = useMemo(() => {
    const lowerSearch = searchText.toLowerCase()
    return messages.filter(m => {
      // Level filter
      if (!activeFilters.has(m.level)) return false
      // Per-incident filter (Feature 5)
      if (incidentFilter !== null) {
        if (m.incidentId?.toString() !== incidentFilter) return false
      }
      // Text search
      if (lowerSearch && !m.text.toLowerCase().includes(lowerSearch)) return false
      return true
    })
  }, [messages, activeFilters, searchText, incidentFilter])

  // ── Feature 9 — Live counts (from ALL messages, not filtered) ────────────
  const anomalyCount = useMemo(
    () => messages.filter(m => ['anomaly', 'warning', 'error'].includes(m.level)).length,
    [messages],
  )
  const actionCount = useMemo(
    () => messages.filter(m => m.level === 'action').length,
    [messages],
  )

  // ── Feature 4 — Chip toggle handler ──────────────────────────────────────
  const toggleChip = useCallback((chip: ChipDef) => {
    setActiveFilters(prev => {
      const next = new Set(prev)
      const allOn = chip.levels.every(l => prev.has(l))
      if (allOn) {
        chip.levels.forEach(l => next.delete(l))
      } else {
        chip.levels.forEach(l => next.add(l))
      }
      return next
    })
  }, [])

  // ── Feature 5 — Incident click handler ───────────────────────────────────
  const handleIncidentClick = useCallback((id: string) => {
    setIncidentFilter(prev => (prev === id ? null : id))
  }, [])

  // ── Feature 6 — Re-enable auto-scroll ────────────────────────────────────
  const enableAutoScroll = () => {
    setAutoScroll(true)
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  // ── Static boot / status lines ───────────────────────────────────────────
  const bootLog = {
    id: 'boot',
    level: 'system' as const,
    timestamp: new Date().toISOString(),
    timeLabel: format(new Date(), 'HH:mm:ss.SSS'),
    text: `ThreatSentinel IDS Terminal — ${WS_URL}`,
    raw: null,
  }

  const statusLog = {
    id: 'status',
    level: status === 'connected' ? 'connected' as const : 'system' as const,
    timestamp: new Date().toISOString(),
    timeLabel: format(new Date(), 'HH:mm:ss.SSS'),
    text: `WebSocket ${status === 'connected' ? 'CONNECTED' : status.toUpperCase()} — auto-reconnect with exponential backoff enabled`,
    raw: null,
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Shared inline-style helpers for filter chips
  // ─────────────────────────────────────────────────────────────────────────

  function chipStyle(chip: ChipDef): React.CSSProperties {
    const active = chip.levels.some(l => activeFilters.has(l))
    return active
      ? {
          background: chip.activeBg,
          border: `1px solid ${chip.activeBorder}`,
          color: chip.activeColor,
        }
      : {
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid rgba(255,255,255,0.08)',
          color: 'rgba(148,163,184,0.45)',
        }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div
      className="flex flex-col flex-1 min-h-0"
      style={{ background: '#000' }}
    >
      {/* ── Terminal header bar ─────────────────────────────────────────── */}
      <div
        className="flex flex-wrap items-center gap-3 px-4 py-3 border-b"
        style={{ borderColor: 'rgba(56,189,248,0.15)', background: 'rgba(0,0,0,0.6)' }}
      >
        <div className="flex items-center gap-2">
          <Terminal size={14} color="var(--col-cyan)" />
          <span className="mono text-xs font-semibold" style={{ color: 'var(--col-cyan)' }}>
            THREAT FEED TERMINAL
          </span>
        </div>

        <span className="mono text-xs opacity-40" style={{ color: 'var(--col-text)' }}>
          {WS_URL}
        </span>

        <div className="ml-auto flex items-center gap-2 flex-wrap">
          {/* Message count */}
          <span
            className="mono text-xs px-2 py-0.5 rounded"
            style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--col-text)' }}
          >
            {messages.length} lines
          </span>

          {/* Connection status */}
          <StatusPill
            variant={wsVariant(status)}
            pulse={status === 'connected' || status === 'connecting'}
          >
            {status === 'connected'  ? 'CONNECTED'  :
             status === 'connecting' ? 'CONNECTING' :
             status === 'error'      ? 'ERROR'      : 'OFFLINE'}
          </StatusPill>
        </div>
      </div>

      {/* ── Controls row ────────────────────────────────────────────────── */}
      <div
        className="flex items-center gap-2 px-4 py-2 border-b"
        style={{ borderColor: 'rgba(255,255,255,0.05)', background: 'rgba(0,0,0,0.4)' }}
      >
        <button
          onClick={() => setPaused(p => !p)}
          className="flex items-center gap-1.5 px-3 py-1 rounded text-xs mono transition-colors duration-150"
          style={{
            background: paused ? 'rgba(251,191,36,0.12)' : 'rgba(255,255,255,0.05)',
            border: `1px solid ${paused ? 'rgba(251,191,36,0.3)' : 'rgba(255,255,255,0.1)'}`,
            color: paused ? 'var(--col-amber)' : 'var(--col-text)',
          }}
        >
          {paused
            ? <><PlayCircle size={13} /> RESUME</>
            : <><PauseCircle size={13} /> PAUSE</>}
        </button>

        <button
          onClick={clearMessages}
          className="flex items-center gap-1.5 px-3 py-1 rounded text-xs mono transition-colors duration-150"
          style={{
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.1)',
            color: 'var(--col-text)',
          }}
        >
          <Trash2 size={13} /> CLEAR
        </button>

        {paused && (
          <span className="mono text-xs" style={{ color: 'var(--col-amber)' }}>
            ⏸ Scroll paused — {messages.length} buffered
          </span>
        )}

        <div className="ml-auto flex items-center gap-1.5 mono text-xs" style={{ color: 'var(--col-text)' }}>
          {status === 'connected'
            ? <Wifi size={12} color="var(--col-green)" />
            : <WifiOff size={12} color="var(--col-red)" />}
          <span className="opacity-50">ws/connect</span>
        </div>
      </div>

      {/* ── Feature 4+5+6 — Sticky filter bar ──────────────────────────── */}
      <div
        className="sticky top-0 z-10 flex flex-wrap items-center gap-2 px-4 py-2 border-b"
        style={{
          borderColor: 'rgba(255,255,255,0.06)',
          background: 'rgba(2,4,8,0.95)',
          backdropFilter: 'blur(8px)',
        }}
      >
        {/* Inner content constrained to max-width (Feature 10) */}
        <div className="flex flex-wrap items-center gap-2 w-full" style={{ maxWidth: '1100px', margin: '0 auto' }}>

          {/* Level-filter chips */}
          {CHIPS.map(chip => (
            <button
              key={chip.label}
              onClick={() => toggleChip(chip)}
              className="px-2.5 py-0.5 rounded text-[11px] font-semibold mono transition-all duration-150"
              style={chipStyle(chip)}
            >
              {chip.label}
            </button>
          ))}

          {/* Feature 5 — Active incident filter badge */}
          {incidentFilter !== null && (
            <span
              className="flex items-center gap-1 px-2.5 py-0.5 rounded text-[11px] font-semibold mono"
              style={{
                background: 'rgba(56,189,248,0.12)',
                border: '1px solid rgba(56,189,248,0.35)',
                color: 'var(--col-cyan)',
              }}
            >
              Incident #{incidentFilter}
              <X
                size={11}
                className="cursor-pointer opacity-70 hover:opacity-100"
                onClick={() => setIncidentFilter(null)}
              />
            </span>
          )}

          {/* Text search */}
          <div className="relative flex items-center ml-1">
            <Search
              size={12}
              className="absolute left-2 pointer-events-none"
              style={{ color: 'rgba(148,163,184,0.4)' }}
            />
            <input
              type="text"
              placeholder="Search logs…"
              value={searchText}
              onChange={e => setSearchText(e.target.value)}
              className="mono text-[11px] pl-6 pr-2 py-0.5 rounded outline-none w-40 focus:w-52 transition-all duration-200"
              style={{
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)',
                color: 'var(--col-text)',
              }}
            />
            {searchText && (
              <X
                size={11}
                className="absolute right-2 cursor-pointer opacity-60 hover:opacity-100"
                style={{ color: 'var(--col-text)' }}
                onClick={() => setSearchText('')}
              />
            )}
          </div>

          {/* Feature 6 — Auto-scroll toggle */}
          <button
            onClick={autoScroll ? () => setAutoScroll(false) : enableAutoScroll}
            className="ml-auto flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[11px] font-semibold mono transition-all duration-150"
            style={
              autoScroll
                ? {
                    background: 'rgba(52,211,153,0.1)',
                    border: '1px solid rgba(52,211,153,0.3)',
                    color: 'var(--col-green)',
                  }
                : {
                    background: 'rgba(251,191,36,0.1)',
                    border: '1px solid rgba(251,191,36,0.3)',
                    color: 'var(--col-amber)',
                  }
            }
          >
            {autoScroll ? <Lock size={11} /> : <Unlock size={11} />}
            Auto-scroll: {autoScroll ? 'ON' : 'OFF'}
          </button>

          {/* Visible count */}
          {(searchText || incidentFilter || activeFilters.size < ALL_LEVELS.size) && (
            <span className="mono text-[11px] opacity-50" style={{ color: 'var(--col-text)' }}>
              {visibleMessages.length}/{messages.length}
            </span>
          )}
        </div>
      </div>

      {/* ── Feature 9 — Live mini-stats strip ───────────────────────────── */}
      <div
        className="flex items-center gap-4 px-4 py-1.5 border-b"
        style={{
          borderColor: 'rgba(255,255,255,0.04)',
          background: 'rgba(0,0,0,0.3)',
        }}
      >
        <div style={{ maxWidth: '1100px', margin: '0 auto', width: '100%' }}
          className="flex items-center gap-4 mono text-[11px]">

          <span style={{ color: 'rgba(148,163,184,0.5)' }}>LIVE</span>

          <span>
            <span style={{ color: 'rgba(148,163,184,0.5)' }}>Anomalies: </span>
            <span style={{ color: anomalyCount > 0 ? 'var(--col-red)' : 'var(--col-text)' }} className="font-semibold">
              {anomalyCount}
            </span>
          </span>

          <span>
            <span style={{ color: 'rgba(148,163,184,0.5)' }}>Actions: </span>
            <span style={{ color: actionCount > 0 ? 'var(--col-amber)' : 'var(--col-text)' }} className="font-semibold">
              {actionCount}
            </span>
          </span>

          <span>
            <span style={{ color: 'rgba(148,163,184,0.5)' }}>Total: </span>
            <span className="font-semibold" style={{ color: 'var(--col-text)' }}>
              {messages.length}
            </span>
          </span>

          <span className="ml-auto flex items-center gap-1.5">
            {status === 'connected'
              ? <><Wifi size={11} color="var(--col-green)" /><span style={{ color: 'var(--col-green)' }}>Connected</span></>
              : <><WifiOff size={11} color="var(--col-red)" /><span style={{ color: 'var(--col-red)' }}>Disconnected</span></>}
          </span>
        </div>
      </div>

      {/* ── Terminal body ────────────────────────────────────────────────── */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto py-4"
        style={{ background: '#020408' }}
      >
        {/* Feature 10 — Content constrained to 1100px, centered */}
        <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '0 16px' }}>

          {/* System boot line */}
          <TerminalLine log={bootLog} />

          {/* Connection status line */}
          <TerminalLine log={statusLog} />

          {/* Divider */}
          <div className="py-1">
            <div className="glow-divider" />
          </div>

          {/* Empty state */}
          {messages.length === 0 && (
            <div className="py-8 text-center">
              <p className="mono text-xs" style={{ color: 'rgba(56,189,248,0.4)' }}>
                Waiting for events from the backend
                <span className="blink ml-0.5">_</span>
              </p>
              <p className="mono text-xs mt-2" style={{ color: 'rgba(148,163,184,0.3)' }}>
                Start the simulator or POST to /predict to see live alerts
              </p>
            </div>
          )}

          {/* No results after filtering */}
          {messages.length > 0 && visibleMessages.length === 0 && (
            <div className="py-6 text-center">
              <p className="mono text-xs" style={{ color: 'rgba(148,163,184,0.3)' }}>
                No messages match the current filter
              </p>
            </div>
          )}

          {/* Message lines — filtered view */}
          <div className="space-y-0.5">
            {visibleMessages.map(msg => (
              <TerminalLine
                key={msg.id}
                log={msg}
                animate={Date.now() - new Date(msg.timestamp).getTime() < 1000}
                onIncidentClick={handleIncidentClick}
              />
            ))}
          </div>

          {/* Scroll anchor */}
          <div ref={bottomRef} />

          {/* Blinking cursor */}
          <div className="flex items-center gap-1 pt-2 mono text-xs" style={{ color: 'var(--col-cyan)' }}>
            <span style={{ color: 'rgba(56,189,248,0.5)' }}>&gt;</span>
            <span className="blink">_</span>
          </div>

        </div>
      </div>
    </div>
  )
}
