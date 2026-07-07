/**
 * src/pages/TerminalPage.tsx
 * ──────────────────────────
 * Full-height terminal emulator fed by the shared WebSocket connection.
 * Reads from WsContext — does NOT open a second socket.
 *
 * Features:
 * - Auto-scroll to bottom on new messages (unless paused)
 * - Pause / Resume toggle
 * - Clear button
 * - Color-coded lines by message type/severity
 * - Blinking cursor at bottom
 */

import { useEffect, useRef, useState } from 'react'
import { Terminal, Wifi, WifiOff, Trash2, PauseCircle, PlayCircle } from 'lucide-react'
import { useWsContext } from '../contexts/WsContext'
import { TerminalLine, type LineType } from '../components/TerminalLine'
import { StatusPill } from '../components/StatusPill'
import { WS_BASE_URL } from '../lib/api'
import type { WsStatus } from '../hooks/useWebSocket'

const WS_URL = `${WS_BASE_URL}/ws/connect`

// ── Helpers ───────────────────────────────────────────────────────────────────

function resolveLineType(data: unknown): LineType {
  if (typeof data !== 'object' || data === null) return 'info'
  const d = data as Record<string, unknown>
  const t = (d.type ?? d.event ?? d.severity ?? '') as string
  if (['anomaly', 'threat', 'intrusion'].includes(t.toLowerCase())) return 'anomaly'
  if (['alert', 'critical', 'warning'].includes(t.toLowerCase()))    return 'alert'
  if (['action', 'remediation', 'block'].includes(t.toLowerCase()))  return 'action'
  if (['connected', 'success', 'ok'].includes(t.toLowerCase()))      return 'connected'
  if (d.is_anomaly === true) return 'anomaly'
  return 'info'
}

function formatPayload(data: unknown): string {
  if (typeof data === 'string') return data
  return JSON.stringify(data, null, 2)
}

function wsVariant(status: WsStatus) {
  if (status === 'connected')   return 'green'  as const
  if (status === 'connecting')  return 'amber'  as const
  if (status === 'error')       return 'red'    as const
  return 'slate' as const
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function TerminalPage() {
  const { status, messages, clearMessages } = useWsContext()
  const [paused, setPaused] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom whenever messages change (unless paused)
  useEffect(() => {
    if (!paused && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, paused])

  return (
    <div
      className="flex flex-col flex-1 min-h-0"
      style={{
        background: '#000',
      }}
    >
      {/* ── Terminal header bar ─────────────────────────────────────── */}
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
            {status === 'connected'   ? 'CONNECTED' :
             status === 'connecting'  ? 'CONNECTING' :
             status === 'error'       ? 'ERROR' : 'OFFLINE'}
          </StatusPill>
        </div>
      </div>

      {/* ── Controls row ─────────────────────────────────────────────── */}
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
            : <><PauseCircle size={13} /> PAUSE</>
          }
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
            : <WifiOff size={12} color="var(--col-red)" />
          }
          <span className="opacity-50">ws/connect</span>
        </div>
      </div>

      {/* ── Terminal body ─────────────────────────────────────────────── */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-4 space-y-0.5"
        style={{ background: '#020408' }}
      >
        {/* System boot line */}
        <TerminalLine timestamp={new Date()} type="system">
          ThreatSentinel IDS Terminal — {WS_URL}
        </TerminalLine>

        {/* Connection status line */}
        <TerminalLine timestamp={new Date()} type={status === 'connected' ? 'connected' : 'system'}>
          WebSocket {status === 'connected' ? 'CONNECTED' : status.toUpperCase()} — auto-reconnect with exponential backoff enabled
        </TerminalLine>

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

        {/* Message lines */}
        {messages.map(msg => (
          <TerminalLine
            key={msg.id}
            timestamp={msg.receivedAt}
            type={resolveLineType(msg.data)}
          >
            {formatPayload(msg.data)}
          </TerminalLine>
        ))}

        {/* Scroll anchor */}
        <div ref={bottomRef} />

        {/* Blinking cursor */}
        <div className="flex items-center gap-1 pt-2 mono text-xs" style={{ color: 'var(--col-cyan)' }}>
          <span style={{ color: 'rgba(56,189,248,0.5)' }}>&gt;</span>
          <span className="blink">_</span>
        </div>
      </div>
    </div>
  )
}
