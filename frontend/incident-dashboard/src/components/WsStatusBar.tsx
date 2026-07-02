/**
 * src/components/WsStatusBar.tsx
 * ────────────────────────────────
 * A slim banner that shows the live WebSocket connection status.
 * Renders a coloured pulsing dot + label matching the current WsStatus.
 */

import type { WsStatus } from '../hooks/useWebSocket'

interface Props {
  status: WsStatus
  messageCount: number
}

const CONFIG: Record<WsStatus, { dot: string; label: string; bg: string }> = {
  connecting:   { dot: 'bg-amber-400 animate-pulse',  label: 'Connecting…',    bg: 'bg-amber-400/10 border-amber-400/30' },
  connected:    { dot: 'bg-emerald-400 animate-pulse', label: 'Connected',      bg: 'bg-emerald-400/10 border-emerald-400/30' },
  disconnected: { dot: 'bg-slate-400',                 label: 'Disconnected — retrying…', bg: 'bg-slate-400/10 border-slate-400/30' },
  error:        { dot: 'bg-red-400 animate-pulse',     label: 'Connection error — retrying…', bg: 'bg-red-400/10 border-red-400/30' },
}

export function WsStatusBar({ status, messageCount }: Props) {
  const { dot, label, bg } = CONFIG[status]

  return (
    <div className={`flex items-center gap-3 px-4 py-2 rounded-lg border text-sm font-mono ${bg}`}>
      {/* Pulsing dot */}
      <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${dot}`} />

      {/* Status label */}
      <span className="text-white/80">
        WebSocket&nbsp;
        <span className="text-white font-semibold">{label}</span>
      </span>

      {/* Message counter badge */}
      <span className="ml-auto text-white/50 text-xs">
        {messageCount} message{messageCount !== 1 ? 's' : ''} received
      </span>

      {/* URL hint */}
      <span className="hidden md:inline text-white/30 text-xs">
        ws://localhost:8000/ws/connect
      </span>
    </div>
  )
}
