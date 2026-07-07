/**
 * src/components/WsMessageFeed.tsx
 * ─────────────────────────────────
 * Scrollable real-time feed of every WebSocket message received from the
 * IDS backend.  Newest message at the top.
 *
 * Each message card shows:
 *  - Received timestamp
 *  - event type (if present in payload)
 *  - Formatted JSON payload
 */

import type { WsMessage } from '../hooks/useWebSocket'

interface Props {
  messages: WsMessage[]
  onClear: () => void
}

/** Colour coding by event type */
function eventColor(event?: string): string {
  switch (event) {
    case 'connected': return 'border-l-emerald-400 text-emerald-300'
    case 'anomaly':   return 'border-l-red-400    text-red-300'
    case 'alert':     return 'border-l-amber-400  text-amber-300'
    default:          return 'border-l-slate-500  text-slate-300'
  }
}

export function WsMessageFeed({ messages, onClear }: Props) {
  // Newest first
  const sorted = [...messages].reverse()

  return (
    <div className="flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-white font-semibold text-base tracking-wide">
          Live Message Feed
        </h2>
        {messages.length > 0 && (
          <button
            onClick={onClear}
            className="text-xs text-slate-400 hover:text-white transition-colors px-2 py-1 rounded border border-slate-600 hover:border-slate-400"
          >
            Clear
          </button>
        )}
      </div>

      {/* Empty state */}
      {messages.length === 0 && (
        <div className="flex flex-col items-center justify-center py-12 text-slate-500 gap-2">
          <span className="text-3xl">📡</span>
          <p className="text-sm">Waiting for messages from the server…</p>
        </div>
      )}

      {/* Message cards */}
      <div className="flex flex-col gap-2 max-h-[480px] overflow-y-auto pr-1">
        {sorted.map((msg) => {
          const event = msg.level
          const colorCls = eventColor(event)

          return (
            <div
              key={msg.id}
              className={`border-l-2 pl-3 py-2 rounded-r bg-white/5 hover:bg-white/8 transition-colors ${colorCls}`}
            >
              {/* Top row: timestamp + event badge */}
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs text-slate-500 font-mono">
                  {msg.timeLabel}
                </span>
                {event && (
                  <span className={`text-xs font-bold uppercase tracking-wider ${colorCls}`}>
                    {event}
                  </span>
                )}
              </div>

              {/* Payload */}
              <pre className="text-xs text-slate-200 font-mono whitespace-pre-wrap break-words leading-relaxed">
                {msg.text}
              </pre>
            </div>
          )
        })}
      </div>
    </div>
  )
}
