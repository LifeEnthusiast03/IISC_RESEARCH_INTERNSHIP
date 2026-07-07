/**
 * src/components/TerminalLine.tsx
 * ────────────────────────────────
 * One line of terminal output with timestamp + type-coded color.
 * Used in TerminalPage to render each WebSocket message.
 */

import { format } from 'date-fns'
import type { ReactNode } from 'react'

export type LineType = 'info' | 'anomaly' | 'alert' | 'action' | 'system' | 'connected'

interface Props {
  timestamp: Date
  type?: LineType
  children: ReactNode
}

const TYPE_COLOR: Record<LineType, string> = {
  info:      'var(--col-cyan)',
  anomaly:   'var(--col-red)',
  alert:     'var(--col-red)',
  action:    'var(--col-amber)',
  system:    'var(--col-text)',
  connected: 'var(--col-green)',
}

const TYPE_PREFIX: Record<LineType, string> = {
  info:      '[INFO]    ',
  anomaly:   '[ANOMALY] ',
  alert:     '[ALERT]   ',
  action:    '[ACTION]  ',
  system:    '[SYS]     ',
  connected: '[CONN]    ',
}

export function TerminalLine({ timestamp, type = 'info', children }: Props) {
  const color = TYPE_COLOR[type]
  const prefix = TYPE_PREFIX[type]

  return (
    <div className="flex items-start gap-2 font-[family-name:var(--font-mono)] text-xs leading-relaxed">
      {/* Timestamp */}
      <span className="shrink-0 opacity-40" style={{ color: 'var(--col-cyan)' }}>
        {format(timestamp, 'HH:mm:ss.SSS')}
      </span>
      {/* Type prefix */}
      <span className="shrink-0 font-semibold" style={{ color }}>
        {prefix}
      </span>
      {/* Content */}
      <span className="break-all" style={{ color }}>
        {children}
      </span>
    </div>
  )
}
