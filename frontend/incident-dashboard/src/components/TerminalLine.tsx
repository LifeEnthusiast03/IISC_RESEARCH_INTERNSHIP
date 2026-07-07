/**
 * src/components/TerminalLine.tsx
 * ────────────────────────────────
 * One line of terminal output with timestamp + type-coded color.
 * Used in TerminalPage to render each WebSocket message.
 */

import { useState, useEffect } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import type { FormattedLog } from '../lib/logFormatter'

interface Props {
  log: FormattedLog
  animate?: boolean
}

const TYPE_COLOR: Record<string, string> = {
  info:      'var(--col-cyan)',
  anomaly:   'var(--col-red)',
  warning:   'var(--col-amber)',
  error:     'var(--col-red)',
  action:    'var(--col-amber)',
  system:    'var(--col-text)',
  connected: 'var(--col-green)',
}

export function TerminalLine({ log, animate = false }: Props) {
  const color = TYPE_COLOR[log.level] ?? TYPE_COLOR.info
  const [expanded, setExpanded] = useState(false)

  const [displayedContent, setDisplayedContent] = useState(animate ? '' : log.text)

  useEffect(() => {
    if (!animate) {
      setDisplayedContent(log.text)
      return
    }
    setDisplayedContent('')
    let i = 0
    const text = log.text
    
    const totalDurationMs = 300
    const frameMs = 16
    const totalFrames = totalDurationMs / frameMs
    const charsPerFrame = Math.max(1, Math.ceil(text.length / totalFrames))

    const interval = setInterval(() => {
      i += charsPerFrame
      setDisplayedContent(text.slice(0, i))
      if (i >= text.length) clearInterval(interval)
    }, frameMs)

    return () => clearInterval(interval)
  }, [log.text, animate])

  return (
    <div className="font-[family-name:var(--font-mono)] text-xs leading-relaxed">
      <div 
        className="flex items-start gap-2 group cursor-pointer transition-colors hover:bg-[rgba(255,255,255,0.02)] py-0.5 rounded-md px-1 -mx-1"
        onClick={() => setExpanded(p => !p)}
      >
        {/* Timestamp */}
        <span className="shrink-0 opacity-40 mt-0.5" style={{ color: 'var(--col-cyan)' }}>
          {log.timeLabel}
        </span>
        {/* Type prefix */}
        <span className="shrink-0 font-semibold mt-0.5" style={{ color }}>
          [{log.level.toUpperCase()}]
        </span>
        {/* Content */}
        <span className="break-all whitespace-pre-wrap flex-1 mt-0.5" style={{ color: log.level === 'error' ? 'var(--col-red)' : color, fontWeight: log.level === 'error' ? 'bold' : 'normal' }}>
          {displayedContent}
        </span>
        {/* Expand toggle */}
        <span className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity mt-0.5" style={{ color: 'var(--col-text)' }}>
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </span>
      </div>
      
      {/* Expanded Raw JSON */}
      {expanded && log.raw && (
        <div className="pl-[104px] pr-4 py-2">
          <pre 
            className="p-3 rounded-lg overflow-x-auto text-[10px]" 
            style={{ 
              background: 'rgba(0,0,0,0.4)', 
              border: '1px solid rgba(255,255,255,0.06)',
              color: 'var(--col-text)' 
            }}
          >
            {typeof log.raw === 'string' ? log.raw : JSON.stringify(log.raw, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
