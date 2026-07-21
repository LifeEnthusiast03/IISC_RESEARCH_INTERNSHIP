/**
 * src/components/TerminalLine.tsx
 * ────────────────────────────────
 * One line of terminal output with timestamp + type-coded color.
 * Used in TerminalPage to render each WebSocket message.
 *
 * Phase 1 additions:
 *   - 3px colored left-border accent keyed off log.level (Feature 1)
 *   - Collapsible multi-line INCIDENT RESPONSE SUMMARY (Feature 2)
 *   - 8px indent + thin connector for grouped/sub-event logs (Feature 3)
 *
 * Phase 3 additions:
 *   - Confidence color pill (Feature 7)
 *   - Hover-reveal copy-to-clipboard icon + "Copied!" tooltip (Feature 8)
 */

import { useState, useEffect, type ReactNode } from 'react'
import { ChevronDown, ChevronUp, Copy } from 'lucide-react'
import type { FormattedLog } from '../lib/logFormatter'

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  log: FormattedLog
  animate?: boolean
  /** Called with the raw incident number string (e.g. "3") when user clicks
   *  an "Incident #N" span in the rendered text (Phase 2 / Feature 5). */
  onIncidentClick?: (id: string) => void
}

// ── Color maps ────────────────────────────────────────────────────────────────

const TYPE_COLOR: Record<string, string> = {
  info:      'var(--col-cyan)',
  anomaly:   'var(--col-red)',
  warning:   'var(--col-amber)',
  error:     'var(--col-red)',
  action:    'var(--col-amber)',
  system:    'var(--col-text)',
  connected: 'var(--col-green)',
}

/** Feature 1 — 3px left-border accent colors */
const BORDER_COLOR: Record<string, string> = {
  anomaly:   'var(--col-red)',
  warning:   'var(--col-amber)',
  error:     'var(--col-red)',
  action:    'var(--col-amber)',
  info:      'var(--col-cyan)',
  connected: 'var(--col-green)',
  system:    'rgba(148,163,184,0.25)',
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Feature 2 — Detect multi-line agent-summary logs.
 * agent_response and normal_traffic_response produce:
 *   "✅ AgentName [Incident #N]\n<full markdown-stripped body>"
 *   "🟢 NormalTrafficBaselineAgent [Incident #N] | ...\n<body>"
 * We collapse these to their first line by default.
 */
function isMultiLineSummary(text: string): boolean {
  return (
    text.includes('\n') &&
    (text.startsWith('✅') || text.startsWith('🟢'))
  )
}

/**
 * Feature 5 helper — Split text on "Incident #\d+" and wrap matches in a
 * clickable <span>. Returns an array of strings and JSX elements.
 * Falls back to raw string when no callback is provided.
 */
function renderWithIncidentLinks(
  text: string,
  onIncidentClick?: (id: string) => void,
): ReactNode {
  if (!onIncidentClick) return text
  const parts = text.split(/(Incident #\d+)/g)
  return parts.map((part, i) => {
    const m = part.match(/^Incident #(\d+)$/)
    if (m) {
      return (
        <span
          key={i}
          className="underline decoration-dotted cursor-pointer opacity-80 hover:opacity-100 transition-opacity"
          style={{ color: 'var(--col-cyan)' }}
          onClick={(e) => { e.stopPropagation(); onIncidentClick(m[1]) }}
        >
          {part}
        </span>
      )
    }
    return <span key={i}>{part}</span>
  })
}

// ── Sub-components ────────────────────────────────────────────────────────────

/** Feature 7 — Confidence color pill */
function ConfidencePill({ value }: { value: number }) {
  const [color, bg, border] =
    value >= 0.85
      ? ['var(--col-green)', 'rgba(52,211,153,0.15)',  'rgba(52,211,153,0.3)' ]
      : value >= 0.5
      ? ['var(--col-amber)', 'rgba(251,191,36,0.15)',  'rgba(251,191,36,0.3)' ]
      : ['var(--col-red)',   'rgba(248,113,113,0.15)', 'rgba(248,113,113,0.3)']
  return (
    <span
      className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold shrink-0 mt-0.5"
      style={{ color, background: bg, border: `1px solid ${border}` }}
    >
      {value.toFixed(2)}
    </span>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function TerminalLine({ log, animate = false, onIncidentClick }: Props) {
  const color       = TYPE_COLOR[log.level]  ?? TYPE_COLOR.info
  const borderColor = BORDER_COLOR[log.level] ?? 'rgba(148,163,184,0.25)'
  const isSummary   = isMultiLineSummary(log.text)
  const summaryLine = isSummary ? log.text.split('\n')[0] : null

  // Expand/collapse state — drives two different expansions:
  //   • summary lines  → shows full text in <pre> below the header row
  //   • regular lines  → shows raw JSON in <pre> below the row
  const [expanded, setExpanded] = useState(false)

  // Feature 8 — copy-to-clipboard feedback
  const [copied, setCopied] = useState(false)

  // Typewriter animation state
  const [displayedContent, setDisplayedContent] = useState(
    animate ? '' : log.text,
  )

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

  // Feature 8 — copy raw payload to clipboard
  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation()
    const payload =
      log.raw != null
        ? (typeof log.raw === 'string' ? log.raw : JSON.stringify(log.raw, null, 2))
        : log.text
    navigator.clipboard.writeText(payload).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }).catch(() => {/* clipboard not available */})
  }

  // Decide what text to show in the inline content span
  // Summary lines always show their first line inline; the rest goes in the
  // expanded <pre> block below — so typewriter runs on a short string.
  const inlineText = isSummary ? (summaryLine ?? log.text) : displayedContent

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    // Feature 3 — grouped sub-event indentation + thin connector line
    <div
      className="font-[family-name:var(--font-mono)] text-xs leading-relaxed"
      style={
        log.isGrouped
          ? {
              marginLeft: '12px',
              paddingLeft: '8px',
              borderLeft: '2px solid rgba(56,189,248,0.18)',
            }
          : undefined
      }
    >
      {/* ── Row ── */}
      <div
        className="flex items-start gap-2 group cursor-pointer transition-colors hover:bg-[rgba(255,255,255,0.02)] py-0.5 rounded-sm relative"
        style={{
          // Feature 1 — left-border accent
          borderLeft: `3px solid ${borderColor}`,
          paddingLeft: '8px',
        }}
        onClick={() => setExpanded(p => !p)}
      >
        {/* Timestamp */}
        <span className="shrink-0 opacity-40 mt-0.5" style={{ color: 'var(--col-cyan)' }}>
          {log.timeLabel}
        </span>

        {/* Level badge */}
        <span className="shrink-0 font-semibold mt-0.5" style={{ color }}>
          [{log.level.toUpperCase()}]
        </span>

        {/* Feature 7 — Confidence pill (attack-classifier events) */}
        {log.confidence !== undefined && (
          <ConfidencePill value={log.confidence} />
        )}

        {/* Main text content */}
        <span
          className="break-all whitespace-pre-wrap flex-1 mt-0.5"
          style={{
            color: log.level === 'error' ? 'var(--col-red)' : color,
            fontWeight: log.level === 'error' ? 'bold' : 'normal',
          }}
        >
          {/* Animate only for non-summary regular lines */}
          {renderWithIncidentLinks(inlineText, onIncidentClick)}
          {/* Summary-collapsed indicator */}
          {isSummary && !expanded && (
            <span className="opacity-30 ml-1 text-[10px]">[+expand]</span>
          )}
        </span>

        {/* Feature 8 — Hover-reveal copy button */}
        <span
          className="shrink-0 relative opacity-0 group-hover:opacity-100 transition-opacity mt-0.5"
          onClick={handleCopy}
          title="Copy raw JSON"
        >
          {/* "Copied!" tooltip */}
          {copied && (
            <span
              className="absolute -top-7 right-0 whitespace-nowrap px-1.5 py-0.5 rounded text-[10px] font-semibold pointer-events-none"
              style={{
                background: 'rgba(52,211,153,0.15)',
                border: '1px solid rgba(52,211,153,0.3)',
                color: 'var(--col-green)',
              }}
            >
              Copied!
            </span>
          )}
          <Copy size={13} style={{ color: 'var(--col-text)' }} />
        </span>

        {/* Expand/collapse chevron */}
        <span
          className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity mt-0.5"
          style={{ color: 'var(--col-text)' }}
        >
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </span>
      </div>

      {/* ── Expanded: Feature 2 — full summary text for multi-line agent responses ── */}
      {expanded && isSummary && (
        <div className="pl-[104px] pr-4 py-2">
          <pre
            className="p-3 rounded-lg overflow-x-auto text-[10px] whitespace-pre-wrap"
            style={{
              background: 'rgba(0,0,0,0.4)',
              border: '1px solid rgba(255,255,255,0.06)',
              color: 'var(--col-text)',
            }}
          >
            {log.text}
          </pre>
        </div>
      )}

      {/* ── Expanded: raw JSON for regular (non-summary) lines ── */}
      {expanded && !isSummary && log.raw != null && (
        <div className="pl-[104px] pr-4 py-2">
          <pre
            className="p-3 rounded-lg overflow-x-auto text-[10px]"
            style={{
              background: 'rgba(0,0,0,0.4)',
              border: '1px solid rgba(255,255,255,0.06)',
              color: 'var(--col-text)',
            }}
          >
            {typeof log.raw === 'string'
              ? log.raw
              : JSON.stringify(log.raw, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
