import { format } from 'date-fns'

/**
 * Convert a Markdown string into clean plain-text suitable for a terminal.
 * Handles: headings, bold, italic, inline-code, horizontal rules, ordered +
 * unordered lists, and trailing whitespace.
 */
function stripMarkdown(md: string): string {
  return md
    .split('\n')
    .map(line => {
      // H3 / H2 / H1 → UPPERCASE LABEL
      line = line.replace(/^###\s+(.*)/, (_, t) => `── ${t.toUpperCase()} ──`)
      line = line.replace(/^##\s+(.*)/, (_, t)  => `━━ ${t.toUpperCase()} ━━`)
      line = line.replace(/^#\s+(.*)/, (_, t)   => `▶ ${t.toUpperCase()}`)
      // Horizontal rules
      line = line.replace(/^[-*_]{3,}\s*$/, '─────────────────────────────')
      // Unordered list bullets
      line = line.replace(/^\s*[-*+]\s+/, '  • ')
      // Ordered list
      line = line.replace(/^\s*(\d+)\.\s+/, (_, n) => `  ${n}. `)
      // Bold+italic, bold, italic, inline-code → plain
      line = line.replace(/\*\*\*(.+?)\*\*\*/g, '$1')
      line = line.replace(/\*\*(.+?)\*\*/g,   '$1')
      line = line.replace(/\*(.+?)\*/g,       '$1')
      line = line.replace(/_(.+?)_/g,         '$1')
      line = line.replace(/`(.+?)`/g,         '$1')
      return line
    })
    .join('\n')
    // Collapse 3+ consecutive blank lines to 2
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export type LogLevel = 'info' | 'anomaly' | 'action' | 'warning' | 'error' | 'connected' | 'system'

export interface FormattedLog {
  id: string
  level: LogLevel
  timestamp: string      // raw ISO string
  timeLabel: string      // HH:mm:ss.SSS formatted
  text: string           // single-line human readable summary
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  raw: any               // the original parsed JSON object (or string if parse failed)
  // ── Optional extended fields (backward-compatible) ──────────────────────────
  incidentId?: string | number  // incident_id when present in payload
  isGrouped?:  boolean          // true for agent sub-event logs (indent + connector)
  confidence?: number           // attack-classifier confidence score (0–1)
}

function generateId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function formatLogMessage(rawMessage: string | object | any): FormattedLog {
  let parsed = rawMessage

  if (typeof rawMessage === 'string') {
    try {
      parsed = JSON.parse(rawMessage)
    } catch {
      // It's just a raw string
      parsed = rawMessage
    }
  }

  const id = generateId()
  const now = new Date()
  let timestampStr = now.toISOString()
  let dateObj = now

  if (parsed && typeof parsed === 'object') {
    if (parsed.timestamp) {
      timestampStr = parsed.timestamp
      dateObj = new Date(parsed.timestamp)
      // Fallback if invalid date
      if (isNaN(dateObj.getTime())) {
        dateObj = now
        timestampStr = now.toISOString()
      }
    }
  }

  const timeLabel = format(dateObj, 'HH:mm:ss.SSS')

  // Resolve level — extended with agent event types
  let level: LogLevel = 'info'
  if (parsed && typeof parsed === 'object') {
    const t = (parsed.type ?? parsed.event ?? parsed.severity ?? '').toLowerCase()
    if (['anomaly', 'threat', 'intrusion'].includes(t))                    level = 'anomaly'
    else if (['warning', 'alert', 'critical'].includes(t))                 level = 'warning'
    else if (['error', 'agent_error', 'normal_traffic_error'].includes(t)) level = 'error'
    else if (['action', 'remediation', 'block'].includes(t))               level = 'action'
    else if (['agent_invocation'].includes(t))                             level = 'action'
    else if (['agent_response'].includes(t))                               level = 'info'
    else if (['normal_traffic_response'].includes(t))                      level = 'connected'
    else if (['connected', 'success', 'ok'].includes(t))                   level = 'connected'
    else if (parsed.is_anomaly === true)                                   level = 'anomaly'
  }

  // Build human-readable text
  let text = ''
  if (typeof parsed === 'string') {
    text = parsed
  } else {
    const evt = (parsed.event ?? parsed.type ?? '').toLowerCase()

    // ── Agent broadcast events ───────────────────────────────────────────────
    if (evt === 'agent_invocation') {
      const id  = parsed.incident_id !== undefined ? `#${parsed.incident_id}` : ''
      const src = parsed.source_ip   ?? '?'
      const dst = parsed.dest_ip     ?? '?'
      const type   = parsed.attack_type ?? 'Unknown'
      const action = parsed.dqn_action  ?? '?'
      text = `🤖 Agent invoked${id ? ` for Incident ${id}` : ''} | ${src} → ${dst} | type=${type} | dqn=${action}`

    } else if (evt === 'agent_response') {
      const id    = parsed.incident_id  !== undefined ? `#${parsed.incident_id}` : ''
      const agent = parsed.handling_agent ?? 'Agent'
      const resp  = parsed.final_response ?? ''
      const header = `✅ ${agent}${id ? ` [Incident ${id}]` : ''}`
      // Strip markdown from the LLM response so the terminal shows clean text
      const cleanResp = resp ? stripMarkdown(resp) : ''
      text = cleanResp ? `${header}\n${cleanResp}` : header

    } else if (evt === 'agent_error') {
      const id  = parsed.incident_id !== undefined ? `Incident #${parsed.incident_id}` : ''
      const err = parsed.error ?? 'Unknown error'
      text = `❌ Agent pipeline failed${id ? ` — ${id}` : ''}: ${err}`

    } else if (evt === 'normal_traffic_response') {
      const id    = parsed.incident_id !== undefined ? `#${parsed.incident_id}` : ''
      const src   = parsed.source_ip ?? '?'
      const dst   = parsed.dest_ip   ?? '?'
      const recon = typeof parsed.recon_error === 'number' ? parsed.recon_error.toFixed(6) : (parsed.recon_error ?? '?')
      const resp  = parsed.final_response ?? ''
      const header = `🟢 NormalTrafficBaselineAgent${id ? ` [Incident ${id}]` : ''} | ${src} → ${dst} | recon=${recon}`
      const cleanResp = resp ? stripMarkdown(resp) : ''
      text = cleanResp ? `${header}\n${cleanResp}` : header

    } else if (evt === 'normal_traffic_error') {
      const id  = parsed.incident_id !== undefined ? `Incident #${parsed.incident_id}` : ''
      const err = parsed.error ?? 'Unknown error'
      text = `❌ Normal traffic agent failed${id ? ` — ${id}` : ''}: ${err}`

    // ── Standard anomaly event ────────────────────────────────────────────────
    } else if (level === 'anomaly') {
      const src   = parsed.source_ip ?? '?'
      const dst   = parsed.dest_ip   ?? '?'
      const recon = parsed.recon_error !== undefined ? parsed.recon_error : (parsed.reconstruction_error ?? '?')
      const type  = parsed.attack_type ?? 'Unknown'
      let reconStr = String(recon)
      if (typeof recon === 'number') reconStr = recon.toFixed(4)
      text = `🚨 ${src} → ${dst} | recon_error=${reconStr} | type=${type}`

    // ── Standard action event ─────────────────────────────────────────────────
    } else if (level === 'action') {
      const incident = parsed.incident_id !== undefined ? `Incident #${parsed.incident_id}` : ''
      const action   = parsed.action_taken ?? parsed.dqn_action ?? parsed.message ?? 'Unknown action'
      const severity = parsed.severity ? ` | severity=${parsed.severity.toUpperCase()}` : ''
      const parts = [incident, action].filter(Boolean)
      text = `⚡ ${parts.join(' | ')}${severity}`

    // ── Generic info / warning with message field ─────────────────────────────
    } else if (parsed.message) {
      text = parsed.message

    // ── Last-resort raw JSON fallback ─────────────────────────────────────────
    } else {
      text = JSON.stringify(parsed)
    }
  }

  // ── Populate optional extended fields ──────────────────────────────────────
  let incidentId: string | number | undefined
  let isGrouped: boolean | undefined
  let confidence: number | undefined

  if (parsed && typeof parsed === 'object') {
    // incidentId — present on almost every pipeline event
    if (parsed.incident_id !== undefined) {
      incidentId = parsed.incident_id
    }

    // isGrouped — mark agent sub-events so TerminalLine can indent them
    const evtForGroup = (parsed.event ?? parsed.type ?? '').toLowerCase()
    if (
      ['agent_invocation', 'agent_response', 'agent_error',
       'normal_traffic_response', 'normal_traffic_error'].includes(evtForGroup)
    ) {
      isGrouped = true
    }

    // confidence — attack classifier payloads carry this field
    if (typeof parsed.confidence === 'number') {
      confidence = parsed.confidence
    }
  }

  return {
    id,
    level,
    timestamp: timestampStr,
    timeLabel,
    text,
    raw: parsed,
    incidentId,
    isGrouped,
    confidence,
  }
}
