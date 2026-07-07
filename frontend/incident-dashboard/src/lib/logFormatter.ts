import { format } from 'date-fns'

export type LogLevel = 'info' | 'anomaly' | 'action' | 'warning' | 'error' | 'connected' | 'system'

export interface FormattedLog {
  id: string
  level: LogLevel
  timestamp: string    // raw ISO string
  timeLabel: string    // HH:mm:ss.SSS formatted
  text: string         // single-line human readable summary
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  raw: any             // the original parsed JSON object (or string if parse failed)
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

  // Resolve level
  let level: LogLevel = 'info'
  if (parsed && typeof parsed === 'object') {
    const t = (parsed.type ?? parsed.event ?? parsed.severity ?? '').toLowerCase()
    if (['anomaly', 'threat', 'intrusion'].includes(t)) level = 'anomaly'
    else if (['warning', 'alert', 'critical'].includes(t)) level = 'warning'
    else if (['error'].includes(t)) level = 'error'
    else if (['action', 'remediation', 'block'].includes(t)) level = 'action'
    else if (['connected', 'success', 'ok'].includes(t)) level = 'connected'
    else if (parsed.is_anomaly === true) level = 'anomaly'
  }

  // Build human-readable text
  let text = ''
  if (typeof parsed === 'string') {
    text = parsed
  } else {
    // Determine the shape based on level / fields
    if (level === 'anomaly') {
      const src = parsed.source_ip ?? '?'
      const dst = parsed.dest_ip ?? '?'
      const recon = parsed.recon_error !== undefined ? parsed.recon_error : (parsed.reconstruction_error ?? '?')
      const type = parsed.attack_type ?? 'Unknown'
      
      let reconStr = String(recon)
      if (typeof recon === 'number') reconStr = recon.toFixed(4)
      
      text = `${src} → ${dst} | recon_error=${reconStr} | type=${type}`
    } else if (level === 'action') {
      const incident = parsed.incident_id !== undefined ? `Incident #${parsed.incident_id}` : ''
      const action = parsed.action_taken ?? parsed.dqn_action ?? parsed.message ?? 'Unknown action'
      const severity = parsed.severity ? ` | severity=${parsed.severity.toUpperCase()}` : ''
      const parts = [incident, action].filter(Boolean)
      text = `${parts.join(' | ')}${severity}`
    } else if (parsed.message) {
      // Generic info with message
      text = parsed.message
    } else {
      // Generic info fallback
      text = JSON.stringify(parsed)
    }
  }

  return {
    id,
    level,
    timestamp: timestampStr,
    timeLabel,
    text,
    raw: parsed,
  }
}
