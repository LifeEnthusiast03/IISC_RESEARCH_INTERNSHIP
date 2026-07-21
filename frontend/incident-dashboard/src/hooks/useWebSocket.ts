/**
 * src/hooks/useWebSocket.ts
 * ─────────────────────────
 * Custom React hook that manages a persistent WebSocket connection to the
 * IDS backend at ws://localhost:8000/ws/connect.
 *
 * Features
 * --------
 * - Connects on mount, cleans up on unmount
 * - Auto-reconnects with exponential backoff (max 30 s) after any disconnect
 * - On each new message, invalidates the ['incidents'] React Query cache so
 *   the Incidents and Analytics pages automatically refresh
 * - Exposes:
 *     status      : 'connecting' | 'connected' | 'disconnected' | 'error'
 *     messages    : chronological list of every JSON message received
 *     lastMessage : most recent WsMessage (or null before first message)
 *     clearMessages : empties the message history
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { WS_BASE_URL } from '../lib/api'
import { formatLogMessage, type FormattedLog } from '../lib/logFormatter'

// ── Types ─────────────────────────────────────────────────────────────────────

export type WsStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

export type WsMessage = FormattedLog // keep export for compatibility if needed, though we use FormattedLog

interface UseWebSocketReturn {
  status: WsStatus
  messages: FormattedLog[]
  lastMessage: FormattedLog | null
  clearMessages: () => void
}

// ── Constants ─────────────────────────────────────────────────────────────────

const WS_URL         = `${WS_BASE_URL}/ws/connect`
const MAX_MESSAGES   = 500         // keep last N messages in memory / sessionStorage
const INITIAL_RETRY_MS = 1_000     // 1 s first retry
const MAX_RETRY_MS   = 30_000      // cap at 30 s
const SESSION_KEY    = 'ts_terminal_logs'  // sessionStorage key

// ── sessionStorage helpers ────────────────────────────────────────────────────

function loadFromSession(): FormattedLog[] {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as FormattedLog[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function saveToSession(msgs: FormattedLog[]): void {
  try {
    // Trim to MAX_MESSAGES before writing so sessionStorage never grows unbounded
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(msgs.slice(-MAX_MESSAGES)))
  } catch {
    // Storage quota exceeded — fail silently
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────


export function useWebSocket(): UseWebSocketReturn {
  const [status, setStatus] = useState<WsStatus>('connecting')
  // Seed from sessionStorage so logs survive F5/refresh within the same tab
  const [messages, setMessages] = useState<FormattedLog[]>(loadFromSession)
  const [lastMessage, setLastMessage] = useState<FormattedLog | null>(null)

  const queryClient = useQueryClient()

  const wsRef        = useRef<WebSocket | null>(null)
  const retryMsRef   = useRef(INITIAL_RETRY_MS)
  const timerRef     = useRef<ReturnType<typeof setTimeout> | null>(null)

  const pushMessage = useCallback((raw: string) => {
    const msg = formatLogMessage(raw)
    setMessages(prev => {
      const next = [...prev.slice(-MAX_MESSAGES + 1), msg]
      saveToSession(next)
      return next
    })
    setLastMessage(msg)

    // Keep React Query cache fresh whenever a live alert arrives
    queryClient.invalidateQueries({ queryKey: ['incidents'] })
  }, [queryClient])

  const connect = useCallback(() => {
    setStatus('connecting')
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      if (wsRef.current !== ws) { ws.close(); return }
      setStatus('connected')
      retryMsRef.current = INITIAL_RETRY_MS   // reset backoff on success
    }

    ws.onmessage = (evt: MessageEvent<string>) => {
      if (wsRef.current !== ws) return
      pushMessage(evt.data)
    }

    ws.onerror = () => {
      if (wsRef.current !== ws) return
      setStatus('error')
    }

    ws.onclose = () => {
      if (wsRef.current !== ws) return
      setStatus('disconnected')
      // Exponential backoff retry
      timerRef.current = setTimeout(() => {
        retryMsRef.current = Math.min(retryMsRef.current * 2, MAX_RETRY_MS)
        connect()
      }, retryMsRef.current)
    }
  }, [pushMessage])

  useEffect(() => {
    connect()

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  const clearMessages = useCallback(() => {
    setMessages([])
    sessionStorage.removeItem(SESSION_KEY)
  }, [])

  return { status, messages, lastMessage, clearMessages }
}
