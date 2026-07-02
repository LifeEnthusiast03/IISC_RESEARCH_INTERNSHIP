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
 * - Exposes:
 *     status   : 'connecting' | 'connected' | 'disconnected' | 'error'
 *     messages : chronological list of every JSON message received
 *     clearMessages : empties the message history
 */

import { useCallback, useEffect, useRef, useState } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────────

export type WsStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

export interface WsMessage {
  /** Monotonically increasing id so React keys stay stable */
  id: number
  /** Browser timestamp when the message arrived */
  receivedAt: Date
  /** Parsed JSON payload (or raw string if not valid JSON) */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any
}

interface UseWebSocketReturn {
  status: WsStatus
  messages: WsMessage[]
  clearMessages: () => void
}

// ── Constants ─────────────────────────────────────────────────────────────────

const WS_URL = 'ws://localhost:8000/ws/connect'
const MAX_MESSAGES = 200          // keep last N messages in state
const INITIAL_RETRY_MS = 1_000   // 1 s first retry
const MAX_RETRY_MS = 30_000      // cap at 30 s

// ── Hook ──────────────────────────────────────────────────────────────────────

let _msgId = 0

export function useWebSocket(): UseWebSocketReturn {
  const [status, setStatus] = useState<WsStatus>('connecting')
  const [messages, setMessages] = useState<WsMessage[]>([])

  const wsRef        = useRef<WebSocket | null>(null)
  const retryMsRef   = useRef(INITIAL_RETRY_MS)
  const timerRef     = useRef<ReturnType<typeof setTimeout> | null>(null)
  const unmountedRef = useRef(false)

  const pushMessage = useCallback((raw: string) => {
    let data: unknown
    try { data = JSON.parse(raw) } catch { data = raw }

    const msg: WsMessage = { id: ++_msgId, receivedAt: new Date(), data }
    setMessages(prev => [...prev.slice(-MAX_MESSAGES + 1), msg])
  }, [])

  const connect = useCallback(() => {
    if (unmountedRef.current) return

    setStatus('connecting')
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      if (unmountedRef.current) { ws.close(); return }
      setStatus('connected')
      retryMsRef.current = INITIAL_RETRY_MS   // reset backoff on success
    }

    ws.onmessage = (evt: MessageEvent<string>) => {
      if (!unmountedRef.current) pushMessage(evt.data)
    }

    ws.onerror = () => {
      if (!unmountedRef.current) setStatus('error')
    }

    ws.onclose = () => {
      if (unmountedRef.current) return
      setStatus('disconnected')
      // Exponential backoff retry
      timerRef.current = setTimeout(() => {
        retryMsRef.current = Math.min(retryMsRef.current * 2, MAX_RETRY_MS)
        connect()
      }, retryMsRef.current)
    }
  }, [pushMessage])

  useEffect(() => {
    unmountedRef.current = false
    connect()

    return () => {
      unmountedRef.current = true
      if (timerRef.current) clearTimeout(timerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const clearMessages = useCallback(() => setMessages([]), [])

  return { status, messages, clearMessages }
}
