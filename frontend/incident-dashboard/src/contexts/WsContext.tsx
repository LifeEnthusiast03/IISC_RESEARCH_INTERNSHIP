/**
 * src/contexts/WsContext.tsx
 * ──────────────────────────
 * Provides a single shared WebSocket connection across all pages.
 * Wrap the app in <WsProvider> (done in RootLayout) so both the Navbar
 * status pill and the TerminalPage read from the SAME socket.
 */

import { createContext, useContext, type ReactNode } from 'react'
import { useWebSocket, type WsStatus, type WsMessage } from '../hooks/useWebSocket'

// ── Context shape ─────────────────────────────────────────────────────────────

interface WsContextValue {
  status: WsStatus
  messages: WsMessage[]
  lastMessage: WsMessage | null
  clearMessages: () => void
}

const WsContext = createContext<WsContextValue | null>(null)

// ── Provider ──────────────────────────────────────────────────────────────────

export function WsProvider({ children }: { children: ReactNode }) {
  const ws = useWebSocket()
  return <WsContext.Provider value={ws}>{children}</WsContext.Provider>
}

// ── Consumer hook ─────────────────────────────────────────────────────────────

export function useWsContext(): WsContextValue {
  const ctx = useContext(WsContext)
  if (!ctx) throw new Error('useWsContext must be used inside <WsProvider>')
  return ctx
}
