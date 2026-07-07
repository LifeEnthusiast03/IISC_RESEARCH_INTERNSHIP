/**
 * src/layouts/RootLayout.tsx
 * ──────────────────────────
 * Persistent shell rendered for every route:
 *   WsProvider → Navbar → ambient background → <Outlet /> → footer
 *
 * WsProvider is mounted here so both the Navbar status pill and the
 * TerminalPage share ONE WebSocket connection via WsContext.
 */

import { Outlet } from 'react-router-dom'
import { WsProvider } from '../contexts/WsContext'
import { Navbar } from '../components/Navbar'

export default function RootLayout() {
  return (
    <WsProvider>
      <div className="relative min-h-screen flex flex-col" style={{ background: 'var(--col-bg)' }}>

        {/* ── Ambient grid background ───────────────────────────────── */}
        <div className="grid-bg" />

        {/* ── Glow orbs ─────────────────────────────────────────────── */}
        <div
          className="orb"
          style={{
            width: 700, height: 700,
            top: -250, left: -200,
            background: 'radial-gradient(circle, rgba(56,189,248,0.05) 0%, transparent 70%)',
          }}
        />
        <div
          className="orb"
          style={{
            width: 500, height: 500,
            bottom: -100, right: -100,
            background: 'radial-gradient(circle, rgba(248,113,113,0.04) 0%, transparent 70%)',
          }}
        />

        {/* ── Navbar ────────────────────────────────────────────────── */}
        <Navbar />

        {/* ── Page content ──────────────────────────────────────────── */}
        <main className="relative z-10 flex-1 flex flex-col pt-24 md:pt-28">
          <Outlet />
        </main>

        {/* ── Footer ────────────────────────────────────────────────── */}
        <footer
          className="relative z-10 px-6 py-4 border-t flex items-center justify-between"
          style={{ borderColor: 'var(--col-border)' }}
        >
          <p className="mono text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>
            ThreatSentinel IDS · IISc Research Internship · 2026
          </p>
          <p className="mono text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>
            PyTorch 2.6.0+cu124 · React 19 · Vite 8 · Tailwind v4
          </p>
        </footer>

      </div>
    </WsProvider>
  )
}
