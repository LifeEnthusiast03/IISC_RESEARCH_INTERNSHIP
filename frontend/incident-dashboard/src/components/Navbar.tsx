/**
 * src/components/Navbar.tsx
 * ─────────────────────────
 * Persistent navigation bar rendered by RootLayout.
 * - Logo badge "TS" + "ThreatSentinel" + version tag
 * - 4 NavLinks with active-state cyan highlight
 * - Right side: WebSocket connection status pill (from WsContext)
 */

import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Terminal, List, BarChart3, Home, Menu, X } from 'lucide-react'
import clsx from 'clsx'
import { useWsContext } from '../contexts/WsContext'
import { StatusPill } from './StatusPill'
import { useScrollPosition } from '../hooks/useScrollPosition'
import type { WsStatus } from '../hooks/useWebSocket'

const NAV_LINKS = [
  { to: '/',          label: 'Home',      icon: Home     },
  { to: '/terminal',  label: 'Terminal',  icon: Terminal  },
  { to: '/incidents', label: 'Incidents', icon: List      },
  { to: '/analytics', label: 'Analytics', icon: BarChart3 },
] as const

function wsVariant(status: WsStatus) {
  if (status === 'connected')   return 'green'  as const
  if (status === 'connecting')  return 'amber'  as const
  if (status === 'error')       return 'red'    as const
  return 'slate' as const
}

function wsLabel(status: WsStatus) {
  if (status === 'connected')   return 'Connected'
  if (status === 'connecting')  return 'Connecting…'
  if (status === 'error')       return 'Conn Error'
  return 'Disconnected'
}

export function Navbar() {
  const { status } = useWsContext()
  const scrollY = useScrollPosition()
  const isScrolled = scrollY > 40
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  return (
    <>
      <header
        className={clsx(
          "fixed top-4 left-1/2 -translate-x-1/2 w-[calc(100%-2rem)] max-w-7xl z-30",
          "flex items-center justify-between px-4 py-2.5 md:px-5 md:py-3 transition-all duration-300",
          isScrolled 
            ? "rounded-2xl border shadow-[0_8px_30px_rgb(0,0,0,0.4)]" 
            : "rounded-xl border shadow-[0_4px_20px_rgb(0,0,0,0.2)]"
        )}
        style={{
          borderColor: isScrolled ? 'rgba(56,189,248,0.2)' : 'var(--col-border)',
          background: isScrolled ? 'rgba(8,11,20,0.95)' : 'rgba(8,11,20,0.7)',
          backdropFilter: 'blur(16px)',
        }}
      >
        {/* ── Logo ──────────────────────────────────────────────────────── */}
        <NavLink to="/" className="flex items-center gap-3 group" aria-label="ThreatSentinel home">
          <div
            className="flex items-center justify-center w-8 h-8 rounded-lg text-sm font-bold transition-shadow duration-200 group-hover:shadow-[0_0_12px_rgba(56,189,248,0.35)]"
            style={{
              background: 'var(--col-cyan-dim)',
              border: '1px solid rgba(56,189,248,0.3)',
              color: 'var(--col-cyan)',
            }}
          >
            TS
          </div>
          <div>
            <p className="text-sm font-semibold leading-none" style={{ color: 'var(--col-text-hi)' }}>
              ThreatSentinel
            </p>
            <p className="text-xs mt-0.5 mono" style={{ color: 'var(--col-text)' }}>
              IDS Dashboard v0.1
            </p>
          </div>
        </NavLink>

        {/* ── Nav links ─────────────────────────────────────────────────── */}
        <nav className="hidden md:flex items-center gap-1" aria-label="Primary navigation">
          {NAV_LINKS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-1.5 px-3 py-2 rounded-full text-sm transition-all duration-150 outline-none',
                  'focus-visible:ring-2 focus-visible:ring-[var(--col-cyan)] focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--col-bg)]',
                  isActive
                    ? 'text-[var(--col-cyan)] bg-[rgba(56,189,248,0.08)] border border-[rgba(56,189,248,0.2)]'
                    : 'text-[var(--col-text)] hover:text-[var(--col-text-hi)] hover:bg-[rgba(255,255,255,0.04)] border border-transparent',
                )
              }
            >
              <Icon size={14} strokeWidth={1.75} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* ── WS status pill & Mobile Menu ──────────────────────────────── */}
        <div className="flex items-center gap-3">
          <StatusPill
            variant={wsVariant(status)}
            pulse={status === 'connected' || status === 'connecting'}
          >
            {wsLabel(status)}
          </StatusPill>
          
          <button
            className="md:hidden flex items-center justify-center w-8 h-8 rounded-lg border transition-colors"
            style={{
              borderColor: 'var(--col-border)',
              color: 'var(--col-text-hi)',
              background: 'rgba(255,255,255,0.03)',
            }}
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-label="Toggle mobile menu"
          >
            {isMobileMenuOpen ? <X size={16} /> : <Menu size={16} />}
          </button>
        </div>
      </header>

      {/* ── Mobile Menu Dropdown ──────────────────────────────────────── */}
      {isMobileMenuOpen && (
        <div
          className="fixed top-20 left-1/2 -translate-x-1/2 w-[calc(100%-2rem)] z-20 rounded-2xl border p-2 shadow-xl md:hidden"
          style={{
            borderColor: 'var(--col-border)',
            background: 'rgba(8,11,20,0.95)',
            backdropFilter: 'blur(16px)',
          }}
        >
          <nav className="flex flex-col gap-1">
            {NAV_LINKS.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                onClick={() => setIsMobileMenuOpen(false)}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 px-4 py-3 rounded-xl text-sm transition-all duration-150',
                    isActive
                      ? 'text-[var(--col-cyan)] bg-[rgba(56,189,248,0.08)] border border-[rgba(56,189,248,0.2)]'
                      : 'text-[var(--col-text)] hover:text-[var(--col-text-hi)] hover:bg-[rgba(255,255,255,0.04)] border border-transparent',
                  )
                }
              >
                <Icon size={16} strokeWidth={1.75} />
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      )}
    </>
  )
}
