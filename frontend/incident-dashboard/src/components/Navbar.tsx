/**
 * src/components/Navbar.tsx
 * ─────────────────────────
 * Persistent navigation bar rendered by RootLayout.
 *
 * Features:
 *  - Logo badge "TS" + "ThreatSentinel" + version tag
 *  - 4 NavLinks with active-state cyan highlight
 *  - Right side: WebSocket connection status pill (from WsContext)
 *  - Scroll-direction hide/show: slides up on scroll-down, reveals on scroll-up
 *  - Narrow floating pill shape: ~82% viewport width on large screens
 *  - Pill/capsule corners: rounded-full
 *
 * Scroll behavior implemented via useNavScroll (rAF-throttled, ref-based — no
 * per-tick re-renders). CSS transform + opacity transition handles the animation.
 *
 * Width breakpoints:
 *   mobile  : w-[calc(100%-1rem)]   (8px each side — don't cramp nav items)
 *   sm      : w-[calc(100%-2rem)]   (16px each side)
 *   lg      : w-[82%] max-w-6xl    (narrows to a floating pill on large screens)
 *
 * Scroll thresholds (defined in useNavScroll):
 *   SCROLL_THRESHOLD : 8 px delta to commit to a direction change
 *   TOP_ZONE         : 80 px — always visible while scrollY < 80
 */

import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Terminal, List, BarChart3, Home, Menu, X } from 'lucide-react'
import clsx from 'clsx'
import { useNavScroll } from '../hooks/useNavScroll'

const NAV_LINKS = [
  { to: '/',          label: 'Home',      icon: Home     },
  { to: '/terminal',  label: 'Terminal',  icon: Terminal  },
  { to: '/incidents', label: 'Incidents', icon: List      },
  { to: '/analytics', label: 'Analytics', icon: BarChart3 },
] as const

export function Navbar() {
  const { isScrolled, isHidden } = useNavScroll()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  return (
    <>
      <header
        className={clsx(
          // ── Positioning: fixed, horizontally centred ──────────────────
          "fixed top-4 left-1/2 z-30",

          // Width: content-fit pill — wraps tightly around nav links
          "w-fit",

          // ── Shape: full pill/capsule corners ─────────────────────────
          "rounded-full",

          // ── Layout ───────────────────────────────────────────────────
          "flex items-center justify-center px-3 py-1.5",

        )}
        style={{
          // ── Frosted glass background ──────────────────────────────────
          backdropFilter: 'blur(18px)',
          background: isScrolled ? 'rgba(8,11,20,0.95)' : 'rgba(8,11,20,0.72)',
          borderColor: isScrolled ? 'rgba(56,189,248,0.22)' : 'var(--col-border)',
          border: '1px solid',
          boxShadow: isScrolled
            ? '0 8px 32px rgba(0,0,0,0.45), inset 0 0 0 1px rgba(56,189,248,0.06)'
            : '0 4px 20px rgba(0,0,0,0.22)',

          // ── Scroll-direction hide/show ─────────────────────────────────
          // translateY(-120%) clears the top: 1rem offset plus the bar's own height
          // opacity fade makes the exit/entrance softer
          transform: isHidden ? 'translateX(-50%) translateY(-120%)' : 'translateX(-50%) translateY(0)',
          opacity: isHidden ? 0 : 1,
          // Separate transitions: transform is quick, opacity a touch slower for softness
          transition: 'transform 280ms cubic-bezier(0.4,0,0.2,1), opacity 260ms ease-out, background 300ms ease, box-shadow 300ms ease, border-color 300ms ease',
          // Pointer-events off while hidden so invisible bar can't intercept clicks
          pointerEvents: isHidden ? 'none' : 'auto',
        }}
      >
        {/* ── Nav links ──────────────────────────────────────────────── */}
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

        {/* ── Mobile Menu toggle (mobile only) ──────────────────────── */}
        <button
          className="md:hidden flex items-center justify-center w-8 h-8 rounded-full border transition-colors"
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
      </header>

      {/* ── Mobile Menu Dropdown ───────────────────────────────────────── */}
      {isMobileMenuOpen && (
        <div
          className="fixed top-20 left-1/2 -translate-x-1/2 w-[calc(100%-2rem)] z-20 rounded-2xl border p-2 shadow-xl md:hidden"
          style={{
            borderColor: 'var(--col-border)',
            background: 'rgba(8,11,20,0.95)',
            backdropFilter: 'blur(18px)',
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
