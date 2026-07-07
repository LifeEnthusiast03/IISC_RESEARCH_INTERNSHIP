/**
 * src/components/StatusPill.tsx
 * ─────────────────────────────
 * Reusable coloured status badge — used in Navbar (WS status), stat cards,
 * and throughout pages. Matches the existing .badge-* CSS pattern.
 */

import clsx from 'clsx'
import type { ReactNode } from 'react'

type Variant = 'cyan' | 'green' | 'red' | 'amber' | 'slate'

interface Props {
  variant: Variant
  dot?: boolean
  pulse?: boolean
  children: ReactNode
  className?: string
}

const VARIANT_STYLES: Record<Variant, { pill: string; dot: string }> = {
  cyan:  { pill: 'bg-[rgba(56,189,248,0.1)] border-[rgba(56,189,248,0.3)] text-[var(--col-cyan)]',  dot: 'bg-[var(--col-cyan)]'  },
  green: { pill: 'bg-[rgba(52,211,153,0.1)] border-[rgba(52,211,153,0.3)] text-[var(--col-green)]', dot: 'bg-[var(--col-green)]' },
  red:   { pill: 'bg-[rgba(248,113,113,0.1)] border-[rgba(248,113,113,0.3)] text-[var(--col-red)]', dot: 'bg-[var(--col-red)]'   },
  amber: { pill: 'bg-[rgba(251,191,36,0.1)] border-[rgba(251,191,36,0.3)] text-[var(--col-amber)]', dot: 'bg-[var(--col-amber)]' },
  slate: { pill: 'bg-[rgba(148,163,184,0.08)] border-[rgba(148,163,184,0.2)] text-[var(--col-text)]', dot: 'bg-[var(--col-text)]' },
}

export function StatusPill({ variant, dot = true, pulse = false, children, className }: Props) {
  const { pill, dot: dotColor } = VARIANT_STYLES[variant]
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[11px] font-semibold tracking-[0.06em] uppercase',
        'font-[family-name:var(--font-mono)]',
        pill,
        className,
      )}
    >
      {dot && (
        <span
          className={clsx('w-1.5 h-1.5 rounded-full shrink-0', dotColor, pulse && 'animate-pulse')}
        />
      )}
      {children}
    </span>
  )
}
