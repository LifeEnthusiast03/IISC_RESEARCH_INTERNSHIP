/**
 * src/components/StatCard.tsx
 * ───────────────────────────
 * Reusable metric card — icon, label, value, optional sub-label.
 * Reuses the existing .stat-card CSS class and --col-* design tokens.
 */

import type { ReactNode } from 'react'

interface Props {
  icon: ReactNode
  label: string
  value: string | number
  subLabel?: string
  accentColor?: string
}

export function StatCard({ icon, label, value, subLabel, accentColor = 'var(--col-cyan)' }: Props) {
  return (
    <div
      className="stat-card rounded-xl p-4 border"
      style={{ background: 'var(--col-surface)', borderColor: 'var(--col-border)' }}
    >
      <div className="flex items-start justify-between">
        <span className="text-xl">{icon}</span>
        {subLabel && (
          <span
            className="mono text-xs px-1.5 py-0.5 rounded"
            style={{ background: 'rgba(255,255,255,0.04)', color: 'var(--col-text)' }}
          >
            {subLabel}
          </span>
        )}
      </div>
      <p
        className="mt-3 text-2xl font-bold tracking-tight mono"
        style={{ color: accentColor }}
      >
        {value}
      </p>
      <p className="mt-1 text-xs" style={{ color: 'var(--col-text)' }}>
        {label}
      </p>
    </div>
  )
}
