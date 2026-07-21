/**
 * src/hooks/useNavScroll.ts
 * ─────────────────────────
 * Tracks scroll position and direction for the Navbar hide/show behavior.
 *
 * Design principles:
 *  - Uses refs for prevScrollY and rAF handle — zero re-renders on scroll ticks
 *  - Only calls setState when the isHidden boolean actually flips (rare)
 *  - Throttles via requestAnimationFrame so it fires at most once per frame
 *  - isHidden: true  → navbar slides up and fades out (user scrolled DOWN)
 *  - isHidden: false → navbar slides in and fades in  (user scrolled UP or at top)
 *
 * Thresholds:
 *  SCROLL_THRESHOLD   : 8 px — minimum scroll delta needed to register a direction
 *                       change; avoids jitter from scroll-wheel noise or bounce.
 *  TOP_ZONE           : 80 px — while scrollY < TOP_ZONE the navbar is always shown,
 *                       even if the user is scrolling down slightly near the top.
 */

import { useState, useEffect, useRef } from 'react'

const SCROLL_THRESHOLD = 8   // px — minimum delta to commit to a direction
const TOP_ZONE        = 80   // px — always visible below this scrollY

interface NavScrollState {
  /** True once the page has scrolled past the TOP_ZONE — used for background/shadow changes */
  isScrolled: boolean
  /** True when the navbar should be hidden (translate up + fade) */
  isHidden: boolean
}

export function useNavScroll(): NavScrollState {
  const [state, setState] = useState<NavScrollState>({
    isScrolled: false,
    isHidden: false,
  })

  const prevScrollY = useRef(0)
  const rafId       = useRef<number | null>(null)
  const pending     = useRef(false)

  useEffect(() => {
    const onScroll = () => {
      if (pending.current) return
      pending.current = true

      rafId.current = requestAnimationFrame(() => {
        pending.current = false
        const currentY = window.scrollY
        const delta    = currentY - prevScrollY.current

        prevScrollY.current = currentY

        const atTop      = currentY < TOP_ZONE
        const isScrolled = currentY > TOP_ZONE

        setState(prev => {
          if (atTop) return { isScrolled: false, isHidden: false }

          let isHidden = prev.isHidden
          if (delta > SCROLL_THRESHOLD)        isHidden = true
          else if (delta < -SCROLL_THRESHOLD)  isHidden = false

          if (isHidden === prev.isHidden && isScrolled === prev.isScrolled) return prev
          return { isScrolled, isHidden }
        })
      })
    }

    window.addEventListener('scroll', onScroll, { passive: true })
    onScroll()

    return () => {
      window.removeEventListener('scroll', onScroll)
      if (rafId.current !== null) cancelAnimationFrame(rafId.current)
    }
  }, [])



  return state
}
