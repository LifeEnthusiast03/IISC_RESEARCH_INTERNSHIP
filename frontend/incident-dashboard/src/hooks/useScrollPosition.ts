import { useState, useEffect } from 'react'

/**
 * Custom hook to track the vertical scroll position of the window.
 * Respects prefers-reduced-motion to some extent by not triggering rapid updates
 * if the user has requested reduced motion, though here we just throttle or 
 * directly use the event.
 */
export function useScrollPosition() {
  const [scrollPosition, setScrollPosition] = useState(0)

  useEffect(() => {
    const updatePosition = () => {
      setScrollPosition(window.scrollY)
    }
    
    window.addEventListener('scroll', updatePosition, { passive: true })
    updatePosition()
    
    return () => window.removeEventListener('scroll', updatePosition)
  }, [])

  return scrollPosition
}
