/**
 * src/hooks/useHealth.ts
 * ──────────────────────
 * React Query hook for GET /health.
 * Refetches every 15 s to keep the system status chip in HomePage live.
 */

import { useQuery } from '@tanstack/react-query'
import { getHealth, type HealthResponse } from '../lib/api'

export function useHealth() {
  return useQuery<HealthResponse, Error>({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 15_000,
    staleTime: 10_000,
  })
}
