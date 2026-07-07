/**
 * src/hooks/useIncidents.ts
 * ─────────────────────────
 * React Query hook for GET /incidents (paginated).
 * staleTime is 30 s; the WebSocket hook also invalidates this cache on
 * every live alert so the table stays fresh without manual refresh.
 */

import { useQuery } from '@tanstack/react-query'
import { getIncidents, type IncidentListResponse } from '../lib/api'

export function useIncidents(page = 1, pageSize = 20) {
  return useQuery<IncidentListResponse, Error>({
    queryKey: ['incidents', page, pageSize],
    queryFn: () => getIncidents(page, pageSize),
    staleTime: 30_000,
    placeholderData: (prev) => prev, // keep previous page data while next loads
  })
}

/** Fetch a larger batch for analytics aggregation (client-side) */
export function useIncidentsAnalytics() {
  return useQuery<IncidentListResponse, Error>({
    queryKey: ['incidents', 1, 100],
    queryFn: () => getIncidents(1, 100),
    staleTime: 30_000,
  })
}
