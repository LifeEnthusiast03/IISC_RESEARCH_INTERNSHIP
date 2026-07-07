/**
 * src/lib/api.ts
 * ──────────────
 * Centralised axios instance + typed API functions for the ThreatSentinel
 * backend (FastAPI at VITE_API_URL, defaulting to http://localhost:8000).
 *
 * All functions return typed data matching the backend Pydantic schemas.
 */

import axios from 'axios'

// ── Base URL ──────────────────────────────────────────────────────────────────

export const BASE_URL: string =
  (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

/** Derive the WebSocket base from the HTTP base URL */
export const WS_BASE_URL: string = BASE_URL.replace(/^http/, 'ws')

// ── Axios instance ────────────────────────────────────────────────────────────

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 10_000,
  headers: { 'Content-Type': 'application/json' },
})

// ── Types (matching backend schemas.py) ───────────────────────────────────────

export interface HealthResponse {
  status: 'ok' | 'degraded'
  autoencoder_loaded: boolean
  dqn_loaded: boolean
  db_connected: boolean
  uptime_seconds: number
}

export interface IncidentOut {
  id: number
  timestamp: string
  source_ip: string
  dest_ip: string
  src_port: number | null
  dst_port: number | null
  reconstruction_error: number
  is_anomaly: boolean
  attack_type_predicted: string | null
  dqn_action: string | null
  action_status: string
  raw_features: Record<string, unknown> | unknown[] | null
  created_at: string
}

export interface IncidentListResponse {
  items: IncidentOut[]
  total: number
  page: number
  page_size: number
}

// ── API functions ─────────────────────────────────────────────────────────────

/** GET /health */
export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health')
  return data
}

/** GET /incidents?page=&page_size= */
export async function getIncidents(
  page = 1,
  pageSize = 20,
): Promise<IncidentListResponse> {
  const { data } = await api.get<IncidentListResponse>('/incidents', {
    params: { page, page_size: pageSize },
  })
  return data
}

/** GET /incidents/{id} */
export async function getIncidentById(id: number): Promise<IncidentOut> {
  const { data } = await api.get<IncidentOut>(`/incidents/${id}`)
  return data
}

export default api
