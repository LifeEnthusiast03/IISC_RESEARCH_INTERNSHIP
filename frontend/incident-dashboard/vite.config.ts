import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  // Vite 6+ uses server.proxy + preview.historyApiFallback differently.
  // For dev, React Router's BrowserRouter is handled by Vite's built-in
  // fallback. For direct URL navigation in dev, add the middleware below.
  server: {
    // Vite 6 handles SPA fallback natively — all non-file requests
    // that would 404 are served index.html automatically.
  },
})
