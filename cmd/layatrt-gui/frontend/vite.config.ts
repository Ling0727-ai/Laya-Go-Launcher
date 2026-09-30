import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

// Dev-server port.
//
// Vite's default is 5173, which is the most contended port on a developer
// machine: whichever Vite project starts first claims it and every other one
// silently drifts to 5174, 5175, … Moving to another fixed number just relocates
// that race, and on a busy box any fixed port can already be taken.
//
// So: default to a less common port, and let LAYATRT_DEV_PORT (or PORT) override
// it. strictPort makes a clash a loud error instead of a silent port change,
// which is what you want when the docs name a port.
const devPort = Number(process.env.LAYATRT_DEV_PORT || process.env.PORT || 5188)

// https://vitejs.dev/config/
export default defineConfig({
  // Relative asset paths are required for the Wails asset server: it serves the
  // bundle from a custom scheme, where Vite's default absolute "/assets/..." URLs
  // do not resolve, leaving the window blank.
  base: './',
  plugins: [react()],

  server: {
    port: devPort,
    // Fail instead of quietly moving to the next port, so a clash is visible.
    strictPort: true,
  },
  preview: {
    port: devPort,
    strictPort: true,
  },

  resolve: {
    alias: {
      // Mirrors the "@/*" path mapping in tsconfig.json.
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
})
