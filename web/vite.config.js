import { defineConfig } from 'vite';

export default defineConfig(({ mode }) => ({
  base: mode === 'production' ? '/Stupidify/' : '/',
  build: {
    target: 'es2022',
    sourcemap: true,
  },
}));
