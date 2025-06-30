// @ts-check
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'astro/config'
import sitemap from '@astrojs/sitemap'
import github from '@astrojs/github'
import mdx from '@astrojs/mdx'

export default defineConfig({
  site: 'https://blog.anumeya.com',
  integrations: [mdx(), sitemap(), github()],
  vite: {
    plugins: [tailwindcss()],
  },
})
