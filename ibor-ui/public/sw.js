const CACHE = 'ibor-v2'

self.addEventListener('install', e => {
  e.waitUntil(self.skipWaiting())
})

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  )
})

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url)
  const isNavigation = e.request.mode === 'navigate' || url.pathname === '/' || url.pathname.endsWith('.html')
  const isApi = url.pathname.startsWith('/api') || url.pathname.startsWith('/analyst')

  // API: always network, never cache (financial data must be fresh)
  if (isApi) {
    e.respondWith(fetch(e.request))
    return
  }

  // HTML / navigation: network-first so users always get the latest asset hashes.
  // Cache fallback only for offline use.
  if (isNavigation) {
    e.respondWith(
      fetch(e.request).then(response => {
        const clone = response.clone()
        caches.open(CACHE).then(c => c.put(e.request, clone))
        return response
      }).catch(() => caches.match(e.request))
    )
    return
  }

  // Hashed static assets (JS/CSS with content hash in filename): cache-first is safe
  // because the URL changes whenever content changes.
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached
      return fetch(e.request).then(response => {
        if (response.ok) {
          const clone = response.clone()
          caches.open(CACHE).then(c => c.put(e.request, clone))
        }
        return response
      })
    })
  )
})
