const CACHE_PREFIX = "pyodide-pytorch-playground-";
const SHELL_CACHE = `${CACHE_PREFIX}shell-v1`;
const RUNTIME_CACHE = `${CACHE_PREFIX}runtime-v1`;
const SHELL_ASSETS = ["./", "./index.html", "./styles.css", "./app.js", "./worker.js"];

async function put(cacheName, request, response) {
  if (!response || (response.status !== 200 && response.type !== "opaque")) return response;
  const cache = await caches.open(cacheName);
  await cache.put(request, response.clone());
  return response;
}

async function cacheFirst(request, cacheName) {
  const cached = await caches.match(request);
  if (cached) return cached;
  return put(cacheName, request, await fetch(request));
}

async function networkFirst(request, cacheName) {
  try {
    return await put(cacheName, request, await fetch(request));
  } catch (error) {
    const cached = await caches.match(request);
    if (cached) return cached;
    throw error;
  }
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter(
              (key) =>
                key.startsWith(CACHE_PREFIX) && key !== SHELL_CACHE && key !== RUNTIME_CACHE,
            )
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  const isManifest = url.pathname.endsWith("/runtime/build-manifest.json");
  const isWheel = url.pathname.includes("/runtime/") && url.pathname.endsWith(".whl");
  const isPyodideAsset = url.hostname === "cdn.jsdelivr.net" && url.pathname.includes("/pyodide/");
  const isPythonPackage =
    url.hostname.endsWith("pythonhosted.org") ||
    url.hostname === "pypi.org" ||
    url.pathname.endsWith(".whl");

  if (request.mode === "navigate" || isManifest) {
    event.respondWith(networkFirst(request, SHELL_CACHE));
  } else if (isWheel || isPyodideAsset || isPythonPackage) {
    event.respondWith(cacheFirst(request, RUNTIME_CACHE));
  } else if (url.origin === self.location.origin) {
    event.respondWith(networkFirst(request, SHELL_CACHE));
  }
});
