/**
 * Streaming download with progress, backed by the Cache API so a reload does not re-fetch
 * hundreds of megabytes of weights.
 */
import type { LoadProgress } from "./types.js";

const CACHE_NAME = "laya-web-v1";
/** Chrome caps a single ArrayBuffer at ~2 GB; refuse anything that cannot be materialised. */
const MAX_BYTES = 2_000_000_000;

export type ProgressFn = (p: LoadProgress) => void;

function cacheAvailable(): boolean {
  return typeof caches !== "undefined";
}

/** Ask the browser to keep our cached weights rather than evicting them under pressure. */
export async function requestPersistentStorage(): Promise<boolean> {
  try {
    if (typeof navigator === "undefined" || !navigator.storage?.persist)
      return false;
    if (await navigator.storage.persisted()) return true;
    return await navigator.storage.persist();
  } catch {
    return false;
  }
}

export async function storageEstimate(): Promise<{
  quota: number;
  usage: number;
} | null> {
  try {
    if (typeof navigator === "undefined" || !navigator.storage?.estimate)
      return null;
    const e = await navigator.storage.estimate();
    return { quota: e.quota ?? 0, usage: e.usage ?? 0 };
  } catch {
    return null;
  }
}

/** True when this URL is already in the weight cache. */
export async function isCached(url: string): Promise<boolean> {
  if (!cacheAvailable()) return false;
  try {
    const cache = await caches.open(CACHE_NAME);
    return (await cache.match(url)) !== undefined;
  } catch {
    return false;
  }
}

export async function clearCache(): Promise<void> {
  if (!cacheAvailable()) return;
  await caches.delete(CACHE_NAME);
}

/**
 * Fetch `url` as bytes, serving from the Cache API when present and populating it when not.
 * Progress is reported per chunk while streaming.
 */
export async function fetchBytes(
  url: string,
  opts: {
    phase: LoadProgress["phase"];
    expectedBytes?: number;
    onProgress?: ProgressFn;
  } = {
    phase: "weights",
  },
): Promise<Uint8Array> {
  const { phase, expectedBytes, onProgress } = opts;
  const file = url.split("/").pop() ?? url;

  if (expectedBytes && expectedBytes > MAX_BYTES) {
    throw new Error(
      `${file} is ${(expectedBytes / 1e9).toFixed(2)} GB, beyond the ~2 GB a browser can hold in one ArrayBuffer`,
    );
  }

  const cache = cacheAvailable()
    ? await caches.open(CACHE_NAME).catch(() => null)
    : null;

  if (cache) {
    const hit = await cache.match(url).catch(() => undefined);
    if (hit) {
      onProgress?.({
        phase,
        file,
        cached: true,
        received: 0,
        total: expectedBytes,
      });
      const buf = new Uint8Array(await hit.arrayBuffer());
      onProgress?.({
        phase,
        file,
        cached: true,
        received: buf.byteLength,
        total: buf.byteLength,
      });
      return buf;
    }
  }

  const res = await fetch(url, { mode: "cors", credentials: "omit" });
  if (!res.ok) {
    const host = new URL(url).host;
    throw new Error(
      `failed to fetch ${file} from ${host}: ${res.status} ${res.statusText}. ` +
        (res.status === 401 || res.status === 403 || res.status === 404
          ? "The file may have moved or been removed; update @wexare/laya-web, or pass your own bundle to Laya.load()."
          : "Check the connection and try again."),
    );
  }

  const header =
    res.headers.get("content-length") ?? res.headers.get("x-linked-size");
  const total = header ? Number(header) : (expectedBytes ?? 0);

  let bytes: Uint8Array;
  if (!res.body) {
    bytes = new Uint8Array(await res.arrayBuffer());
    onProgress?.({
      phase,
      file,
      received: bytes.byteLength,
      total: bytes.byteLength,
    });
  } else {
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    let lastReport = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      // report at most every 2 MB so the UI thread is not flooded
      if (received - lastReport > 2_000_000) {
        lastReport = received;
        onProgress?.({ phase, file, received, total });
      }
    }
    bytes = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) {
      bytes.set(c, off);
      off += c.byteLength;
    }
    onProgress?.({ phase, file, received, total: received });
  }

  if (cache) {
    // Hand Response the underlying buffer rather than a clone: at ~900 MB that second copy is the
    // difference between fitting in a tab and not. `bytes` owns its buffer on every path here,
    // but fall back to a copy if that ever stops being true.
    const body =
      bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? (bytes.buffer as ArrayBuffer)
        : (bytes.slice().buffer as ArrayBuffer);
    // A failed write (a quota error, say) must not fail the load.
    await cache.put(url, new Response(body)).catch(() => undefined);
  }
  return bytes;
}

export async function fetchJson<T>(
  url: string,
  onProgress?: ProgressFn,
  phase: LoadProgress["phase"] = "config",
): Promise<T> {
  const bytes = await fetchBytes(url, { phase, onProgress });
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

export function formatBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} kB`;
  return `${n} B`;
}
