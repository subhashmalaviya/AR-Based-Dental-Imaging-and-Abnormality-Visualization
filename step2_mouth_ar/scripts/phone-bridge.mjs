import { randomBytes } from 'node:crypto';
import { networkInterfaces } from 'node:os';

const token = () => randomBytes(24).toString('hex');
const PREFIX = '/__phone/';
const TTL = 30 * 60 * 1000;
// This local-only relay keeps one pending capture per pairing, never writes files.
export function phoneBridgeMiddleware({ maxBytes = 200_000_000, totalBytes = 256_000_000, ttl = TTL } = {}) {
  const sessions = new Map(); let reserved = 0;
  const remove = id => {
    const s = sessions.get(id);
    if (s?.media) reserved -= s.media.data.length;
    sessions.delete(id);
  };
  const clean = () => { for (const [id, s] of sessions) if (Date.now() - s.touched > ttl) remove(id); };
  const timer = setInterval(clean, Math.min(ttl, 60_000)); timer.unref();
  const middleware = async (req, res, next) => {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    if (!pathname.startsWith(PREFIX)) return next();
    res.setHeader('Cache-Control', 'no-store'); res.setHeader('X-Content-Type-Options', 'nosniff');
    const json = (code, value) => { res.statusCode = code; res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(value)); };
    try {
      clean();
      if (req.headers.origin && new URL(req.headers.origin).host !== req.headers.host)
        return json(403, { error: 'Open the phone link on this server; cross-origin transfer is disabled.' });
      if (req.headers['x-dental-client'] !== '1') return json(403, { error: 'Use the Dental AR capture page.' });
      if (pathname === `${PREFIX}session` && req.method === 'POST') {
        if (sessions.size >= 4) return json(429, { error: 'Close an existing pairing or wait for it to expire.' });
        const id = token(), receiver = token(), sender = token();
        sessions.set(id, { receiver, sender, touched: Date.now(), media: null, uploading: false });
        const protocol = req.socket.encrypted ? 'https:' : 'http:';
        const port = new URL(`${protocol}//${req.headers.host}`).port;
        const hosts = [...new Set([req.headers.host, ...Object.values(networkInterfaces()).flat()
          .filter(n => n && !n.internal && n.family === 'IPv4').map(n => `${n.address}${port ? `:${port}` : ''}`)])];
        return json(201, { id, receiver, sender, expiresInMinutes: ttl / 60000,
          origins: hosts.map(h => `${protocol}//${h}`) });
      }
      const [, , id, action] = pathname.split('/'), s = sessions.get(id);
      if (!s) return json(404, { error: 'Pairing expired or server restarted. Create a new phone link on the computer.' });
      const auth = req.headers.authorization?.replace(/^Bearer /, '');
      const sending = action === 'upload';
      if (auth !== (sending ? s.sender : s.receiver)) return json(403, { error: 'Invalid pairing key.' });
      s.touched = Date.now();
      if (sending && req.method === 'POST') {
        if (s.media || s.uploading) return json(409, { error: 'The previous capture is waiting on the computer. Load or dismiss it first.' });
        const size = Number(req.headers['content-length']);
        if (!Number.isSafeInteger(size) || size <= 0 || size > maxBytes) return json(413, { error: 'Capture must be nonempty and below 200 MB.' });
        const type = String(req.headers['content-type'] ?? '').split(';')[0];
        if (!/^(image|video)\/[a-z0-9.+-]+$/i.test(type) && type !== 'application/octet-stream')
          return json(415, { error: 'Send a photo or video file.' });
        if (reserved + size > totalBytes) return json(429, { error: 'Transfer memory is full. Load or dismiss pending captures first.' });
        const name = decodeURIComponent(String(req.headers['x-file-name'] ?? 'capture')).slice(0, 180).replace(/[\x00-\x1f/\\]/g, '_');
        const kind = req.headers['x-media-kind'];
        if (!['photo', 'video'].includes(kind)) return json(400, { error: 'Choose photo or video.' });
        s.uploading = true; reserved += size;
        let kept = false;
        try {
          const chunks = []; let count = 0;
          for await (const chunk of req) {
            count += chunk.length;
            if (count > size) throw new Error('Capture exceeded its declared size.');
            chunks.push(chunk);
          }
          if (count !== size || !sessions.has(id)) throw new Error('Transfer interrupted or pairing closed.');
          s.media = { id: token(), name, type, kind, data: Buffer.concat(chunks) };
          s.touched = Date.now(); kept = true;
          return json(201, { received: true });
        } finally { s.uploading = false; if (!kept) reserved -= size; }
      }
      if (action === 'pending' && req.method === 'GET') {
        const m = s.media;
        return json(200, m ? { id: m.id, name: m.name, type: m.type, kind: m.kind, size: m.data.length } : null);
      }
      if (action === 'media' && req.method === 'GET' && s.media) {
        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Content-Length', s.media.data.length); return res.end(s.media.data);
      }
      if (action === 'media' && req.method === 'DELETE') {
        if (s.media) reserved -= s.media.data.length;
        s.media = null; return json(200, { cleared: true });
      }
      if (action === 'close' && req.method === 'DELETE') { remove(id); return json(200, { closed: true }); }
      return json(404, { error: 'Capture unavailable.' });
    } catch (e) { if (!res.headersSent) json(400, { error: e.message }); else res.end(); }
  };
  middleware.close = () => { clearInterval(timer); for (const id of sessions.keys()) remove(id); };
  return middleware;
}

export function phoneBridge() {
  const configure = server => {
    const middleware = phoneBridgeMiddleware(); server.middlewares.use(middleware);
    server.httpServer?.once('close', middleware.close);
  };
  return { name: 'local-phone-capture', configureServer: configure, configurePreviewServer: configure };
}
