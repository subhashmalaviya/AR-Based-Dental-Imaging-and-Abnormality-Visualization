import { readdir, realpath, stat, open } from 'node:fs/promises';
import { resolve, relative, isAbsolute, sep } from 'node:path';
import { randomBytes } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const allowedFile = /\.(json|png|jpe?g|webp|bmp|gif|dng|heic|heif|avif|tiff?)$/i;
const inside = (root, path) => { const rel = relative(root, path); return rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel); };

export async function scanPatientFolder(folder, allowedRoot) {
  const root = await realpath(allowedRoot), base = await realpath(resolve(folder));
  if (!inside(root, base)) throw new Error('Path must be inside this project. Use Choose main folder for folders elsewhere.');
  const files = []; let visited = 0;
  async function walk(dir, depth) {
    if (depth > 20) throw new Error('Folder nesting exceeds 20 levels');
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      if (++visited > 20000) throw new Error('Folder is too large. Select a smaller dataset folder.');
      if (entry.name.startsWith('.') || entry.isSymbolicLink() || ['node_modules', 'dist'].includes(entry.name)) continue;
      const path = resolve(dir, entry.name);
      if (entry.isDirectory()) await walk(path, depth + 1);
      else if (entry.isFile() && allowedFile.test(entry.name)) {
        if (files.length >= 5000) throw new Error('Select at most 5000 image/JSON files');
        const size = (await stat(path)).size;
        files.push({ path: relative(base, path).split(sep).join('/'), size, absolute: path });
      }
    }
  }
  await walk(base, 0); return { base, files };
}

export function patientFolderMiddleware(allowedRoot) {
  const manifests = new Map();
  return async (req, res, next) => {
    const path = new URL(req.url, 'http://localhost').pathname;
    if (!path.startsWith('/__batch/')) return next();
    const json = (code, value) => { res.statusCode = code; res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(value)); };
    res.setHeader('Cache-Control', 'no-store'); res.setHeader('X-Content-Type-Options', 'nosniff');
    try {
      const local = ['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress);
      if (!local || req.headers['x-dental-client'] !== '1'
        || (req.headers.origin && new URL(req.headers.origin).host !== req.headers.host)) return json(403, { error: 'Folder paths are available only from this computer. Use Choose main folder instead.' });
      for (const [id, item] of manifests) if (Date.now() - item.touched > 3600000) manifests.delete(id);
      if (path === '/__batch/scan' && req.method === 'POST') {
        let body = ''; for await (const chunk of req) { body += chunk; if (body.length > 8192) throw new Error('Folder path is too long'); }
        const { folder } = JSON.parse(body);
        if (typeof folder !== 'string' || !isAbsolute(folder)) throw new Error('Enter an absolute main folder path');
        const result = await scanPatientFolder(folder, allowedRoot);
        if (manifests.size >= 8) manifests.delete(manifests.keys().next().value);
        const id = randomBytes(24).toString('hex');
        manifests.set(id, { ...result, touched: Date.now() });
        return json(200, { files: result.files.map((f, index) => ({ path: f.path, size: f.size, url: `/__batch/file/${id}/${index}` })) });
      }
      const match = /^\/__batch\/file\/([a-f0-9]{48})\/(\d+)$/.exec(path);
      const manifest = match && manifests.get(match[1]), file = manifest?.files[Number(match?.[2])];
      if (req.method !== 'GET' || !file) return json(404, { error: 'Folder scan expired. Scan again.' });
      const resolved = await realpath(file.absolute);
      if (!inside(manifest.base, resolved)) throw new Error('File moved outside selected folder');
      const handle = await open(resolved, 'r');
      try {
        const info = await handle.stat();
        if (!info.isFile() || info.size > 80 * 1024 * 1024) throw new Error('File is not a regular file below 80 MB');
        manifest.touched = Date.now();
        res.setHeader('Content-Type', 'application/octet-stream');
        res.end(await handle.readFile());
      } finally { await handle.close(); }
    } catch (e) { if (!res.headersSent) json(400, { error: e.message }); else res.end(); }
  };
}

export function patientFolder() {
  const configure = server => { server.middlewares.use(patientFolderMiddleware(fileURLToPath(new URL('../../', import.meta.url)))); };
  return { name: 'local-patient-folder', configureServer: configure, configurePreviewServer: configure };
}
