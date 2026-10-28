import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { createServer } from 'node:http';
import { imageMime, checkImageDimensions, checkEditCrop } from '../src/ui/imageInput.js';
import { phoneBridgeMiddleware } from '../scripts/phone-bridge.mjs';

test('JPEG data with a DNG extension and full-resolution PNG are recognized without rescaling', async () => {
  assert.equal(imageMime(Uint8Array.from([255, 216, 255]), 'image/dng'), 'image/jpeg');
  assert.equal(imageMime(Uint8Array.from([137, 80, 78, 71])), 'image/png');
  assert.equal(imageMime(new TextEncoder().encode('II*\0')), 'image/tiff');
  assert.doesNotThrow(() => checkImageDimensions(3024, 4032));
  const fullPhoto = [0, 0, 3024, 4032];
  assert.doesNotThrow(() => checkEditCrop(fullPhoto));
  assert.deepEqual(fullPhoto, [0, 0, 3024, 4032], 'full-photo comparison preserves original coordinates');
  assert.doesNotThrow(() => checkEditCrop([0, 0, 8000, 4000]));
  assert.throws(() => checkEditCrop([0, 0, 8000, 4001]), /32 megapixels/);
  assert.throws(() => checkEditCrop([0, 0, NaN, 4032]), /Invalid comparison/);
  assert.doesNotThrow(() => checkEditCrop([1000, 1500, 1000, 1000]));
  assert.throws(() => checkImageDimensions(20000, 20000), /32 megapixels/);
});

test('phone relay authenticates roles, preserves bytes until acknowledged and revokes pairing', async t => {
  const middleware = phoneBridgeMiddleware({ maxBytes: 100, totalBytes: 150 });
  const server = createServer((req, res) => middleware(req, res, () => { res.statusCode = 404; res.end(); }));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => { middleware.close(); await new Promise(resolve => server.close(resolve)); });
  const base = `http://127.0.0.1:${server.address().port}/__phone/`;
  const request = (path, key, options = {}) => fetch(base + path, { ...options, headers: {
    'X-Dental-Client': '1', ...(key ? { Authorization: `Bearer ${key}` } : {}), ...options.headers,
  } });
  assert.equal((await fetch(base + 'session', { method: 'POST' })).status, 403);
  assert.equal((await request('session', null, { method: 'POST', headers: { Origin: 'https://unrelated.example' } })).status, 403);
  const session = await (await request('session', null, { method: 'POST' })).json();
  const { id, receiver, sender } = session;
  assert.notEqual(receiver, sender);
  const upload = { method: 'POST', headers: { 'Content-Type': 'image/png', 'X-Media-Kind': 'photo', 'X-File-Name': encodeURIComponent('phone photo.png') }, body: Buffer.from([0, 1, 2, 255]) };
  assert.equal((await request(`${id}/upload`, receiver, upload)).status, 403);
  assert.equal((await request(`${id}/upload`, sender, { ...upload, body: Buffer.alloc(101) })).status, 413);
  assert.equal((await request(`${id}/upload`, sender, upload)).status, 201);
  assert.equal((await request(`${id}/upload`, sender, upload)).status, 409);
  assert.equal((await request(`${id}/pending`, sender)).status, 403);
  const pending = await (await request(`${id}/pending`, receiver)).json();
  assert.equal(pending.name, 'phone photo.png'); assert.equal(pending.kind, 'photo');
  for (let i = 0; i < 2; i++) assert.deepEqual(new Uint8Array(await (await request(`${id}/media`, receiver)).arrayBuffer()), new Uint8Array(upload.body));
  await request(`${id}/media`, receiver, { method: 'DELETE' });
  assert.equal(await (await request(`${id}/pending`, receiver)).json(), null);
  assert.equal((await request(`${id}/upload`, sender, upload)).status, 201);
  await request(`${id}/close`, receiver, { method: 'DELETE' });
  assert.equal((await request(`${id}/upload`, sender, upload)).status, 404);
});

test('actual supplied DNG exports contain JPEG data and all supplied PNGs fit the new import limit', async t => {
  const folder = new URL('../../testvideophoto/', import.meta.url);
  let names;
  try { names = await readdir(folder); } catch { t.skip('Optional local user fixtures unavailable'); return; }
  for (const name of names.filter(n => /^IMG_.*\.(DNG|png)$/i.test(n))) {
    const data = await readFile(new URL(name, folder));
    if (/\.dng$/i.test(name)) assert.equal(imageMime(data.subarray(0, 32)), 'image/jpeg', name);
    else { assert.equal(imageMime(data), 'image/png'); checkImageDimensions(data.readUInt32BE(16), data.readUInt32BE(20)); }
  }
});
