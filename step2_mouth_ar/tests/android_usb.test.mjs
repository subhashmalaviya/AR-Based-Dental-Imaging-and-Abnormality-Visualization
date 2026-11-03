import test from 'node:test';
import assert from 'node:assert/strict';
import { createUsbKeeper, parseDevices } from '../scripts/android-usb.mjs';

test('USB forwarding is restored after disconnect without switching phones', async () => {
  let devices = 'List of devices attached\nphoneA device\n', rules = '', restores = 0;
  const check = createUsbKeeper({ run: async args => {
    if (args[0] === 'devices') return devices;
    assert.equal(args[1], 'phoneA');
    if (args.at(-1) === '--list') return rules;
    assert.deepEqual(args.slice(2), ['reverse', 'tcp:5173', 'tcp:5173']);
    restores++; rules = 'UsbFfs tcp:5173 tcp:5173\n'; return '';
  } });
  assert.match(await check(), /USB ready/);
  await check(); assert.equal(restores, 1);
  devices = 'List of devices attached\nphoneB device\n'; rules = '';
  assert.match(await check(), /Waiting/); assert.equal(restores, 1);
  devices = 'List of devices attached\nphoneA unauthorized\n';
  assert.match(await check(), /allow USB debugging/); assert.equal(restores, 1);
  devices = 'List of devices attached\nphoneA device\n';
  assert.match(await check(), /USB ready/); assert.equal(restores, 2);
});

test('USB helper waits for a single device and validates ports', async () => {
  assert.deepEqual(parseDevices('List of devices attached\nabc offline\n\n'), [{ serial: 'abc', state: 'offline' }]);
  assert.throws(() => createUsbKeeper({ port: 0 }), /Port/);
  const check = createUsbKeeper({ run: async args => {
    assert.deepEqual(args, ['devices']); return 'List of devices attached\na device\nb device\n';
  } });
  assert.match(await check(), /Multiple devices/);
});
