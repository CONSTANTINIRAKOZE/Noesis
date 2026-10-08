// Prints Walter-in-JavaScript's outputs as JSON, for tests/test_web_parity.py to compare with PyTorch.
import { readFileSync } from 'node:fs';
import { Walter } from '../web/walter.js';

const dir = new URL('../web/', import.meta.url);
const meta = JSON.parse(readFileSync(new URL('walter.json', dir)));
const buf = readFileSync(new URL('walter.bin', dir));
const w = new Walter(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const cases = JSON.parse(process.argv[2]);
console.log(JSON.stringify(cases.map(([fn, ...args]) => {
  try { return w[fn](...args); } catch (e) { return { error: e.message }; }
})));
