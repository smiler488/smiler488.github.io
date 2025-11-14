export function _getUsers() {
  const raw = typeof window !== 'undefined' ? window.localStorage.getItem('app_users') : null;
  if (!raw) return [];
  try { return JSON.parse(raw) || []; } catch { return []; }
}

export function _saveUsers(users) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem('app_users', JSON.stringify(users));
}

function toHex(buf) {
  return Array.prototype.map.call(new Uint8Array(buf), (x) => ('00' + x.toString(16)).slice(-2)).join('');
}

function fromHex(hex) {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  return bytes.buffer;
}

function genSaltHex() {
  const s = crypto.getRandomValues(new Uint8Array(16));
  return toHex(s.buffer);
}

async function deriveHashHex(password, saltHex) {
  const pwUtf8 = new TextEncoder().encode(password);
  const saltBuf = fromHex(saltHex);
  const baseKey = await crypto.subtle.importKey('raw', pwUtf8, 'PBKDF2', false, ['deriveBits']);
  const bits = await crypto.subtle.deriveBits({ name: 'PBKDF2', salt: saltBuf, iterations: 150000, hash: 'SHA-256' }, baseKey, 256);
  return toHex(bits);
}

export async function register(email, password) {
  const users = _getUsers();
  const exists = users.find((u) => u.email.toLowerCase() === String(email).toLowerCase());
  if (exists) throw new Error('Email already registered');
  const saltHex = genSaltHex();
  const hashHex = await deriveHashHex(password, saltHex);
  users.push({ email, saltHex, hashHex, createdAt: new Date().toISOString() });
  _saveUsers(users);
  setSession({ email });
  return { email };
}

export async function login(email, password) {
  const users = _getUsers();
  const u = users.find((x) => x.email.toLowerCase() === String(email).toLowerCase());
  if (!u) throw new Error('Account not found');
  const hashHex = await deriveHashHex(password, u.saltHex);
  if (hashHex !== u.hashHex) throw new Error('Invalid credentials');
  setSession({ email });
  return { email };
}

export function getSession() {
  if (typeof window === 'undefined') return null;
  const raw = window.localStorage.getItem('app_session');
  if (!raw) return null;
  try { return JSON.parse(raw) || null; } catch { return null; }
}

export function setSession(sess) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem('app_session', JSON.stringify({ email: sess.email, at: Date.now() }));
}

export function logout() {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem('app_session');
}