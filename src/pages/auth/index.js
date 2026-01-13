import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import { register, login, getSession, logout } from '../../lib/auth';
import { getSupabase } from '../../lib/supabaseClient';

export default function AccountPage() {
  const [supabaseUser, setSupabaseUser] = useState(null);
  const sess = getSession();
  const [tab, setTab] = useState((sess || supabaseUser) ? 'profile' : 'login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [password2, setPassword2] = useState('');
  const [msg, setMsg] = useState('');

  useEffect(() => {
    const sb = getSupabase();
    if (!sb) return;
    sb.auth.getUser().then(({ data }) => {
      if (data?.user) {
        setSupabaseUser(data.user);
        setTab('profile');
      }
    });
    const { data: sub } = sb.auth.onAuthStateChange((_event, session) => {
      setSupabaseUser(session?.user || null);
      if (session?.user) setTab('profile');
    });
    return () => sub?.subscription?.unsubscribe?.();
  }, []);

  async function doLogin() {
    setMsg('');
    const sb = getSupabase();
    if (sb) {
      const { data, error } = await sb.auth.signInWithPassword({ email, password });
      if (error) { setMsg(error.message); return; }
      setSupabaseUser(data.user);
      setTab('profile');
      return;
    }
    try { await login(email, password); setMsg('Logged in'); setTab('profile'); } catch (e) { setMsg(String(e.message || e)); }
  }

  async function doRegister() {
    setMsg('');
    if (!email || !password) { setMsg('Enter email and password'); return; }
    if (password !== password2) { setMsg('Passwords do not match'); return; }
    const sb = getSupabase();
    if (sb) {
      const { data, error } = await sb.auth.signUp({ email, password });
      if (error) { setMsg(error.message); return; }
      setMsg('Registered. Please check your email to confirm.');
      return;
    }
    try { await register(email, password); setMsg('Registered and logged in'); setTab('profile'); } catch (e) { setMsg(String(e.message || e)); }
  }

  async function doLogout() {
    const sb = getSupabase();
    if (sb) { await sb.auth.signOut(); setSupabaseUser(null); }
    logout();
    setTab('login');
    setMsg('Logged out');
  }

  return (
    <Layout title="Account">
      <div className="app-container" style={{ maxWidth: 560 }}>
        <div className="app-header" style={{ marginBottom: 12 }}>
          <h1 className="app-title">Account</h1>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="button button--secondary" onClick={() => setTab('login')}>Login</button>
            <button className="button button--secondary" onClick={() => setTab('register')}>Register</button>
          </div>
        </div>

        {tab === 'login' && (
          <div className="app-card">
            <div style={{ display: 'grid', gap: 12 }}>
              <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} style={{ padding: 12, border: '1px solid var(--ifm-border-color)', borderRadius: 8 }} />
              <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} style={{ padding: 12, border: '1px solid var(--ifm-border-color)', borderRadius: 8 }} />
              <button className="button button--primary" onClick={doLogin}>Login</button>
            </div>
          </div>
        )}

        {tab === 'register' && (
          <div className="app-card">
            <div style={{ display: 'grid', gap: 12 }}>
              <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} style={{ padding: 12, border: '1px solid var(--ifm-border-color)', borderRadius: 8 }} />
              <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} style={{ padding: 12, border: '1px solid var(--ifm-border-color)', borderRadius: 8 }} />
              <input type="password" placeholder="Confirm Password" value={password2} onChange={(e) => setPassword2(e.target.value)} style={{ padding: 12, border: '1px solid var(--ifm-border-color)', borderRadius: 8 }} />
              <button className="button button--primary" onClick={doRegister}>Register</button>
            </div>
          </div>
        )}

        {tab === 'profile' && (
          <div className="app-card" style={{ display: 'grid', gap: 12 }}>
            <div>Signed in as {supabaseUser?.email || getSession()?.email}</div>
            <div style={{ display: 'flex', gap: 8 }}>
              <a className="button button--secondary" href="/app">Go to Apps</a>
              <button className="button button--secondary" onClick={doLogout}>Logout</button>
            </div>
          </div>
        )}

        {msg && (
          <p className="app-muted" style={{ marginTop: 12 }}>{msg}</p>
        )}
      </div>
    </Layout>
  );
}