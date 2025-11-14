import React, { useEffect, useState } from 'react';
import { getSession } from '../lib/auth';
import { getSupabase } from '../lib/supabaseClient';

export default function RequireAuthBanner({ children }) {
  const [authed, setAuthed] = useState(false);
  useEffect(() => {
    const sb = getSupabase();
    if (sb) {
      sb.auth.getUser().then(({ data }) => {
        const ok = !!data?.user;
        setAuthed(ok);
        if (typeof window !== 'undefined') window.__APP_AUTH_OK__ = ok;
      });
      const { data: sub } = sb.auth.onAuthStateChange((_event, session) => {
        const ok = !!session?.user;
        setAuthed(ok);
        if (typeof window !== 'undefined') window.__APP_AUTH_OK__ = ok;
      });
      return () => sub?.subscription?.unsubscribe?.();
    }
    const local = !!getSession();
    setAuthed(local);
    if (typeof window !== 'undefined') window.__APP_AUTH_OK__ = local;
  }, []);

  return (
    <div>
      {!authed && (
        <div className="app-card" style={{ marginBottom: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="app-muted">Please login to use app features</span>
            <a className="button button--secondary" href="/auth">Login / Register</a>
          </div>
        </div>
      )}
      {children}
    </div>
  );
}