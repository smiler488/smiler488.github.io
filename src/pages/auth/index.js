import React, { useEffect, useState } from "react";
import Layout from "@theme/Layout";
import { getSupabase } from "../../lib/supabaseClient";

export default function AccountPage() {
  const [supabaseUser, setSupabaseUser] = useState(null);
  const [available, setAvailable] = useState(null);
  const [tab, setTab] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [password2, setPassword2] = useState("");
  const [msg, setMsg] = useState("");

  useEffect(() => {
    const sb = getSupabase();
    setAvailable(Boolean(sb));
    if (!sb) return undefined;
    sb.auth.getUser().then(({ data }) => {
      if (data?.user) {
        setSupabaseUser(data.user);
        setTab("profile");
      }
    });
    const { data: sub } = sb.auth.onAuthStateChange((_event, session) => {
      setSupabaseUser(session?.user || null);
      if (session?.user) setTab("profile");
    });
    return () => sub?.subscription?.unsubscribe?.();
  }, []);

  async function doLogin() {
    setMsg("");
    const sb = getSupabase();
    if (!sb) {
      setMsg("Account service is not configured.");
      return;
    }
    const { data, error } = await sb.auth.signInWithPassword({
      email,
      password,
    });
    if (error) {
      setMsg(error.message);
      return;
    }
    setSupabaseUser(data.user);
    setTab("profile");
  }

  async function doRegister() {
    setMsg("");
    if (!email || !password) {
      setMsg("Enter email and password");
      return;
    }
    if (password !== password2) {
      setMsg("Passwords do not match");
      return;
    }
    const sb = getSupabase();
    if (!sb) {
      setMsg("Account service is not configured.");
      return;
    }
    const { error } = await sb.auth.signUp({ email, password });
    if (error) {
      setMsg(error.message);
      return;
    }
    setMsg("Registered. Please check your email to confirm.");
  }

  async function doLogout() {
    const sb = getSupabase();
    if (sb) {
      await sb.auth.signOut();
      setSupabaseUser(null);
    }
    setTab("login");
    setMsg("Logged out");
  }

  return (
    <Layout title="Account">
      <div className="app-container" style={{ maxWidth: 560 }}>
        <div className="app-header" style={{ marginBottom: 12 }}>
          <h1 className="app-title">Account</h1>
          {available && (
            <div style={{ display: "flex", gap: 8 }}>
              <button
                className="button button--secondary"
                onClick={() => setTab("login")}
              >
                Login
              </button>
              <button
                className="button button--secondary"
                onClick={() => setTab("register")}
              >
                Register
              </button>
            </div>
          )}
        </div>

        {available === false && (
          <div className="app-card">
            <strong>Account service is not configured</strong>
            <p className="app-muted" style={{ marginBottom: 0, marginTop: 8 }}>
              Local browser-only accounts have been disabled because they do not
              provide real access control. Configure Supabase at build time and
              enforce Row Level Security before enabling sign-in.
            </p>
          </div>
        )}

        {available === null && (
          <p className="app-muted">Checking account service…</p>
        )}

        {available && tab === "login" && (
          <div className="app-card">
            <div style={{ display: "grid", gap: 12 }}>
              <input
                type="email"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                style={{
                  padding: 12,
                  border: "1px solid var(--ifm-border-color)",
                  borderRadius: 8,
                }}
              />
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{
                  padding: 12,
                  border: "1px solid var(--ifm-border-color)",
                  borderRadius: 8,
                }}
              />
              <button className="button button--primary" onClick={doLogin}>
                Login
              </button>
            </div>
          </div>
        )}

        {available && tab === "register" && (
          <div className="app-card">
            <div style={{ display: "grid", gap: 12 }}>
              <input
                type="email"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                style={{
                  padding: 12,
                  border: "1px solid var(--ifm-border-color)",
                  borderRadius: 8,
                }}
              />
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{
                  padding: 12,
                  border: "1px solid var(--ifm-border-color)",
                  borderRadius: 8,
                }}
              />
              <input
                type="password"
                placeholder="Confirm Password"
                value={password2}
                onChange={(e) => setPassword2(e.target.value)}
                style={{
                  padding: 12,
                  border: "1px solid var(--ifm-border-color)",
                  borderRadius: 8,
                }}
              />
              <button className="button button--primary" onClick={doRegister}>
                Register
              </button>
            </div>
          </div>
        )}

        {available && tab === "profile" && (
          <div className="app-card" style={{ display: "grid", gap: 12 }}>
            <div>Signed in as {supabaseUser?.email}</div>
            <div style={{ display: "flex", gap: 8 }}>
              <a className="button button--secondary" href="/app">
                Go to Apps
              </a>
              <button className="button button--secondary" onClick={doLogout}>
                Logout
              </button>
            </div>
          </div>
        )}

        {msg && (
          <p className="app-muted" style={{ marginTop: 12 }}>
            {msg}
          </p>
        )}
      </div>
    </Layout>
  );
}
