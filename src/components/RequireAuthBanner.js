import React, { useEffect, useState } from "react";
import Link from "@docusaurus/Link";
import { getSupabase } from "../lib/supabaseClient";

export default function RequireAuthBanner({ children }) {
  const [authed, setAuthed] = useState(false);
  const [available, setAvailable] = useState(null);
  useEffect(() => {
    const sb = getSupabase();
    setAvailable(Boolean(sb));
    if (sb) {
      sb.auth.getUser().then(({ data }) => {
        const ok = !!data?.user;
        setAuthed(ok);
      });
      const { data: sub } = sb.auth.onAuthStateChange((_event, session) => {
        const ok = !!session?.user;
        setAuthed(ok);
      });
      return () => sub?.subscription?.unsubscribe?.();
    }
    return undefined;
  }, []);

  return (
    <div>
      {!authed && (
        <div className="app-card" style={{ marginBottom: 12 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span className="app-muted">
              {available === false
                ? "Account service is unavailable."
                : "Please sign in to use this protected feature."}
            </span>
            {available && (
              <Link className="button button--secondary" to="/auth">
                Sign in
              </Link>
            )}
          </div>
        </div>
      )}
      {authed ? children : null}
    </div>
  );
}
