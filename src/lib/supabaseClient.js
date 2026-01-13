import { createClient } from '@supabase/supabase-js';

let client = null;
export function getSupabase() {
  if (client) return client;
  if (typeof window === 'undefined') return null;
  const url = window.__SUPABASE_URL__;
  const key = window.__SUPABASE_ANON_KEY__;
  if (!url || !key) return null;
  client = createClient(url, key, {
    auth: { persistSession: true, autoRefreshToken: true },
  });
  return client;
}