import { createClient } from "@supabase/supabase-js";
import siteConfig from "@generated/docusaurus.config";

let client = null;
export function getSupabase() {
  if (client) return client;
  if (typeof window === "undefined") return null;
  const url = siteConfig.customFields?.supabaseUrl || window.__SUPABASE_URL__;
  const key =
    siteConfig.customFields?.supabaseAnonKey || window.__SUPABASE_ANON_KEY__;
  if (!url || !key) return null;
  try {
    const parsedUrl = new URL(url);
    if (parsedUrl.protocol !== "https:" && parsedUrl.hostname !== "localhost")
      return null;
    client = createClient(url, key, {
      auth: { persistSession: true, autoRefreshToken: true },
    });
  } catch {
    return null;
  }
  return client;
}
