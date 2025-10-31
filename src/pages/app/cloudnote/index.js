// src/pages/app/cloudnote/index.js
import React, { useEffect, useState } from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
const BACKEND_BASE = (typeof window !== 'undefined' && window.__CLOUDNOTE_BACKEND_URL__) ? window.__CLOUDNOTE_BACKEND_URL__ : "";
/*
  Cloud Note (static) — zero-backend sharing via encrypted URL fragment
  - Creates shareable links like https://.../app/cloudnote#note=<BASE64_PAYLOAD>
  - If no password: generated random encryption key is appended (so link-holder can decrypt).
  - If password provided: salt is stored and recipient must enter password to decrypt.
  - Fields: name, content, expires (ISO datetime, optional), password (optional), readonly boolean.
  - When opening a link, the page tries to parse #note=... and prompt for password if needed.
*/

function buf2hex(buffer) {
  return Array.prototype.map
    .call(new Uint8Array(buffer), (x) => ("00" + x.toString(16)).slice(-2))
    .join("");
}
function hex2buf(hex) {
  if (!hex) return new Uint8Array();
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return bytes.buffer;
}
function b64encode(buf) {
  return btoa(String.fromCharCode(...new Uint8Array(buf)));
}
function b64decode(s) {
  const str = atob(s);
  const arr = new Uint8Array(str.length);
  for (let i = 0; i < str.length; i++) arr[i] = str.charCodeAt(i);
  return arr.buffer;
}

// derive key from password using PBKDF2 (returns CryptoKey for AES-GCM)
async function deriveKeyFromPassword(password, saltHex) {
  const saltBuf = hex2buf(saltHex);
  const pwUtf8 = new TextEncoder().encode(password);
  const baseKey = await window.crypto.subtle.importKey(
    "raw",
    pwUtf8,
    "PBKDF2",
    false,
    ["deriveKey"]
  );
  const key = await window.crypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      salt: saltBuf,
      iterations: 200_000,
      hash: "SHA-256",
    },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
  return key;
}

// generate random key (CryptoKey), export as hex
async function generateRawKeyHex() {
  const key = await crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, true, [
    "encrypt",
    "decrypt",
  ]);
  const raw = await crypto.subtle.exportKey("raw", key);
  return buf2hex(raw);
}
async function importRawKeyFromHex(hex) {
  const buf = hex2buf(hex);
  return window.crypto.subtle.importKey("raw", buf, "AES-GCM", false, ["decrypt", "encrypt"]);
}

// encrypt JSON object with AES-GCM and key CryptoKey (or key hex import)
async function encryptWithKeyObj(keyCryptoKey, jsonObj) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const plain = new TextEncoder().encode(JSON.stringify(jsonObj));
  const ct = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, keyCryptoKey, plain);
  return {
    iv: buf2hex(iv.buffer),
    ct: b64encode(ct),
  };
}

// decrypt payload using CryptoKey
async function decryptWithKeyObj(keyCryptoKey, ivHex, ctB64) {
  const iv = hex2buf(ivHex);
  const ct = b64decode(ctB64);
  const plainBuf = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, keyCryptoKey, ct);
  const txt = new TextDecoder().decode(plainBuf);
  return JSON.parse(txt);
}

// helper: returns hex salt
function genSaltHex() {
  const s = crypto.getRandomValues(new Uint8Array(16));
  return buf2hex(s.buffer);
}
// -------------------- Optional backend storage helpers --------------------
// If you deploy a tiny backend and set `window.__CLOUDNOTE_BACKEND_URL__` to its base URL,
// the app will attempt to save/fetch encrypted payloads under note names so different
// devices can open by name + password. If BACKEND_BASE is empty, behavior falls back
// to localStorage only (single-browser only).

async function backendSavePayload(name, payload) {
  if (!BACKEND_BASE || !name) return;
  try {
    await fetch(`${BACKEND_BASE.replace(/\/$/, '')}/notes/${encodeURIComponent(name)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    console.warn('backendSavePayload failed', e);
  }
}

async function backendFetchPayload(name) {
  if (!BACKEND_BASE || !name) return null;
  try {
    const r = await fetch(`${BACKEND_BASE.replace(/\/$/, '')}/notes/${encodeURIComponent(name)}`);
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    console.warn('backendFetchPayload failed', e);
    return null;
  }
}

// -------------------------------------------------------------------------
// Build share token and put it in URL fragment
// If password provided: derive key from password and store salt/iv/ct in payload -> recipient needs password
// If no password: create random rawKeyHex, encrypt using that key, and append rawKeyHex to fragment so link-holder can decrypt
// Build share token and put it in URL fragment
// If password provided: derive key from password and store salt/iv/ct in payload -> recipient needs password
// If no password: create random rawKeyHex, encrypt using that key, and append rawKeyHex to fragment so link-holder can decrypt
async function buildShareFragment(noteObj, password = "") {
  // noteObj: { name, content, createdAtISO, expiresAtISO|null, readOnly: bool }
  if (password && password.length > 0) {
    // password-protected: derive key, but do not include key in link
    const salt = genSaltHex();
    const key = await deriveKeyFromPassword(password, salt);
    const enc = await encryptWithKeyObj(key, noteObj);
    const payload = {
      mode: "pw",
      salt, // hex
      iv: enc.iv,
      ct: enc.ct,
      meta: { v: 1 },
    };
    // persist payload locally so it can be opened by name+password later (client-only storage)
    try {
      if (noteObj && noteObj.name) {
        try {
          window.localStorage.setItem(
            `cloudnote_store:${noteObj.name}`,
            JSON.stringify(payload)
          );
        } catch (e) {
          // ignore localStorage errors (quota, private mode)
          console.warn("Could not save cloudnote to localStorage:", e);
        }
      // also attempt to save to optional backend so other devices can fetch by name
        try {
         await backendSavePayload(noteObj && noteObj.name, payload);
        } catch (e) {
         console.warn('backend save failed', e);
        }
      }
    } catch (e) {
      console.warn("localStorage unavailable:", e);
    }
    const b = btoa(JSON.stringify(payload));
    return `note=${b}`; // link: #note=BASE64_JSON
  } else {
    // no password: generate random key and include it in fragment (anyone with link can decrypt)
    const rawKeyHex = await generateRawKeyHex();
    const keyCrypto = await importRawKeyFromHex(rawKeyHex);
    const enc = await encryptWithKeyObj(keyCrypto, noteObj);
    const payload = {
      mode: "key",
      iv: enc.iv,
      ct: enc.ct,
      meta: { v: 1 },
    };
    const b = btoa(JSON.stringify(payload));
    // include key after a dot, URL-safe
    return `note=${b}.${rawKeyHex}`;
  }
}

// parse fragment like '#note=<b64>' or '#note=<b64>.<rawHexKey>'
function parseFragmentHash() {
  const h = location.hash || "";
  if (!h.includes("note=")) return null;
  const after = h.split("note=")[1];
  if (!after) return null;
  const [b64, maybeKey] = after.split(".");
  try {
    const payload = JSON.parse(atob(b64));
    return { payload, rawKeyHex: maybeKey || null };
  } catch (e) {
    return null;
  }
}

// helper to retrieve stored payload by note name
async function getStoredPayloadByName(name) {
  try {
    const s = window.localStorage.getItem(`cloudnote_store:${name}`);
    if (s) {
      try { return JSON.parse(s); } catch (e) { /* fallback to backend */ }
    }
  } catch (e) {
    console.warn("Failed to read stored payload from localStorage:", e);
  }
  // if not in localStorage and backend configured, try fetch
  if (BACKEND_BASE) {
    try {
      const remote = await backendFetchPayload(name);
      if (remote) return remote;
    } catch (e) {
      console.warn('backendFetchPayload error', e);
    }
  }
  return null;
}

// 这个函数将被移动到组件内部

export default function CloudNotePage() {
  const [name, setName] = useState("");
  const [content, setContent] = useState("");
  const [expires, setExpires] = useState("");
  const [password, setPassword] = useState("");
  const [readOnly, setReadOnly] = useState(false);
  const [generatedLink, setGeneratedLink] = useState("");
  const [status, setStatus] = useState("");
  const [openedNote, setOpenedNote] = useState(null);
  const [requirePassword, setRequirePassword] = useState(false);
  const [openPassword, setOpenPassword] = useState("");
  const [parsedFrag, setParsedFrag] = useState(null);
  const [lookupName, setLookupName] = useState("");
  const [lookupPassword, setLookupPassword] = useState("");

  // on mount, try parse fragment
  useEffect(() => {
    const parsed = parseFragmentHash();
    setParsedFrag(parsed);
    if (parsed) {
      // if mode==pw -> require password; if mode==key -> try auto-decrypt
      if (parsed.payload.mode === "pw") {
        setRequirePassword(true);
        setStatus("Protected note — enter password to open.");
      } else if (parsed.payload.mode === "key") {
        // raw key is present? (likely included in fragment)
        if (parsed.rawKeyHex) {
          (async () => {
            try {
              const keyCrypto = await importRawKeyFromHex(parsed.rawKeyHex);
              const noteObj = await decryptWithKeyObj(keyCrypto, parsed.payload.iv, parsed.payload.ct);
              // check expiry
              const ok = checkExpiryAndReturn(noteObj);
              if (ok) {
                setOpenedNote(noteObj);
                setStatus("Opened note from link.");
              }
            } catch (e) {
              console.error(e);
              setStatus("Failed to decrypt with key from link.");
            }
          })();
        } else {
          setStatus("Shared link missing key — cannot open.");
        }
      } else {
        setStatus("Unknown link mode.");
      }
    }
  }, []);

  function checkExpiryAndReturn(noteObj) {
    if (!noteObj) return false;
    if (noteObj.expiresAtISO) {
      const exp = new Date(noteObj.expiresAtISO);
      if (isNaN(exp.getTime())) {
        // ignore invalid
      } else {
        if (Date.now() > exp.getTime()) {
          setStatus("This note has expired.");
          return false;
        }
      }
    }
    return true;
  }

  async function onOpenWithPassword() {
    if (!parsedFrag) return;
    try {
      const key = await deriveKeyFromPassword(openPassword, parsedFrag.payload.salt);
      const noteObj = await decryptWithKeyObj(key, parsedFrag.payload.iv, parsedFrag.payload.ct);
      if (!checkExpiryAndReturn(noteObj)) return;
      setOpenedNote(noteObj);
      setStatus("Opened note with password.");
      setRequirePassword(false);
    } catch (e) {
      console.error(e);
      setStatus("Wrong password or decryption failed.");
    }
  }
  
  // Open a stored password-protected note by name + password
  async function onOpenByNamePassword() {
    if (!lookupName) {
      setStatus("Please enter a note name to open.");
      return;
    }
    const stored = await getStoredPayloadByName(lookupName);
    if (!stored) {
      setStatus("No stored note with that name found in this browser.");
      return;
    }
    if (stored.mode !== "pw") {
      setStatus("Stored note is not password-protected or cannot be opened by name/password.");
      return;
    }
    if (!lookupPassword) {
      setStatus("Please enter the password to open this note.");
      return;
    }

    try {
      const key = await deriveKeyFromPassword(lookupPassword, stored.salt);
      const noteObj = await decryptWithKeyObj(key, stored.iv, stored.ct);
      if (!checkExpiryAndReturn(noteObj)) return;
      setOpenedNote(noteObj);
      setStatus("Opened note by name + password.");
      // If we successfully opened by name+password, clear parsedFrag/password-prompt state
      setRequirePassword(false);
      setParsedFrag(null);
    } catch (e) {
      console.error(e);
      setStatus("Wrong password or decryption failed.");
    }
  }

  async function onGenerateLink() {
    try {
      setStatus("Generating link...");
      const noteObj = {
        name: name || "Untitled",
        content: content || "",
        createdAtISO: new Date().toISOString(),
        expiresAtISO: expires ? new Date(expires).toISOString() : null,
        readOnly: !!readOnly,
      };
      const frag = await buildShareFragment(noteObj, password);
      const full = `${location.origin}${location.pathname}#${frag}`;
      setGeneratedLink(full);
      setStatus("Link generated. Copy and share.");
    } catch (e) {
      console.error(e);
      setStatus("Failed to generate link: " + e.message);
    }
  }

  function copyLink() {
    if (!generatedLink) return;
    navigator.clipboard
      .writeText(generatedLink)
      .then(() => setStatus("Link copied to clipboard."))
      .catch((e) => setStatus("Copy failed: " + e.message));
  }

  function clearFragmentView() {
    // remove fragment from URL without reload
    history.replaceState(null, "", location.pathname + location.search);
    setParsedFrag(null);
    setOpenedNote(null);
    setStatus("");
  }

  // allow editing the openedNote only if opened and note.readOnly==false AND link provided edit key (for 'key' mode we can allow)
  function canEditOpened() {
    if (!openedNote) return false;
    // For password-protected notes, we don't have server-side control; treat as editable if not marked readOnly
    return !openedNote.readOnly;
  }

  async function onSaveEditAndRegenerate() {
    if (!openedNote) return;
    // produce new fragment using existing openPassword if it was password-protected, else generate new link with key
    const noteObj = {
      ...openedNote,
      content: openedNote.content,
      createdAtISO: openedNote.createdAtISO || new Date().toISOString(),
    };
    // We attempt to reuse openPassword when parsedFrag.mode==='pw' and user provided it
    let pw = "";
    if (parsedFrag && parsedFrag.payload.mode === "pw") pw = openPassword;
    const frag = await buildShareFragment(noteObj, pw);
    const full = `${location.origin}${location.pathname}#${frag}`;
    setGeneratedLink(full);
    setStatus("Regenerated link for edited note. Copy to share.");
  }

  return (
    <Layout title="Cloud Sticky Notes">
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div style={{ maxWidth: 900, margin: "28px auto", padding: "12px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h1>Cloud Sticky Note (static)</h1>
          <a 
            href="/docs/tutorial-apps/cloud-sticky-note-tutorial" 
            style={{
              padding: "8px 16px",
              backgroundColor: "#000000",
              color: "#ffffff",
              textDecoration: "none",
              borderRadius: "10px",
              fontSize: "14px",
              fontWeight: "500",
              border: "1px solid #000000",
              transition: "all 0.2s ease"
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = "#333333";
              e.target.style.borderColor = "#333333";
              e.target.style.transform = "translateY(-1px)";
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = "#000000";
              e.target.style.borderColor = "#000000";
              e.target.style.transform = "translateY(0)";
            }}
          >
             Tutorial
          </a>
        </div>
        <p style={{ color: "#444" }}>
          Create a temporary/shareable note. This zero-backend version stores the note encrypted in the URL fragment.
          Sharing the link allows others to open it. Use a password for higher privacy (recipient must know the password).
        </p>

        <div style={{ display: "grid", gap: 10, marginTop: 12 }}>
          <label>
            Note name
            <input style={styles.input} value={name} onChange={(e) => setName(e.target.value)} />
          </label>

          <label>
            Content
            <textarea rows={8} style={styles.textarea} value={content} onChange={(e) => setContent(e.target.value)} />
          </label>

          <div style={{ display: "flex", gap: 12 }}>
            <label style={{ flex: 1 }}>
              Expires (optional)
              <input type="datetime-local" style={styles.input} value={expires} onChange={(e) => setExpires(e.target.value)} />
            </label>
            <label style={{ flexBasis: 200 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <input type="checkbox" checked={readOnly} onChange={(e) => setReadOnly(e.target.checked)} />
                <div>Read-only link</div>
              </div>
              <small style={{ color: "#666" }}>Marking read-only is advisory in static links.</small>
            </label>
            <label style={{ flexBasis: 260 }}>
              <div>Protect with password (optional)</div>
              <input style={styles.input} value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Optional password" />
              <small style={{ color: "#666" }}>
                If set, recipients will need the password to open the note.
              </small>
            </label>
          </div>

          <div style={{ display: "flex", gap: 10 }}>
            <button onClick={onGenerateLink} style={styles.btnPrimary}>Generate share link</button>
            <button onClick={copyLink} style={styles.btnSecondary} disabled={!generatedLink}>Copy link</button>
            <button onClick={() => { navigator.clipboard.writeText(content || ""); }} style={styles.btnGhost}>Copy content</button>
            <button onClick={() => { setName(""); setContent(""); setExpires(""); setPassword(""); setReadOnly(false); setGeneratedLink(""); setStatus(""); }} style={styles.btnGhost}>Reset</button>
          </div>

          {generatedLink && (
            <div style={{ background: "#f5f5f7", padding: 8, borderRadius: 6 }}>
              <div style={{ fontSize: 13 }}>Share link (fragment)</div>
              <textarea readOnly rows={2} style={styles.shareArea} value={generatedLink} />
            </div>
          )}

          <div style={{ marginTop: 6 }}>
            <strong>Status:</strong> {status}
          </div>

          <hr />

          <h3>Open shared note (from URL)</h3>
        
          <div style={{ marginTop: 12 }}>
            <h4>Open by name & password</h4>
            <div style={{ display: "grid", gap: 8, maxWidth: 520 }}>
              <label>
                Note name
                <input style={styles.input} value={lookupName} onChange={(e) => setLookupName(e.target.value)} />
              </label>
              <label>
                Password
                <input type="password" style={styles.input} value={lookupPassword} onChange={(e) => setLookupPassword(e.target.value)} />
              </label>

              <div style={{ display: "flex", gap: 8 }}>
                <button onClick={onOpenByNamePassword} style={styles.btnPrimary}>Open</button>
                <button onClick={() => { setLookupName(""); setLookupPassword(""); setStatus(""); }} style={styles.btnGhost}>Clear</button>
              </div>
              <div style={{ color: "#666", fontSize: 13 }}>
                Notes saved by password-protected generation (this browser) are accessible by entering the same name and password.
              </div>
            </div>
          </div>

          {parsedFrag ? (
            <div>
              <div style={{ padding: 8, background: "#f5f5f7", borderRadius: 6 }}>
                Parsed shared link found in URL fragment. Mode: <b>{parsedFrag.payload.mode}</b>.
              </div>
              {parsedFrag.payload.mode === "pw" && (
                <div style={{ marginTop: 8 }}>
                  <label>Enter password to open</label>
                  <input style={styles.input} value={openPassword} onChange={(e) => setOpenPassword(e.target.value)} />
                  <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                    <button onClick={onOpenWithPassword} style={styles.btnPrimary}>Open</button>
                    <button onClick={clearFragmentView} style={styles.btnGhost}>Clear</button>
                  </div>
                </div>
              )}

              {parsedFrag.payload.mode === "key" && (
                <div style={{ marginTop: 8 }}>
                  <div>Key present in fragment (auto-decrypt attempted).</div>
                  <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                    <button onClick={clearFragmentView} style={styles.btnGhost}>Clear</button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div>No share fragment found in URL. To open a note, paste a link with <code>#note=...</code> and reload.</div>
          )}

          {openedNote && (
            <div style={{ marginTop: 12, padding: 12, borderRadius: 8, background: "#f5f5f7" }}>
              <h3>{openedNote.name}</h3>
              <div style={{ color: "#666", fontSize: 13 }}>
                Created: {openedNote.createdAtISO}
                {openedNote.expiresAtISO && <span> · Expires: {openedNote.expiresAtISO}</span>}
                {openedNote.readOnly && <span> · Read-only</span>}
              </div>

              <div style={{ marginTop: 8 }}>
                <textarea style={styles.textarea} rows={8} value={openedNote.content} readOnly={!canEditOpened()}
                  onChange={(e) => setOpenedNote({ ...openedNote, content: e.target.value })} />
              </div>

              <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                <button disabled={!canEditOpened()} onClick={async () => { await onSaveEditAndRegenerate(); }} style={styles.btnPrimary}>
                  Save edits & regenerate link
                </button>
                <button onClick={() => { navigator.clipboard.writeText(openedNote.content || ""); }} style={styles.btnSecondary}>Copy content</button>
                <button onClick={clearFragmentView} style={styles.btnGhost}>Close</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}

const styles = {
  input: { width: "100%", padding: "8px 10px", marginTop: 6, borderRadius: 6, border: "1px solid #ccc" },
  textarea: { width: "100%", padding: "8px 10px", marginTop: 6, borderRadius: 6, border: "1px solid #ccc", fontFamily: "inherit" },
  shareArea: { width: "100%", borderRadius: 6, border: "1px solid #ddd", padding: 6 },
  btnPrimary: { background: "#6c6c70", color: "#ffffff", padding: "8px 12px", borderRadius: 6, border: "1px solid #6c6c70", cursor: "pointer" },
  btnSecondary: { background: "#6c6c70", color: "#ffffff", padding: "8px 12px", borderRadius: 6, border: "1px solid #6c6c70", cursor: "pointer" },
  btnGhost: { background: "#fff", color: "#111", padding: "8px 12px", borderRadius: 6, border: "1px solid #ddd", cursor: "pointer" },
};