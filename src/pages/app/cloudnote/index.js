// src/pages/app/cloudnote/index.js
import React, { useEffect, useState } from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
import CitationNotice from "../../../components/CitationNotice";
import styles from "./styles.module.css";

const BACKEND_BASE = (typeof window !== 'undefined' && window.__CLOUDNOTE_BACKEND_URL__) ? window.__CLOUDNOTE_BACKEND_URL__ : "";

/* 
  Encryption Helpers 
  (Logic unchanged from original)
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

async function deriveKeyFromPassword(password, saltHex) {
  const saltBuf = hex2buf(saltHex);
  const pwUtf8 = new TextEncoder().encode(password);
  const baseKey = await window.crypto.subtle.importKey(
    "raw", pwUtf8, "PBKDF2", false, ["deriveKey"]
  );
  return window.crypto.subtle.deriveKey(
    { name: "PBKDF2", salt: saltBuf, iterations: 200_000, hash: "SHA-256" },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
}

async function generateRawKeyHex() {
  const key = await crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, true, ["encrypt", "decrypt"]);
  const raw = await crypto.subtle.exportKey("raw", key);
  return buf2hex(raw);
}

async function importRawKeyFromHex(hex) {
  const buf = hex2buf(hex);
  return window.crypto.subtle.importKey("raw", buf, "AES-GCM", false, ["decrypt", "encrypt"]);
}

async function encryptWithKeyObj(keyCryptoKey, jsonObj) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const plain = new TextEncoder().encode(JSON.stringify(jsonObj));
  const ct = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, keyCryptoKey, plain);
  return { iv: buf2hex(iv.buffer), ct: b64encode(ct) };
}

async function decryptWithKeyObj(keyCryptoKey, ivHex, ctB64) {
  const iv = hex2buf(ivHex);
  const ct = b64decode(ctB64);
  const plainBuf = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, keyCryptoKey, ct);
  return JSON.parse(new TextDecoder().decode(plainBuf));
}

function genSaltHex() {
  const s = crypto.getRandomValues(new Uint8Array(16));
  return buf2hex(s.buffer);
}

/* Backend Storage Helpers (Optional) */
async function backendSavePayload(name, payload) {
  if (!BACKEND_BASE || !name) return;
  try {
    await fetch(`${BACKEND_BASE.replace(/\/$/, '')}/notes/${encodeURIComponent(name)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (e) { console.warn('backendSavePayload failed', e); }
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

async function buildShareFragment(noteObj, password = "") {
  if (password && password.length > 0) {
    const salt = genSaltHex();
    const key = await deriveKeyFromPassword(password, salt);
    const enc = await encryptWithKeyObj(key, noteObj);
    const payload = { mode: "pw", salt, iv: enc.iv, ct: enc.ct, meta: { v: 1 } };

    try {
      if (noteObj?.name) {
        window.localStorage.setItem(`cloudnote_store:${noteObj.name}`, JSON.stringify(payload));
        await backendSavePayload(noteObj.name, payload);
      }
    } catch (e) { console.warn("storage error:", e); }

    return `note=${btoa(JSON.stringify(payload))}`;
  } else {
    const rawKeyHex = await generateRawKeyHex();
    const keyCrypto = await importRawKeyFromHex(rawKeyHex);
    const enc = await encryptWithKeyObj(keyCrypto, noteObj);
    const payload = { mode: "key", iv: enc.iv, ct: enc.ct, meta: { v: 1 } };
    return `note=${btoa(JSON.stringify(payload))}.${rawKeyHex}`;
  }
}

function parseFragmentHash() {
  const h = location.hash || "";
  if (!h.includes("note=")) return null;
  const after = h.split("note=")[1];
  if (!after) return null;
  const [b64, maybeKey] = after.split(".");
  try {
    return { payload: JSON.parse(atob(b64)), rawKeyHex: maybeKey || null };
  } catch (e) { return null; }
}

async function getStoredPayloadByName(name) {
  try {
    const s = window.localStorage.getItem(`cloudnote_store:${name}`);
    if (s) return JSON.parse(s);
  } catch (e) { }
  if (BACKEND_BASE) {
    try { return await backendFetchPayload(name); } catch (e) { }
  }
  return null;
}

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

  useEffect(() => {
    const parsed = parseFragmentHash();
    setParsedFrag(parsed);
    if (parsed) {
      if (parsed.payload.mode === "pw") {
        setRequirePassword(true);
        setStatus("🔒 Protected note — password required.");
      } else if (parsed.payload.mode === "key") {
        if (parsed.rawKeyHex) {
          (async () => {
            try {
              const keyCrypto = await importRawKeyFromHex(parsed.rawKeyHex);
              const noteObj = await decryptWithKeyObj(keyCrypto, parsed.payload.iv, parsed.payload.ct);
              if (checkExpiryAndReturn(noteObj)) {
                setOpenedNote(noteObj);
                setStatus("✅ Opened note from link.");
              }
            } catch (e) {
              console.error(e);
              setStatus("❌ Failed to decrypt.");
            }
          })();
        } else {
          setStatus("❌ Shared link missing key.");
        }
      } else {
        setStatus("❌ Unknown link mode.");
      }
    }
  }, []);

  function checkExpiryAndReturn(noteObj) {
    if (!noteObj) return false;
    if (noteObj.expiresAtISO) {
      const exp = new Date(noteObj.expiresAtISO);
      if (!isNaN(exp.getTime()) && Date.now() > exp.getTime()) {
        setStatus("⚠️ This note has expired.");
        return false;
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
      setStatus("✅ Opened note with password.");
      setRequirePassword(false);
    } catch (e) {
      setStatus("❌ Wrong password or decryption failed.");
    }
  }

  async function onOpenByNamePassword() {
    if (!lookupName) return setStatus("⚠️ Enter note name.");
    const stored = await getStoredPayloadByName(lookupName);
    if (!stored) return setStatus("❌ No stored note found.");
    if (stored.mode !== "pw") return setStatus("❌ Not a password-protected note.");
    if (!lookupPassword) return setStatus("⚠️ Enter password.");

    try {
      const key = await deriveKeyFromPassword(lookupPassword, stored.salt);
      const noteObj = await decryptWithKeyObj(key, stored.iv, stored.ct);
      if (!checkExpiryAndReturn(noteObj)) return;
      setOpenedNote(noteObj);
      setStatus("✅ Opened note by name/password.");
      setRequirePassword(false);
      setParsedFrag(null);
    } catch (e) {
      setStatus("❌ Wrong password or decryption failed.");
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
      setStatus("✅ Link generated. Ready to share.");
    } catch (e) {
      setStatus("❌ Failed to generate: " + e.message);
    }
  }

  function copyLink() {
    if (!generatedLink) return;
    navigator.clipboard.writeText(generatedLink)
      .then(() => setStatus("📋 Copied to clipboard."))
      .catch((e) => setStatus("❌ Copy failed."));
  }

  function clearFragmentView() {
    history.replaceState(null, "", location.pathname + location.search);
    setParsedFrag(null);
    setOpenedNote(null);
    setStatus("");
    setOpenPassword("");
  }

  function canEditOpened() {
    return openedNote && !openedNote.readOnly;
  }

  async function onSaveEditAndRegenerate() {
    if (!openedNote) return;
    const noteObj = { ...openedNote, content: openedNote.content, createdAtISO: new Date().toISOString() };
    let pw = "";
    if (parsedFrag && parsedFrag.payload.mode === "pw") pw = openPassword;

    const frag = await buildShareFragment(noteObj, pw);
    const full = `${location.origin}${location.pathname}#${frag}`;
    setGeneratedLink(full);
    setStatus("✅ Saved & Regenerated link.");
  }

  return (
    <Layout title="Cloud Sticky Notes">
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className={styles.container}>
        <div className={styles.header}>
          <h1 className={styles.title}>📝 Cloud Sticky Note</h1>
          <a className="button button--secondary button--sm" href="/docs/tutorial-apps/cloud-sticky-note-tutorial">Tutorial</a>
        </div>

        <p className={styles.intro}>
          A secure, zero-backend way to share encrypted notes. Data is stored in the URL fragment.
          Share the link to give access. Add a password for extra security.
        </p>

        {/* --- Create/Share Section --- */}
        <div className={styles.card}>
          <h2 className={styles.cardTitle}>Create New Note</h2>

          <div className={styles.inputGroup}>
            <label className={styles.label}>Note Name</label>
            <input className={styles.input} placeholder="e.g. Meeting Minutes" value={name} onChange={(e) => setName(e.target.value)} />
          </div>

          <div className={styles.inputGroup}>
            <label className={styles.label}>Content</label>
            <textarea className={styles.textarea} placeholder="Write your note here..." value={content} onChange={(e) => setContent(e.target.value)} />
          </div>

          <div className={styles.optionsGrid}>
            <div className={styles.inputGroup}>
              <label className={styles.label}>Expires (Optional)</label>
              <input type="datetime-local" className={styles.input} value={expires} onChange={(e) => setExpires(e.target.value)} />
            </div>

            <div className={styles.inputGroup}>
              <label className={styles.label}>Password Protection (Optional)</label>
              <input type="password" className={styles.input} placeholder="Recipient must enter this" value={password} onChange={(e) => setPassword(e.target.value)} />
            </div>

            <div className={styles.inputGroup} style={{ alignSelf: 'end' }}>
              <label className={styles.checkboxLabel}>
                <input type="checkbox" checked={readOnly} onChange={(e) => setReadOnly(e.target.checked)} />
                <span>Read-only link</span>
              </label>
              <small className={styles.smallText}>Advisory only.</small>
            </div>
          </div>

          <div className={styles.buttonGroup}>
            <button onClick={onGenerateLink} className={`${styles.btn} ${styles.btnPrimary}`}>Generete Share Link</button>
            <button onClick={copyLink} className={`${styles.btn} ${styles.btnSecondary}`} disabled={!generatedLink}>Copy Link</button>
            <button onClick={() => navigator.clipboard.writeText(content)} className={`${styles.btn} ${styles.btnGhost}`}>Copy Text</button>
            <button onClick={() => { setName(""); setContent(""); setExpires(""); setPassword(""); setReadOnly(false); setGeneratedLink(""); setStatus(""); }} className={`${styles.btn} ${styles.btnGhost}`}>Reset form</button>
          </div>

          {generatedLink && (
            <div style={{ marginTop: 20 }}>
              <div className={styles.label}>Generated Link:</div>
              <textarea readOnly rows={3} className={styles.shareArea} value={generatedLink} />
            </div>
          )}

          {status && <div className={styles.status}>{status}</div>}
        </div>

        {/* --- Open Section --- */}
        <div className={styles.card}>
          <h2 className={styles.cardTitle}>Open Note</h2>

          {parsedFrag ? (
            <div style={{ padding: 16, background: 'var(--ifm-background-surface-color)', borderRadius: 8 }}>
              {parsedFrag.payload.mode === "pw" && (
                <div>
                  <label className={styles.label}>This note is password protected</label>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <input type="password" className={styles.input} placeholder="Enter Password" value={openPassword} onChange={(e) => setOpenPassword(e.target.value)} />
                    <button onClick={onOpenWithPassword} className={`${styles.btn} ${styles.btnPrimary}`}>Unlock</button>
                  </div>
                </div>
              )}
              {parsedFrag.payload.mode === "key" && (
                <div>Processing link... <button className={`${styles.btn} ${styles.btnGhost}`} onClick={clearFragmentView}>Cancel</button></div>
              )}
            </div>
          ) : (
            <div style={{ marginBottom: 16, color: 'var(--ifm-color-emphasis-700)' }}>
              Opening a shared link? It should load automatically.  <br />
              Or open a previously saved password-protected note below:
            </div>
          )}

          {!parsedFrag && (
            <div className={styles.optionsGrid} style={{ alignItems: 'end' }}>
              <div className={styles.inputGroup}>
                <label className={styles.label}>Stored Note Name</label>
                <input className={styles.input} value={lookupName} onChange={(e) => setLookupName(e.target.value)} />
              </div>
              <div className={styles.inputGroup}>
                <label className={styles.label}>Password</label>
                <input type="password" className={styles.input} value={lookupPassword} onChange={(e) => setLookupPassword(e.target.value)} />
              </div>
              <div className={styles.buttonGroup} style={{ marginTop: 0, marginBottom: 16 }}>
                <button onClick={onOpenByNamePassword} className={`${styles.btn} ${styles.btnSecondary}`}>Open Saved</button>
              </div>
            </div>
          )}
        </div>

        {/* --- Display Note --- */}
        {openedNote && (
          <div className={styles.card} style={{ border: '2px solid var(--ifm-color-primary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
              <h2 className={styles.cardTitle}>{openedNote.name || "Untitled Note"}</h2>
              <button onClick={clearFragmentView} className={`${styles.btn} ${styles.btnGhost} button--sm`}>Close</button>
            </div>

            <div className={styles.noteDisplay}>
              <div className={styles.noteMeta}>
                Created: {new Date(openedNote.createdAtISO).toLocaleString()}
                {openedNote.expiresAtISO && <span> • Expires: {new Date(openedNote.expiresAtISO).toLocaleString()}</span>}
                {openedNote.readOnly && <span> • 👁️ Read Only</span>}
              </div>
              <textarea
                className={styles.textarea}
                style={{ background: 'transparent', border: 'none', boxShadow: 'none', padding: 0, minHeight: 300, fontSize: '1.1rem' }}
                value={openedNote.content}
                readOnly={!canEditOpened()}
                onChange={(e) => setOpenedNote({ ...openedNote, content: e.target.value })}
              />
            </div>

            <div className={styles.buttonGroup}>
              <button disabled={!canEditOpened()} onClick={onSaveEditAndRegenerate} className={`${styles.btn} ${styles.btnPrimary}`}>Save Edits & Get New Link</button>
              <button onClick={() => navigator.clipboard.writeText(openedNote.content)} className={`${styles.btn} ${styles.btnSecondary}`}>Copy Text</button>
            </div>
          </div>
        )}

        <CitationNotice />
      </div>
    </Layout>
  );
}
