// src/pages/app/cloudnote/index.js
import React, { useEffect, useState } from "react";
import Heading from "@theme/Heading";
import AppScaffold from "../../../components/AppScaffold";
import CitationNotice from "../../../components/CitationNotice";
import styles from "./styles.module.css";

const BACKEND_BASE =
  typeof window !== "undefined" && window.__CLOUDNOTE_BACKEND_URL__
    ? window.__CLOUDNOTE_BACKEND_URL__
    : "";
const MAX_NOTE_CHARS = 3000;
const MAX_NAME_CHARS = 80;

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
    "raw",
    pwUtf8,
    "PBKDF2",
    false,
    ["deriveKey"]
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
  const key = await crypto.subtle.generateKey(
    { name: "AES-GCM", length: 256 },
    true,
    ["encrypt", "decrypt"]
  );
  const raw = await crypto.subtle.exportKey("raw", key);
  return buf2hex(raw);
}

async function importRawKeyFromHex(hex) {
  const buf = hex2buf(hex);
  return window.crypto.subtle.importKey("raw", buf, "AES-GCM", false, [
    "decrypt",
    "encrypt",
  ]);
}

async function encryptWithKeyObj(keyCryptoKey, jsonObj) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const plain = new TextEncoder().encode(JSON.stringify(jsonObj));
  const ct = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    keyCryptoKey,
    plain
  );
  return { iv: buf2hex(iv.buffer), ct: b64encode(ct) };
}

async function decryptWithKeyObj(keyCryptoKey, ivHex, ctB64) {
  const iv = hex2buf(ivHex);
  const ct = b64decode(ctB64);
  const plainBuf = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv },
    keyCryptoKey,
    ct
  );
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
    await fetch(
      `${BACKEND_BASE.replace(/\/$/, "")}/notes/${encodeURIComponent(name)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );
  } catch (e) {
    console.warn("backendSavePayload failed", e);
  }
}

async function backendFetchPayload(name) {
  if (!BACKEND_BASE || !name) return null;
  try {
    const r = await fetch(
      `${BACKEND_BASE.replace(/\/$/, "")}/notes/${encodeURIComponent(name)}`
    );
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    console.warn("backendFetchPayload failed", e);
    return null;
  }
}

async function buildShareFragment(noteObj, password = "") {
  if (password && password.length > 0) {
    const salt = genSaltHex();
    const key = await deriveKeyFromPassword(password, salt);
    const enc = await encryptWithKeyObj(key, noteObj);
    const payload = {
      mode: "pw",
      salt,
      iv: enc.iv,
      ct: enc.ct,
      meta: { v: 1 },
    };

    try {
      if (noteObj?.name) {
        window.localStorage.setItem(
          `cloudnote_store:${noteObj.name}`,
          JSON.stringify(payload)
        );
        await backendSavePayload(noteObj.name, payload);
      }
    } catch (e) {
      console.warn("storage error:", e);
    }

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
  } catch (e) {
    return null;
  }
}

async function getStoredPayloadByName(name) {
  try {
    const s = window.localStorage.getItem(`cloudnote_store:${name}`);
    if (s) return JSON.parse(s);
  } catch (e) {}
  if (BACKEND_BASE) {
    try {
      return await backendFetchPayload(name);
    } catch (e) {}
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
  const [openedProtection, setOpenedProtection] = useState({
    mode: "key",
    password: "",
  });
  const [openPassword, setOpenPassword] = useState("");
  const [parsedFrag, setParsedFrag] = useState(null);
  const [lookupName, setLookupName] = useState("");
  const [lookupPassword, setLookupPassword] = useState("");

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

  useEffect(() => {
    let cancelled = false;

    async function openFromFragment() {
      const parsed = parseFragmentHash();
      if (cancelled) return;
      setParsedFrag(parsed);
      if (!parsed) return;

      if (parsed.payload.mode === "pw") {
        setStatus("🔒 Protected note — password required.");
      } else if (parsed.payload.mode === "key") {
        if (parsed.rawKeyHex) {
          try {
            const keyCrypto = await importRawKeyFromHex(parsed.rawKeyHex);
            const noteObj = await decryptWithKeyObj(
              keyCrypto,
              parsed.payload.iv,
              parsed.payload.ct
            );
            if (cancelled) return;
            if (checkExpiryAndReturn(noteObj)) {
              setOpenedNote(noteObj);
              setOpenedProtection({ mode: "key", password: "" });
              setStatus("✅ Opened note from link.");
            }
          } catch (e) {
            if (cancelled) return;
            console.error(e);
            setStatus("❌ Failed to decrypt.");
          }
        } else {
          setStatus("❌ Shared link missing key.");
        }
      } else {
        setStatus("❌ Unknown link mode.");
      }
    }

    const timeoutId = window.setTimeout(() => void openFromFragment(), 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, []);

  async function onOpenWithPassword() {
    if (!parsedFrag) return;
    if (!openPassword) {
      setStatus("⚠️ Enter the note password.");
      return;
    }
    try {
      const key = await deriveKeyFromPassword(
        openPassword,
        parsedFrag.payload.salt
      );
      const noteObj = await decryptWithKeyObj(
        key,
        parsedFrag.payload.iv,
        parsedFrag.payload.ct
      );
      if (!checkExpiryAndReturn(noteObj)) return;
      setOpenedNote(noteObj);
      setOpenedProtection({ mode: "pw", password: openPassword });
      setStatus("✅ Opened note with password.");
    } catch (e) {
      setStatus("❌ Wrong password or decryption failed.");
    }
  }

  async function onOpenByNamePassword() {
    if (!lookupName) return setStatus("⚠️ Enter note name.");
    const stored = await getStoredPayloadByName(lookupName);
    if (!stored) return setStatus("❌ No stored note found.");
    if (stored.mode !== "pw")
      return setStatus("❌ Not a password-protected note.");
    if (!lookupPassword) return setStatus("⚠️ Enter password.");

    try {
      const key = await deriveKeyFromPassword(lookupPassword, stored.salt);
      const noteObj = await decryptWithKeyObj(key, stored.iv, stored.ct);
      if (!checkExpiryAndReturn(noteObj)) return;
      setOpenedNote(noteObj);
      setOpenedProtection({ mode: "pw", password: lookupPassword });
      setStatus("✅ Opened note by name/password.");
      setParsedFrag(null);
    } catch (e) {
      setStatus("❌ Wrong password or decryption failed.");
    }
  }

  async function onGenerateLink() {
    try {
      if (name.length > MAX_NAME_CHARS) {
        setStatus(
          `⚠️ Note name must be ${MAX_NAME_CHARS} characters or fewer.`
        );
        return;
      }
      if (content.length > MAX_NOTE_CHARS) {
        setStatus(
          `⚠️ Note content must be ${MAX_NOTE_CHARS.toLocaleString()} characters or fewer.`
        );
        return;
      }
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

  async function copyText(value, successMessage) {
    if (!value) return;
    try {
      await navigator.clipboard.writeText(value);
      setStatus(successMessage);
    } catch (_) {
      setStatus("❌ Copy failed. Select the text and copy it manually.");
    }
  }

  function clearFragmentView() {
    history.replaceState(null, "", location.pathname + location.search);
    setParsedFrag(null);
    setOpenedNote(null);
    setStatus("");
    setOpenPassword("");
    setOpenedProtection({ mode: "key", password: "" });
  }

  function canEditOpened() {
    return openedNote && !openedNote.readOnly;
  }

  async function onSaveEditAndRegenerate() {
    if (!openedNote) return;
    const noteObj = {
      ...openedNote,
      content: openedNote.content,
      createdAtISO: new Date().toISOString(),
    };
    const pw = openedProtection.mode === "pw" ? openedProtection.password : "";

    const frag = await buildShareFragment(noteObj, pw);
    const full = `${location.origin}${location.pathname}#${frag}`;
    setGeneratedLink(full);
    setStatus("✅ Saved & Regenerated link.");
  }

  return (
    <AppScaffold appId="cloudnote">
      <div className={styles.container}>
        <aside className={styles.boundaryNotice} role="note">
          Encryption happens in this browser. Shared data lives in the URL
          fragment; password-protected copies are kept only in this
          browser&apos;s local storage unless an optional backend is configured.
          Expiry and read-only flags are advisory, not revocation controls.
        </aside>

        {status && (
          <div className={styles.status} role="status" aria-live="polite">
            {status}
          </div>
        )}

        {/* --- Create/Share Section --- */}
        <section className={styles.card}>
          <Heading as="h2" className={styles.cardTitle}>
            Create New Note
          </Heading>

          <div className={styles.inputGroup}>
            <label className={styles.label} htmlFor="cloudnote-name">
              Note Name
            </label>
            <input
              id="cloudnote-name"
              maxLength={MAX_NAME_CHARS}
              className={styles.input}
              placeholder="e.g. Meeting Minutes"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className={styles.inputGroup}>
            <div className={styles.labelRow}>
              <label className={styles.label} htmlFor="cloudnote-content">
                Content
              </label>
              <span className={styles.counter}>
                {content.length.toLocaleString()} /{" "}
                {MAX_NOTE_CHARS.toLocaleString()}
              </span>
            </div>
            <textarea
              id="cloudnote-content"
              maxLength={MAX_NOTE_CHARS}
              className={styles.textarea}
              placeholder="Write your note here..."
              value={content}
              onChange={(e) => setContent(e.target.value)}
            />
          </div>

          <div className={styles.optionsGrid}>
            <div className={styles.inputGroup}>
              <label className={styles.label} htmlFor="cloudnote-expires">
                Expires (Optional)
              </label>
              <input
                id="cloudnote-expires"
                type="datetime-local"
                className={styles.input}
                value={expires}
                onChange={(e) => setExpires(e.target.value)}
              />
            </div>

            <div className={styles.inputGroup}>
              <label className={styles.label} htmlFor="cloudnote-password">
                Password Protection (Optional)
              </label>
              <input
                id="cloudnote-password"
                type="password"
                className={styles.input}
                placeholder="Recipient must enter this"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            <div className={styles.inputGroup} style={{ alignSelf: "end" }}>
              <label className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={readOnly}
                  onChange={(e) => setReadOnly(e.target.checked)}
                />
                <span>Read-only link</span>
              </label>
              <small className={styles.smallText}>Advisory only.</small>
            </div>
          </div>

          <div className={styles.buttonGroup}>
            <button
              onClick={onGenerateLink}
              className={`${styles.btn} ${styles.btnPrimary}`}
            >
              Generate Share Link
            </button>
            <button
              onClick={() =>
                copyText(generatedLink, "📋 Link copied to clipboard.")
              }
              className={`${styles.btn} ${styles.btnSecondary}`}
              disabled={!generatedLink}
            >
              Copy Link
            </button>
            <button
              onClick={() =>
                copyText(content, "📋 Note text copied to clipboard.")
              }
              className={`${styles.btn} ${styles.btnGhost}`}
              disabled={!content}
            >
              Copy Text
            </button>
            <button
              onClick={() => {
                setName("");
                setContent("");
                setExpires("");
                setPassword("");
                setReadOnly(false);
                setGeneratedLink("");
                setStatus("");
              }}
              className={`${styles.btn} ${styles.btnGhost}`}
            >
              Reset form
            </button>
          </div>

          {generatedLink && (
            <div style={{ marginTop: 20 }}>
              <label className={styles.label} htmlFor="cloudnote-share-link">
                Generated Link
              </label>
              <textarea
                id="cloudnote-share-link"
                aria-describedby="cloudnote-share-help"
                readOnly
                rows={3}
                className={styles.shareArea}
                value={generatedLink}
              />
              <small id="cloudnote-share-help" className={styles.smallText}>
                Anyone with a key-mode link can decrypt it. Password-mode links
                still require the password.
              </small>
            </div>
          )}
        </section>

        {/* --- Open Section --- */}
        <section className={styles.card}>
          <Heading as="h2" className={styles.cardTitle}>
            Open Note
          </Heading>

          {parsedFrag ? (
            <div
              style={{
                padding: 16,
                background: "var(--ifm-background-surface-color)",
                borderRadius: 8,
              }}
            >
              {parsedFrag.payload.mode === "pw" && (
                <div>
                  <label
                    className={styles.label}
                    htmlFor="cloudnote-open-password"
                  >
                    This note is password protected
                  </label>
                  <div className={styles.unlockRow}>
                    <input
                      id="cloudnote-open-password"
                      type="password"
                      className={styles.input}
                      placeholder="Enter Password"
                      value={openPassword}
                      onChange={(e) => setOpenPassword(e.target.value)}
                    />
                    <button
                      onClick={onOpenWithPassword}
                      className={`${styles.btn} ${styles.btnPrimary}`}
                    >
                      Unlock
                    </button>
                  </div>
                </div>
              )}
              {parsedFrag.payload.mode === "key" && (
                <div>
                  Processing link...{" "}
                  <button
                    className={`${styles.btn} ${styles.btnGhost}`}
                    onClick={clearFragmentView}
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div
              style={{
                marginBottom: 16,
                color: "var(--ifm-color-emphasis-700)",
              }}
            >
              Opening a shared link? It should load automatically. <br />
              Or open a previously saved password-protected note below:
            </div>
          )}

          {!parsedFrag && (
            <div className={styles.optionsGrid} style={{ alignItems: "end" }}>
              <div className={styles.inputGroup}>
                <label className={styles.label} htmlFor="cloudnote-lookup-name">
                  Stored Note Name
                </label>
                <input
                  id="cloudnote-lookup-name"
                  className={styles.input}
                  value={lookupName}
                  onChange={(e) => setLookupName(e.target.value)}
                />
              </div>
              <div className={styles.inputGroup}>
                <label
                  className={styles.label}
                  htmlFor="cloudnote-lookup-password"
                >
                  Password
                </label>
                <input
                  id="cloudnote-lookup-password"
                  type="password"
                  className={styles.input}
                  value={lookupPassword}
                  onChange={(e) => setLookupPassword(e.target.value)}
                />
              </div>
              <div
                className={styles.buttonGroup}
                style={{ marginTop: 0, marginBottom: 16 }}
              >
                <button
                  onClick={onOpenByNamePassword}
                  className={`${styles.btn} ${styles.btnSecondary}`}
                >
                  Open Saved
                </button>
              </div>
            </div>
          )}
        </section>

        {/* --- Display Note --- */}
        {openedNote && (
          <section className={`${styles.card} ${styles.openedCard}`}>
            <div className={styles.openedHeader}>
              <Heading as="h2" className={styles.cardTitle}>
                {openedNote.name || "Untitled Note"}
              </Heading>
              <button
                onClick={clearFragmentView}
                className={`${styles.btn} ${styles.btnGhost} button--sm`}
              >
                Close
              </button>
            </div>

            <div className={styles.noteDisplay}>
              <div className={styles.noteMeta}>
                Created: {new Date(openedNote.createdAtISO).toLocaleString()}
                {openedNote.expiresAtISO && (
                  <span>
                    {" "}
                    • Expires:{" "}
                    {new Date(openedNote.expiresAtISO).toLocaleString()}
                  </span>
                )}
                {openedNote.readOnly && <span> • 👁️ Read Only</span>}
              </div>
              <textarea
                className={styles.textarea}
                style={{
                  background: "transparent",
                  border: "none",
                  boxShadow: "none",
                  padding: 0,
                  minHeight: 300,
                  fontSize: "1.1rem",
                }}
                value={openedNote.content}
                aria-label="Opened note content"
                readOnly={!canEditOpened()}
                onChange={(e) =>
                  setOpenedNote({ ...openedNote, content: e.target.value })
                }
              />
            </div>

            <div className={styles.buttonGroup}>
              <button
                disabled={!canEditOpened()}
                onClick={onSaveEditAndRegenerate}
                className={`${styles.btn} ${styles.btnPrimary}`}
              >
                Save Edits & Get New Link
              </button>
              <button
                onClick={() =>
                  copyText(
                    openedNote.content,
                    "📋 Note text copied to clipboard."
                  )
                }
                className={`${styles.btn} ${styles.btnSecondary}`}
              >
                Copy Text
              </button>
            </div>
          </section>
        )}

        <CitationNotice />
      </div>
    </AppScaffold>
  );
}
