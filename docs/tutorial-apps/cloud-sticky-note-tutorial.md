---
title: "Encrypted Note"
description: "Encrypt a text note in the browser, share it through a URL fragment or keep a password-protected copy in local storage."
sidebar_label: "Encrypted Note"
sidebar_position: 13
hide_title: true
keywords:
  - "note"
  - "encrypt"
  - "share"
  - "password"
  - "local"
app_route: "/app/cloudnote"
app_icon: "TXT"
app_category: "Utilities"
app_runtime: "Local encryption and storage"
app_tone: "rose"
app_badges:
  - "Web Crypto"
  - "Local storage"
  - "Text only"
---

## What it does

Encrypted Note encrypts a text note in your browser and places the encrypted payload in a shareable URL fragment. You can use a random-key link or add a password. Password-protected payloads are also stored under the note name in this browser's local storage.

:::info Text-only utility
The current app accepts a note name and up to 3,000 characters of text. It does not support file attachments, accounts or built-in cross-device synchronization.
:::

## Before you start

- Use a recent browser with Web Crypto, localStorage and JavaScript support.
- A note name can contain up to 80 characters; content can contain up to 3,000.
- Treat an unprotected link as a bearer secret: anyone with the complete URL can decrypt it.
- For password mode, share the password through a different trusted channel.
- Some chat, email or QR tools truncate long URLs, so test the received link.

## Quick workflow

1. [Open Encrypted Note](/app/cloudnote).
2. Under **Create New Note**, enter **Note Name** and **Content**.
3. Optionally set **Expires**, **Password Protection** and **Read-only link**.
4. Select **Generate Share Link**, then **Copy Link**.
5. Send the complete link. A random-key link opens automatically; a password link asks the recipient to select **Unlock** after entering the password.
6. To reopen a password-protected copy in the same browser, enter **Stored Note Name** and **Password**, then select **Open Saved**.
7. An editable opened note can use **Save Edits & Get New Link**. Use **Copy Text** or **Close** as needed.

## Controls & outputs

| Control or output                  | Purpose                                                                                                           |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Note Name**                      | Identifies the note and, in password mode, becomes the local-storage lookup name.                                 |
| **Content**                        | Stores text only, with a 3,000-character counter and limit.                                                       |
| **Expires (Optional)**             | Adds a client-checked expiration timestamp.                                                                       |
| **Password Protection (Optional)** | Derives the encryption key from a password instead of embedding a random key in the link.                         |
| **Read-only link**                 | Opens the note as non-editable in this app's UI; this is advisory only.                                           |
| **Generate Share Link**            | Encrypts the current note and creates a URL fragment.                                                             |
| **Copy Link / Copy Text**          | Copies the generated URL or current text to the clipboard.                                                        |
| **Reset form**                     | Clears the creation form; it does not delete saved browser data or revoke an old link.                            |
| **Open Saved**                     | Looks up a password-protected payload by exact name in this browser, then decrypts it with the supplied password. |
| **Save Edits & Get New Link**      | Re-encrypts an editable opened note and creates a new link.                                                       |

## How it works

The note object contains its name, content, creation time, optional expiration and read-only flag. Encryption and decryption use the browser's Web Crypto API.

Without a password, the app generates a random 256-bit AES-GCM key. The encrypted payload and raw key are both placed in the URL fragment, so possession of the full link is sufficient to decrypt the note.

With a password, the app derives a 256-bit AES-GCM key using PBKDF2-SHA-256, a random 16-byte salt and 200,000 iterations. The URL contains the salt, IV and ciphertext but not the password-derived key. The encrypted payload is also saved to localStorage under the note name.

Each encryption uses a new random IV. Editing and regenerating creates a new payload and link; the old link remains valid. Expiration and read-only behavior are checked by this page's client code and are not cryptographic revocation controls.

## Data, privacy & external services

In the standard deployment, encryption, fragment parsing and saved-note lookup happen locally. URL fragments are normally not included in an HTTP request to the hosting server.

A complete random-key link exposes both ciphertext and its key to anyone who obtains it through browser history, clipboard access, screenshots, extensions, synced tabs or the receiving application. Password mode separates the password from the link, but security still depends on password strength and how it is shared.

The code supports an optional developer-configured backend, but the public UI has no backend selector. If such a backend is configured, a password-protected encrypted payload and the note name are sent to it for save/retrieval. Do not assume zero-backend behavior on a modified deployment.

:::caution No revocation or enforced access control
Deleting a message containing the URL does not invalidate copies of that link. Expiry and read-only flags can be bypassed outside this UI, and the app has no account identity, server-side authorization or link-revocation service.
:::

## Limitations

- Text only: no attachments, rich text or file transfer.
- No account, note list, synchronization, audit log, version history or local-delete control.
- Password strength is not enforced.
- Local browser storage can be cleared, blocked or overwritten; reusing the same note name replaces its saved payload.
- **Reset form** does not remove an already saved password-protected payload.
- Long encrypted URLs may exceed limits in browsers, messaging platforms or QR tools.
- An expired note is blocked by this UI but its encrypted payload is not automatically erased.
- Saving an edit generates a new link but does not revoke the previous one.

## Troubleshooting

- **The shared link does not open:** copy the complete URL, including everything after `#note=`.
- **Wrong password or decryption failed:** passwords are case-sensitive; confirm both the password and link are unchanged.
- **No stored note found:** **Open Saved** works only for password-protected notes saved under the exact name in this browser or browser profile.
- **This note has expired:** the client-side expiration time has passed; the app does not provide an override.
- **Copy failed:** select the generated link or note text and copy it manually.
- **Web Crypto is unavailable:** update the browser and use the HTTPS site.
- **A generated URL is truncated:** shorten the note or share the link through a channel that preserves long URL fragments.
- **An old link still opens after editing:** this is expected; distribute the new link and treat the old one as still valid.

[Open Encrypted Note →](/app/cloudnote)

[← Back to App Lab](/app)
