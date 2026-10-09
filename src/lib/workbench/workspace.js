/**
 * Local workspace (design/DESIGN_SPEC.md §10.2): an IndexedDB store of
 * typed artifacts that tools hand to each other. Everything stays in this
 * browser; clearing site data deletes it.
 */
const DB_NAME = "smiler488-lab";
const STORE = "artifacts";
export const WORKSPACE_EVENT = "lab:workspace";

let dbPromise = null;

function openDb() {
  if (typeof indexedDB === "undefined") {
    return Promise.reject(new Error("IndexedDB unavailable"));
  }
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);
      request.onupgradeneeded = () => {
        const store = request.result.createObjectStore(STORE, {
          keyPath: "id",
        });
        store.createIndex("type", "type");
        store.createIndex("createdAt", "createdAt");
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  return dbPromise;
}

function tx(mode, fn) {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const t = db.transaction(STORE, mode);
        const result = fn(t.objectStore(STORE));
        t.oncomplete = () => resolve(result?.result ?? result);
        t.onerror = () => reject(t.error);
      })
  );
}

function notify() {
  window.dispatchEvent(new CustomEvent(WORKSPACE_EVENT));
}

/** Save an artifact: { type, appId, label, data }. Returns the stored item. */
export async function saveArtifact({ type, appId, label, data }) {
  const item = {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    type,
    appId,
    label,
    data,
    createdAt: new Date().toISOString(),
  };
  await tx("readwrite", (store) => store.put(item));
  notify();
  return item;
}

/** All artifacts, newest first; optionally only the given types. */
export async function listArtifacts(types) {
  const items = await tx("readonly", (store) => store.getAll());
  return (items ?? [])
    .filter((item) => !types || types.includes(item.type))
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function getArtifact(id) {
  return tx("readonly", (store) => store.get(id));
}

export async function deleteArtifact(id) {
  await tx("readwrite", (store) => store.delete(id));
  notify();
}

export async function clearWorkspace() {
  await tx("readwrite", (store) => store.clear());
  notify();
}
