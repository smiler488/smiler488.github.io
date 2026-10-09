/**
 * Parameter records for exports (design/DESIGN_SPEC.md §8.2).
 *
 * Any tool — React or a legacy static script — announces an export by
 * dispatching a `lab:export` event on window:
 *
 *   window.dispatchEvent(new CustomEvent("lab:export", { detail: {
 *     files: [{ name, blob? , text? }],   // what was exported
 *     parameters: { ... },                // settings that produced it
 *     inputs: [{ name, blob?, text? }],   // optional source files
 *   }}));
 *
 * The tool shell (AppScaffold) listens, hashes the files with Web Crypto and
 * shows a downloadable `<name>.provenance.json`. Only parameters the tool
 * chooses to pass are recorded — never device data or location unless the
 * tool's own output already contains it.
 */
export const EXPORT_EVENT = "lab:export";
export const PROVENANCE_SCHEMA = "smiler488.provenance/v1";

/** Announce an export from React code. */
export function recordExport(detail) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(EXPORT_EVENT, { detail }));
}

async function sha256Hex(data) {
  if (!data || !globalThis.crypto?.subtle) return null;
  const buffer =
    typeof data === "string"
      ? new TextEncoder().encode(data)
      : await data.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function describeFile(file) {
  const data = file.blob ?? file.text ?? null;
  const bytes =
    file.blob?.size ??
    (file.text != null ? new TextEncoder().encode(file.text).length : null);
  return {
    name: file.name,
    ...(bytes != null ? { bytes } : {}),
    ...(data ? { sha256: await sha256Hex(data) } : {}),
  };
}

/** Build the full record for one export. */
export async function buildProvenance({ app, detail, build, siteUrl }) {
  const files = await Promise.all((detail.files ?? []).map(describeFile));
  const inputs = await Promise.all((detail.inputs ?? []).map(describeFile));
  return {
    schema: PROVENANCE_SCHEMA,
    tool: {
      id: app.id,
      version: app.version,
      maturity: app.maturity,
    },
    site: {
      build: build || null,
      url: `${siteUrl.replace(/\/$/, "")}${app.route}`,
    },
    createdAt: new Date().toISOString(),
    ...(inputs.length ? { inputs } : {}),
    parameters: detail.parameters ?? {},
    outputs: files,
    cite: "https://doi.org/10.5281/zenodo.17544584",
  };
}
