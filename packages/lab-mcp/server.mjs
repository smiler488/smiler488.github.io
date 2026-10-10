#!/usr/bin/env node
/**
 * smiler488 Lab MCP server (design/DESIGN_SPEC.md §9.3).
 *
 * Exposes the website's science layer (src/lib/science) as MCP tools over
 * stdio, so an AI assistant computes exactly what the App Lab tools compute.
 * Zero dependencies: MCP's stdio transport is newline-delimited JSON-RPC 2.0.
 * stdout carries protocol messages only; diagnostics go to stderr.
 */
import { createInterface } from "node:readline";
import {
  BRDF_BOUNDS,
  SOLAR_METHOD,
  fieldArea,
  polygonCentroid,
  principalPlane,
  solarPosition,
} from "../../src/lib/science/index.js";

const SERVER_INFO = { name: "smiler488-lab", version: "0.1.0" };
const PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"];
const ISO_WITH_ZONE =
  /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})$/;

class InputError extends Error {}

function requireNumber(value, name, min, max) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new InputError(`${name} must be a finite number`);
  }
  if (value < min || value > max) {
    throw new InputError(`${name} must be between ${min} and ${max}`);
  }
  return value;
}

const TOOLS = [
  {
    name: "field_area",
    title: "Field area",
    description:
      "Area of a field boundary given as latitude/longitude vertices, in square metres, hectares and mu. Same method as the Land Surveyor tool on smiler488.github.io: shoelace formula on a local equirectangular projection. Suited to field-sized polygons, not regions spanning hundreds of kilometres.",
    inputSchema: {
      type: "object",
      properties: {
        points: {
          type: "array",
          minItems: 3,
          description:
            "Boundary vertices in order; do not repeat the first point at the end.",
          items: {
            type: "object",
            properties: {
              lat: { type: "number", minimum: -90, maximum: 90 },
              lng: { type: "number", minimum: -180, maximum: 180 },
            },
            required: ["lat", "lng"],
          },
        },
      },
      required: ["points"],
    },
    run({ points }) {
      if (!Array.isArray(points) || points.length < 3) {
        throw new InputError("points must be an array of at least 3 vertices");
      }
      const clean = points.map((p, i) => ({
        lat: requireNumber(p?.lat, `points[${i}].lat`, -90, 90),
        lng: requireNumber(p?.lng, `points[${i}].lng`, -180, 180),
      }));
      const result = { ...fieldArea(clean), centroid: polygonCentroid(clean) };
      const text = `Area ${result.area_m2.toFixed(
        2
      )} m² (${result.area_ha.toFixed(4)} ha, ${result.area_mu.toFixed(
        2
      )} mu) from ${
        result.vertices
      } vertices. Centroid ${result.centroid.lat.toFixed(
        6
      )}, ${result.centroid.lng.toFixed(6)}. Method: ${result.method}.`;
      return { text, structured: result };
    },
  },
  {
    name: "solar_position",
    title: "Solar position",
    description:
      "Sun elevation and azimuth (degrees, azimuth clockwise from north) for a location and moment, using the same simplified approximation as the Sensor Recorder tool. The local time zone is explicit so results do not depend on where this server runs.",
    inputSchema: {
      type: "object",
      properties: {
        latitude: { type: "number", minimum: -90, maximum: 90 },
        longitude: { type: "number", minimum: -180, maximum: 180 },
        datetime: {
          type: "string",
          description:
            "ISO 8601 moment with an explicit zone, e.g. 2026-06-21T12:00:00+08:00 or 2026-06-21T04:00:00Z.",
        },
        utc_offset_minutes: {
          type: "integer",
          minimum: -720,
          maximum: 840,
          description:
            "Local time zone offset from UTC in minutes (480 for UTC+8). Defines the local calendar day used by the approximation.",
        },
      },
      required: ["latitude", "longitude", "datetime", "utc_offset_minutes"],
    },
    run({ latitude, longitude, datetime, utc_offset_minutes: offset }) {
      requireNumber(latitude, "latitude", -90, 90);
      requireNumber(longitude, "longitude", -180, 180);
      if (typeof datetime !== "string" || !ISO_WITH_ZONE.test(datetime)) {
        throw new InputError(
          "datetime must be ISO 8601 with Z or ±hh:mm, e.g. 2026-06-21T12:00:00+08:00"
        );
      }
      if (!Number.isInteger(offset))
        throw new InputError("utc_offset_minutes must be an integer");
      requireNumber(offset, "utc_offset_minutes", -720, 840);
      const { elevation, azimuth } = solarPosition(
        latitude,
        longitude,
        datetime,
        offset
      );
      const result = {
        elevation_deg: elevation,
        azimuth_deg: azimuth,
        method: SOLAR_METHOD,
      };
      const text = `Sun elevation ${elevation.toFixed(
        3
      )}°, azimuth ${azimuth.toFixed(3)}° (clockwise from north)${
        elevation < 0 ? "; the sun is below the horizon" : ""
      }. Method: ${SOLAR_METHOD}.`;
      return { text, structured: result };
    },
  },
  {
    name: "leaf_brdf",
    title: "Leaf BRDF",
    description:
      "Leaf bidirectional reflectance (sr⁻¹) in the principal plane from the Cook–Torrance model of Deng et al. (2025), Plant Phenomics 7(4):100135, doi:10.1016/j.plaphe.2025.100135, as implemented in the study's fitting code. Parameters are limited to the study's fitting bounds. View angles are signed: positive on the specular (mirror) side.",
    inputSchema: {
      type: "object",
      properties: {
        rho: {
          type: "number",
          minimum: BRDF_BOUNDS.rho[0],
          maximum: BRDF_BOUNDS.rho[1],
          description: "Surface roughness σ (named rho in the fitting code)",
        },
        k: {
          type: "number",
          minimum: BRDF_BOUNDS.k[0],
          maximum: BRDF_BOUNDS.k[1],
          description: "Diffuse coefficient k",
        },
        n: {
          type: "number",
          minimum: BRDF_BOUNDS.n[0],
          maximum: BRDF_BOUNDS.n[1],
          description: "Refractive index n",
        },
        incidence_deg: {
          type: "number",
          minimum: 0,
          maximum: 85,
          description: "Light zenith angle θi",
        },
        view_deg: {
          type: "array",
          minItems: 1,
          maxItems: 181,
          items: { type: "number", minimum: -85, maximum: 85 },
          description: "Signed view zenith angles; positive = specular side",
        },
      },
      required: ["rho", "k", "n", "incidence_deg", "view_deg"],
    },
    run({ rho, k, n, incidence_deg: incidence, view_deg: views }) {
      requireNumber(rho, "rho", ...BRDF_BOUNDS.rho);
      requireNumber(k, "k", ...BRDF_BOUNDS.k);
      requireNumber(n, "n", ...BRDF_BOUNDS.n);
      requireNumber(incidence, "incidence_deg", 0, 85);
      if (!Array.isArray(views) || !views.length || views.length > 181) {
        throw new InputError("view_deg must be an array of 1–181 angles");
      }
      views.forEach((v, i) => requireNumber(v, `view_deg[${i}]`, -85, 85));
      const values = principalPlane({ rho, k, n }, incidence, views).map(
        (p) => ({
          view_deg: p.angle,
          total: p.total,
          specular: p.specular,
          diffuse: p.diffuse,
        })
      );
      const peak = values.reduce((best, v) =>
        v.total > best.total ? v : best
      );
      const result = {
        incidence_deg: incidence,
        parameters: { rho, k, n },
        values,
      };
      const text = `BRDF at ${
        values.length
      } view angle(s); peak ${peak.total.toPrecision(4)} sr⁻¹ at ${
        peak.view_deg
      }°. Source: Deng et al. 2025, doi:10.1016/j.plaphe.2025.100135.`;
      return { text, structured: result };
    },
  },
];

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function reply(id, result) {
  send({ jsonrpc: "2.0", id, result });
}

function fail(id, code, message) {
  send({ jsonrpc: "2.0", id, error: { code, message } });
}

function handle(message) {
  const { id, method, params } = message;
  const isRequest = id !== undefined && id !== null;
  if (!isRequest) return; // notifications (initialized, cancelled, …) need no reply

  switch (method) {
    case "initialize": {
      const requested = params?.protocolVersion;
      reply(id, {
        protocolVersion: PROTOCOL_VERSIONS.includes(requested)
          ? requested
          : PROTOCOL_VERSIONS[0],
        capabilities: { tools: {} },
        serverInfo: SERVER_INFO,
        instructions:
          "Science functions from the App Lab at smiler488.github.io. Results are identical to the website's tools. Cite the BRDF model as Deng et al. 2025, doi:10.1016/j.plaphe.2025.100135.",
      });
      return;
    }
    case "ping":
      reply(id, {});
      return;
    case "tools/list":
      reply(id, {
        tools: TOOLS.map(({ name, title, description, inputSchema }) => ({
          name,
          title,
          description,
          inputSchema,
        })),
      });
      return;
    case "tools/call": {
      const tool = TOOLS.find((t) => t.name === params?.name);
      if (!tool) {
        fail(id, -32602, `Unknown tool: ${params?.name}`);
        return;
      }
      try {
        const { text, structured } = tool.run(params?.arguments ?? {});
        reply(id, {
          content: [{ type: "text", text }],
          structuredContent: structured,
          isError: false,
        });
      } catch (error) {
        // Tool-level errors are reported in the result so the model can correct its input.
        reply(id, {
          content: [{ type: "text", text: `Error: ${error.message}` }],
          isError: true,
        });
        if (!(error instanceof InputError)) console.error(error);
      }
      return;
    }
    default:
      fail(id, -32601, `Method not found: ${method}`);
  }
}

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  if (!line.trim()) return;
  let message;
  try {
    message = JSON.parse(line);
  } catch {
    fail(null, -32700, "Parse error");
    return;
  }
  try {
    handle(message);
  } catch (error) {
    console.error(error);
    if (message?.id != null) fail(message.id, -32603, "Internal error");
  }
});
rl.on("close", () => process.exit(0));
