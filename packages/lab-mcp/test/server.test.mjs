/**
 * Talks to the server over stdio exactly as an MCP client would, and checks
 * that its answers equal the website tools' reference values.
 * Run: npm test (in packages/lab-mcp) or node --test packages/lab-mcp/test/
 */
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { createInterface } from "node:readline";

const reference = JSON.parse(
  readFileSync(
    new URL(
      "../../../src/lib/science/__fixtures__/reference.json",
      import.meta.url
    ),
    "utf8"
  )
);

let child;
let nextId = 1;
const pending = new Map();

function request(method, params) {
  const id = nextId++;
  child.stdin.write(
    `${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`
  );
  return new Promise((resolve) => pending.set(id, resolve));
}

function call(name, args) {
  return request("tools/call", { name, arguments: args });
}

before(async () => {
  child = spawn(
    process.execPath,
    [new URL("../server.mjs", import.meta.url).pathname],
    {
      stdio: ["pipe", "pipe", "inherit"],
    }
  );
  createInterface({ input: child.stdout }).on("line", (line) => {
    const message = JSON.parse(line); // stdout must carry protocol JSON only
    pending.get(message.id)?.(message);
    pending.delete(message.id);
  });
  const init = await request("initialize", {
    protocolVersion: "2025-06-18",
    capabilities: {},
    clientInfo: { name: "test", version: "0" },
  });
  assert.equal(init.result.protocolVersion, "2025-06-18");
  assert.deepEqual(init.result.capabilities, { tools: {} });
  child.stdin.write(
    `${JSON.stringify({
      jsonrpc: "2.0",
      method: "notifications/initialized",
    })}\n`
  );
});

after(() => child.kill());

test("lists the three science tools with input schemas", async () => {
  const { result } = await request("tools/list", {});
  assert.deepEqual(
    result.tools.map((t) => t.name),
    ["field_area", "solar_position", "leaf_brdf"]
  );
  for (const tool of result.tools)
    assert.equal(tool.inputSchema.type, "object");
});

test("field_area equals Land Surveyor (10,252.04 m² test field)", async () => {
  const field = reference.area.at(-1);
  const { result } = await call("field_area", { points: field.points });
  assert.equal(result.isError, false);
  assert.equal(result.structuredContent.area_m2, field.area);
  assert.equal(result.structuredContent.area_m2.toFixed(2), "10252.04");
  assert.match(result.content[0].text, /10252\.04 m²/);
});

test("solar_position equals Sensor Recorder in several time zones", async () => {
  for (const r of reference.solar.filter((_, i) => i % 7 === 0)) {
    const iso = new Date(r.iso).toISOString();
    const { result } = await call("solar_position", {
      latitude: r.lat,
      longitude: r.lon,
      datetime: iso,
      utc_offset_minutes: r.utcOffsetMinutes,
    });
    assert.equal(result.isError, false, result.content[0].text);
    assert.equal(result.structuredContent.elevation_deg, r.elevation);
    assert.equal(result.structuredContent.azimuth_deg, r.azimuth);
  }
});

test("leaf_brdf equals the study's fitting code", async () => {
  for (const c of reference.brdf) {
    const { result } = await call("leaf_brdf", {
      rho: c.rho,
      k: c.k,
      n: c.n,
      incidence_deg: c.incidence,
      view_deg: [c.view],
    });
    const got = result.structuredContent.values[0].total;
    assert.ok(Math.abs(got - c.f) / c.f < 1e-9);
  }
});

test("rejects bad input as a tool error the model can correct", async () => {
  const outOfBounds = await call("leaf_brdf", {
    rho: 2,
    k: 0.3,
    n: 1.47,
    incidence_deg: 30,
    view_deg: [10],
  });
  assert.equal(outOfBounds.result.isError, true);
  assert.match(outOfBounds.result.content[0].text, /rho must be between/);

  const naiveTime = await call("solar_position", {
    latitude: 40,
    longitude: 116,
    datetime: "2026-06-21T12:00:00",
    utc_offset_minutes: 480,
  });
  assert.equal(naiveTime.result.isError, true);

  const tooFew = await call("field_area", { points: [{ lat: 1, lng: 1 }] });
  assert.equal(tooFew.result.isError, true);
});

test("answers JSON-RPC errors for unknown tools and methods", async () => {
  const unknownTool = await call("nope", {});
  assert.equal(unknownTool.error.code, -32602);
  const unknownMethod = await request("resources/list", {});
  assert.equal(unknownMethod.error.code, -32601);
  const pong = await request("ping", {});
  assert.deepEqual(pong.result, {});
});
