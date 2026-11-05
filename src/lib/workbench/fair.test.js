import { test } from "node:test";
import assert from "node:assert/strict";
import {
  FAIR_VARIABLES,
  MIAPPE_VERSION,
  buildMiappeSheets,
  missingMeta,
  summarizeObservations,
} from "./fair.js";

const META = {
  investigationTitle: "Cotton leaf angle survey",
  studyTitle: "Shihezi 2026 field",
  contactInst: "China Agricultural University",
  locationCountry: "China",
  siteName: "Shihezi experimental station",
  genus: "Gossypium",
  species: "hirsutum",
  personName: "Test Person",
};

const FAIR = {
  variables: [FAIR_VARIABLES.leafInclination],
  obsUnitType: "leaf",
  observations: [
    {
      obsUnitId: "L1",
      timestamp: "2026-07-01T10:00:00.000Z",
      latitude: 44.3,
      longitude: 86.0,
      values: { leaf_inclination_deg: 32.123456 },
    },
    {
      obsUnitId: "L2",
      timestamp: "2026-07-02T10:00:00.000Z",
      latitude: 44.5,
      longitude: 86.2,
      values: { leaf_inclination_deg: 58 },
    },
    {
      obsUnitId: "L1",
      timestamp: "2026-07-03T10:00:00.000Z",
      values: { leaf_inclination_deg: 30 },
    },
  ],
};

const TOOL = {
  name: "Sensor Recorder",
  url: "https://smiler488.github.io/app/sensor",
  version: "2026.10.0",
};

// Column names of the official MIAPPE minimal spreadsheet template
const TEMPLATE_HEADERS = {
  Investigation: ["investigationId", "investigationTitle", "miappeVersion"],
  Study: ["studyId", "studyTitle", "studyStartDate", "locationCountry", "siteName"],
  Person: ["studyId", "personName", "personAffiliation"],
  "Data file": ["studyId", "dataFileLink", "dataFileDesc", "dataFileVersion"],
  "Biological Material": ["studyId", "biologicalMaterialId", "genus", "species"],
  "Observation Unit": ["biologicalMaterialId", "obsUnitId", "obsUnitType"],
  "Observed Variable": [
    "variableId",
    "traitName",
    "traitAccNumber",
    "methodName",
    "scaleName",
    "scaleAccNumber",
  ],
};

test("missingMeta lists empty required fields", () => {
  assert.deepEqual(missingMeta(META), []);
  assert.deepEqual(missingMeta({ ...META, siteName: "  " }), ["siteName"]);
});

test("summarizeObservations averages location and spans the dates", () => {
  const s = summarizeObservations(FAIR.observations);
  assert.equal(s.latitude, 44.4);
  assert.ok(Math.abs(s.longitude - 86.1) < 1e-9);
  assert.equal(s.altitude, null);
  assert.equal(s.startDate, "2026-07-01");
  assert.equal(s.endDate, "2026-07-03");
});

test("buildMiappeSheets follows the MIAPPE template and keeps ids linked", () => {
  const sheets = buildMiappeSheets({
    meta: META,
    fair: FAIR,
    tool: TOOL,
    createdAt: "2026-07-04T00:00:00.000Z",
  });
  for (const [name, columns] of Object.entries(TEMPLATE_HEADERS)) {
    for (const column of columns) {
      assert.ok(sheets[name][0].includes(column), `${name}.${column}`);
    }
  }
  const cell = (sheet, row, column) =>
    sheets[sheet][row][sheets[sheet][0].indexOf(column)];

  assert.equal(cell("Investigation", 1, "miappeVersion"), MIAPPE_VERSION);
  const studyId = cell("Study", 1, "studyId");
  assert.equal(cell("Person", 1, "studyId"), studyId);
  assert.equal(cell("Observed Variable", 1, "studyId"), studyId);
  assert.equal(cell("Study", 1, "studyStartDate"), "2026-07-01");

  // One observation unit per distinct id, all tied to the material
  const material = cell("Biological Material", 1, "biologicalMaterialId");
  assert.equal(sheets["Observation Unit"].length, 1 + 2);
  assert.equal(cell("Observation Unit", 1, "biologicalMaterialId"), material);
  assert.equal(cell("Observation Unit", 2, "obsUnitType"), "leaf");

  // Trait, entity and scale accessions as looked up in OLS
  assert.equal(cell("Observed Variable", 1, "traitAccNumber"), "TO:0000206");
  assert.equal(cell("Observed Variable", 1, "traitEntityAccessionNumber"), "PO:0025034");
  assert.equal(cell("Observed Variable", 1, "scaleAccNumber"), "UO:0000185");

  // Data sheet: one row per observation, a column per variable id
  assert.equal(sheets.Data.length, 1 + 3);
  assert.equal(cell("Data", 1, "leaf_inclination_deg"), 32.1235);
  assert.equal(cell("Data", 3, "latitude"), "");
  assert.equal(cell("Data file", 1, "dataFileLink"), "sheet:Data");
});
