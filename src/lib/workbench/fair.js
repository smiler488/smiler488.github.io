/**
 * FAIR phenotype export (design/DESIGN_SPEC.md §8.4, D-022): turns a tool's
 * observations into a MIAPPE spreadsheet — the sheet and column names of the
 * official minimal template (github.com/MIAPPE/MIAPPE, Templates/
 * MIAPPE_Minimal_Spreadsheet_Template.xlsx) — plus a Data sheet holding the
 * observations themselves. Pure functions; the browser writes the workbook
 * with SheetJS (writeMiappeWorkbook).
 *
 * A tool opts in by adding `fair` to its lab:export event detail:
 *   fair: {
 *     variables: [VARIABLE, ...],          // e.g. FAIR_VARIABLES.leafInclination
 *     observations: [{ obsUnitId, timestamp, latitude?, longitude?,
 *                      altitude?, values: { [variableId]: number } }],
 *     obsUnitType: "leaf" | "plant" | "plot" ...,
 *     methodNote?: "…",                    // free text added to the study
 *   }
 * Ontology accessions below were checked against EBI OLS (2026-11):
 * TO = Plant Trait Ontology, PO = Plant Ontology, PATO, UO = Units of
 * Measurement. Never add an accession that has not been looked up.
 */

export const MIAPPE_VERSION = "1.1";

/** Observed variables (trait + method + scale) used by App Lab tools. */
export const FAIR_VARIABLES = {
  leafInclination: {
    variableId: "leaf_inclination_deg",
    variableName: "Leaf inclination angle from horizontal (phone sensors)",
    traitName: "leaf angle",
    traitAccNumber: "TO:0000206",
    traitEntity: "leaf",
    traitEntityAccessionNumber: "PO:0025034",
    traitCharacteristic: "angle",
    traitCharacteristicAccessionNumber: "PATO:0002326",
    methodName: "Phone laid flat on the leaf blade; screen-plane inclination",
    methodDesc:
      "Inclination of the leaf blade from horizontal, computed from the browser deviceorientation angles as arccos(cos β · cos γ); 0° is horizontal, 90° vertical. Recorded with the Sensor Recorder (smiler488.github.io/app/sensor).",
    methodRef: "https://smiler488.github.io/app/sensor",
    scaleName: "degree",
    scaleAccNumber: "UO:0000185",
    timeScale: "ISO 8601 date-time",
  },
};

/** Fields the export form must have before a workbook is offered. */
export const REQUIRED_META = [
  "investigationTitle",
  "studyTitle",
  "contactInst",
  "locationCountry",
  "siteName",
  "genus",
  "species",
  "personName",
];

export function missingMeta(meta) {
  return REQUIRED_META.filter((key) => !String(meta?.[key] ?? "").trim());
}

function slug(text) {
  return (
    String(text ?? "")
      .normalize("NFKD")
      .replace(/[^\w]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .toLowerCase()
      .slice(0, 40) || "study"
  );
}

const mean = (xs) =>
  xs.length ? xs.reduce((sum, x) => sum + x, 0) / xs.length : null;

/** Site location and date range, derived from the observations. */
export function summarizeObservations(observations = []) {
  const num = (key) =>
    observations
      .map((o) => o[key])
      .filter((v) => typeof v === "number" && Number.isFinite(v));
  const times = observations
    .map((o) => Date.parse(o.timestamp))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  const day = (t) => new Date(t).toISOString().slice(0, 10);
  return {
    latitude: mean(num("latitude")),
    longitude: mean(num("longitude")),
    altitude: mean(num("altitude")),
    startDate: times.length ? day(times[0]) : null,
    endDate: times.length ? day(times[times.length - 1]) : null,
  };
}

const round = (v, digits) =>
  typeof v === "number" && Number.isFinite(v) ? Number(v.toFixed(digits)) : "";

/**
 * Build the workbook as { sheetName: rows[][] } (first row = headers).
 * `meta` comes from the export form; `fair` from the tool's export event.
 */
export function buildMiappeSheets({ meta, fair, tool, createdAt }) {
  const { variables = [], observations = [], obsUnitType = "plant" } = fair;
  const site = summarizeObservations(observations);
  const investigationId = `${slug(meta.investigationTitle)}`;
  const studyId = `${investigationId}-${slug(meta.studyTitle)}`;
  const materialId = `${studyId}-material-1`;
  const dataSheet = "Data";
  const submission = (createdAt ?? new Date().toISOString()).slice(0, 10);

  const units = [...new Set(observations.map((o) => o.obsUnitId))];

  return {
    Investigation: [
      [
        "investigationId",
        "investigationTitle",
        "investigationDescription",
        "submissionDate",
        "publicReleaseDate",
        "license",
        "miappeVersion",
        "associatedPublication",
      ],
      [
        investigationId,
        meta.investigationTitle,
        meta.investigationDescription ?? "",
        submission,
        "",
        meta.license ?? "",
        MIAPPE_VERSION,
        meta.associatedPublication ?? "",
      ],
    ],
    Study: [
      [
        "studyId",
        "studyTitle",
        "studyDescription",
        "studyStartDate",
        "studyEndDate",
        "contactInst",
        "locationCountry",
        "siteName",
        "locationLatitude",
        "locationLongitude",
        "locationAltitude",
        "expeDesignDesc",
        "expeDesignType",
        "obsUnitLevelHierarchy",
        "obsUnitDesc",
        "growthFacilityDesc",
        "growthFacilityType",
        "culturalPractice",
        "expeDesignMap",
      ],
      [
        studyId,
        meta.studyTitle,
        [meta.studyDescription, fair.methodNote].filter(Boolean).join(" "),
        site.startDate ?? submission,
        site.endDate ?? "",
        meta.contactInst,
        meta.locationCountry,
        meta.siteName,
        round(site.latitude, 6),
        round(site.longitude, 6),
        round(site.altitude, 1),
        meta.expeDesignDesc ?? "",
        "",
        `study>${obsUnitType}`,
        meta.obsUnitDesc ??
          `One observation unit per ${obsUnitType}; ids as entered in ${tool.name}.`,
        meta.growthFacilityDesc ?? "",
        "",
        "",
        "",
      ],
    ],
    Person: [
      [
        "studyId",
        "personName",
        "personEmail",
        "personId",
        "personRole",
        "personAffiliation",
      ],
      [
        studyId,
        meta.personName,
        meta.personEmail ?? "",
        meta.personId ?? "",
        "data collector",
        meta.contactInst,
      ],
    ],
    "Data file": [
      ["studyId", "dataFileLink", "dataFileDesc", "dataFileVersion"],
      [
        studyId,
        `sheet:${dataSheet}`,
        `Observations exported from ${tool.name} (${tool.url}), tool version ${tool.version}. One row per observation; columns are observed variable ids.`,
        tool.version,
      ],
    ],
    "Biological Material": [
      [
        "studyId",
        "biologicalMaterialId",
        "biologicalMaterialExtId",
        "organism",
        "genus",
        "species",
        "infraspecificName",
      ],
      [
        studyId,
        materialId,
        "",
        meta.organism ?? "",
        meta.genus,
        meta.species,
        meta.infraspecificName ?? "",
      ],
    ],
    "Observation Unit": [
      [
        "biologicalMaterialId",
        "obsUnitId",
        "obsUnitType",
        "externalId",
        "spatialDistribution",
        "obsUnitFactorValue",
      ],
      ...units.map((id) => [materialId, id, obsUnitType, "", "", ""]),
    ],
    "Observed Variable": [
      [
        "studyId",
        "variableId",
        "variableName",
        "variableAccNumber",
        "traitName",
        "traitEntity",
        "traitEntityAccessionNumber",
        "traitCharacteristic",
        "traitCharacteristicAccessionNumber",
        "traitAccNumber",
        "methodName",
        "methodAccNumber",
        "methodDesc",
        "methodRef",
        "scaleName",
        "scaleAccNumber",
        "timeScale",
      ],
      ...variables.map((v) => [
        studyId,
        v.variableId,
        v.variableName,
        v.variableAccNumber ?? "",
        v.traitName,
        v.traitEntity ?? "",
        v.traitEntityAccessionNumber ?? "",
        v.traitCharacteristic ?? "",
        v.traitCharacteristicAccessionNumber ?? "",
        v.traitAccNumber ?? "",
        v.methodName,
        v.methodAccNumber ?? "",
        v.methodDesc,
        v.methodRef ?? "",
        v.scaleName,
        v.scaleAccNumber ?? "",
        v.timeScale ?? "",
      ]),
    ],
    [dataSheet]: [
      [
        "obsUnitId",
        "timestamp",
        "latitude",
        "longitude",
        "altitude",
        ...variables.map((v) => v.variableId),
      ],
      ...observations.map((o) => [
        o.obsUnitId,
        o.timestamp,
        round(o.latitude, 7),
        round(o.longitude, 7),
        round(o.altitude, 2),
        ...variables.map((v) => round(o.values?.[v.variableId], 4)),
      ]),
    ],
  };
}

/** Browser only: write the sheets to an .xlsx Blob with SheetJS. */
export async function writeMiappeWorkbook(sheets) {
  const XLSX = await import("xlsx");
  const book = XLSX.utils.book_new();
  for (const [name, rows] of Object.entries(sheets)) {
    XLSX.utils.book_append_sheet(book, XLSX.utils.aoa_to_sheet(rows), name);
  }
  const array = XLSX.write(book, { bookType: "xlsx", type: "array" });
  return new Blob([array], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
}
