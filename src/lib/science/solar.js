/**
 * Solar position (design/DESIGN_SPEC.md §9.3). Pure function, no DOM.
 *
 * The same simplified approximation the Sensor Recorder uses: Cooper's
 * declination, a standard equation-of-time approximation and the hour-angle
 * formulae. Unlike the original, which read the device clock, the local time
 * zone is an explicit parameter, so a server in another zone returns the
 * same answer as the visitor's browser.
 */

export const SOLAR_METHOD =
  "Simplified approximation: Cooper declination, equation-of-time approximation, hour-angle formulae (as in Sensor Recorder)";

/**
 * @param {number} latitude   degrees, north positive
 * @param {number} longitude  degrees, east positive
 * @param {Date|string|number} instant  the moment of observation
 * @param {number} utcOffsetMinutes  local time zone offset from UTC, e.g. 480 for UTC+8
 * @returns {{ elevation: number, azimuth: number }} degrees; azimuth 0–360 clockwise from north
 */
export function solarPosition(latitude, longitude, instant, utcOffsetMinutes) {
  if (typeof latitude !== "number" || typeof longitude !== "number") {
    return { elevation: null, azimuth: null };
  }
  const rad = Math.PI / 180;
  const deg = 180 / Math.PI;
  const ms = new Date(instant).getTime();
  // Local wall-clock fields, read through UTC getters of a shifted date.
  const local = new Date(ms + utcOffsetMinutes * 60000);
  const year = local.getUTCFullYear();
  const n = Math.floor(
    (Date.UTC(year, local.getUTCMonth(), local.getUTCDate()) -
      Date.UTC(year, 0, 0)) /
      86400000
  );

  const B = (2 * Math.PI * (n - 81)) / 364;
  const EoT = 9.87 * Math.sin(2 * B) - 7.53 * Math.cos(B) - 1.5 * Math.sin(B); // minutes
  const decl = 23.45 * Math.sin(((2 * Math.PI) / 365) * (284 + n)); // degrees

  const localMinutes =
    local.getUTCHours() * 60 +
    local.getUTCMinutes() +
    local.getUTCSeconds() / 60;
  const tz = utcOffsetMinutes / 60;
  const solarMinutes = localMinutes + 4 * longitude + EoT - 60 * tz;
  const HRA = 15 * (solarMinutes / 60 - 12); // degrees

  const latRad = latitude * rad;
  const declRad = decl * rad;
  const hraRad = HRA * rad;

  const sinAlt =
    Math.sin(latRad) * Math.sin(declRad) +
    Math.cos(latRad) * Math.cos(declRad) * Math.cos(hraRad);
  const elevation = Math.asin(Math.max(-1, Math.min(1, sinAlt))) * deg;

  const azRad = Math.atan2(
    Math.sin(hraRad),
    Math.cos(hraRad) * Math.sin(latRad) - Math.tan(declRad) * Math.cos(latRad)
  );
  const azimuth = (azRad * deg + 180 + 360) % 360;

  return { elevation, azimuth };
}
