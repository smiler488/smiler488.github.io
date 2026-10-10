/**
 * Solar position (design/DESIGN_SPEC.md §9.3). Pure function, no DOM.
 *
 * NOAA solar position algorithm (the NOAA Global Monitoring Laboratory
 * solar calculator, after Meeus, Astronomical Algorithms). science.test.js
 * checks it against NREL SPA (pvlib) for 1950–2090 at latitudes ±65°:
 * elevation and azimuth agree within a few hundredths of a degree.
 * Elevation is geometric, without atmospheric refraction.
 */

export const SOLAR_METHOD =
  "NOAA solar position algorithm (Meeus); geometric elevation without refraction; validated against NREL SPA";

const rad = Math.PI / 180;
const deg = 180 / Math.PI;

/**
 * @param {number} latitude   degrees, north positive
 * @param {number} longitude  degrees, east positive
 * @param {Date|string|number} instant  the moment of observation
 * @param {number} [utcOffsetMinutes]  local time zone of the observer. The
 *   position depends only on the instant; the offset is accepted so callers
 *   can keep recording the wall-clock zone, and does not change the result.
 * @returns {{ elevation: number, azimuth: number }} degrees; azimuth 0–360 clockwise from north
 */
// eslint-disable-next-line no-unused-vars
export function solarPosition(latitude, longitude, instant, utcOffsetMinutes) {
  if (typeof latitude !== "number" || typeof longitude !== "number") {
    return { elevation: null, azimuth: null };
  }
  const ms = new Date(instant).getTime();
  const julianDay = ms / 86400000 + 2440587.5;
  const T = (julianDay - 2451545) / 36525; // Julian centuries since J2000.0

  const L0 = (280.46646 + T * (36000.76983 + T * 0.0003032)) % 360; // mean longitude
  const M = 357.52911 + T * (35999.05029 - 0.0001537 * T); // mean anomaly
  const e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T); // eccentricity
  const C =
    Math.sin(M * rad) * (1.914602 - T * (0.004817 + 0.000014 * T)) +
    Math.sin(2 * M * rad) * (0.019993 - 0.000101 * T) +
    Math.sin(3 * M * rad) * 0.000289; // equation of centre
  const trueLong = L0 + C;
  const omega = 125.04 - 1934.136 * T;
  const apparentLong = trueLong - 0.00569 - 0.00478 * Math.sin(omega * rad);
  const meanObliquity =
    23 +
    (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60;
  const obliquity = meanObliquity + 0.00256 * Math.cos(omega * rad);
  const decl = Math.asin(
    Math.sin(obliquity * rad) * Math.sin(apparentLong * rad)
  );

  const y = Math.tan((obliquity / 2) * rad) ** 2;
  const eqTime =
    4 *
    deg *
    (y * Math.sin(2 * L0 * rad) -
      2 * e * Math.sin(M * rad) +
      4 * e * y * Math.sin(M * rad) * Math.cos(2 * L0 * rad) -
      0.5 * y * y * Math.sin(4 * L0 * rad) -
      1.25 * e * e * Math.sin(2 * M * rad)); // minutes

  const utcMinutes = (((ms % 86400000) + 86400000) % 86400000) / 60000;
  const trueSolarMinutes =
    (((utcMinutes + eqTime + 4 * longitude) % 1440) + 1440) % 1440;
  const hourAngle = trueSolarMinutes / 4 - 180; // degrees, 0 at solar noon

  const lat = latitude * rad;
  const cosZenith = Math.max(
    -1,
    Math.min(
      1,
      Math.sin(lat) * Math.sin(decl) +
        Math.cos(lat) * Math.cos(decl) * Math.cos(hourAngle * rad)
    )
  );
  const zenith = Math.acos(cosZenith);
  const elevation = 90 - zenith * deg;

  const azimuth =
    (Math.atan2(
      Math.sin(hourAngle * rad),
      Math.cos(hourAngle * rad) * Math.sin(lat) - Math.tan(decl) * Math.cos(lat)
    ) *
      deg +
      180 +
      360) %
    360;

  return { elevation, azimuth };
}
