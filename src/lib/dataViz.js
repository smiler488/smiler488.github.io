/**
 * Data colours (design/DESIGN_SPEC.md §7.4). Kept separate from the UI
 * palette in src/css/tokens.css: the UI stays neutral, data gets colour.
 *
 * CATEGORICAL is the Okabe–Ito colour-blind-safe set, ordered for white or
 * near-black backgrounds (black is omitted so series stay visible in dark
 * mode; yellow comes last because of its low contrast on white).
 * SEQUENTIAL is viridis, sampled at eight stops.
 */
export const CATEGORICAL = [
  "#0072B2",
  "#E69F00",
  "#009E73",
  "#D55E00",
  "#56B4E9",
  "#CC79A7",
  "#F0E442",
];

export const SEQUENTIAL = [
  "#440154",
  "#46327E",
  "#365C8D",
  "#277F8E",
  "#1FA187",
  "#4AC16D",
  "#A0DA39",
  "#FDE725",
];

export const CHART_FONT_FAMILY =
  "'Hanken Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
