/**
 * Wording of the data integrity screen in English and Chinese: what each
 * check looks for, why the pattern is unusual, and the ordinary explanations
 * a reviewer should rule out first.
 */

export const CHECKS = {
  duplicate_rows: {
    title: { en: "Duplicated records", zh: "重复记录" },
    what: {
      en: "Rows with identical values in every screened column.",
      zh: "所有被筛查列的数值完全相同的行。",
    },
    why: {
      en: "Independent measurements rarely coincide in every variable; the expected number of such coincidences, from each column's value frequencies, is given.",
      zh: "独立测量很少在所有变量上同时一致；报告给出了按各列数值频率估算的偶然一致期望次数。",
    },
    innocent: {
      en: "A record entered or exported twice; a design in which rows repeat on purpose (pooled samples, templates).",
      zh: "同一条记录被重复录入或导出；设计上有意重复的行（混合样本、模板）。",
    },
  },
  near_duplicate_rows: {
    title: { en: "Near-identical records", zh: "近似重复记录" },
    what: {
      en: "Pairs of rows identical in every screened column except one.",
      zh: "除一列外其余被筛查列都完全相同的成对行。",
    },
    why: {
      en: "The pattern a copied row leaves after one value is edited.",
      zh: "这是复制一行后再改动其中一个数值所留下的典型痕迹。",
    },
    innocent: {
      en: "Repeated measurements of the same unit in which only one variable changes; coarse values that often coincide.",
      zh: "对同一对象的重复测量中只有一个变量变化；数值精度较粗、经常相同。",
    },
  },
  identical_columns: {
    title: { en: "Identical columns", zh: "完全相同的列" },
    what: {
      en: "Two columns with the same value in every row where both have data.",
      zh: "两列在都有数据的每一行上数值完全相同。",
    },
    why: {
      en: "Separate variables or replicates should not agree exactly.",
      zh: "不同变量或重复之间不应完全一致。",
    },
    innocent: {
      en: "The same variable included twice, or a column copied as a template.",
      zh: "同一变量被收录两次，或某列被复制作为模板。",
    },
  },
  linear_columns: {
    title: { en: "Columns related exactly", zh: "列之间存在精确线性关系" },
    what: {
      en: "One column equals a × other column + b in every row, to the recorded digits.",
      zh: "某列在每一行上都等于另一列 × a + b（在记录精度内完全成立）。",
    },
    why: {
      en: "Measured variables are never exactly proportional; an exact relation means one column was computed from the other.",
      zh: "实测变量不会精确成比例；精确关系说明其中一列由另一列计算得到。",
    },
    innocent: {
      en: "A derived column (unit conversion, percentage, total, normalized value) — expected if the computation is documented.",
      zh: "派生列（单位换算、百分比、合计、归一化值）——若计算方法有说明则属正常。",
    },
  },
  repeated_run: {
    title: { en: "Repeated sequence of values", zh: "重复出现的数值序列" },
    what: {
      en: "A run of consecutive values that appears again elsewhere, in the same order.",
      zh: "一段连续数值在别处按相同顺序再次出现。",
    },
    why: {
      en: "The trace of copying and pasting a block of cells. The expected number of such repeats by chance, over every pair of columns and positions, is given.",
      zh: "这是复制粘贴一段单元格留下的痕迹。报告给出了在所有列对和位置上偶然出现此类重复的期望次数。",
    },
    innocent: {
      en: "The same samples measured or listed twice; a reference series repeated by design.",
      zh: "同一批样本被测量或列出两次；设计上重复使用的参考序列。",
    },
  },
  outliers: {
    title: { en: "Outlying values", zh: "离群值" },
    what: {
      en: "Values far from the column median, by the modified z-score (Iglewicz & Hoaglin).",
      zh: "按修正 z 分数（Iglewicz & Hoaglin）远离该列中位数的数值。",
    },
    why: {
      en: "A data-quality check: outliers are not signs of fabrication but should be traceable to the raw records.",
      zh: "这是数据质量检查：离群值并非造假迹象，但应能追溯到原始记录。",
    },
    innocent: {
      en: "Genuine extreme observations, mixed units, or typing errors (a misplaced decimal point).",
      zh: "真实的极端观测、单位混用或录入错误（小数点错位）。",
    },
  },
  terminal_digits: {
    title: { en: "Uneven final digits", zh: "末位数字分布不均" },
    what: {
      en: "The last recorded digit (third significant place or later) is not evenly distributed.",
      zh: "最后一位记录数字（第三位有效数字及以后）分布不均匀。",
    },
    why: {
      en: "In measured data such digits are close to uniform; invented numbers usually are not, because people favour some digits (Mosimann et al. 1995).",
      zh: "实测数据中这一位数字接近均匀分布；编造的数字通常不是，因为人会偏好某些数字（Mosimann 等, 1995）。",
    },
    innocent: {
      en: "Values read by eye from a scale, converted units, or a mixture of instruments with different resolutions.",
      zh: "肉眼读取刻度、单位换算后的数值，或混用了分辨率不同的仪器。",
    },
  },
  rounding_heaping: {
    title: { en: "Preference for 0 and 5", zh: "末位偏好 0 和 5" },
    what: {
      en: "More final digits 0 and 5 (or 5 among 1–9) than chance allows.",
      zh: "末位为 0 和 5（或在 1–9 中为 5）的比例高于偶然水平。",
    },
    why: {
      en: "The mark of values rounded or estimated by hand rather than read from an instrument.",
      zh: "这是数值被人工取整或估读、而非由仪器读出的标志。",
    },
    innocent: {
      en: "Manual readings (rulers, scores, visual ratings) are routinely rounded to 0 or 5.",
      zh: "人工读数（直尺、评分、目测分级）常被取整到 0 或 5。",
    },
  },
  benford: {
    title: {
      en: "First digits differ from Benford's law",
      zh: "首位数字偏离本福特定律",
    },
    what: {
      en: "First significant digits of positive values spanning three or more orders of magnitude, compared with Benford's law (MAD, Nigrini 2012).",
      zh: "跨越三个及以上数量级的正值，其首位有效数字与本福特定律比较（MAD，Nigrini 2012）。",
    },
    why: {
      en: "Data that grow multiplicatively often follow Benford's law; strong departures can indicate invented values.",
      zh: "按倍数增长的数据常符合本福特定律；明显偏离可能提示数值系编造。",
    },
    innocent: {
      en: "Many measurements do not follow Benford's law at all (bounded or normally distributed quantities); reported as a note only.",
      zh: "许多测量量本就不符合本福特定律（有界或正态分布的量）；因此只作为提示。",
    },
  },
  equal_steps: {
    title: { en: "Equal increments", zh: "等步长序列" },
    what: {
      en: "Consecutive values that rise or fall by exactly the same step.",
      zh: "相邻数值以完全相同的步长递增或递减。",
    },
    why: {
      en: "Measurements with noise do not form exact arithmetic sequences; filled-in or interpolated values do. The expected number of such runs by chance is given.",
      zh: "带噪声的测量不会形成精确等差数列，而补填或插值得到的数值会。报告给出了偶然出现此类序列的期望次数。",
    },
    innocent: {
      en: "A set-point or dose series measured in that order, or interpolation that is documented.",
      zh: "按设定梯度或剂量顺序记录的数值，或有说明的插值。",
    },
  },
  common_step: {
    title: { en: "Values on a common step", zh: "数值落在统一步长上" },
    what: {
      en: "Every value is a multiple of a step coarser than the recorded precision.",
      zh: "所有数值都是某个步长的整数倍，而该步长比记录精度更粗。",
    },
    why: {
      en: "The data are coarser than their decimals suggest; digit tests are not applied to such a column.",
      zh: "数据的实际精度比小数位数显示的更粗；此类列不做末位数字检验。",
    },
    innocent: {
      en: "The instrument's resolution (for example 0.5 units) or values converted from a coarser scale.",
      zh: "仪器分辨率（例如 0.5 个单位），或由更粗刻度换算而来的数值。",
    },
  },
  grim: {
    title: { en: "Impossible mean (GRIM)", zh: "不可能的均值（GRIM）" },
    what: {
      en: "A mean of n integer observations that no integer total can produce at its reported decimals (Brown & Heathers 2017).",
      zh: "由 n 个整数观测得到的均值，在其报告的小数位下没有任何整数总和能产生（Brown & Heathers 2017）。",
    },
    why: {
      en: "Such a mean cannot come from the stated n integer values.",
      zh: "这样的均值不可能来自所述的 n 个整数。",
    },
    innocent: {
      en: "A typo in the mean or n, missing responses (a smaller n), or data that are not integers.",
      zh: "均值或 n 录入错误、存在缺失应答（实际 n 更小），或数据并非整数。",
    },
  },
};

export const REFERENCES = [
  "Al-Marzouki S, Evans S, Marshall T, Roberts I (2005). Are these data real? Statistical methods for the detection of data fabrication in clinical trials. BMJ 331:267–270.",
  "Brown NJL, Heathers JAJ (2017). The GRIM test: a simple technique detects numerous anomalies in the reporting of results in psychology. Soc Psychol Personal Sci 8:363–369.",
  "Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate. J R Stat Soc B 57:289–300.",
  "Iglewicz B, Hoaglin DC (1993). How to Detect and Handle Outliers. ASQC Quality Press.",
  "Mosimann JE, Wiseman CV, Edelman RE (1995). Data fabrication: can people generate random digits? Account Res 4:31–55.",
  "Nigrini MJ (2012). Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection. Wiley.",
];
