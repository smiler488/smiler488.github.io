/**
 * Cotton 3D datasets shown in the research notes (design/DESIGN_SPEC.md §7.2).
 * Generated from the author's files (sample cotton_20240109-84-5 and the
 * canopy model) by quantizing positions to 16-bit integers over each
 * bounding box. Binary layouts:
 *   points: Uint16 xyz × count, then Uint8 class × count (sorted by class)
 *   mesh:   Uint16 xyz for leaf then non-leaf vertices, then Uint16 index
 *           triples for leaf then non-leaf triangles (indices per group)
 * The canopy was simplified for display (quadric decimation per optical
 * class); `full` holds statistics computed from the complete model.
 */
export const COTTON3D = {
  sfm: {
    kind: "points",
    count: 39765,
    sourceCount: 39765,
    min: [0.24950001, 0.0094, -1.03312504],
    max: [0.48629999, 0.4118, -0.7184],
    classCounts: {
      0: 2862,
      1: 2817,
      2: 34086,
    },
    bytes: 278355,
  },
  hy3d: {
    kind: "points",
    count: 60000,
    sourceCount: 177357,
    min: [0.2519900088888889, 0.022262335, -1.016200452222222],
    max: [0.4934674966666667, 0.406886535, -0.7214752866666666],
    classCounts: {
      0: 3881,
      1: 4627,
      2: 51492,
    },
    bytes: 420000,
  },
  canopy: {
    kind: "mesh",
    min: [-0.475139, 0.002237225, -1.0801292],
    max: [1.0761758499999998, 0.4113439, -0.22416435],
    leaf: {
      vertices: 57552,
      triangles: 89999,
    },
    nonleaf: {
      vertices: 18560,
      triangles: 30000,
    },
    full: {
      triangles: 1613632,
      leafTriangles: 1214544,
      nonleafTriangles: 399088,
      leafArea_m2: 1.2745,
      nonleafArea_m2: 0.4344,
      height_m: 0.414,
      plants: 24,
      // label1 = reflectance, label2 = transmittance (confirmed by the author).
      optical: {
        leaf: {
          reflectance: 0.1,
          transmittance: 0.05,
        },
        nonleaf: {
          reflectance: 0.15,
          transmittance: 0.0,
        },
      },
    },
  },
};

/** Point-cloud classes, as confirmed by the author. */
export const POINT_CLASSES = {
  0: { en: "Main stem", zh: "主茎" },
  1: { en: "Branches and petioles", zh: "分枝与叶柄" },
  2: { en: "Leaves", zh: "叶片" },
};
