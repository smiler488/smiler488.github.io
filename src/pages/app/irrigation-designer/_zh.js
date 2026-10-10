/** Chinese interface text for this tool (src/lib/i18n/toolText.js). */
export default {
  "Pressure-compensating": "压力补偿式",
  "Mainline velocity {0} m/s exceeds {1} m/s; use a larger mainline or fewer submains per shift.":
    "干管流速 {0} m/s 超过 {1} m/s；请加大干管管径或减少每组轮灌的支管数。",
  "Submain inlet velocity {0} m/s exceeds {1} m/s; use a larger submain or closer submains.":
    "支管入口流速 {0} m/s 超过 {1} m/s；请加大支管管径或减小支管间距。",
  "Flow per shift {0} m³/h exceeds the pump's {1} m³/h; open fewer submains per shift.":
    "每组轮灌流量 {0} m³/h 超过水泵的 {1} m³/h；请减少每组同时开启的支管数。",
  "Pressure at the farthest submain inlet is {0} kPa short of what the subunit needs; raise pump pressure or enlarge the mainline.":
    "最远支管入口压力比灌水小区所需压力低 {0} kPa；请提高水泵压力或加大干管管径。",
  "Pressure variation within a subunit is {0}% (limit {1}%); shorten laterals, enlarge the submain or use pressure-compensating emitters.":
    "灌水小区内压力偏差为 {0}%（上限 {1}%）；请缩短毛管、加大支管管径或改用压力补偿滴头。",
  "Emitter flow variation is {0}%, above the 20% usually considered acceptable.":
    "灌水器流量偏差为 {0}%，超过通常可接受的 20%。",
  ", and emitter flow variation": "，灌水器流量偏差",
  "Mainline and submains: Hazen–Williams,": "干管和支管：Hazen–Williams 公式，",
  "Underlying formulas & references": "计算公式与参考文献",
  Tips: "设计建议",
  Reset: "重置",
  "Export SVG": "导出 SVG",
  "Tape inner diameter (mm)": "滴灌带内径（mm）",
  "Operating pressure (kPa)": "工作压力（kPa）",
  "Emitter flow (L/h)": "滴头流量（L/h）",
  "Emitter spacing (cm)": "滴头间距（cm）",
  "Tape spacing (m)": "滴灌带间距（m）",
  "Drip laterals": "滴灌带（毛管）",
  "Submains per shift": "每组轮灌支管数",
  "Diameter (mm)": "管径（mm）",
  "Spacing (m)": "间距（m）",
  Submains: "支管",
  "Ring / two-end feed": "环状 / 两端供水",
  Centerline: "中线",
  "Field edge": "田块边缘",
  Location: "位置",
  "PVC (C≈150)": "PVC（C≈150）",
  "PE (C≈140)": "PE（C≈140）",
  Material: "材质",
  Mainline: "干管",
  "Max velocity (m/s)": "最大流速（m/s）",
  "Allowable pressure variation (%)": "允许压力偏差（%）",
  "Fertigation skid": "施肥装置",
  "Filter loss (kPa)": "过滤器损失（kPa）",
  "Max flow (m³/h)": "最大流量（m³/h）",
  "Pump pressure (kPa)": "水泵压力（kPa）",
  "Pump pressure, filter/fertigation losses, and allowable variation determine available head.":
    "水泵压力、过滤器与施肥装置损失以及允许偏差，共同决定可用水头。",
  "Headworks & Constraints": "首部与约束条件",
  "Slope along width (%)": "垂直行方向坡度（%）",
  "Slope along length (%)": "顺行方向坡度（%）",
  "clockwise from N": "自正北顺时针",
  "Orientation (°)": "方向（°）",
  "Width (m)": "宽度（m）",
  "Length (m)": "长度（m）",
  "Orientation measured clockwise from true north. Slopes convert to head differences.":
    "方向自正北顺时针计量；坡度会换算为水头差。",
  "Field & Terrain": "田块与地形",
  "Design check for drip systems.": "滴灌系统设计校核。",
  "Reset defaults": "恢复默认值",
  Warnings: "警示",
  "All screened parameters are within the configured limits. Continue with detailed hydraulic and zoning checks.":
    "各项参数均在设定范围内。可继续进行详细的水力与分区校核。",
  "Hydraulic summary": "水力计算结果",
  "pressure-compensating, within its range": "压力补偿式，处于调压范围内",
  "Emitter flow variation": "灌水器流量偏差",
  "within a subunit": "灌水小区内",
  "Pressure variation": "压力偏差",
  "Pressure at farthest submain": "最远支管入口压力",
  "Mainline headloss": "干管水头损失",
  "Flow per shift": "每组轮灌流量",
  "Submain headloss": "支管水头损失",
  "Lateral headloss": "毛管水头损失",
  "Lateral run": "毛管长度",
  Layout: "布局",
  "Layout Preview": "布局预览",
  Headworks: "首部",
  "Scaled field diagram showing the mainline, submains, drip laterals and headworks.":
    "按比例绘制的田块示意图，显示干管、支管、滴灌带和首部。",
  "Irrigation layout preview": "灌溉布局预览",
  "Use pressure-compensating emitters on slopes above about 0.5% or where pressure variation cannot be kept low.":
    "坡度超过约 0.5% 或无法把压力偏差控制在较低水平时，请使用压力补偿滴头。",
  "Shorter laterals (closer submains) cut lateral losses sharply: friction grows with roughly the 2.75th power of lateral length.":
    "缩短毛管（减小支管间距）能大幅降低毛管损失：摩阻约与毛管长度的 2.75 次方成正比。",
  "Keep pipe velocities at or below 1.5 m/s to limit water hammer and energy losses.":
    "管道流速宜不超过 1.5 m/s，以减小水锤和能量损失。",
  "needed {0} kPa · margin {1}": "需要 {0} kPa · 余量 {1}",
  "velocity {0} m/s": "流速 {0} m/s",
  "inlet {0} m/s · {1} m³/h": "入口 {0} m/s · {1} m³/h",
  "inlet velocity {0} m/s": "入口流速 {0} m/s",
  "every {0} m · {1} laterals": "间距 {0} m · 共 {1} 条毛管",
  "Validated in the site's test suite: F factors against Christiansen's table and the F-factor losses against segment-by-segment summation (within 2%). References: Keller & Karmeli (1974), Trans. ASAE 17(4): 678–684; Christiansen (1942), Univ. California Agric. Exp. Stn. Bull. 670; ASABE EP405.":
    "已在网站测试套件中验证：多口系数与 Christiansen 原表对照，多口系数法的损失与逐段累加摩阻对照（误差 2% 以内）。参考文献：Keller & Karmeli (1974), Trans. ASAE 17(4): 678–684；Christiansen (1942), Univ. California Agric. Exp. Stn. Bull. 670；ASABE EP405。",
  "Laterals follow the crop rows. Tape spacing is the row spacing; emitter data come from the tape's datasheet.":
    "滴灌带沿作物行铺设。滴灌带间距即行距；滴头参数取自滴灌带产品说明书。",
  "Cross the field width. Spacing sets the number of submains and the lateral length (half the spacing on each side); a centreline mainline feeds them from the middle.":
    "支管横跨田宽。间距决定支管条数和毛管长度（每侧为间距的一半）；干管位于中线时从支管中部供水。",
  "Runs along the field length. Material sets the Hazen–Williams C; a ring (two-end) feed halves the run and its flow.":
    "干管沿田块长度方向布置。材质决定 Hazen–Williams 系数 C；环状（两端）供水时干管长度和流量各减半。",
  "Friction in the mainline, submains and laterals, elevation, pressure at the farthest subunit and emitter flow variation are computed from standard hydraulics. Minor losses at fittings, transients and manufacturing variation of emitters are not included; use the tape's datasheet for emitter flow, exponent and inner diameter.":
    "干管、支管和毛管的沿程损失、高差、最远灌水小区的压力以及灌水器流量偏差，均按标准水力学方法计算。未计入管件局部损失、瞬变流和灌水器制造偏差；滴头流量、流态指数和内径请以滴灌带说明书为准。",
  "Field drawn to scale with rotation; headworks shown at origin.":
    "田块按比例绘制并可旋转；首部位于原点。",
  "Keep pressure variation within a subunit (submain plus laterals) below about 20% for non-compensating emitters; this gives about 10% flow variation.":
    "采用非压力补偿滴头时，灌水小区（支管加毛管）内的压力偏差宜控制在约 20% 以内，对应流量偏差约 10%。",
  "。灌水小区入口压力按 Keller & Karmeli 方法计算，":
    "。灌水小区入口压力按 Keller & Karmeli 方法计算，",
  "。有 N 个等间距出口的管道乘以 Christiansen 多口系数":
    "。有 N 个等间距出口的管道乘以 Christiansen 多口系数",
  "。毛管：Darcy–Weisbach 公式，摩阻系数采用 Blasius 公式":
    "。毛管：Darcy–Weisbach 公式，摩阻系数采用 Blasius 公式",
  "{0} submain(s) open · pump {1}": "开启 {0} 条支管 · 水泵 {1}",
  "{0} emitters · {1} L/h": "{0} 个滴头 · {1} L/h",
};
