/** Chinese interface text for this tool (src/lib/i18n/toolText.js). */
export default {
  "Solar time is computed from the UTC instant, the longitude and the equation of time. Validated against NREL SPA: elevation within 0.05° and azimuth within 0.1°. Elevation is geometric, without atmospheric refraction.":
    "真太阳时由 UTC 时刻、经度和时差计算。已用 NREL SPA 验证：高度角误差 0.05° 以内，方位角误差 0.1° 以内。高度角为几何高度角，未做大气折射校正。",
  "solar declination": "太阳赤纬",
  "geographic latitude": "地理纬度",
  Symbols: "符号",
  "rad × (180 / π) · Azimuth reported from North, clockwise [0 … 360°]":
    "rad × (180 / π) · 方位角以正北为起点顺时针计量 [0 … 360°]",
  "atan2(": "atan2(",
  "arcsin(": "arcsin(",
  "Solar angle formulas": "太阳角度公式",
  "No data yet. Enter ID and click “Capture Sample”.":
    "暂无数据。输入编号后点击“采集样本”。",
  "Motion access:": "运动权限：",
  "Recorded rows:": "已记录行数：",
  "Session status": "会话状态",
  "Refresh Location": "刷新定位",
  "Accuracy:": "精度：",
  "Altitude:": "海拔：",
  "Longitude:": "经度：",
  "Latitude:": "纬度：",
  "Latest location": "最新定位",
  "Please allow motion/orientation access in browser settings.":
    "请在浏览器设置中允许访问运动与方向传感器。",
  "Gamma (Y, roll):": "Gamma（Y 轴，横滚）：",
  "Beta (X, pitch):": "Beta（X 轴，俯仰）：",
  "Alpha (Z, yaw):": "Alpha（Z 轴，偏航）：",
  "Current orientation": "当前姿态",
  "Export CSV": "导出 CSV",
  "e.g. Plot-04-Leaf-12": "如 Plot-04-Leaf-12",
  "Leaf or sample ID": "叶片或样本编号",
  "Allow motion/orientation and location access from an explicit tap.":
    "请通过点击按钮授权访问运动/方向传感器和定位。",
  "Sensor readiness": "传感器状态",
  "No orientation reading arrived. Keep the phone awake, check motion access, and try again.":
    "没有收到姿态读数。请保持手机屏幕常亮、检查运动权限后重试。",
  "Motion permission was denied or is not available. Please enable motion/orientation access for this site in your browser settings and try again.":
    "运动权限被拒绝或不可用。请在浏览器设置中为本网站开启运动/方向权限后重试。",
  "This device or browser does not provide motion sensors. Orientation data may not be available. Try using a mobile phone with gyroscope/accelerometer.":
    "此设备或浏览器不提供运动传感器，可能无法获取姿态数据。请使用带陀螺仪和加速度计的手机。",
  "Unable to access location. Your browser or device may have blocked geolocation for this site.":
    "无法获取定位。浏览器或设备可能已禁止本网站使用定位。",
  "Geolocation is not supported on this device or browser.":
    "此设备或浏览器不支持定位。",
  "Not requested": "未请求",
  Unavailable: "不可用",
  Enabled: "已开启",
  "Enable Motion Permission": "开启运动权限",
  "Motion Permission Granted": "已获得运动权限",
  "Capture Sample": "采集样本",
  "Capturing…": "正在采集…",
  "Enable sensors": "开启传感器",
  "Sensors enabled": "传感器已开启",
  "Solar declination (δ) and the equation of time come from the NOAA solar position algorithm (after Meeus); elevation (h) and azimuth (A) then follow from:":
    "太阳赤纬（δ）和时差采用 NOAA 太阳位置算法（基于 Meeus）计算；高度角（h）和方位角（A）再由下式求得：",
  "{0} GPS points · {1}": "{0} 个 GPS 点 · {1}",
};
