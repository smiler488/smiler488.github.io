---
title: 双目视觉工作台
description: "对已标定的并排双目视频流做立体校正，用块匹配计算米制深度，并以毫米为单位导出深度。"
sidebar_label: 双目视觉
sidebar_position: 8
hide_title: true
keywords: [stereo, camera, depth, rectification, block matching]
app_route: /app/stereo
app_icon: "3D"
app_category: "Imaging & vision"
app_runtime: "Local camera processing"
app_tone: violet
app_badges: ["Camera input", "Stereo pair", "Metric depth"]
---

## 功能简介

双目视觉工作台读取左右并排的双目相机视频流，用该相机组的标定参数对两路图像做立体校正，再通过块匹配计算以毫米为单位的深度图。可以保存校正后的图像对、深度预览图和米制深度（16 位 PGM 及记录参数的 JSON），并打包为 ZIP。全部处理在浏览器中完成。

## 准备工作

- 连接一台在同一帧中左右并排输出两路画面的相机。
- 使用 HTTPS 或 localhost，浏览器需支持 `MediaDevices` 和 Canvas。
- 只有点击 **Start camera** 后才会开启摄像头。
- 深度计算使用内置的一组 1280 × 480 并排相机的标定参数（每路 640 × 480）。其他相机和分辨率可以预览画面，但只在标定分辨率下计算深度。

## 快速流程

1. 在 **Camera configuration** 中选择 **Video device**，将 **Total width** 设为 1280、**Height** 设为 480。
2. 填写可用作文件名的 **Sample ID**，或保留默认前缀 `sample`。
3. 点击 **Start camera** 并允许浏览器访问摄像头。页面显示校正后的左右视图；视频流与标定参数一致时，**Compute depth** 按钮可用。
4. 点击 **Capture stereo** 保存当前校正后的图像对。
5. 点击 **Compute depth**。状态栏显示有效测量像素的比例，以及深度的中位数和 5%–95% 范围。
6. 点击 **Save depth map** 将深度结果加入本次会话。
7. 点击 **Download ZIP** 打包下载全部文件；断开相机前先点击 **Stop**。

## 控件与输出

| 控件 | 作用 |
| --- | --- |
| Video device | 选择检测到的相机。授权前设备名称可能显示为通用名称。 |
| Total width / Height | 请求的视频尺寸。深度计算要求 1280 × 480。 |
| Start camera / Stop | 开启或释放所选视频流。 |
| Sample ID | 文件名前缀（会自动去除不安全字符）。 |
| Capture stereo | 将校正后的左右 PNG 加入会话 ZIP。 |
| Compute depth | 校正当前帧并计算米制深度。 |
| Save depth map | 将深度预览图、以毫米为单位的 16 位深度和参数记录加入 ZIP。 |
| Download ZIP | 下载会话文件 `stereo_captures.zip`。 |

文件：

```text
sample_stereo_001_left_rectified.png
sample_stereo_001_right_rectified.png
sample_depth_001_depth_preview.png   # 灰度预览，越近越亮
sample_depth_001_depth_mm.pgm        # 16 位深度，单位 mm，0 表示无测量
sample_depth_001_depth.json          # 焦距、基线、匹配参数和深度统计
```

在 Python 中可用 `cv2.imread(path, cv2.IMREAD_UNCHANGED)` 读取 PGM（数值单位为毫米），也可用 ImageJ/Fiji 打开。

## 工作原理

1. **立体校正。** 根据两台相机的内参矩阵、畸变系数以及相机间的旋转和平移，用 Bouguet 算法计算校正旋转（与 OpenCV `stereoRectify` 相同，零视差主点），并生成去畸变校正映射（与 `initUndistortRectifyMap` 相同）。两路图像经双线性插值重映射后，对应点位于同一行。
2. **视差。** 校正后的图像转为灰度，用基于积分图的绝对差和（SAD）块匹配计算视差（15 × 15 窗口，64 个视差）。只有满足以下条件的匹配才被保留：结果唯一（其他视差的代价都不在最优代价的 10% 以内）、左右一致性检验误差不超过 1 像素、窗口内纹理足够；再用代价的抛物线拟合得到亚像素视差。
3. **深度。** Z = f · B / d，其中校正后焦距 f ≈ 528 px，基线 B ≈ 59.9 mm。

该实现（`static/js/stereo_core.js`）已用合成场景验证：将位于 600 mm 和 900 mm 的纹理平面按本相机组的标定参数（含镜头畸变和相机间旋转）渲染成左右图像，恢复出的深度中位数误差小于 1%（实测 0.2%），有效测量像素占 98%–100%。

## 数据、隐私与外部服务

- 视频帧、图像处理、采集列表和 ZIP 打包都在浏览器中完成，视频流不会上传。
- JSZip 从 jsDelivr 加载。
- 采集结果只保存在页面内存中的 ZIP 里，刷新或离开页面即清空。
- 所选相机的 ID 保存在本地，以便下次恢复选择。

## 局限

- 标定参数只适用于这一台 1280 × 480 相机组。更换镜头、相机移动、重新对焦或换用其他设备后需要重新标定。
- 在 f ≈ 528 px、B ≈ 59.9 mm、64 个视差的设置下，可测的最近距离约为 0.49 m。深度分辨率随距离增大而下降：在 1 m 处，1 个像素的视差约对应 3 cm。
- 缺乏纹理、反光、重复纹理或被遮挡的区域无法可靠匹配，结果留空（PGM 中为 0）。
- 在笔记本电脑上每帧深度计算约需 1 秒，移动设备上更慢。

## 常见问题

| 问题 | 检查内容 |
| --- | --- |
| 没有列出相机 | 连接设备、使用 HTTPS、授予权限，浏览器显示设备名称后重新打开设备列表。 |
| 无法访问相机 | 关闭其他占用相机的应用，检查网站权限，再次点击 **Start camera**。 |
| **Compute depth** 一直不可用 | 视频流不是 1280 × 480 并排格式；请求该尺寸或使用已标定的相机组。 |
| 深度图大部分为空 | 增加纹理、使用漫射光、避免反光，并让拍摄对象距离相机 0.5 m 以上。 |
| 校正后左右行不对齐 | 相机组已与标定参数不符（镜头移动、重新对焦），需要重新标定。 |
| 无法下载 ZIP | 先采集至少一组图像对或深度图，并确认 JSZip 已加载。 |

[打开双目视觉工作台](/app/stereo)

[返回实验室](/app)
