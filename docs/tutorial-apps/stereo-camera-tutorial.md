---
title: Stereo Vision Workspace
description: "Split and rectify a side-by-side camera stream, inspect stereo pairs and explore depth output with an explicit calibration profile."
sidebar_label: Stereo Vision
sidebar_position: 8
hide_title: true
keywords: [stereo, camera, depth, opencv, rectification]
app_route: /app/stereo
app_icon: "3D"
app_category: "Imaging & vision"
app_runtime: "Local camera processing"
app_tone: violet
app_badges: ["Camera input", "Stereo pair", "Depth preview"]
---

## What it does

Stereo Vision Workspace reads a side-by-side camera stream, separates the two eye views, applies a bundled rectification profile when OpenCV is available, and produces an exploratory grayscale depth preview. You can save stereo pairs and depth images individually or package the session files as a ZIP.

:::caution Exploratory depth only

The depth canvas is a per-frame visualization, not calibrated raw depth data. It does not preserve millimetre values and must not be used directly for leaf thickness, dimensional metrology, or comparisons between independently normalized frames.

:::

## Before you start

- Connect a camera that outputs left and right views side by side in one video frame.
- Use HTTPS or localhost and a browser that supports `MediaDevices`, Canvas, and WebAssembly.
- Grant camera permission only after selecting **Start camera**; the page does not start capture automatically.
- For the bundled calibration profile, use the original 1280 × 480 stream: 640 × 480 pixels per eye. Other devices or resolutions are not calibrated by this profile.
- Keep an internet connection available while OpenCV.js and JSZip load from external CDNs.
- Expect substantial CPU and memory use during live rectification and depth computation.

## Quick workflow

1. Under **Camera configuration**, choose **Video device** and set **Total width** and **Height**.
2. Enter a filesystem-safe **Sample ID** or keep the default `sample` prefix.
3. Select **Start camera** and approve the browser permission request.
4. Check the status panel. Calibrated rectification requires OpenCV and successfully initialized rectification maps; **basic mode** is only a fallback preview.
5. Select **Capture stereo** to save the current left and right canvases.
6. Select **Compute depth**, inspect the grayscale result, then select **Save depth map** if it is useful as a visual record.
7. Select **Download ZIP** to package all captured session files.
8. Select **Stop** before disconnecting the camera or leaving the page.

## Controls & outputs

| Control              | Current behaviour                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------- |
| Video device         | Selects a detected camera. Labels may remain generic until permission is granted.           |
| Total width / Height | Requests a stream size. These values do not create a new calibration profile.               |
| Start camera / Stop  | Starts or releases the selected media stream.                                               |
| Sample ID            | Supplies the sanitized prefix used in captured filenames.                                   |
| Capture stereo       | Adds left and right PNGs to the session ZIP and provides individual download links.         |
| Compute depth        | Computes one disparity-derived grayscale depth visualization from the current pair.         |
| Save depth map       | Adds the current grayscale depth PNG to the session. It is enabled after depth computation. |
| Download ZIP         | Downloads the in-memory session as `stereo_captures.zip`.                                   |

Stereo-pair filenames follow:

```text
sample_stereo_001_left_rectified.png
sample_stereo_001_right_rectified.png
```

Depth filenames follow:

```text
sample_depth_001_depth_precision.png
```

The filename describes the intended workflow. Confirm the status panel reported working OpenCV rectification; fallback split images can otherwise retain the `_rectified` suffix without calibrated remapping.

## How it works

Each video frame is copied to a hidden Canvas and split at half its width. With OpenCV ready, the app uses fixed intrinsic, distortion, rotation, and translation values to build 640 × 480 rectification maps for one stereo rig, then remaps the two eye images.

**Compute depth** converts the pair to grayscale and runs OpenCV StereoBM with 64 disparities and a 15-pixel block. Disparity is converted internally using the bundled focal length and approximately 59.94 mm baseline, but the app then maps the valid minimum and maximum of that single frame to 0–255. Only this normalized image is displayed and saved.

If OpenCV cannot load, the app falls back to a simple split and a browser-based block matcher. That fallback is slower and is not a calibrated replacement for the OpenCV path.

## Data, privacy & external services

- Camera frames, image processing, capture lists, and ZIP assembly stay in the browser; the app does not upload the video stream.
- OpenCV.js is attempted from several public CDNs, and JSZip is loaded from jsDelivr. Their availability and policies are external to this site.
- Captures remain only in the page's in-memory ZIP until downloaded. Refreshing or leaving the page clears the session.
- The selected camera ID is stored locally so the browser can restore the preference later.

## Limitations

- The bundled calibration is specific to one 1280 × 480 side-by-side rig. Lens changes, camera movement, focus changes, resizing, or another device invalidate it.
- With the configured focal length, baseline, and 64-disparity search, the nearest representable depth is approximately 0.49 m. The workspace is therefore not configured for a 5–30 cm measurement range.
- The saved depth PNG contains relative grayscale only; it does not include the frame's normalization limits, raw disparity, confidence, or metric depth.
- Per-frame normalization means equal gray values in different images need not represent equal distances.
- Textureless, reflective, repetitive, occluded, or poorly synchronized scenes can produce invalid matches.
- Live remapping and the fallback matcher can be slow on mobile or low-power hardware.
- No calibration editor, raw-depth export, point cloud, leaf contour, or thickness analysis is provided.

## Troubleshooting

| Problem                           | What to check                                                                                                   |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| No camera is listed               | Connect the device, use HTTPS, grant permission, and reopen the device list after the browser reveals labels.   |
| Camera access fails               | Close other camera applications, check site permission, and try **Start camera** again.                         |
| The two eye views are not aligned | Confirm side-by-side ordering, request 1280 × 480, and verify the physical rig matches the bundled calibration. |
| Status reports basic mode         | Check the network and reload so OpenCV.js can initialize; do not treat fallback output as calibrated.           |
| Depth is mostly black or noisy    | Improve texture and diffuse lighting, reduce reflections, keep both eyes synchronized, and avoid occlusion.     |
| **Save depth map** is disabled    | Start the camera and complete **Compute depth** first.                                                          |
| ZIP download is unavailable       | Capture at least one stereo pair or depth map and ensure the JSZip CDN loaded.                                  |
| The page becomes unresponsive     | Stop the stream, reduce resolution, close other heavy tabs, and avoid the basic fallback on low-power devices.  |

[Open Stereo Vision Workspace](/app/stereo)

[Back to App Lab](/app)
