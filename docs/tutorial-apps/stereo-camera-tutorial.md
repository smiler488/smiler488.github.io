---
title: Stereo Vision Workspace
description: "Rectify a calibrated side-by-side stereo stream, compute metric depth by block matching and export depth in millimetres."
sidebar_label: Stereo Vision
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

## What it does

Stereo Vision Workspace reads a side-by-side stereo camera stream, rectifies the two views with the rig's calibration, and computes a depth map in millimetres by block matching. Stereo pairs, a depth preview and the metric depth (16-bit PGM plus a JSON record of the parameters) can be saved and packaged as a ZIP. All processing runs in the browser.

## Before you start

- Connect a camera that outputs left and right views side by side in one video frame.
- Use HTTPS or localhost and a browser that supports `MediaDevices` and Canvas.
- Camera access starts only after you select **Start camera**.
- Depth uses the bundled calibration of one 1280 × 480 side-by-side rig (640 × 480 per eye). Other cameras and resolutions can be previewed, but depth is computed only at the calibrated resolution.

## Quick workflow

1. Under **Camera configuration**, choose **Video device** and set **Total width** 1280 and **Height** 480.
2. Enter a filesystem-safe **Sample ID** or keep the default `sample` prefix.
3. Select **Start camera** and approve the browser permission request. The rectified left and right views appear; **Compute depth** is enabled when the stream matches the calibration.
4. Select **Capture stereo** to save the current rectified pair.
5. Select **Compute depth**. The status line reports the fraction of pixels measured and the median and 5–95 % depth range.
6. Select **Save depth map** to add the depth to the session.
7. Select **Download ZIP** to package all captured files, then **Stop** before disconnecting the camera.

## Controls & outputs

| Control              | Behaviour                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------- |
| Video device         | Selects a detected camera. Labels may remain generic until permission is granted.           |
| Total width / Height | Requests a stream size. Depth requires 1280 × 480.                                           |
| Start camera / Stop  | Starts or releases the selected media stream.                                               |
| Sample ID            | Supplies the sanitized prefix used in captured filenames.                                   |
| Capture stereo       | Adds the rectified left and right PNGs to the session ZIP.                                   |
| Compute depth        | Rectifies the current frame and computes metric depth.                                       |
| Save depth map       | Adds the depth preview, the 16-bit depth in millimetres and the parameter record to the ZIP. |
| Download ZIP         | Downloads the session as `stereo_captures.zip`.                                              |

Files:

```text
sample_stereo_001_left_rectified.png
sample_stereo_001_right_rectified.png
sample_depth_001_depth_preview.png   # grey preview, near = bright
sample_depth_001_depth_mm.pgm        # 16-bit depth in mm, 0 = no measurement
sample_depth_001_depth.json          # focal length, baseline, matcher settings, depth statistics
```

Read the PGM in Python with `cv2.imread(path, cv2.IMREAD_UNCHANGED)` (values in millimetres), or open it in ImageJ/Fiji.

## How it works

1. **Rectification.** From the rig's intrinsic matrices, distortion coefficients and the rotation and translation between the cameras, the workspace computes rectifying rotations with Bouguet's algorithm (as OpenCV `stereoRectify`, zero-disparity principal point) and builds undistort-rectify maps (as `initUndistortRectifyMap`). Both views are remapped with bilinear interpolation, so corresponding points lie on the same image row.
2. **Disparity.** The rectified views are converted to grey and matched with sum-of-absolute-differences block matching (15 × 15 blocks, 64 disparities) computed on integral images. A match is kept only if it is unique (no other disparity within 10 % of the best cost), consistent left-to-right and right-to-left (within 1 pixel), and in a textured block; disparity is refined to sub-pixel by a parabola through the costs.
3. **Depth.** Z = f · B / d, with the rectified focal length f ≈ 528 px and the baseline B ≈ 59.9 mm.

The implementation (`static/js/stereo_core.js`) is validated on a synthetic scene: textured planes at 600 mm and 900 mm rendered through this rig's calibration, including lens distortion and the inter-camera rotation, are recovered with a median depth error below 1 % (measured 0.2 %) and 98–100 % of pixels measured.

## Data, privacy & external services

- Camera frames, processing, capture lists and ZIP assembly stay in the browser; the video stream is not uploaded.
- JSZip is loaded from jsDelivr.
- Captures remain only in the page's in-memory ZIP until downloaded; refreshing or leaving the page clears the session.
- The selected camera ID is stored locally so the browser can restore the preference.

## Limitations

- The calibration belongs to one 1280 × 480 rig. Lens changes, camera movement, focus changes or another device require recalibration.
- With f ≈ 528 px, B ≈ 59.9 mm and 64 disparities, the nearest measurable depth is about 0.49 m. Depth resolution falls with distance: at 1 m one pixel of disparity corresponds to about 3 cm.
- Textureless, reflective, repetitive or occluded regions have no reliable match and are left empty (0 in the PGM).
- Depth computation takes about a second per frame on a laptop and longer on mobile devices.

## Troubleshooting

| Problem                           | What to check                                                                                                   |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| No camera is listed               | Connect the device, use HTTPS, grant permission, and reopen the device list after the browser reveals labels.   |
| Camera access fails               | Close other camera applications, check site permission, and try **Start camera** again.                         |
| **Compute depth** stays disabled  | The stream is not 1280 × 480 side by side; request that size or use the calibrated rig.                          |
| Depth is mostly empty             | Add texture, use diffuse light, avoid reflections, and keep the subject beyond about 0.5 m.                     |
| Rectified rows do not line up     | The rig no longer matches its calibration (moved lens, refocus); recalibrate.                                   |
| ZIP download is unavailable       | Capture at least one stereo pair or depth map and check that JSZip loaded.                                      |

[Open Stereo Vision Workspace](/app/stereo)

[Back to App Lab](/app)
