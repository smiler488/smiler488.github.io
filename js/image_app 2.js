// ✅ 等待 OpenCV 加载
let isOpenCVLoaded = false;

window.onload = () => {
  if (typeof cv !== "undefined") {
    cv.onRuntimeInitialized = () => {
      console.log("✅ OpenCV.js 加载成功！");
      isOpenCVLoaded = true;
    };
  } else {
    console.error("❌ OpenCV.js 加载失败，请检查网络！");
  }
};

// ✅ 全局变量
let leafData = []; // 存储叶片数据
let currentImageID = 1; // 当前图像编号

// ✅ 监听拍照事件
document.getElementById("photoInput").addEventListener("change", handleImageUpload);

// ✅ 处理上传的图像
function handleImageUpload(event) {
  const file = event.target.files[0];
  if (!file) {
    alert("❌ 未选择任何图片，请重新拍照！");
    return;
  }

  // ✅ 读取文件并显示原始图像
  const reader = new FileReader();
  reader.onload = function (e) {
    const imgElement = document.createElement("img");
    imgElement.src = e.target.result;
    imgElement.onload = function () {
      displayOriginalImage(imgElement);
    };
  };
  reader.readAsDataURL(file);
}

// ✅ 显示原始图像
function displayOriginalImage(imgElement) {
  const captureCanvas = document.getElementById("captureCanvas");
  const ctx = captureCanvas.getContext("2d");

  const scaleFactor = Math.min(
    captureCanvas.width / imgElement.width,
    captureCanvas.height / imgElement.height
  );
  const newWidth = imgElement.width * scaleFactor;
  const newHeight = imgElement.height * scaleFactor;

  ctx.clearRect(0, 0, captureCanvas.width, captureCanvas.height);
  ctx.drawImage(imgElement, 0, 0, newWidth, newHeight);

  console.log("✅ 图像加载完成！");
  document.getElementById("processBtn").disabled = false; // ✅ 启用按钮
}

// ✅ 图像处理功能：灰度 + 去噪 + 二值化 + 轮廓检测
document.getElementById("processBtn").addEventListener("click", processImage);

function processImage() {
  if (!isOpenCVLoaded) {
    alert("⚠️ OpenCV.js 尚未加载完成，请稍后重试！");
    return;
  }

  const captureCanvas = document.getElementById("captureCanvas");
  const processCanvas = document.getElementById("processCanvas");

  if (captureCanvas.width === 0 || captureCanvas.height === 0) {
    alert("⚠️ 无有效图像，请先拍摄图像！");
    return;
  }

  try {
    // ✅ 读取 Canvas 数据到 OpenCV Mat
    let src = cv.imread(captureCanvas);
    let gray = new cv.Mat();
    let blurred = new cv.Mat();
    let binary = new cv.Mat();
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();

    // ✅ 转换为灰度图像
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // ✅ 去除小噪声（使用中值滤波）
    cv.medianBlur(gray, blurred, 5);

    // ✅ 高斯模糊 + 二值化
    cv.GaussianBlur(blurred, binary, new cv.Size(5, 5), 0);
    cv.threshold(binary, binary, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

    // ✅ 查找外部轮廓
    cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    // ✅ 创建一个空白的 Mat 作为掩码
    let mask = cv.Mat.zeros(binary.rows, binary.cols, cv.CV_8UC1);

    // ✅ 只绘制有效轮廓
    for (let i = 0; i < contours.size(); i++) {
      const rect = cv.boundingRect(contours.get(i));

      // ✅ 忽略 Canvas 边界上的误识别区域
      if (
        rect.width > 10 &&
        rect.height > 10 &&
        rect.x > 5 &&
        rect.y > 5 &&
        rect.x + rect.width < binary.cols - 5 &&
        rect.y + rect.height < binary.rows - 5
      ) {
        cv.drawContours(mask, contours, i, new cv.Scalar(255, 255, 255, 255), -1);
      }
    }

    // ✅ 只保留有效区域的轮廓
    cv.bitwise_and(binary, mask, binary);

    // ✅ 计算叶片面积并绘制轮廓
    let color = new cv.Scalar(0, 255, 0, 255);
    let leafAreas = calculateLeafAreas(contours);

    for (let i = 0; i < contours.size(); i++) {
      cv.drawContours(src, contours, i, color, 2);
    }

    // ✅ 显示处理后的图像
    cv.imshow(processCanvas, src);
    updateLeafData(leafAreas);

    // ✅ 释放内存
    src.delete();
    gray.delete();
    blurred.delete();
    binary.delete();
    contours.delete();
    hierarchy.delete();
    mask.delete();

    console.log("✅ 图像处理完成！");
  } catch (error) {
    console.error("❌ 图像处理失败:", error);
    alert("❌ 图像处理出错，请检查图像是否有效！");
  }
}

// ✅ 计算叶片面积
function calculateLeafAreas(contours) {
  let leafAreas = [];
  for (let i = 0; i < contours.size(); i++) {
    const area = cv.contourArea(contours.get(i)); // 计算轮廓面积
    if (area > 100) {
      // 过滤小噪声
      leafAreas.push({ leafID: i + 1, area: Math.round(area) });
    }
  }
  return leafAreas;
}

// ✅ 更新叶片数据
function updateLeafData(leafAreas) {
  const imageNumber = document.getElementById("imageNumber").value || currentImageID;

  leafAreas.forEach((leaf) => {
    leafData.push({
      imageID: imageNumber,
      leafID: leaf.leafID,
      area: leaf.area,
    });
  });

  document.getElementById("downloadBtn").disabled = false;
  currentImageID++;

  // ✅ 更新叶片信息
  let leafInfo = document.getElementById("leafInfo");
  let leafHTML = `<h3>🌿 Leaf Info</h3><p>Detected ${leafAreas.length} leaves.</p>`;
  leafAreas.forEach((leaf) => {
    leafHTML += `<p>Leaf ${leaf.leafID}: ${leaf.area} pixels²</p>`;
  });
  leafInfo.innerHTML = leafHTML;
}

// ✅ 下载数据为 CSV
document.getElementById("downloadBtn").addEventListener("click", downloadCSV);

function downloadCSV() {
  if (leafData.length === 0) {
    alert("⚠️ No leaf data available. Please process an image first.");
    return;
  }

  let csvContent = "data:text/csv;charset=utf-8,";
  csvContent += "ImageID,LeafID,Area (pixels²)\n";

  leafData.forEach((leaf) => {
    csvContent += `${leaf.imageID},${leaf.leafID},${leaf.area}\n`;
  });

  // ✅ 创建下载链接
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "leaf_data.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  console.log("✅ CSV file generated successfully!");
}

// ✅ 重置数据
function resetData() {
  leafData = []; // 清空数据
  document.getElementById("captureCanvas").getContext("2d").clearRect(0, 0, 400, 300);
  document.getElementById("processCanvas").getContext("2d").clearRect(0, 0, 400, 300);
  document.getElementById("leafInfo").innerHTML =
    "<h3>🌿 Leaf Info</h3><p>No leaf data available yet.</p>";

  document.getElementById("processBtn").disabled = true;
  document.getElementById("downloadBtn").disabled = true;
  console.log("✅ Data has been reset!");
}

// ✅ 启动时初始化
console.log("✅ image_app.js loaded successfully!");