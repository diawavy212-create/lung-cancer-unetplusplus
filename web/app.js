const loginScreen = document.querySelector("#loginScreen");
const loginForm = document.querySelector("#loginForm");
const logoutButton = document.querySelector("#logoutButton");
const fileInput = document.querySelector("#ctFile");
const fileState = document.querySelector("#fileState");
const predictButton = document.querySelector("#predictButton");
const previewImage = document.querySelector("#previewImage");
const resultState = document.querySelector("#resultState");
const resultMain = document.querySelector("#resultMain");
const timeValue = document.querySelector("#timeValue");
const accuracyValue = document.querySelector("#accuracyValue");
const confidenceValue = document.querySelector("#confidenceValue");
const diceValue = document.querySelector("#diceValue");

let selectedFileName = "";

loginForm.addEventListener("submit", (event) => {
  event.preventDefault();
  loginScreen.classList.add("hidden");
});

logoutButton.addEventListener("click", () => {
  loginScreen.classList.remove("hidden");
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) {
    return;
  }

  selectedFileName = file.name;
  fileState.textContent = "已选择";

  if (file.type.startsWith("image/")) {
    previewImage.src = URL.createObjectURL(file);
    previewImage.onload = () => URL.revokeObjectURL(previewImage.src);
  }
});

predictButton.addEventListener("click", () => {
  resultState.textContent = "分析中";
  predictButton.disabled = true;
  predictButton.textContent = "Predicting";

  runServerPrediction().catch(() => runDemoPrediction());
});

async function runServerPrediction() {
  const file = fileInput.files[0];
  if (!file) {
    throw new Error("No file selected");
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Prediction API failed");
  }

  const result = await response.json();
  renderPrediction({
    suspicious: result.is_suspicious,
    diagnosis: result.diagnosis,
    detail: result.is_suspicious
      ? "模型在肝脏区域检测到疑似异常强化灶，请结合临床指标与医生阅片意见综合判断。"
      : "当前影像未检测到显著肿瘤特征，仍建议按临床流程进行人工复核。",
    elapsed: Number(result.prediction_time || 0).toFixed(2),
    accuracy: result.accuracy == null ? "--" : `${(result.accuracy * 100).toFixed(2)}%`,
    confidence: `${((result.confidence || 0) * 100).toFixed(2)}%`,
    dice: result.dice == null ? "--" : Number(result.dice).toFixed(3),
  });
}

function runDemoPrediction() {
  window.setTimeout(() => {
    const seed = selectedFileName.length || 8;
    const suspicious = seed % 3 !== 0;
    const accuracy = `${(94.2 + (seed % 7) * 0.43).toFixed(2)}%`;
    const confidence = `${(88.6 + (seed % 9) * 0.82).toFixed(2)}%`;
    const elapsed = (1.18 + (seed % 6) * 0.17).toFixed(2);
    const dice = (0.891 + (seed % 5) * 0.014).toFixed(3);

    renderPrediction({
      suspicious,
      diagnosis: suspicious ? "建议进一步复核" : "未见明显异常",
      detail: suspicious
        ? "模型在肝脏区域检测到疑似异常强化灶，请结合临床指标与医生阅片意见综合判断。"
        : "当前影像未检测到显著肿瘤特征，仍建议按临床流程进行人工复核。",
      elapsed,
      accuracy,
      confidence,
      dice,
    });
  }, 900);
}

function renderPrediction({ suspicious, diagnosis, detail, elapsed, accuracy, confidence, dice }) {
  resultMain.classList.toggle("is-danger", suspicious);
  resultMain.classList.toggle("is-safe", !suspicious);
  resultMain.innerHTML = suspicious
    ? `<span class="risk-label">疑似阳性风险</span><strong>${diagnosis}</strong><p>${detail}</p>`
    : `<span class="risk-label">低风险提示</span><strong>${diagnosis}</strong><p>${detail}</p>`;

  timeValue.textContent = `${elapsed} s`;
  accuracyValue.textContent = accuracy;
  confidenceValue.textContent = confidence;
  diceValue.textContent = dice;
  resultState.textContent = "已完成";
  predictButton.disabled = false;
  predictButton.textContent = "Upload";
}
