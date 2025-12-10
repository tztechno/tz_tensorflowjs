// =========================================================================
// LICENSE PLATE DETECTOR CLASS (YOLOv8 Letterbox Preprocessing対応版)
// =========================================================================
class LicensePlateDetector {
    constructor() {
        this.model = null;
        this.videoStream = null;
        this.modelSize = 640;
        this.confTh = 0.25; // 推論結果の信頼度閾値
        this.iouTh = 0.45;  // Non-Maximum Suppression (NMS)の閾値

        this.canvas = document.getElementById("inputCanvas");
        this.ctx = this.canvas.getContext("2d");
        this.video = document.getElementById("cameraFeed");

        this.imageInput = document.getElementById("imageInput");
        this.cameraBtn = document.getElementById("cameraBtn");
        this.stopCameraBtn = document.getElementById("stopCameraBtn");
        this.detectBtn = document.getElementById("detectBtn");
        this.status = document.getElementById("status");
        this.resultsDiv = document.getElementById("results");

        this.bindEvents();
        this.loadModel();
    }

    bindEvents() {
        this.imageInput.onchange = e => this.loadImage(e.target.files[0]);
        this.cameraBtn.onclick = () => this.startCamera();
        this.stopCameraBtn.onclick = () => this.stopCamera();
        this.detectBtn.onclick = () => this.detect();
    }

    updateStatus(msg) {
        this.status.textContent = msg;
    }

    async loadModel() {
        this.updateStatus("Loading model...");
        try {
            // !!! ここに変換した best.onnx への正しいパスを設定 !!!
            this.model = await ort.InferenceSession.create(
                "best.onnx", // 例: 変換したモデル名に置き換える
                { executionProviders: ["wasm"] }
            );
            this.updateStatus("Model loaded. Ready.");
        } catch (e) {
            this.updateStatus(`Error loading model: ${e.message}`);
            console.error("Model Load Error:", e);
        }
    }

    loadImage(file) {
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.video.style.display = 'none';
                this.drawInput(img);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    startCamera() {
        this.stopCamera();
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
            .then(stream => {
                this.videoStream = stream;
                this.video.srcObject = stream;
                this.video.style.display = 'block';
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    this.drawInput(this.video);
                };
                this.updateStatus("Camera started.");
            })
            .catch(err => {
                this.updateStatus(`Could not start camera: ${err.message}`);
                console.error("Camera error:", err);
            });
    }

    stopCamera() {
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(track => track.stop());
            this.videoStream = null;
            this.video.style.display = 'none';
            this.updateStatus("Camera stopped. Ready to detect.");
        }
    }

    drawInput(source) {
        this.canvas.width = source.videoWidth || source.width;
        this.canvas.height = source.videoHeight || source.height;
        this.ctx.drawImage(source, 0, 0, this.canvas.width, this.canvas.height);
    }

    async detect() {
        if (!this.model) {
            this.updateStatus("Model not loaded yet.");
            return;
        }

        this.updateStatus("Detecting...");
        this.resultsDiv.innerHTML = '';
        
        // 1. Preprocessing (Letterbox + Normalization)
        const { tensor, xRatio, yRatio, xPad, yPad, scale } = this.preprocess();
        
        // 2. Inference
        const feeds = { 'images': tensor }; 
        const outputMap = await this.model.run(feeds);
        const output = outputMap[this.model.outputNames[0]].data; 

        // 3. Postprocessing (YOLOv8 format)
        const detections = this.postprocess(output, xRatio, yRatio, xPad, yPad, scale);
        
        // 4. Draw Results
        this.drawResults(detections);
        this.updateStatus(`Detection finished. Found ${detections.length} plates.`);
    }

    /**
     * YOLOv8 Letterbox Preprocessing (アスペクト比保持) を正確に再現
     * @returns {Object} 処理されたテンソルと後処理に必要な比率
     */
    preprocess() {
        // 元の画像サイズ (canvasから取得)
        const originalWidth = this.canvas.width;
        const originalHeight = this.canvas.height;
        
        // ターゲットサイズ
        const targetWidth = this.modelSize;
        const targetHeight = this.modelSize;

        // Letterbox比率の計算
        const widthRatio = targetWidth / originalWidth;
        const heightRatio = targetHeight / originalHeight;
        const scale = Math.min(widthRatio, heightRatio); // 小さい方の比率を採用

        const resizedWidth = Math.round(originalWidth * scale);
        const resizedHeight = Math.round(originalHeight * scale);

        const xPad = Math.floor((targetWidth - resizedWidth) / 2); // パディング量
        const yPad = Math.floor((targetHeight - resizedHeight) / 2);

        // 新しいキャンバスを用意し、Letterbox処理を再現
        const letterboxCanvas = document.createElement('canvas');
        letterboxCanvas.width = targetWidth;
        letterboxCanvas.height = targetHeight;
        const lb_ctx = letterboxCanvas.getContext('2d');
        lb_ctx.fillStyle = "#808080"; // 灰色パディング
        lb_ctx.fillRect(0, 0, targetWidth, targetHeight);
        
        // 元の画像をアスペクト比を保ちつつ描画
        lb_ctx.drawImage(this.canvas, xPad, yPad, resizedWidth, resizedHeight);
        
        // テンソル化と正規化 ([0, 255] -> [0, 1] for ONNX)
        const imageData = lb_ctx.getImageData(0, 0, targetWidth, targetHeight).data;
        const floatData = new Float32Array(3 * targetWidth * targetHeight);
        let k = 0;
        for (let i = 0; i < imageData.length; i += 4) {
            // R, G, B チャンネル順に格納し、[0, 1]に正規化
            floatData[k++] = imageData[i] / 255.0;     // R
            floatData[k++] = imageData[i + 1] / 255.0; // G
            floatData[k++] = imageData[i + 2] / 255.0; // B
        }

        const tensor = new ort.Tensor('float32', floatData, [1, 3, targetHeight, targetWidth]);

        return { 
            tensor: tensor, 
            xRatio: originalWidth / resizedWidth, // 検出結果を元画像サイズに戻すための比率
            yRatio: originalHeight / resizedHeight, 
            xPad: xPad,
            yPad: yPad,
            scale: scale
        };
    }

    /**
     * YOLOv8の推論結果を処理し、NMSを適用してバウンディングボックスを抽出
     */
    postprocess(output, xRatio, yRatio, xPad, yPad, scale) {
        const dets = [];
        const numClasses = 1; // License Plateのみ
        const numAnchors = 8400; // YOLOv8のデフォルトアンカー数 (640x640)
        const stride = numClasses + 4; // Bbox (4) + Class (1) = 5
        const data = output;
        
        // YOLOv8の出力は [5 + num_classes, num_anchors] のフラット配列として出力されることが多い
        // (x, y, w, h, conf, class_scores...)

        for (let i = 0; i < numAnchors; i++) {
            // conf: 検出信頼度
            const conf = data[4 * numAnchors + i]; // YOLOv8のデフォルト出力形式を前提
            
            if (conf < this.confTh) continue;

            const boxIndex = i;
            
            // Bounding Box の座標 (正規化されたセンター座標 x, y, w, h)
            const x = data[boxIndex];
            const y = data[1 * numAnchors + boxIndex];
            const w = data[2 * numAnchors + boxIndex];
            const h = data[3 * numAnchors + boxIndex];
            
            // Letterboxとモデル入力サイズから元の画像サイズへ座標を復元 (rescale)
            
            // 1. パディングを考慮した変換 (モデルサイズ -> Letterbox描画サイズ)
            const x_center = (x - xPad) / scale;
            const y_center = (y - yPad) / scale;
            const box_width = w / scale;
            const box_height = h / scale;
            
            // 2. 正規化されたセンター座標 (x, y, w, h) からコーナー座標 (x1, y1, w, h) へ変換
            const x1 = x_center - (box_width / 2);
            const y1 = y_center - (box_height / 2);
            
            // 3. 実際のピクセル値に変換 (元の画像サイズ)
            const final_x1 = x1 * originalWidth;
            const final_y1 = y1 * originalHeight;
            const final_width = box_width * originalWidth;
            const final_height = box_height * originalHeight;
            
            dets.push({
                x: final_x1,
                y: final_y1,
                width: final_width,
                height: final_height,
                confidence: conf,
                label: 'License Plate' 
            });
        }

        // NMS (Non-Maximum Suppression) を適用
        return this.nms(dets);
    }
    
    /**
     * Non-Maximum Suppression (NMS) の実装
     */
    nms(dets) {
        dets.sort((a, b) => b.confidence - a.confidence);
        const res = [];

        while (dets.length) {
            const best = dets.shift();
            res.push(best);
            dets = dets.filter(d => this.iou(best, d) < this.iouTh);
        }
        return res;
    }

    /**
     * Intersection over Union (IoU) の計算
     */
    iou(a, b) {
        const x1 = Math.max(a.x, b.x);
        const y1 = Math.max(a.y, b.y);
        const x2 = Math.min(a.x + a.width, b.x + b.width);
        const y2 = Math.min(a.y + a.height, b.y + b.height);

        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const ua = a.width * a.height + b.width * b.height - inter;
        return inter / ua;
    }

    drawResults(detections) {
        this.ctx.strokeStyle = '#00FF00'; // 緑色
        this.ctx.lineWidth = 3;
        this.ctx.font = "18px Arial";
        this.ctx.fillStyle = '#00FF00';

        this.resultsDiv.innerHTML = '';

        detections.forEach((d, index) => {
            // バウンディングボックスの描画
            this.ctx.strokeRect(d.x, d.y, d.width, d.height);
            
            // ラベルの描画 (確信度付き)
            const labelText = `${d.label} (${(d.confidence * 100).toFixed(1)}%)`;
            this.ctx.fillText(labelText, d.x, d.y > 20 ? d.y - 5 : d.y + 20);

            // 結果パネルにテキストで出力
            this.resultsDiv.innerHTML += `
                <div class="result-item">
                    <h4>Detection #${index + 1}</h4>
                    <p>Confidence: ${(d.confidence * 100).toFixed(2)}%</p>
                    <p>Box: X=${d.x.toFixed(0)}, Y=${d.y.toFixed(0)}, W=${d.width.toFixed(0)}, H=${d.height.toFixed(0)}</p>
                </div>
            `;
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new LicensePlateDetector();
});
