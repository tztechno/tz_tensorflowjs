class LicensePlateDetector {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.videoStream = null;
        this.isCameraActive = false;
        this.currentImage = null;
        this.originalImageData = null;
        
        // YOLO model parameters
        this.modelSize = 640;
        this.confidenceThreshold = 0.25;
        this.iouThreshold = 0.45;
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadModel();
    }
    
    initializeElements() {
        this.canvas = document.getElementById('inputCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.video = document.getElementById('cameraFeed');
        this.imageInput = document.getElementById('imageInput');
        this.cameraBtn = document.getElementById('cameraBtn');
        this.stopCameraBtn = document.getElementById('stopCameraBtn');
        this.detectBtn = document.getElementById('detectBtn');
        this.status = document.getElementById('status');
        this.results = document.getElementById('results');
        
        // キャンバスサイズを設定
        this.canvas.width = 800;
        this.canvas.height = 600;
    }
    
    async loadModel() {
        try {
            this.updateStatus('Loading YOLO model...');
            console.log('Loading model from:', 'https://huggingface.co/datasets/stpete2/onnx_model/resolve/main/license_plate_yolo.onnx');
            
            // ONNX Runtimeセッションを作成
            this.model = await ort.InferenceSession.create(
                'https://huggingface.co/datasets/stpete2/onnx_model/resolve/main/license_plate_yolo.onnx',
                {
                    executionProviders: ['wasm']
                }
            );
            
            this.isModelLoaded = true;
            console.log('Model loaded successfully:', this.model);
            console.log('Model input names:', this.model.inputNames);
            console.log('Model output names:', this.model.outputNames);
            
            this.updateStatus('Model loaded successfully. Ready to detect.');
            
        } catch (error) {
            console.error('Error loading model:', error);
            this.updateStatus('Error loading model: ' + error.message);
        }
    }
    
    setupEventListeners() {
        this.imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        this.cameraBtn.addEventListener('click', () => this.startCamera());
        this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
        this.detectBtn.addEventListener('click', () => this.detect());
    }
    
    updateStatus(message) {
        this.status.textContent = message;
    }
    
    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            const img = new Image();
            const url = URL.createObjectURL(file);
            
            img.onload = () => {
                this.currentImage = img;
                this.originalImageData = null; // リセット
                this.drawImageToCanvas(img);
                this.updateStatus('Image loaded. Click "Detect" to find license plates.');
                URL.revokeObjectURL(url);
                
                // キャンバスをクリアして新しい画像を表示
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
            };
            
            img.onerror = (error) => {
                console.error('Image loading error:', error);
                this.updateStatus('Error loading image');
                URL.revokeObjectURL(url);
            };
            
            img.src = url;
        } catch (error) {
            console.error('Error loading image:', error);
            this.updateStatus('Error loading image: ' + error.message);
        }
    }
    
    async startCamera() {
        try {
            this.stopCamera(); // 既存のカメラを停止
            
            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };
            
            this.videoStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.videoStream;
            this.video.style.display = 'block';
            this.isCameraActive = true;
            
            this.video.onloadedmetadata = () => {
                this.video.play();
                this.updateStatus('Camera active. Click "Detect" to find license plates.');
            };
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.updateStatus('Error accessing camera: ' + error.message);
        }
    }
    
    stopCamera() {
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(track => track.stop());
            this.videoStream = null;
            this.video.style.display = 'none';
            this.isCameraActive = false;
            this.updateStatus('Camera stopped.');
        }
    }
    
    drawImageToCanvas(img) {
        // キャンバスのサイズを画像のアスペクト比に合わせて調整
        const maxWidth = 800;
        const maxHeight = 600;
        let width = img.width;
        let height = img.height;
        
        if (width > maxWidth) {
            height = (maxWidth / width) * height;
            width = maxWidth;
        }
        
        if (height > maxHeight) {
            width = (maxHeight / height) * width;
            height = maxHeight;
        }
        
        this.canvas.width = width;
        this.canvas.height = height;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(img, 0, 0, width, height);
        
        // 元の画像データを保存
        this.originalImageData = this.ctx.getImageData(0, 0, width, height);
    }
    
    async detect() {
        if (!this.isModelLoaded) {
            this.updateStatus('Model not loaded yet. Please wait.');
            return;
        }
        
        try {
            this.updateStatus('Detecting license plates...');
            this.results.innerHTML = '<div class="result-item">Processing...</div>';
            
            let imageData;
            
            if (this.isCameraActive && this.video.readyState === this.video.HAVE_ENOUGH_DATA) {
                // カメラからのキャプチャ
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
                imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
                this.originalImageData = imageData;
            } else if (this.currentImage || this.originalImageData) {
                // アップロードされた画像を使用
                if (this.originalImageData) {
                    imageData = this.originalImageData;
                } else {
                    // キャンバスから画像データを取得
                    imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
                    this.originalImageData = imageData;
                }
            } else {
                this.updateStatus('No image source available. Please upload an image or start camera.');
                return;
            }
            
            console.log('Image dimensions:', this.canvas.width, this.canvas.height);
            
            // 画像を前処理
            const inputTensor = this.preprocessImage(imageData);
            console.log('Input tensor shape:', inputTensor.dims);
            
            // 推論を実行
            const startTime = performance.now();
            const outputs = await this.model.run({ [this.model.inputNames[0]]: inputTensor });
            const endTime = performance.now();
            
            console.log('Inference time:', endTime - startTime, 'ms');
            console.log('Model outputs:', outputs);
            
            // 結果を処理
            const detections = this.processYOLOOutput(outputs);
            console.log('Detections:', detections);
            
            // バウンディングボックスを描画
            this.drawDetections(detections);
            
            // 結果を表示
            this.displayResults(detections);
            
            this.updateStatus(`Detection complete. Found ${detections.length} license plate(s).`);
            
        } catch (error) {
            console.error('Error during detection:', error);
            this.updateStatus('Error during detection: ' + error.message);
        }
    }
    
    preprocessImage(imageData) {
        const { width, height } = this.canvas;
        
        // リサイズ用の一時キャンバスを作成
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.modelSize;
        tempCanvas.height = this.modelSize;
        const tempCtx = tempCanvas.getContext('2d');
        
        // 画像を描画してリサイズ
        const img = new Image();
        img.src = this.canvas.toDataURL();
        
        // 白い背景で描画（余白を埋める）
        tempCtx.fillStyle = '#ffffff';
        tempCtx.fillRect(0, 0, this.modelSize, this.modelSize);
        
        // アスペクト比を維持してリサイズ
        const scale = Math.min(this.modelSize / width, this.modelSize / height);
        const newWidth = width * scale;
        const newHeight = height * scale;
        const xOffset = (this.modelSize - newWidth) / 2;
        const yOffset = (this.modelSize - newHeight) / 2;
        
        tempCtx.drawImage(
            this.canvas, 
            0, 0, width, height,
            xOffset, yOffset, newWidth, newHeight
        );
        
        // 画像データを取得
        const resizedData = tempCtx.getImageData(0, 0, this.modelSize, this.modelSize);
        
        // Float32Arrayに変換（正規化: 0-255 -> 0-1）
        const float32Data = new Float32Array(this.modelSize * this.modelSize * 3);
        
        for (let i = 0; i < resizedData.data.length; i += 4) {
            const pixelIndex = Math.floor(i / 4);
            float32Data[pixelIndex * 3] = resizedData.data[i] / 255.0;         // R
            float32Data[pixelIndex * 3 + 1] = resizedData.data[i + 1] / 255.0; // G
            float32Data[pixelIndex * 3 + 2] = resizedData.data[i + 2] / 255.0; // B
        }
        
        // テンソルを作成 (NCHW形式: [1, 3, height, width])
        const tensor = new ort.Tensor(
            'float32', 
            float32Data, 
            [1, 3, this.modelSize, this.modelSize]
        );
        
        return tensor;
    }
    
    processYOLOOutput(outputs) {
        const detections = [];
        
        // モデルの出力構造を確認
        console.log('Output keys:', Object.keys(outputs));
        
        // 一般的なYOLO出力形式に対応
        for (const [outputName, outputTensor] of Object.entries(outputs)) {
            console.log(`Output ${outputName}:`, outputTensor.dims, outputTensor.data.length);
            
            const data = outputTensor.data;
            const dims = outputTensor.dims;
            
            // 様々なYOLO出力形式を処理
            if (dims.length === 3) {
                // 形式: [1, 84, 8400] または類似
                this.processDetectionOutput(data, dims, detections);
            } else if (dims.length === 2) {
                // 形式: [1, 8400, 84]
                this.processDetectionOutput(data, dims, detections);
            } else if (dims.length === 4) {
                // 形式: [1, 84, 80, 80] (特徴マップ)
                this.processGridOutput(data, dims, detections);
            }
        }
        
        console.log('Raw detections before NMS:', detections.length);
        
        // Non-Maximum Suppressionを適用
        const filteredDetections = this.nonMaximumSuppression(detections);
        console.log('Filtered detections after NMS:', filteredDetections.length);
        
        return filteredDetections;
    }
    
    processDetectionOutput(data, dims, detections) {
        const numDetections = dims[1] || dims[2];
        const numClasses = 1; // ナンバープレートのみ
        
        console.log('Processing detection output:', dims, 'numDetections:', numDetections);
        
        for (let i = 0; i < numDetections; i++) {
            let offset;
            
            if (dims.length === 3 && dims[0] === 1) {
                // [1, 84, 8400] 形式
                offset = i * dims[1];
            } else if (dims.length === 2) {
                // [1, 8400, 84] 形式
                offset = i * (4 + 1 + numClasses);
            } else {
                continue;
            }
            
            // バウンディングボックス情報を抽出
            const x = data[offset] * this.canvas.width;
            const y = data[offset + 1] * this.canvas.height;
            const width = data[offset + 2] * this.canvas.width;
            const height = data[offset + 3] * this.canvas.height;
            const confidence = data[offset + 4];
            
            if (confidence > this.confidenceThreshold) {
                detections.push({
                    x: x - width / 2,
                    y: y - height / 2,
                    width,
                    height,
                    confidence,
                    class: 'license_plate'
                });
            }
        }
    }
    
    processGridOutput(data, dims, detections) {
        // グリッドベースの出力を処理（より高度なYOLOバージョン用）
        const gridSize = dims[2];
        const cellSize = this.modelSize / gridSize;
        
        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                const offset = (i * gridSize + j) * dims[1];
                const confidence = data[offset + 4];
                
                if (confidence > this.confidenceThreshold) {
                    const x = (j + data[offset]) * cellSize;
                    const y = (i + data[offset + 1]) * cellSize;
                    const width = data[offset + 2] * this.modelSize;
                    const height = data[offset + 3] * this.modelSize;
                    
                    // オリジナル画像サイズにスケーリング
                    const scaleX = this.canvas.width / this.modelSize;
                    const scaleY = this.canvas.height / this.modelSize;
                    
                    detections.push({
                        x: (x - width / 2) * scaleX,
                        y: (y - height / 2) * scaleY,
                        width: width * scaleX,
                        height: height * scaleY,
                        confidence,
                        class: 'license_plate'
                    });
                }
            }
        }
    }
    
    nonMaximumSuppression(detections) {
        if (detections.length === 0) return [];
        
        // 信頼度でソート
        detections.sort((a, b) => b.confidence - a.confidence);
        
        const suppressed = [];
        
        while (detections.length > 0) {
            const current = detections[0];
            suppressed.push(current);
            
            // 現在の検出をリストから削除
            detections.splice(0, 1);
            
            // 残りの検出をフィルタリング
            for (let i = detections.length - 1; i >= 0; i--) {
                const iou = this.calculateIoU(current, detections[i]);
                if (iou > this.iouThreshold) {
                    detections.splice(i, 1);
                }
            }
        }
        
        return suppressed;
    }
    
    calculateIoU(box1, box2) {
        const x1 = Math.max(box1.x, box2.x);
        const y1 = Math.max(box1.y, box2.y);
        const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
        const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
        
        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const area1 = box1.width * box1.height;
        const area2 = box2.width * box2.height;
        const union = area1 + area2 - intersection;
        
        return intersection / union;
    }
    
    drawDetections(detections) {
        // オリジナル画像を復元
        if (this.originalImageData) {
            this.ctx.putImageData(this.originalImageData, 0, 0);
        } else if (this.currentImage) {
            this.ctx.drawImage(this.currentImage, 0, 0, this.canvas.width, this.canvas.height);
        }
        
        // バウンディングボックスを描画
        detections.forEach((det, index) => {
            const { x, y, width, height, confidence } = det;
            
            // バウンディングボックス
            this.ctx.strokeStyle = '#00ff00';
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x, y, width, height);
            
            // ラベル背景
            this.ctx.fillStyle = '#00ff00';
            const label = `Plate ${(confidence * 100).toFixed(1)}%`;
            const textMetrics = this.ctx.measureText(label);
            const textWidth = textMetrics.width;
            
            // ラベルボックス
            this.ctx.fillRect(x, y - 30, textWidth + 20, 30);
            
            // ラベルテキスト
            this.ctx.fillStyle = '#000';
            this.ctx.font = 'bold 16px Arial';
            this.ctx.fillText(label, x + 10, y - 10);
            
            // インデックス番号
            this.ctx.fillStyle = '#ff0000';
            this.ctx.beginPath();
            this.ctx.arc(x + width - 15, y + 15, 15, 0, Math.PI * 2);
            this.ctx.fill();
            
            this.ctx.fillStyle = '#fff';
            this.ctx.font = 'bold 14px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText((index + 1).toString(), x + width - 15, y + 15);
        });
    }
    
    displayResults(detections) {
        this.results.innerHTML = '';
        
        if (detections.length === 0) {
            this.results.innerHTML = '<div class="result-item">No license plates detected</div>';
            return;
        }
        
        detections.forEach((det, index) => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result-item';
            resultDiv.innerHTML = `
                <h4>License Plate #${index + 1}</h4>
                <p><strong>Confidence:</strong> ${(det.confidence * 100).toFixed(2)}%</p>
                <p><strong>Position:</strong> X=${Math.round(det.x)}, Y=${Math.round(det.y)}</p>
                <p><strong>Size:</strong> ${Math.round(det.width)} × ${Math.round(det.height)} pixels</p>
                <p><strong>Area:</strong> ${Math.round(det.width * det.height)} pixels²</p>
            `;
            this.results.appendChild(resultDiv);
        });
    }
    
    // デバッグ用: モデル情報を表示
    async debugModel() {
        if (!this.model) {
            console.log('Model not loaded');
            return;
        }
        
        console.log('=== Model Debug Info ===');
        console.log('Input Names:', this.model.inputNames);
        console.log('Output Names:', this.model.outputNames);
        
        // ダミー入力でテスト
        const dummyInput = new ort.Tensor(
            'float32',
            new Float32Array(this.modelSize * this.modelSize * 3).fill(0.5),
            [1, 3, this.modelSize, this.modelSize]
        );
        
        try {
            const testOutput = await this.model.run({ [this.model.inputNames[0]]: dummyInput });
            console.log('Test output:', testOutput);
        } catch (error) {
            console.error('Test inference error:', error);
        }
    }
}

// アプリケーションを初期化
window.addEventListener('DOMContentLoaded', () => {
    const detector = new LicensePlateDetector();
    
    // グローバルに公開（デバッグ用）
    window.detector = detector;
    
    console.log('License Plate Detector initialized');
});
