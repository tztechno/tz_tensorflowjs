class LicensePlateDetector {
    constructor() {
        this.model = null;
        this.videoStream = null;
        this.modelSize = 640;
        this.confTh = 0.25;
        this.iouTh = 0.45;

        this.canvas = document.getElementById("inputCanvas");
        this.ctx = this.canvas.getContext("2d");
        this.video = document.getElementById("cameraFeed");

        this.imageInput = document.getElementById("imageInput");
        this.cameraBtn = document.getElementById("cameraBtn");
        this.stopCameraBtn = document.getElementById("stopCameraBtn");
        this.detectBtn = document.getElementById("detectBtn");
        this.status = document.getElementById("status");

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
        this.model = await ort.InferenceSession.create(
            "https://huggingface.co/datasets/stpete2/onnx_model/resolve/main/license_plate_yolo.onnx",
            { executionProviders: ["wasm"] }
        );
        this.updateStatus("Model ready.");
    }

    loadImage(file) {
        const img = new Image();
        img.onload = () => {
            this.canvas.width = img.width;
            this.canvas.height = img.height;
            this.ctx.drawImage(img, 0, 0);
        };
        img.src = URL.createObjectURL(file);
    }

    async startCamera() {
        this.stopCamera();
        this.videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
        this.video.srcObject = this.videoStream;
        await this.video.play();
    }

    stopCamera() {
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(t => t.stop());
            this.videoStream = null;
        }
    }

    async detect() {
        if (!this.model) return;

        if (this.videoStream) {
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.ctx.drawImage(this.video, 0, 0);
        }

        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const input = this.preprocess(imageData);

        const output = await this.model.run({ images: input });
        const dets = this.postprocess(output[Object.keys(output)[0]]);
        this.draw(dets);
    }

    preprocess(img) {
        const temp = document.createElement("canvas");
        temp.width = this.modelSize;
        temp.height = this.modelSize;

        const tctx = temp.getContext("2d");
        tctx.drawImage(this.canvas, 0, 0, this.modelSize, this.modelSize);

        const { data } = tctx.getImageData(0, 0, this.modelSize, this.modelSize);

        const arr = new Float32Array(this.modelSize * this.modelSize * 3);
        for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
            arr[j] = data[i] / 255;
            arr[j + 1] = data[i + 1] / 255;
            arr[j + 2] = data[i + 2] / 255;
        }
        return new ort.Tensor("float32", arr, [1, 3, this.modelSize, this.modelSize]);
    }

    postprocess(tensor) {
        const d = tensor.data;
        const w = this.canvas.width;
        const h = this.canvas.height;
        const stride = 6; // [x,y,w,h,conf,class]

        const dets = [];

        for (let i = 0; i < d.length; i += stride) {
            const conf = d[i + 4];
            if (conf < this.confTh) continue;

            const bx = d[i] * w - (d[i + 2] * w) / 2;
            const by = d[i + 1] * h - (d[i + 3] * h) / 2;

            dets.push({
                x: bx,
                y: by,
                width: d[i + 2] * w,
                height: d[i + 3] * h,
                confidence: conf
            });
        }

        return this.nms(dets);
    }

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

    iou(a, b) {
        const x1 = Math.max(a.x, b.x);
        const y1 = Math.max(a.y, b.y);
        const x2 = Math.min(a.x + a.width, b.x + b.width);
        const y2 = Math.min(a.y + a.height, b.y + b.height);

        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const ua = a.width * a.height + b.width * b.height - inter;
        return inter / ua;
    }

    draw(dets) {
        this.ctx.lineWidth = 3;
        this.ctx.strokeStyle = "#00FF00";
        this.ctx.font = "16px Arial";

        dets.forEach(d => {
            this.ctx.strokeRect(d.x, d.y, d.width, d.height);
            const label = `${(d.confidence * 100).toFixed(1)}%`;
            this.ctx.fillText(label, d.x, d.y - 5);
        });
    }
}
