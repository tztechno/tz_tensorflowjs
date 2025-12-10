# YOLO ONNX Web Demo


Place `best.onnx` in this repo (or set model URL in the app). Open via GitHub Pages.


## Deploy
1. Push to GitHub
2. Settings → Pages → Deploy from `main` branch `/ (root)`
3. Wait a minute and visit `https://<username>.github.io/<repo>/`


**If ONNX i



            // ONNX Runtimeセッションを作成
            this.model = await ort.InferenceSession.create(
                'https://huggingface.co/datasets/stpete2/onnx_model/resolve/main/license_plate_yolo.onnx',
                {
                    executionProviders: ['webgl', 'wasm'],
                    graphOptimizationLevel: 'all'
                }
            );

以下のように簡略してもOK

            // ONNX Runtimeセッションを作成
            this.model = await ort.InferenceSession.create(
                'https://huggingface.co/datasets/stpete2/onnx_model/resolve/main/license_plate_yolo.onnx',
                {
                    executionProviders: ['wasm'],
                }
            );
