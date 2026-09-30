.PHONY: data features train-a train-b train-c eval eval-all quantize report test smoke clean

data:            ## download UCF101 subset (CRCV official; fallback: hf)
	python -m src.data.download_ucf101 --data-dir data --source crcv

data-hf:         ## fallback: 10-class HF mirror (no RAR tooling needed)
	python -m src.data.download_ucf101 --data-dir data --source hf

features:        ## cache frozen ResNet-18 frame features (models A/B)
	python -m src.data.extract_features --data-dir data --out-dir data/features_resnet18

train-a:         ## model A: mean pool over frozen features
	python -m src.train --config configs/framepool.yaml

train-b:         ## model B: temporal transformer over frozen features
	python -m src.train --config configs/temporal_transformer.yaml

train-c:         ## model C: r3d_18 fine-tune (GPU: ~20 min on T4; CPU: see r3d18_cpu.yaml)
	python -m src.train --config configs/r3d18.yaml

eval-all:        ## test-split metrics + plots + efficiency bench, all models
	python -m src.evaluate --config configs/framepool.yaml
	python -m src.evaluate --config configs/temporal_transformer.yaml
	python -m src.evaluate --config configs/r3d18.yaml

quantize:        ## ONNX export + INT8 static quantization + CPU benchmark
	python -m src.export_quantize --config configs/r3d18.yaml

report:          ## regenerate README tables/plots from run artifacts
	python -m src.make_report

test:            ## unit + e2e tests (synthetic data; no download needed)
	pytest -q

clean:
	rm -rf results/checkpoints results/logs results/quantized
