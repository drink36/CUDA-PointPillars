## visualize
pip install open3d numpy
python plot.py --bin lidar_front_cropped\seq_3_frame_144.bin --txt results\seq_3_frame_144.bin.txt

## export pth to onnx

docker run --gpus all -it --shm-size=8g `
  -v ${PWD}:/workspace `
  pcdet-onnx

cd OpenPCDet

python -m pip install -e . --no-build-isolation (take some time to build it)

cd ..

python3 tool/export_onnx.py --ckpt ckpts/pointpillar_7728.pth --out_dir model

## jetson inference

docker run -it --rm   --gpus all   --network host   -v "$(pwd)":/workspace   -w /workspace   ghcr.io/buckeye-autodrive/autodrive_jetson_humble_tensorrt:latest   bash

sh tool/build_trt_engine.sh

rm -rf build
mkdir build && cd build
cmake ..
make -j
cd ..

cd build && ./pointpillar ../lidar/ ../result/ --timer




