rm -rf build
mkdir build && cd build
cmake ..
make -j
cd ..

rm -rf built
mkdir built && cd built
cmake ..
make -j
cd ..


sh tool/build_trt_engine.sh

cd build && ./pointpillar ../lidar/ ../result/ --timer

docker run -it --rm   --gpus all   --network host   -v "$(pwd)":/workspace   -w /workspace   ghcr.io/buckeye-autodrive/autodrive_jetson_humble_tensorrt:latest   bash




## visualize
python plot.py --bin lidar_front_cropped\seq_3_frame_144.bin --txt results\seq_3_frame_144.bin.txt
pip install open3d numpy






## export pth to onnx

docker run --gpus all -it --shm-size=8g `
  -v ${PWD}:/workspace `
  pcdet-onnx



cd OpenPCDet

python -m pip install -e . --no-build-isolation
