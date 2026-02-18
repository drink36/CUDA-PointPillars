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