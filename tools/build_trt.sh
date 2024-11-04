$TensorRT_DIR/bin/trtexec --onnx=./config/iassd_hvcsx2_4x8_80e_kitti_3cls\(export\).onnx \
 --fp16 \
 --plugins=plugins/lib/librd3d_trt_plugin.so \
 --saveEngine=./config/iassd.engine \
 --verbose \
 --dumpLayerInfo \
 --dumpProfile \
 --separateProfileRun \
 --profilingVerbosity=detailed > config/iassd.log 2>&1