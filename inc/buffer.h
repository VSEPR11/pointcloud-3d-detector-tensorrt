//
// Created by nrsl on 23-4-5.
//

#ifndef POINT_DETECTION_BUFFER_H
#define POINT_DETECTION_BUFFER_H

#include <cassert>

#include <vector>
#include <memory>
#include <utility>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace PointDetection {

class BufferManager {
 public:
    explicit BufferManager(nvinfer1::IExecutionContext &context, int batch_size = 1) {
        auto &engine = context.getEngine();
        for (int i = 0; i < engine.getNbIOTensors(); ++i) {
            auto tensor_name = engine.getIOTensorName(i);
            bool is_output = engine.getTensorIOMode(tensor_name) != nvinfer1::TensorIOMode::kINPUT;
            auto type = engine.getTensorDataType(tensor_name);
            auto dims = context.getTensorShape(tensor_name);
            if (dims.d[0] == -1) {
                fprintf(stderr, "dims[0] == -1, please call context.setBindingDimensions\n");
                dims.d[0] = batch_size;
            }
            size_t size = SizeOf(type) * Volume(dims);
            if (!is_output)
            {

                void* deviceMem;
                cudaMalloc(&deviceMem, size);
                devicePtrs.push_back(deviceMem);
                context.setTensorAddress(tensor_name, devicePtrs.back());
                inputSizes.push_back(size);
            }
            else
            {
                void* hostMem;
                cudaHostAlloc(&hostMem, size, 0);
                hostPtrs.push_back(hostMem);
                context.setTensorAddress(tensor_name, hostPtrs.back());
                outputSizes.push_back(size);
                void* output;
                cudaMalloc(&output, size);
                outputs.push_back(output);
            }
        }
    }

    void ToDevice(const cudaStream_t &stream) {
        for (int i = 0; i < devicePtrs.size(); i++)
        {
            cudaMemcpyAsync(devicePtrs[i], inputs[i], inputSizes[i], cudaMemcpyHostToDevice, stream);
        }
    }

    void ToHost(const cudaStream_t &stream) {
        for (int i = 0; i < outputSizes.size(); i++)
        {
            cudaMemcpyAsync(outputs[i], hostPtrs[i], outputSizes[i], cudaMemcpyDeviceToHost, stream);
        }
    }

    template<class T, int N>
    inline const typeof(T[N]) *ReadOutput(int index) {
        return reinterpret_cast<typeof(T[N]) *>(outputs[index]);
    }

    inline void SetInputs(const std::vector<void *> &in_ptr) {
        inputs = in_ptr;
    }

    inline auto *IO() {
        return bindings.data();
    }

    ~BufferManager()
    {
        for (auto ptr: devicePtrs)
        {
            cudaFree(ptr);
        }
        for (auto ptr: hostPtrs)
        {
            cudaFree(ptr);
        }
        for (auto ptr: outputs)
        {
            cudaFree(ptr);
        }
    }

 private:
    static int SizeOf(const nvinfer1::DataType &type) {
        switch (type) {
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8:
            return 1;
        default:
            assert(false);
        }
    }

    static int Volume(const nvinfer1::Dims &dims) {
        int size = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            size *= dims.d[i];
        }
        return size;
    }

    std::vector<void*> devicePtrs;
    std::vector<void*> hostPtrs;
    std::vector<size_t> inputSizes;
    std::vector<size_t> outputSizes;
    std::vector<void *> bindings;

    std::vector<void*> inputs;
    std::vector<void*> outputs;
};

}  // namespace PointDetection

#endif //POINT_DETECTION_BUFFER_H
