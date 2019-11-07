#pragma once

#include <QJsonObject>

#include <NvInfer.h>

namespace gestureserver
{
namespace nn
{
class TensorRtExecution
{
public:
    explicit TensorRtExecution(QJsonObject const& settings);
    ~TensorRtExecution();

    int tsnInputSize() const;
    int tsnOutputSize() const;
    int mlpInputSize() const;
    int mlpOutputSize() const;

    void createContextes(nvinfer1::ICudaEngine* tsn, nvinfer1::ICudaEngine* mlp);
    void executeTsn(float const* input, float* output);
    void executeMlp(float const* input, float* output);

private:
    int getSize(nvinfer1::Dims const& dims);
    void applySoftmax(float* dst, int len) const;

private:
    int m_tsnBatches = 0;
    int m_mlpBatches = 0;

    nvinfer1::IExecutionContext* m_tsnContext = nullptr;
    nvinfer1::IExecutionContext* m_mlpContext = nullptr;

    int m_tsnInputSize = 0;
    int m_tsnOutputSize = 0;
    int m_mlpInputSize = 0;
    int m_mlpOutputSize = 0;

    static constexpr size_t COUNT_BINDINGS = 2;
    void* m_tsnDeviceMemory[COUNT_BINDINGS] = {nullptr, nullptr};
    void* m_mlpDeviceMemory[COUNT_BINDINGS] = {nullptr, nullptr};
    size_t m_tsnDeviceMemorySizes[COUNT_BINDINGS] = {0};
    size_t m_mlpDeviceMemorySizes[COUNT_BINDINGS] = {0};
};
}
}
