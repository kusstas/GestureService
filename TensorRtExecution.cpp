#include "TensorRtExecution.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cassert>

#define CHECK_CUDA(x)               if ((x) != ::cudaSuccess) qFatal("Fatal check: \"" #x "\"")

namespace gestureserver
{
namespace nn
{
TensorRtExecution::TensorRtExecution(QJsonObject const& settings)
{
    m_tsnBatches = settings["tsnMaxBatches"].toInt();
    m_mlpBatches = settings["mlpMaxBatches"].toInt();

    assert(m_tsnBatches > 0);
    assert(m_mlpBatches > 0);
}

TensorRtExecution::~TensorRtExecution()
{
    if (m_tsnContext)
    {
        m_tsnContext->destroy();
    }

    if (m_mlpContext)
    {
        m_mlpContext->destroy();
    }

    for (auto ptr : m_tsnDeviceMemory)
    {
        if (ptr)
        {
            CHECK_CUDA(cudaFree(ptr));
        }
    }

    for (auto ptr : m_mlpDeviceMemory)
    {
        if (ptr)
        {
            CHECK_CUDA(cudaFree(ptr));
        }
    }
}

int TensorRtExecution::tsnInputSize() const
{
    return m_tsnInputSize;
}

int TensorRtExecution::tsnOutputSize() const
{
    return m_tsnOutputSize;
}

int TensorRtExecution::mlpInputSize() const
{
    return m_mlpInputSize;
}

int TensorRtExecution::mlpOutputSize() const
{
    return m_mlpOutputSize;
}

void TensorRtExecution::createContextes(nvinfer1::ICudaEngine* tsn, nvinfer1::ICudaEngine* mlp)
{
    assert(tsn && mlp);
    assert(!m_tsnContext && !m_mlpContext);

    m_tsnContext = tsn->createExecutionContext();
    assert(m_tsnContext);

    m_mlpContext = mlp->createExecutionContext();
    assert(m_mlpContext);

    m_tsnInputSize = getSize(tsn->getBindingDimensions(0)) * m_tsnBatches;
    m_tsnOutputSize = getSize(tsn->getBindingDimensions(1)) * m_tsnBatches;
    m_mlpInputSize = getSize(mlp->getBindingDimensions(0)) * m_mlpBatches;
    m_mlpOutputSize = getSize(mlp->getBindingDimensions(1)) * m_mlpBatches;

    assert(m_tsnInputSize > 0);
    assert(m_tsnOutputSize > 0);
    assert(m_mlpInputSize > 0);
    assert(m_mlpOutputSize > 0);

    m_tsnDeviceMemorySizes[0] = static_cast<size_t>(m_tsnInputSize) * sizeof(float);
    m_tsnDeviceMemorySizes[1] = static_cast<size_t>(m_tsnOutputSize) * sizeof(float);
    m_mlpDeviceMemorySizes[0] = static_cast<size_t>(m_mlpInputSize) * sizeof(float);
    m_mlpDeviceMemorySizes[1] = static_cast<size_t>(m_mlpOutputSize) * sizeof(float);

    for (size_t i = 0; i < COUNT_BINDINGS; i++)
    {
        CHECK_CUDA(cudaMalloc(&m_tsnDeviceMemory[i], m_tsnDeviceMemorySizes[i]));
    }

    for (size_t i = 0; i < COUNT_BINDINGS; i++)
    {
        CHECK_CUDA(cudaMalloc(&m_mlpDeviceMemory[i], m_mlpDeviceMemorySizes[i]));
    }
}

void TensorRtExecution::executeTsn(float const* input, float* output)
{
    assert(input && output);
    assert(m_tsnContext);

    CHECK_CUDA(cudaMemcpy(m_tsnDeviceMemory[0], input, m_tsnDeviceMemorySizes[0], cudaMemcpyHostToDevice));

    assert(m_tsnContext->execute(m_tsnBatches, m_tsnDeviceMemory));

    CHECK_CUDA(cudaMemcpy(output, m_tsnDeviceMemory[1], m_tsnDeviceMemorySizes[1], cudaMemcpyDeviceToHost));
}

void TensorRtExecution::executeMlp(float const* input, float* output)
{
    assert(input && output);
    assert(m_mlpContext);

    CHECK_CUDA(cudaMemcpy(m_mlpDeviceMemory[0], input, m_mlpDeviceMemorySizes[0], cudaMemcpyHostToDevice));

    assert(m_mlpContext->execute(m_mlpBatches, m_mlpDeviceMemory));

    CHECK_CUDA(cudaMemcpy(output, m_mlpDeviceMemory[1], m_mlpDeviceMemorySizes[1], cudaMemcpyDeviceToHost));

    applySoftmax(output, m_mlpDeviceMemorySizes[1] / sizeof (float));
}

int TensorRtExecution::getSize(nvinfer1::Dims const& dims)
{
    int size = dims.nbDims > 0 ? 1 : 0;

    for (int i = 0; i < dims.nbDims; i++)
    {
        size *= dims.d[i];
    }

    return size;
}

void TensorRtExecution::applySoftmax(float* dst, int len) const
{
    float sum = 0;
    for (size_t i = 0; i < len; i++)
    {
        dst[i] = std::exp(dst[i]);
        sum += dst[i];
    }

    for (size_t i = 0; i < len; i++)
    {
        dst[i] /= sum;
    }
}
}
}
