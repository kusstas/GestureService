#pragma once

#include <QJsonObject>

namespace nvinfer1
{
class ICudaEngine;
class IBuilder;
}

namespace gestureserver
{
namespace nn
{
struct Engines
{
    nvinfer1::ICudaEngine* tsn;
    nvinfer1::ICudaEngine* mlp;
};

class TensorRtEngineBuilder
{
public:
    explicit TensorRtEngineBuilder(QJsonObject const& settings);

    Engines build() const;

private:
    nvinfer1::ICudaEngine* buildTsn(nvinfer1::IBuilder* builder) const;
    nvinfer1::ICudaEngine* buildMlp(nvinfer1::IBuilder* builder) const;

private:
    QString m_tsnOnnxPath{};
    QString m_mlpOnnxPath{};
    QString m_tsnSerializePath{};
    QString m_mlpSerializePath{};
    size_t m_tsnMaxWorkspace = 0;
    size_t m_mlpMaxWorkspace = 0;
    int m_tsnMaxBatches = 0;
    int m_mlpMaxBatches = 0;
};
}
}
