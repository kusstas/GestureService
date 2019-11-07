#include "TensorRtEngineBuilder.h"

#include <NvInferRuntimeCommon.h>
#include <NvInferRuntime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <QDebug>
#include <QFile>

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* msg) override
    {
        static constexpr auto TAG = "[TENSOR_RT]";
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            qCritical() << TAG << msg;
            break;
        case Severity::kERROR:
            qCritical() << TAG << msg;
            break;
        case Severity::kWARNING:
            qWarning() << TAG << msg;
            break;
        case Severity::kINFO:
            qInfo() << TAG << msg;
            break;
        case Severity::kVERBOSE:
            qDebug() << TAG << msg;
            break;
        }
    }
};

static Logger gLogger;

namespace gestureserver
{
namespace nn
{
TensorRtEngineBuilder::TensorRtEngineBuilder(QJsonObject const& settings)
{
    m_tsnOnnxPath = settings["tsnOnnxPath"].toString();
    m_mlpOnnxPath = settings["mlpOnnxPath"].toString();
    m_tsnSerializePath = settings["tsnSerializePath"].toString();
    m_mlpSerializePath = settings["mlpSerializePath"].toString();
    m_tsnMaxWorkspace = static_cast<size_t>(settings["tsnMaxWorkspace"].toInt());
    m_mlpMaxWorkspace = static_cast<size_t>(settings["mlpMaxWorkspace"].toInt());
    m_tsnMaxBatches = settings["tsnMaxBatches"].toInt();
    m_mlpMaxBatches = settings["mlpMaxBatches"].toInt();

    assert(!m_tsnOnnxPath.isEmpty());
    assert(!m_mlpOnnxPath.isEmpty());
    assert(!m_tsnSerializePath.isEmpty());
    assert(!m_mlpSerializePath.isEmpty());
    assert(m_tsnMaxWorkspace > 0);
    assert(m_mlpMaxWorkspace > 0);
    assert(m_tsnMaxBatches > 0);
    assert(m_mlpMaxBatches > 0);
}

Engines TensorRtEngineBuilder::build() const
{
    Engines out;
    bool needToBuildTsn = true;
    bool needToBuildMlp = true;

    bool const tsnExists = QFile::exists(m_tsnSerializePath);
    bool const mlpExists = QFile::exists(m_mlpSerializePath);

    if (tsnExists || mlpExists)
    {
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);

        if (tsnExists)
        {
            QFile file(m_tsnSerializePath);

            if (file.open(QFile::ReadOnly))
            {
                auto data = file.readAll();
                out.tsn = runtime->deserializeCudaEngine(data.data(), static_cast<size_t>(data.size()));
                needToBuildTsn = out.tsn == nullptr;
            }
        }

        if (mlpExists)
        {
            QFile file(m_mlpSerializePath);

            if (file.open(QFile::ReadOnly))
            {
                auto data = file.readAll();
                out.mlp = runtime->deserializeCudaEngine(data.data(), static_cast<size_t>(data.size()));
                needToBuildTsn = out.mlp == nullptr;
            }
        }

        runtime->destroy();
    }

    if (needToBuildTsn || needToBuildMlp)
    {
        auto const builder = nvinfer1::createInferBuilder(gLogger);
        assert(builder);

        if (needToBuildTsn)
        {
            out.tsn = buildTsn(builder);
        }
        if (needToBuildMlp)
        {
            out.mlp = buildMlp(builder);
        }

        builder->destroy();
    }


    return out;
}

nvinfer1::ICudaEngine *TensorRtEngineBuilder::buildTsn(nvinfer1::IBuilder* builder) const
{
    auto const networkTsn = builder->createNetworkV2(static_cast<nvinfer1::NetworkDefinitionCreationFlags>(
                                                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    assert(networkTsn);

    auto const parserTsn = nvonnxparser::createParser(*networkTsn, gLogger);
    assert(parserTsn);

    assert(parserTsn->parseFromFile(qUtf8Printable(m_tsnOnnxPath),
                                      static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE)));

    auto config = builder->createBuilderConfig();
    assert(config);


    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    config->setMaxWorkspaceSize(m_tsnMaxWorkspace);
    builder->setMaxBatchSize(m_tsnMaxBatches);
    auto const engineTsn = builder->buildEngineWithConfig(*networkTsn, *config);
    assert(engineTsn);

    auto hostMemory = engineTsn->serialize();

    QFile file(m_tsnSerializePath);
    assert(file.open(QFile::WriteOnly));

    file.write(static_cast<char*>(hostMemory->data()), static_cast<qint64>(hostMemory->size()));

    hostMemory->destroy();
    config->destroy();
    parserTsn->destroy();
    networkTsn->destroy();

    return engineTsn;
}

nvinfer1::ICudaEngine *TensorRtEngineBuilder::buildMlp(nvinfer1::IBuilder* builder) const
{
    auto const networkMlp = builder->createNetworkV2(static_cast<nvinfer1::NetworkDefinitionCreationFlags>(
                                                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    assert(networkMlp);

    auto const parserMlp = nvonnxparser::createParser(*networkMlp, gLogger);
    assert(parserMlp);

    assert(parserMlp->parseFromFile(qUtf8Printable(m_mlpOnnxPath),
                                      static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE)));


    auto config = builder->createBuilderConfig();
    assert(config);

    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    config->setMaxWorkspaceSize(m_mlpMaxWorkspace);
    builder->setMaxBatchSize(m_mlpMaxBatches);
    auto const engineMlp = builder->buildEngineWithConfig(*networkMlp, *config);
    assert(engineMlp);

    auto hostMemory = engineMlp->serialize();

    QFile file(m_mlpSerializePath);
    assert(file.open(QFile::WriteOnly));

    file.write(static_cast<char*>(hostMemory->data()), static_cast<qint64>(hostMemory->size()));

    hostMemory->destroy();
    config->destroy();
    parserMlp->destroy();
    networkMlp->destroy();

    return engineMlp;
}
}
}
