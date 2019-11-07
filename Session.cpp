#include "Session.h"

#include <QJsonArray>
#include <QCoreApplication>

#include <opencv2/opencv.hpp>

#include "TensorRtEngineBuilder.h"
#include <QDebug>

namespace gestureserver
{
Session::Session(QJsonObject const& settings)
    : m_execution(settings)
    , m_imageConvertor(settings)
{
    m_camera = settings["camera"].toInt();
    m_countFramesTrigger = settings["countFramesTrigger"].toInt();
    m_countFramesProcessing = settings["countFramesProcessing"].toInt();
    m_successThreshold = settings["successThreshold"].toDouble();

    for (auto const& value : settings["gestureNames"].toArray())
    {
        m_gestureNames.push_back(value.toString());
    }

    gestureserver::nn::TensorRtEngineBuilder builder(settings);

    m_engines = builder.build();
    m_execution.createContextes(m_engines.tsn, m_engines.mlp);

    assert(m_server.enable(settings["urlService"].toString()));
}

Session::~Session()
{
    if (m_engines.mlp)
    {
        m_engines.mlp->destroy();
    }
    if (m_engines.tsn)
    {
        m_engines.tsn->destroy();
    }
}

void Session::execute()
{
    m_inputTsn = std::vector<float>(m_execution.tsnInputSize());
    m_inputMlp = std::vector<float>(m_execution.mlpInputSize());
    m_outputMlp = std::vector<float>(m_execution.mlpOutputSize());

    cv::VideoCapture cap(m_camera);
    m_running = true;

    while (m_running)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            break;
        }

        forward(frame);
        qApp->processEvents(); // for handle server events
    }

    m_running = false;
}

void Session::forward(cv::Mat const& frame)
{
    forwardTsn(frame);

    if (m_outputsTsn.size() == m_countFramesProcessing)
    {
        auto inputMlpIt = m_inputMlp.begin();
        for (auto const& outputTsn : m_outputsTsn)
        {
            std::copy(outputTsn.begin(), outputTsn.end(), inputMlpIt);
            inputMlpIt += outputTsn.size();
        }

        m_execution.executeMlp(m_inputMlp.data(), m_outputMlp.data());
        handleResult();
    }
}

void Session::forwardTsn(cv::Mat const& frame)
{
    auto brg = m_imageConvertor.convert(frame);

    auto inputTsnIt = m_inputTsn.begin();
    for (auto const& ch : brg)
    {
        void const* const rawData = ch.data;
        auto const floatData = static_cast<float const*>(rawData);
        std::copy(floatData, floatData + ch.total(), inputTsnIt);
        inputTsnIt += ch.total();
    }

    if (m_outputsTsn.size() >= m_countFramesProcessing)
    {
        m_outputsTsn.push_back(std::move(m_outputsTsn.front()));
        m_outputsTsn.pop_front();
    }
    else
    {
        m_outputsTsn.emplace_back(m_execution.tsnOutputSize());
    }

    m_execution.executeTsn(m_inputTsn.data(), m_outputsTsn.back().data());
}

void Session::handleResult()
{
    struct Element
    {
        float value;
        int gesture;
    };

    std::vector<Element> result;
    result.reserve(m_outputMlp.size());

    for (size_t i = 0; i < m_outputMlp.size(); i++)
    {
        result.push_back({m_outputMlp[i], static_cast<int>(i)});
    }

    std::sort(result.begin(), result.end(), [] (Element const& l, Element const& r)
    {
        return l.value > r.value;
    });

    Element const& top = result.front();
    if (top.value >= m_successThreshold)
    {
        m_sequenceFramesSuccess++;
        if (top.gesture != m_previousGesture)
        {
            m_sequenceFramesSuccess = 1;
        }
        m_previousGesture = top.gesture;
    }
    else
    {
        m_sequenceFramesSuccess = 0;
        m_previousGesture = -1;
    }

    if (m_sequenceFramesSuccess >= m_countFramesTrigger)
    {
        qDebug() << m_gestureNames[top.gesture] << top.value << m_sequenceFramesSuccess;
        if (top.gesture > 0)
        {
            m_server.postGesture(m_gestureNames[top.gesture], m_sequenceFramesSuccess - m_countFramesTrigger);
        }
    }
}
}
