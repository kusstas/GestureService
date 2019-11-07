#pragma once

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStringList>
#include <cassert>

#include <vector>
#include <deque>

#include <GestureServer.h>

#include "TensorRtEngineBuilder.h"
#include "TensorRtExecution.h"
#include "ImageConvertor.h"

namespace gestureserver
{
class Session
{
public:
    explicit Session(QJsonObject const& settings);
    ~Session();

    void execute();

private:
    void forward(cv::Mat const& frame);
    void forwardTsn(cv::Mat const& frame);
    void handleResult();

private:
    int m_camera = 0;
    int m_countFramesTrigger = 0;
    double m_successThreshold = 0;
    size_t m_countFramesProcessing = 0;
    QStringList m_gestureNames = {};

    nn::Engines m_engines;
    nn::TensorRtExecution m_execution;
    vision::ImageConvertor m_imageConvertor;
    GestureServer m_server;

    std::vector<float> m_inputTsn{};
    std::vector<float> m_inputMlp{};
    std::vector<float> m_outputMlp{};
    std::deque<std::vector<float>> m_outputsTsn{};

    bool m_running = false;
    int m_previousGesture = -1;
    int m_sequenceFramesSuccess = 0;
};
}
