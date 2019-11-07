#include "ImageConvertor.h"

#include <QJsonArray>

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cassert>

namespace gestureserver
{
namespace vision
{
ImageConvertor::ImageConvertor(QJsonObject const& settings)
{
    m_resizeSize = settings["resizeTargetSize"].toInt();
    m_cropSize = settings["cropTargetSize"].toInt();

    auto mean = settings["meanImg"].toArray();
    auto std = settings["stdImg"].toArray();

    assert(static_cast<size_t>(mean.size()) == m_mean.size());
    assert(static_cast<size_t>(std.size()) == m_std.size());

    for (int i = 0; i < mean.size(); i++)
    {
        m_mean[static_cast<size_t>(i)] = mean[i].toDouble();
    }

    for (int i = 0; i < std.size(); i++)
    {
        m_std[static_cast<size_t>(i)] = std[i].toDouble();
    }

    assert(m_cropSize > 0);
    assert(m_resizeSize > 0);
}

std::vector<cv::Mat> ImageConvertor::convert(cv::Mat const& source) const
{
    std::vector<cv::Mat> out;

    cv::Size resize;

    if (source.size().height > source.size().width)
    {
        resize.width = m_resizeSize;
        resize.height = static_cast<int>(m_resizeSize *
                                         static_cast<float>(source.size().height) / source.size().width);
    }
    else
    {
        resize.height = m_resizeSize;
        resize.width = static_cast<int>(m_resizeSize *
                                         static_cast<float>(source.size().width) / source.size().height);
    }

    cv::Mat img;
    cv::resize(source, img, resize);

    cv::Rect roi;
    roi.x = (img.size().width - m_cropSize) / 2;
    roi.y = (img.size().height - m_cropSize) / 2;
    roi.width = m_cropSize;
    roi.height = m_cropSize;
    img = img(roi);

    cv::split(img, out);

    for (size_t ch = 0; ch < out.size(); ch++)
    {
        out[ch].convertTo(out[ch], CV_32FC1);
        out[ch] /= 255;
        out[ch] -= m_mean[ch];
        out[ch] /= m_std[ch];
    }

    return out;
}
}
}
