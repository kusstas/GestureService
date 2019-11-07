#pragma once

#include <vector>

#include <QJsonObject>
#include <opencv2/core/core.hpp>

namespace gestureserver
{
namespace vision
{
class ImageConvertor
{
public:
    explicit ImageConvertor(QJsonObject const& settings);

    std::vector<cv::Mat> convert(cv::Mat const& source) const;

private:
    int m_resizeSize = 0;
    int m_cropSize = 0;

    std::array<double, 3> m_mean = {0, 0, 0};
    std::array<double, 3> m_std = {1, 1, 1};
};
}
}
