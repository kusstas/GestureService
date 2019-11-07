#include <QCoreApplication>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>

#include "Session.h"


int main(int argc, char* argv[])
{
    QCoreApplication a(argc, argv);

    QJsonObject settings;
    {
        QFile file("settings.json");
        assert(file.open(QFile::ReadOnly | QFile::Text));
        settings = QJsonDocument::fromJson(file.readAll()).object();
    }

    gestureserver::Session session(settings);

    session.execute();

    return a.exec();
}
