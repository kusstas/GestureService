QT -= gui

CONFIG += c++17 console
CONFIG -= app_bundle

DEFINES += QT_DEPRECATED_WARNINGS

HEADERS += \
    ImageConvertor.h \
    Session.h \
    TensorRtEngineBuilder.h \
    TensorRtExecution.h

SOURCES += \
    ImageConvertor.cpp \
    Session.cpp \
    TensorRtEngineBuilder.cpp \
    TensorRtExecution.cpp \
    main.cpp

qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

unix:!macx: LIBS += -L$$PWD/../../Libraries/TensorRT-6.0.1.5/lib/ -lnvinfer -lnvinfer_plugin -lnvonnxparser -lnvonnxparser_runtime -lnvparsers
unix:!macx: LIBS += -L/usr/local/cuda/lib64 -lcudart -lcublas -lcudnn
unix:!macx: LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio
unix:!macx: LIBS += -L$$PWD/../GesturePlugin/GestureServer/libs/ -lGestureServer

INCLUDEPATH += $$PWD/../../Libraries/TensorRT-6.0.1.5/include
DEPENDPATH += $$PWD/../../Libraries/TensorRT-6.0.1.5/include

INCLUDEPATH += /usr/local/cuda/include
DEPENDPATH += /usr/local/cuda/include

INCLUDEPATH += /usr/include/opencv2
DEPENDPATH += /usr/include/opencv2

INCLUDEPATH += $$PWD/../GesturePlugin/GestureServer
DEPENDPATH += $$PWD/../GesturePlugin/GestureServer
