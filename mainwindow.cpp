#include <opencv2/opencv.hpp>
#include <opencv2/face/facemark.hpp>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    cv::CascadeClassifier cc("./haarcascade_frontalface_default.xml");
    cv::Ptr<cv::face::Facemark> markDetector = cv::face::createFacemarkLBF();
    markDetector->loadModel("lbfmodel.yaml");

    cv::VideoCapture vc(0);
    if (!vc.isOpened())
        return;

    cv::Mat frame;
    std::vector<cv::Rect> faces;
    cv::Mat grayFrame;
    cv::Scalar color = cv::Scalar(0, 0, 255);   // red
    std::vector<std::vector<cv::Point2f>> shapes;
    while (true) {
        if (cv::waitKey(1) >= 0)
            break;
        vc >> frame;
        if (frame.empty())
            break;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        cc.detectMultiScale(grayFrame, faces, 1.3, 5);

        if (markDetector->fit(frame, faces, shapes))
            // draw facial land marks
            for (unsigned long i = 0; i < faces.size(); ++i)
                for(unsigned long k = 0; k < shapes[i].size(); ++k)
                    cv::circle(frame, shapes[i][k], 2, color, cv::FILLED);

        cv::imshow("Video", frame);
    }
    vc.release();
    cv::destroyAllWindows();
}
