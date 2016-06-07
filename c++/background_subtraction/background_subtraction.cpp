#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{
    cv::VideoCapture capture(0);

    namedWindow("Frame");
    namedWindow("Background");
    namedWindow("FG Mask MOG 2");

    Mat kernel = getStructuringElement(CV_SHAPE_ELLIPSE,Size(5,5));
    Mat frame;
    Mat background;
    Mat fgMaskMOG2;
    Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2(1000, 30, false);
    pMOG2->setVarInit(15);
    pMOG2->setVarMin(4);
    pMOG2->setVarMax(75);

    unsigned n = 0;

    while( capture.read(frame) ){

        pMOG2->apply(frame, fgMaskMOG2);
        pMOG2->getBackgroundImage(background);

        dilate(fgMaskMOG2,fgMaskMOG2,kernel);
        erode(fgMaskMOG2,fgMaskMOG2,kernel);
        
        imshow("Frame", frame);
        imshow("Background", background);
        imshow("FG Mask MOG 2", fgMaskMOG2);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27)
            break;
        else if (key == 32){
            ++n;
            imwrite("frame"+std::to_string(n)+".png", frame);
            imwrite("hand"+std::to_string(n)+".png", fgMaskMOG2);
        }
    }
    destroyAllWindows();
    return 0;
}