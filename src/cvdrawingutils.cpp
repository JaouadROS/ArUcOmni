/**
Copyright 2017 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
*/
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cvdrawingutils.h"
#include "cameraparameters.h"
using namespace cv;
namespace aruco
{
void CvDrawingUtils::draw3dAxis(cv::Mat& Image, const CameraParameters& CP, const cv::Mat& Rvec,
                                   const cv::Mat& Tvec, float axis_size)
   {
       Mat objectPoints(4, 1, CV_32FC3);
       objectPoints.at<cv::Vec3f>(0, 0)[0] = 0;
       objectPoints.at<cv::Vec3f>(0, 0)[1] = 0;
       objectPoints.at<cv::Vec3f>(0, 0)[2] = 0;
       objectPoints.at<cv::Vec3f>(1, 0)[0] = axis_size;
       objectPoints.at<cv::Vec3f>(1, 0)[1] = 0;
       objectPoints.at<cv::Vec3f>(1, 0)[2] = 0;
       objectPoints.at<cv::Vec3f>(2, 0)[0] = 0;
       objectPoints.at<cv::Vec3f>(2, 0)[1] = axis_size;
       objectPoints.at<cv::Vec3f>(2, 0)[2] = 0;
       objectPoints.at<cv::Vec3f>(3, 0)[0] = 0;
       objectPoints.at<cv::Vec3f>(3, 0)[1] = 0;
       objectPoints.at<cv::Vec3f>(3, 0)[2] = axis_size;
       
       std::vector<Point2f> imagePoints;
	   projectPointsOmni(objectPoints, imagePoints, Rvec, Tvec, CP.CameraMatrix, CP.xi);
       // draw lines of different colours
       cv::line(Image, imagePoints[0], imagePoints[1], Scalar(0, 0, 255, 255), 1, cv::LINE_AA);
       cv::line(Image, imagePoints[0], imagePoints[2], Scalar(0, 255, 0, 255), 1, cv::LINE_AA);
       cv::line(Image, imagePoints[0], imagePoints[3], Scalar(255, 0, 0, 255), 1, cv::LINE_AA);
       putText(Image, "x", imagePoints[1], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255, 255), 2);
       putText(Image, "y", imagePoints[2], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0, 255), 2);
       putText(Image, "z", imagePoints[3], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0, 255), 2);
   }
    /****
     *
     *
     *
     ****/
    void CvDrawingUtils::draw3dAxis(cv::Mat& Image, Marker& m, const CameraParameters& CP,int lineSize)
    {
        float size = m.ssize*15;
       Mat objectPoints(4, 1, CV_32FC3);
       objectPoints.at<cv::Vec3f>(0, 0)[0] = 0;
       objectPoints.at<cv::Vec3f>(0, 0)[1] = 0;
       objectPoints.at<cv::Vec3f>(0, 0)[2] = 0;
       objectPoints.at<cv::Vec3f>(1, 0)[0] = size;
       objectPoints.at<cv::Vec3f>(1, 0)[1] = 0;
       objectPoints.at<cv::Vec3f>(1, 0)[2] = 0;
       objectPoints.at<cv::Vec3f>(2, 0)[0] = 0;
       objectPoints.at<cv::Vec3f>(2, 0)[1] = size;
       objectPoints.at<cv::Vec3f>(2, 0)[2] = 0;
       objectPoints.at<cv::Vec3f>(3, 0)[0] = 0;
       objectPoints.at<cv::Vec3f>(3, 0)[1] = 0;
       objectPoints.at<cv::Vec3f>(3, 0)[2] = size;

/*std::cout<<"objectPoints: "<<objectPoints<<std::endl;
std::cout<<"rvec: "<<m.Rvec<<std::endl;
std::cout<<"tvec: "<<m.Tvec<<std::endl;
std::cout<<"K: "<<CP.CameraMatrix<<std::endl;
std::cout<<"xi: "<<CP.xi<<std::endl;
std::cout<<"Distorsion: "<<CP.Distorsion<<std::endl;*/

        std::vector<Point2f> imagePoints;
		projectPointsOmni(objectPoints, imagePoints, m.Rvec, m.Tvec, CP.CameraMatrix, CP.xi);
		//std::cout<<"imagePoints: "<<imagePoints<<std::endl;
        // draw lines of different colours
        cv::line(Image, imagePoints[0], imagePoints[1], Scalar(0, 0, 255, 255), lineSize, cv::LINE_AA);
        cv::line(Image, imagePoints[0], imagePoints[2], Scalar(0, 255, 0, 255), lineSize, cv::LINE_AA);
        cv::line(Image, imagePoints[0], imagePoints[3], Scalar(255, 0, 0, 255), lineSize, cv::LINE_AA);
        putText(Image, "x", imagePoints[1], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255, 255), 2);
        putText(Image, "y", imagePoints[2], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0, 255), 2);
        putText(Image, "z", imagePoints[3], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0, 255), 2);
    }

    /****
     *
     *
     *
     ****/
    void CvDrawingUtils::draw3dCube(cv::Mat& Image, Marker& m, const CameraParameters& CP, int lineSize, bool setYperpendicular)
    {
        Mat objectPoints(8, 3, CV_32FC1);
        float halfSize = m.ssize / 2.f;

        if (setYperpendicular)
        {
            objectPoints.at<float>(0, 0) = -halfSize;
            objectPoints.at<float>(0, 1) = 0;
            objectPoints.at<float>(0, 2) = -halfSize;
            objectPoints.at<float>(1, 0) = halfSize;
            objectPoints.at<float>(1, 1) = 0;
            objectPoints.at<float>(1, 2) = -halfSize;
            objectPoints.at<float>(2, 0) = halfSize;
            objectPoints.at<float>(2, 1) = 0;
            objectPoints.at<float>(2, 2) = halfSize;
            objectPoints.at<float>(3, 0) = -halfSize;
            objectPoints.at<float>(3, 1) = 0;
            objectPoints.at<float>(3, 2) = halfSize;

            objectPoints.at<float>(4, 0) = -halfSize;
            objectPoints.at<float>(4, 1) = m.ssize;
            objectPoints.at<float>(4, 2) = -halfSize;
            objectPoints.at<float>(5, 0) = halfSize;
            objectPoints.at<float>(5, 1) = m.ssize;
            objectPoints.at<float>(5, 2) = -halfSize;
            objectPoints.at<float>(6, 0) = halfSize;
            objectPoints.at<float>(6, 1) = m.ssize;
            objectPoints.at<float>(6, 2) = halfSize;
            objectPoints.at<float>(7, 0) = -halfSize;
            objectPoints.at<float>(7, 1) = m.ssize;
            objectPoints.at<float>(7, 2) = halfSize;
        }
        else
        {
            objectPoints.at<float>(0, 0) = -halfSize;
            objectPoints.at<float>(0, 1) = -halfSize;
            objectPoints.at<float>(0, 2) = 0;
            objectPoints.at<float>(1, 0) = halfSize;
            objectPoints.at<float>(1, 1) = -halfSize;
            objectPoints.at<float>(1, 2) = 0;
            objectPoints.at<float>(2, 0) = halfSize;
            objectPoints.at<float>(2, 1) = halfSize;
            objectPoints.at<float>(2, 2) = 0;
            objectPoints.at<float>(3, 0) = -halfSize;
            objectPoints.at<float>(3, 1) = halfSize;
            objectPoints.at<float>(3, 2) = 0;

            objectPoints.at<float>(4, 0) = -halfSize;
            objectPoints.at<float>(4, 1) = -halfSize;
            objectPoints.at<float>(4, 2) = m.ssize;
            objectPoints.at<float>(5, 0) = halfSize;
            objectPoints.at<float>(5, 1) = -halfSize;
            objectPoints.at<float>(5, 2) = m.ssize;
            objectPoints.at<float>(6, 0) = halfSize;
            objectPoints.at<float>(6, 1) = halfSize;
            objectPoints.at<float>(6, 2) = m.ssize;
            objectPoints.at<float>(7, 0) = -halfSize;
            objectPoints.at<float>(7, 1) = halfSize;
            objectPoints.at<float>(7, 2) = m.ssize;
        }

        std::vector<Point2f> imagePoints;
        projectPoints(objectPoints, m.Rvec, m.Tvec, CP.CameraMatrix, CP.Distorsion, imagePoints);
        // draw lines of different colours
        for (int i = 0; i < 4; i++)
            cv::line(Image, imagePoints[i], imagePoints[(i + 1) % 4], Scalar(0, 0, 255, 255), lineSize, cv::LINE_AA);

        for (int i = 0; i < 4; i++)
            cv::line(Image, imagePoints[i + 4], imagePoints[4 + (i + 1) % 4], Scalar(0, 0, 255, 255), lineSize, cv::LINE_AA);

        for (int i = 0; i < 4; i++)
            cv::line(Image, imagePoints[i], imagePoints[i + 4], Scalar(0, 0, 255, 255), lineSize, cv::LINE_AA);
    }
    
void CvDrawingUtils::projectPointsOmni(InputArray objectPoints, OutputArray imagePoints,
                InputArray rvec, InputArray tvec, InputArray K, double xi)
{
    CV_Assert(objectPoints.type() == CV_64FC3 || objectPoints.type() == CV_32FC3);
    CV_Assert((rvec.depth() == CV_64F || rvec.depth() == CV_32F) && rvec.total() == 3);
    CV_Assert((tvec.depth() == CV_64F || tvec.depth() == CV_32F) && tvec.total() == 3);
    CV_Assert((K.type() == CV_64F || K.type() == CV_32F) && K.size() == Size(3,3));
    //CV_Assert((D.type() == CV_64F || D.type() == CV_32F) && D.total() == 4);

    imagePoints.create(objectPoints.size(), CV_MAKETYPE(objectPoints.depth(), 2));

    int n = (int)objectPoints.total();

    Vec3d om = rvec.depth() == CV_32F ? (Vec3d)*rvec.getMat().ptr<Vec3f>() : *rvec.getMat().ptr<Vec3d>();
    Vec3d T  = tvec.depth() == CV_32F ? (Vec3d)*tvec.getMat().ptr<Vec3f>() : *tvec.getMat().ptr<Vec3d>();

    Vec2d f,c;
    double s;
    if (K.depth() == CV_32F)
    {
        Matx33f Kc = K.getMat();
        f = Vec2f(Kc(0,0), Kc(1,1));
        c = Vec2f(Kc(0,2),Kc(1,2));
        s = (double)Kc(0,1);
    }
    else
    {
        Matx33d Kc = K.getMat();
        f = Vec2d(Kc(0,0), Kc(1,1));
        c = Vec2d(Kc(0,2),Kc(1,2));
        s = Kc(0,1);
    }

    //Vec4d kp = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>() : *D.getMat().ptr<Vec4d>();
    //Vec<double, 4> kp= (Vec<double,4>)*D.getMat().ptr<Vec<double,4> >();

    const Vec3d* Xw_alld = objectPoints.getMat().ptr<Vec3d>();
    const Vec3f* Xw_allf = objectPoints.getMat().ptr<Vec3f>();
    Vec2d* xpd = imagePoints.getMat().ptr<Vec2d>();
    Vec2f* xpf = imagePoints.getMat().ptr<Vec2f>();

    Matx33d R;
    Matx<double, 3, 9> dRdom;
    Rodrigues(om, R, dRdom);

    for (int i = 0; i < n; i++)
    {
        // convert to camera coordinate
        Vec3d Xw = objectPoints.depth() == CV_32F ? (Vec3d)Xw_allf[i] : Xw_alld[i];

        Vec3d Xc = (Vec3d)(R*Xw + T);

/*double rho = cv::norm(a);
            auto x = cv::Vec3d(a.at<double>(0,0)/(a.at<double>(2,0) + rho*xi), a.at<double>(1,0)/(a.at<double>(2,0) + rho*xi), 1);
  */          
            
        // convert to unit sphere
        //Vec3d Xs = Xc/cv::norm(Xc);

        // convert to normalized image plane
        Vec2d xu = Vec2d(Xc[0]/(Xc[2]+cv::norm(Xc)*xi), Xc[1]/(Xc[2]+cv::norm(Xc)*xi));

        // convert to pixel coordinate
        Vec2d final;
        final[0] = f[0]*xu[0]+s*xu[1]+c[0];
        final[1] = f[1]*xu[1]+c[1];

        if (objectPoints.depth() == CV_32F)
        {
            xpf[i] = final;
        }
        else
        {
            xpd[i] = final;
        }
        
    }
}    
}
