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

#include "marker.h"

/// \todo set this definition in the cmake code
#define _USE_MATH_DEFINES

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdio>
#include <math.h>
#include "cameraparameters.h"
#include "ippe.h"


namespace aruco
{
    /**
     *
     */
    Marker::Marker()
    {
        id = -1;
        ssize = -1;
        Rvec.create(3, 1, CV_32FC1);
        Tvec.create(3, 1, CV_32FC1);
        for (int i = 0; i < 3; i++)
            Tvec.at<float>(i, 0) = Rvec.at<float>(i, 0) = -999999;
    }
    /**
     *
     */
    Marker::Marker(int _id)
    {
        id = _id;
        ssize = -1;
        Rvec.create(3, 1, CV_32FC1);
        Tvec.create(3, 1, CV_32FC1);
        for (int i = 0; i < 3; i++)
            Tvec.at<float>(i, 0) = Rvec.at<float>(i, 0) = -999999;
    }
    /**
     *
     */
    Marker::Marker(const Marker& M)
        : std::vector<cv::Point2f>(M)
    {
        M.copyTo(*this);
    }

    /**
     *
     */
    Marker::Marker(const std::vector<cv::Point2f>& corners, int _id)
        : std::vector<cv::Point2f>(corners)
    {
        id = _id;
        ssize = -1;
        Rvec.create(3, 1, CV_32FC1);
        Tvec.create(3, 1, CV_32FC1);
        for (int i = 0; i < 3; i++)
            Tvec.at<float>(i, 0) = Rvec.at<float>(i, 0) = -999999;
    }

    void  Marker::copyTo(Marker &m)const{
        m.id=id;
        // size of the markers sides in meters
        m.ssize=ssize;
        // matrices of rotation and translation respect to the camera
        Rvec.copyTo(m.Rvec );
        Tvec.copyTo(m.Tvec);
        m.resize(size());
        for(size_t i=0;i<size();i++)
            m.at(i)=at(i);
        m.dict_info=dict_info;
        m.contourPoints=contourPoints;
        for (int i = 0; i < 4; i++)
            m.sCorners[i]=sCorners[i];
    }
    /**compares ids
    */
    Marker &  Marker::operator=(const Marker& m)
    {
        m.copyTo(*this);
        return *this;
    }
    /**
     *
     */
    void Marker::glGetModelViewMatrix(double modelview_matrix[16])
    {
        // check if paremeters are valid
        bool invalid = false;
        for (int i = 0; i < 3 && !invalid; i++)
        {
            if (Tvec.at<float>(i, 0) != -999999)
                invalid |= false;
            if (Rvec.at<float>(i, 0) != -999999)
                invalid |= false;
        }
        if (invalid)
            throw cv::Exception(9003, "extrinsic parameters are not set", "Marker::getModelViewMatrix", __FILE__,
                    __LINE__);
        cv::Mat Rot(3, 3, CV_32FC1), Jacob;
        cv::Rodrigues(Rvec, Rot, Jacob);

        double para[3][4];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                para[i][j] = Rot.at<float>(i, j);
        // now, add the translation
        para[0][3] = Tvec.at<float>(0, 0);
        para[1][3] = Tvec.at<float>(1, 0);
        para[2][3] = Tvec.at<float>(2, 0);
        double scale = 1;

        modelview_matrix[0 + 0 * 4] = para[0][0];
        // R1C2
        modelview_matrix[0 + 1 * 4] = para[0][1];
        modelview_matrix[0 + 2 * 4] = para[0][2];
        modelview_matrix[0 + 3 * 4] = para[0][3];
        // R2
        modelview_matrix[1 + 0 * 4] = para[1][0];
        modelview_matrix[1 + 1 * 4] = para[1][1];
        modelview_matrix[1 + 2 * 4] = para[1][2];
        modelview_matrix[1 + 3 * 4] = para[1][3];
        // R3
        modelview_matrix[2 + 0 * 4] = -para[2][0];
        modelview_matrix[2 + 1 * 4] = -para[2][1];
        modelview_matrix[2 + 2 * 4] = -para[2][2];
        modelview_matrix[2 + 3 * 4] = -para[2][3];
        modelview_matrix[3 + 0 * 4] = 0.0;
        modelview_matrix[3 + 1 * 4] = 0.0;
        modelview_matrix[3 + 2 * 4] = 0.0;
        modelview_matrix[3 + 3 * 4] = 1.0;
        if (scale != 0.0)
        {
            modelview_matrix[12] *= scale;
            modelview_matrix[13] *= scale;
            modelview_matrix[14] *= scale;
        }
    }

    /****
     *
     */
    void Marker::OgreGetPoseParameters(double position[3], double orientation[4])
    {
        // check if paremeters are valid
        bool invalid = false;
        for (int i = 0; i < 3 && !invalid; i++)
        {
            if (Tvec.at<float>(i, 0) != -999999)
                invalid |= false;
            if (Rvec.at<float>(i, 0) != -999999)
                invalid |= false;
        }
        if (invalid)
            throw cv::Exception(9003, "extrinsic parameters are not set", "Marker::getModelViewMatrix", __FILE__,
                    __LINE__);

        // calculate position vector
        position[0] = -Tvec.ptr<float>(0)[0];
        position[2] = +Tvec.ptr<float>(0)[2];

        // now calculare orientation quaternion
        cv::Mat Rot(3, 3, CV_32FC1);
        cv::Rodrigues(Rvec, Rot);

        // calculate axes for quaternion
        double stAxes[3][3];
        // x axis
        stAxes[0][0] = -Rot.at<float>(0, 0);
        stAxes[0][1] = -Rot.at<float>(1, 0);
        stAxes[0][2] = +Rot.at<float>(2, 0);
        // y axis
        stAxes[1][0] = -Rot.at<float>(0, 1);
        stAxes[1][1] = -Rot.at<float>(1, 1);
        stAxes[1][2] = +Rot.at<float>(2, 1);
        // for z axis, we use cross product
        stAxes[2][0] = stAxes[0][1] * stAxes[1][2] - stAxes[0][2] * stAxes[1][1];
        stAxes[2][1] = -stAxes[0][0] * stAxes[1][2] + stAxes[0][2] * stAxes[1][0];
        stAxes[2][2] = stAxes[0][0] * stAxes[1][1] - stAxes[0][1] * stAxes[1][0];

        // transposed matrix
        double axes[3][3];
        axes[0][0] = stAxes[0][0];
        axes[1][0] = stAxes[0][1];
        axes[2][0] = stAxes[0][2];

        axes[0][1] = stAxes[1][0];
        axes[1][1] = stAxes[1][1];
        axes[2][1] = stAxes[1][2];

        axes[0][2] = stAxes[2][0];
        axes[1][2] = stAxes[2][1];
        axes[2][2] = stAxes[2][2];

        // Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes
        // article "Quaternion Calculus and Fast Animation".
        double fTrace = axes[0][0] + axes[1][1] + axes[2][2];
        double fRoot;

        if (fTrace > 0.0)
        {
            // |w| > 1/2, may as well choose w > 1/2
            fRoot = sqrt(fTrace + 1.0);  // 2w
            orientation[0] = 0.5 * fRoot;
            fRoot = 0.5 / fRoot;  // 1/(4w)
            orientation[1] = (axes[2][1] - axes[1][2]) * fRoot;
            orientation[2] = (axes[0][2] - axes[2][0]) * fRoot;
            orientation[3] = (axes[1][0] - axes[0][1]) * fRoot;
        }
        else
        {
            // |w| <= 1/2
            static unsigned int s_iNext[3] = {1, 2, 0};
            unsigned int i = 0;
            if (axes[1][1] > axes[0][0])
                i = 1;
            if (axes[2][2] > axes[i][i])
                i = 2;
            unsigned int j = s_iNext[i];
            unsigned int k = s_iNext[j];

            fRoot = sqrt(axes[i][i] - axes[j][j] - axes[k][k] + 1.0);
            double* apkQuat[3] = {&orientation[1], &orientation[2], &orientation[3]};
            *apkQuat[i] = 0.5 * fRoot;
            fRoot = 0.5 / fRoot;
            orientation[0] = (axes[k][j] - axes[j][k]) * fRoot;
            *apkQuat[j] = (axes[j][i] + axes[i][j]) * fRoot;
            *apkQuat[k] = (axes[k][i] + axes[i][k]) * fRoot;
        }
    }

    void Marker::draw(cv::Mat& in,  cv::Scalar color, int lineWidth, bool writeId, bool writeInfo) const
    {

        auto _to_string=[](int i){
            std::stringstream str;str<<i;return str.str();
        };

        if (size() != 4)
            return;
        if (lineWidth == -1)  // auto
            lineWidth = static_cast<int>(std::max(1.f, float(in.cols) / 1000.f));
        cv::line(in, (*this)[0], (*this)[1], color, lineWidth);
        cv::line(in, (*this)[1], (*this)[2], color, lineWidth);
        cv::line(in, (*this)[2], (*this)[3], color, lineWidth);
        cv::line(in, (*this)[3], (*this)[0], color, lineWidth);

        auto p2 =  cv::Point2f(2.f * static_cast<float>(lineWidth), 2.f * static_cast<float>(lineWidth));
        cv::rectangle(in, (*this)[0] - p2, (*this)[0] + p2, cv::Scalar(0, 0, 255, 255), -1, cv::LINE_AA);
        cv::rectangle(in, (*this)[1] - p2, (*this)[1] + p2, cv::Scalar(0, 255, 0, 255), lineWidth, cv::LINE_AA);
        cv::rectangle(in, (*this)[2] - p2, (*this)[2] + p2, cv::Scalar(255, 0, 0, 255), lineWidth, cv::LINE_AA);



        if (writeId)
        {
            // determine the centroid
            cv::Point cent(0, 0);
            for (int i = 0; i < 4; i++)
            {
                cent.x += static_cast<int>((*this)[i].x);
                cent.y +=  static_cast<int>((*this)[i].y);
            }
            cent.x /= 4;
            cent.y /= 4;
            std::string str;
            if(writeInfo) str+= dict_info +":";
            if(writeId)str+=_to_string(id);
            cv::putText(in,str, cent,  cv::FONT_HERSHEY_SIMPLEX, std::max(0.5f, float(lineWidth) * 0.3f),
                    cv::Scalar(255 - color[0], 255 - color[1], 255 - color[2], 255), std::max(lineWidth, 2));
        }
    }

    /**
    */
    void Marker::calculateExtrinsics(float markerSize, const CameraParameters& CP,
            bool setYPerpendicular)
    {
        if (!CP.isValid())
            throw cv::Exception(9004,
                    "!CP.isValid(): invalid camera parameters. It is not possible to calculate extrinsics",
                    "calculateExtrinsics", __FILE__, __LINE__);
        calculateExtrinsics(markerSize, CP.CameraMatrix, CP.Distorsion, setYPerpendicular);
    }

    void print(cv::Point3f p, std::string cad)
    {
        std::cout << cad << " " << p.x << " " << p.y << " " << p.z << std::endl;
    }
    /**
    */
    void Marker::calculateExtrinsics(float markerSizeMeters, cv::Mat camMatrix, cv::Mat distCoeff,
            bool setYPerpendicular)
    {
        if (!isValid())
            throw cv::Exception(9004, "!isValid(): invalid marker. It is not possible to calculate extrinsics",
                    "calculateExtrinsics", __FILE__, __LINE__);
        if (markerSizeMeters <= 0)
            throw cv::Exception(9004, "markerSize<=0: invalid markerSize", "calculateExtrinsics", __FILE__, __LINE__);
        if (camMatrix.rows == 0 || camMatrix.cols == 0)
            throw cv::Exception(9004, "CameraMatrix is empty", "calculateExtrinsics", __FILE__, __LINE__);

        vector<cv::Point3f> objpoints = get3DPoints(markerSizeMeters);


        cv::Mat raux, taux;
        //        cv::solvePnP(objpoints, *this, camMatrix, distCoeff, raux, taux);
        solvePnP(objpoints, *this,camMatrix, distCoeff,raux,taux);
        raux.convertTo(Rvec, CV_32F);
        taux.convertTo(Tvec, CV_32F);
        // rotate the X axis so that Y is perpendicular to the marker plane
        if (setYPerpendicular)
            rotateXAxis(Rvec);
        ssize = markerSizeMeters;
        // cout<<(*this)<<endl;

        //        auto setPrecision=[](double f, double prec){
        //            int x=roundf(f*prec);
        //            return  double(x)/prec;
        //        };
        //        for(int i=0;i<3;i++){
        //            Rvec.ptr<float>(0)[i]=setPrecision(Rvec.ptr<float>(0)[i],100);
        //            Tvec.ptr<float>(0)[i]=setPrecision(Tvec.ptr<float>(0)[i],1000);
        //        }

    }

    void Marker::calculateExtrinsicsOmni(float markerSizeMeters, cv::Mat camMatrix, double xi, cv::Mat distCoeff,
            bool setYPerpendicular)
    {
        if (!isValid())
            throw cv::Exception(9004, "!isValid(): invalid marker. It is not possible to calculate extrinsics",
                    "calculateExtrinsics", __FILE__, __LINE__);
        if (markerSizeMeters <= 0)
            throw cv::Exception(9004, "markerSize<=0: invalid markerSize", "calculateExtrinsics", __FILE__, __LINE__);
        if (camMatrix.rows == 0 || camMatrix.cols == 0)
            throw cv::Exception(9004, "CameraMatrix is empty", "calculateExtrinsics", __FILE__, __LINE__);

        vector<cv::Point3f> listP = get3DPoints(markerSizeMeters);

        std::vector<cv::Point3f> sP{4};
        for(int ii = 0; ii<4 ; ii++)
            sP[ii] = (*this).sCorners[ii];

        cv::Matx44d c, sqrDistance;
        for (int i = 0; i < 3; i++)
        {
            for (int j = i+1; j < 4; j++)
            {
                c(i,j) = c(j,i) = -2.0*(sP[i].x*sP[j].x + sP[i].y*sP[j].y + sP[i].z*sP[j].z);
                cv::Point3d oP = listP[i] - listP[j];
                sqrDistance(i,j) = sqrDistance(j,i) = oP.dot(oP);
            }
        }

        cv::Mat A = cv::Mat::zeros(cv::Size(24,24), CV_64FC1);
        A.at<double>(0, 0) = 1.0; A.at<double>(0, 4) = c(0, 1); A.at<double>(0, 7) = 1.0;      A.at<double>(0, 20) = sqrDistance(0, 1);
        A.at<double>(1, 1) = 1.0; A.at<double>(1, 4) = 1.0;     A.at<double>(1, 7) = c(0, 1);  A.at<double>(1, 21) = sqrDistance(0, 1);
        A.at<double>(2, 5) = 1.0; A.at<double>(2, 8) = 1.0;     A.at<double>(2, 16) = c(0, 1); A.at<double>(2, 22) = sqrDistance(0, 1);
        A.at<double>(3, 6) = 1.0; A.at<double>(3, 9) = 1.0;     A.at<double>(3, 17) = c(0, 1); A.at<double>(3, 23) = sqrDistance(0, 1);

        A.at<double>(4, 0) = 1.0; A.at<double>(4, 5) = c(0, 2);  A.at<double>(4, 11) = 1.0;      A.at<double>(4, 20) = sqrDistance(0, 2);
        A.at<double>(5, 4) = 1.0; A.at<double>(5, 10) = 1.0;     A.at<double>(5, 16) = c(0, 2);  A.at<double>(5, 21) = sqrDistance(0, 2);
        A.at<double>(6, 2) = 1.0; A.at<double>(6, 5) = 1.0;      A.at<double>(6, 11) = c(0, 2);  A.at<double>(6, 22) = sqrDistance(0, 2);
        A.at<double>(7, 6) = 1.0; A.at<double>(7, 12) = 1.0;     A.at<double>(7, 18) = c(0, 2);  A.at<double>(7, 23) = sqrDistance(0, 2);

        A.at<double>(8, 0) = 1.0;  A.at<double>(8, 6) = c(0, 3);    A.at<double>(8, 15) = 1.0;       A.at<double>(8, 20) = sqrDistance(0, 3);
        A.at<double>(9, 4) = 1.0;  A.at<double>(9, 13) = 1.0;       A.at<double>(9, 17) = c(0, 3);   A.at<double>(9, 21) = sqrDistance(0, 3);
        A.at<double>(10, 5) = 1.0; A.at<double>(10, 14) = 1.0;      A.at<double>(10, 18) = c(0, 3);  A.at<double>(10, 22) = sqrDistance(0, 3);
        A.at<double>(11, 3) = 1.0; A.at<double>(11, 6) = 1.0;       A.at<double>(11, 15) = c(0, 3);  A.at<double>(11, 23) = sqrDistance(0, 3);

        A.at<double>(12, 7) = 1.0;  A.at<double>(12, 11) = 1.0;      A.at<double>(12, 16) = c(1, 2);   A.at<double>(12, 20) = sqrDistance(1, 2);
        A.at<double>(13, 1) = 1.0;  A.at<double>(13, 8) = c(1, 2);   A.at<double>(13, 10) = 1.0;       A.at<double>(13, 21) = sqrDistance(1, 2);
        A.at<double>(14, 2) = 1.0;  A.at<double>(14, 8) = 1.0;       A.at<double>(14, 10) = c(1, 2);   A.at<double>(14, 22) = sqrDistance(1, 2);
        A.at<double>(15, 9) = 1.0;  A.at<double>(15, 12) = 1.0;      A.at<double>(15, 19) = c(1, 2);   A.at<double>(15, 23) = sqrDistance(1, 2);

        A.at<double>(16, 7) = 1.0;  A.at<double>(16, 15) = 1.0;     A.at<double>(16, 17) = c(1, 3);   A.at<double>(16, 20) = sqrDistance(1, 3);
        A.at<double>(17, 1) = 1.0;  A.at<double>(17, 9) = c(1, 3);  A.at<double>(17, 13) = 1.0;       A.at<double>(17, 21) = sqrDistance(1, 3);
        A.at<double>(18, 8) = 1.0;  A.at<double>(18, 14) = 1.0;     A.at<double>(18, 19) = c(1, 3);   A.at<double>(18, 22) = sqrDistance(1, 3);
        A.at<double>(19, 3) = 1.0;  A.at<double>(19, 9) = 1.0;      A.at<double>(19, 13) = c(1, 3);   A.at<double>(19, 23) = sqrDistance(1, 3);

        A.at<double>(20, 11) = 1.0;    A.at<double>(20, 15) = 1.0; A.at<double>(20, 17) = c(2, 3);   A.at<double>(20, 20) = sqrDistance(2, 3);
        A.at<double>(21, 9) = c(2, 3); A.at<double>(21, 10) = 1.0; A.at<double>(21, 13) = 1.0;       A.at<double>(21, 21) = sqrDistance(2, 3);
        A.at<double>(22, 2) = 1.0;     A.at<double>(22, 14) = 1.0; A.at<double>(22, 19) = c(2, 3);   A.at<double>(22, 22) = sqrDistance(2, 3);
        A.at<double>(23, 3) = 1.0;     A.at<double>(23, 12) = 1.0; A.at<double>(23, 13) = c(2, 3);   A.at<double>(23, 23) = sqrDistance(2, 3);
        //std::cout<<"A OpenCV"<<A<<std::endl;

        cv::Mat Va;
        cv::Mat Sa;
        cv::Mat tm;
        cv::SVD::compute(A, Sa, tm, Va, cv::SVD::MODIFY_A);
        Va = Va.t();

        //std::cout<<"Sa OpenCV" << Sa << std::endl;
        //std::cout << std::endl;
        //std::cout<<"Va OpenCV" << Va << std::endl;

        // Pour chaque point, on recherche sa distance au centre de projection
        double x[4] = {0.0,0.0,0.0,0.0};

        for(int i = 0 ; i < 4 ; i++)
            x[i] = sqrt(fabs(Va.at<double>(i,23) / Va.at<double>(20+i,23)));

        //std::cout <<"x = " << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << std::endl;


        cv::Point3d cP[4], oP[4];
        // Calcul des 4 points dans le repère caméra
        for(int i=0; i<4; i++)
        {
            cP[i] = x[i]*sP[i];
            oP[i] = listP[i];
        }

        //Calcul de la matrice de transformation du repère objet/monde au repère caméra cMo
        //Calcul des 4 bases différentes possibles avec 4 points dans les deux repères pour calculer la rotation d'abord
        cv::Mat cBase, oBase;//TODO use Matx
        cBase.create(cv::Size(12,3), CV_64FC1);
        oBase.create(cv::Size(12,3), CV_64FC1);

        cv::Point3d X, Y, Z;
        int ix, iy, iz;

        //Repère caméra
        //Premiere base
        X = cP[1]-cP[0]; Y = cP[2]-cP[0];
        Z = X.cross(Y);
        Y = Z.cross(X);
        X = (cv::norm(X)==0)?cv::Point3d(0,0,0):X/cv::norm(X); Y = (cv::norm(Y)==0)?cv::Point3d(0,0,0):Y/cv::norm(Y); Z = (cv::norm(Z)==0)?cv::Point3d(0,0,0):Z/cv::norm(Z); 
        ix = 0 ; iy = ix + 1 ; iz = iy + 1;
        cBase.at<double>(0,ix) = X.x; cBase.at<double>(0,iy) = Y.x; cBase.at<double>(0,iz) = Z.x;
        cBase.at<double>(1,ix) = X.y; cBase.at<double>(1,iy) = Y.y; cBase.at<double>(1,iz) = Z.y;
        cBase.at<double>(2,ix) = X.z; cBase.at<double>(2,iy) = Y.z; cBase.at<double>(2,iz) = Z.z;


        //Deuxieme base
        X = cP[2]-cP[1]; Y = cP[3]-cP[1];
        Z = X.cross(Y);
        Y = Z.cross(X);
        X = (cv::norm(X)==0)?cv::Point3d(0,0,0):X/cv::norm(X); Y = (cv::norm(Y)==0)?cv::Point3d(0,0,0):Y/cv::norm(Y); Z = (cv::norm(Z)==0)?cv::Point3d(0,0,0):Z/cv::norm(Z); 
        ix = iz + 1 ; iy = ix + 1 ; iz = iy + 1;
        cBase.at<double>(0,ix) = X.x; cBase.at<double>(0,iy) = Y.x; cBase.at<double>(0,iz) = Z.x;
        cBase.at<double>(1,ix) = X.y; cBase.at<double>(1,iy) = Y.y; cBase.at<double>(1,iz) = Z.y;
        cBase.at<double>(2,ix) = X.z; cBase.at<double>(2,iy) = Y.z; cBase.at<double>(2,iz) = Z.z;


        //Troisieme base
        X = cP[1]-cP[0]; Y = cP[3]-cP[0];
        Z = X.cross(Y);
        Y = Z.cross(X);
        X = (cv::norm(X)==0)?cv::Point3d(0,0,0):X/cv::norm(X); Y = (cv::norm(Y)==0)?cv::Point3d(0,0,0):Y/cv::norm(Y); Z = (cv::norm(Z)==0)?cv::Point3d(0,0,0):Z/cv::norm(Z); 
        ix = iz+1; iy = ix + 1 ; iz = iy + 1;
        cBase.at<double>(0,ix) = X.x; cBase.at<double>(0,iy) = Y.x; cBase.at<double>(0,iz) = Z.x;
        cBase.at<double>(1,ix) = X.y; cBase.at<double>(1,iy) = Y.y; cBase.at<double>(1,iz) = Z.y;
        cBase.at<double>(2,ix) = X.z; cBase.at<double>(2,iy) = Y.z; cBase.at<double>(2,iz) = Z.z;



        //Quatrieme base
        X = cP[2]-cP[0]; Y = cP[3]-cP[0];
        Z = X.cross(Y);
        Y = Z.cross(X);
        X = (cv::norm(X)==0)?cv::Point3d(0,0,0):X/cv::norm(X); Y = (cv::norm(Y)==0)?cv::Point3d(0,0,0):Y/cv::norm(Y); Z = (cv::norm(Z)==0)?cv::Point3d(0,0,0):Z/cv::norm(Z); 
        ix = iz + 1 ; iy = ix + 1 ; iz = iy + 1;
        cBase.at<double>(0,ix) = X.x; cBase.at<double>(0,iy) = Y.x; cBase.at<double>(0,iz) = Z.x;
        cBase.at<double>(1,ix) = X.y; cBase.at<double>(1,iy) = Y.y; cBase.at<double>(1,iz) = Z.y;
        cBase.at<double>(2,ix) = X.z; cBase.at<double>(2,iy) = Y.z; cBase.at<double>(2,iz) = Z.z;


        //Repère objet/monde
        //Premiere base
        X = oP[1]-oP[0]; Y = oP[2]-oP[0];
        Z = X.cross(Y);
        Y = Z.cross(X);
        X = (cv::norm(X)==0)?cv::Point3d(0,0,0):X/cv::norm(X); Y = (cv::norm(Y)==0)?cv::Point3d(0,0,0):Y/cv::norm(Y); Z = (cv::norm(Z)==0)?cv::Point3d(0,0,0):Z/cv::norm(Z); 
        ix = 0 ; iy = ix + 1 ; iz = iy + 1;
        oBase.at<double>(0,ix) = X.x; oBase.at<double>(0,iy) = Y.x; oBase.at<double>(0,iz) = Z.x;
        oBase.at<double>(1,ix) = X.y; oBase.at<double>(1,iy) = Y.y; oBase.at<double>(1,iz) = Z.y;
        oBase.at<double>(2,ix) = X.z; oBase.at<double>(2,iy) = Y.z; oBase.at<double>(2,iz) = Z.z;


        //Deuxieme base
        X = oP[2]-oP[1]; Y = oP[3]-oP[1];
        Z = X.cross(Y);
        Y = Z.cross(X);
        X = (cv::norm(X)==0)?cv::Point3d(0,0,0):X/cv::norm(X); Y = (cv::norm(Y)==0)?cv::Point3d(0,0,0):Y/cv::norm(Y); Z = (cv::norm(Z)==0)?cv::Point3d(0,0,0):Z/cv::norm(Z); 
        ix = iz + 1 ; iy = ix + 1 ; iz = iy + 1;
        oBase.at<double>(0,ix) = X.x; oBase.at<double>(0,iy) = Y.x; oBase.at<double>(0,iz) = Z.x;
        oBase.at<double>(1,ix) = X.y; oBase.at<double>(1,iy) = Y.y; oBase.at<double>(1,iz) = Z.y;
        oBase.at<double>(2,ix) = X.z; oBase.at<double>(2,iy) = Y.z; oBase.at<double>(2,iz) = Z.z;


        //Troisieme base
        X = oP[1]-oP[0]; Y = oP[3]-oP[0];
        Z = X.cross(Y);
        Y = Z.cross(X);
        X = (cv::norm(X)==0)?cv::Point3d(0,0,0):X/cv::norm(X); Y = (cv::norm(Y)==0)?cv::Point3d(0,0,0):Y/cv::norm(Y); Z = (cv::norm(Z)==0)?cv::Point3d(0,0,0):Z/cv::norm(Z); 
        ix = iz+1; iy = ix + 1 ; iz = iy + 1;
        oBase.at<double>(0,ix) = X.x; oBase.at<double>(0,iy) = Y.x; oBase.at<double>(0,iz) = Z.x;
        oBase.at<double>(1,ix) = X.y; oBase.at<double>(1,iy) = Y.y; oBase.at<double>(1,iz) = Z.y;
        oBase.at<double>(2,ix) = X.z; oBase.at<double>(2,iy) = Y.z; oBase.at<double>(2,iz) = Z.z;



        //Quatrieme base
        X = oP[2]-oP[0]; Y = oP[3]-oP[0];
        Z = X.cross(Y);
        Y = Z.cross(X);
        X = (cv::norm(X)==0)?cv::Point3d(0,0,0):X/cv::norm(X); Y = (cv::norm(Y)==0)?cv::Point3d(0,0,0):Y/cv::norm(Y); Z = (cv::norm(Z)==0)?cv::Point3d(0,0,0):Z/cv::norm(Z); 
        ix = iz + 1 ; iy = ix + 1 ; iz = iy + 1;
        oBase.at<double>(0,ix) = X.x; oBase.at<double>(0,iy) = Y.x; oBase.at<double>(0,iz) = Z.x;
        oBase.at<double>(1,ix) = X.y; oBase.at<double>(1,iy) = Y.y; oBase.at<double>(1,iz) = Z.y;
        oBase.at<double>(2,ix) = X.z; oBase.at<double>(2,iy) = Y.z; oBase.at<double>(2,iz) = Z.z;

        //std::cout<<"cBase o: "<<cBase<<"\n oBase"<<oBase<<std::endl;

        cv::Mat R = cBase * ( oBase.t() * (oBase * oBase.t()).inv());

        cv::Mat U, V, S;
        cv::SVD::compute(R, S, U, V, cv::SVD::MODIFY_A);

        //elimination des facteurs d'échelle introduits par du bruit
        R = U * V;

        //std::cout<<"R: "<<R<<std::endl;
        cv::Point3d rcP[4], mrcP, mcP, t;
        for(int i = 0 ; i < 4 ; i++)
        {
            rcP[i].x = R.at<double>(0,0)*oP[i].x + R.at<double>(0,1)*oP[i].y + R.at<double>(0,2)*oP[i].z;
            rcP[i].y = R.at<double>(1,0)*oP[i].x + R.at<double>(1,1)*oP[i].y + R.at<double>(1,2)*oP[i].z;
            rcP[i].z = R.at<double>(2,0)*oP[i].x + R.at<double>(2,1)*oP[i].y + R.at<double>(2,2)*oP[i].z;

            //rcP[i] = R * cv::Mat(oP[i]);

            mrcP += rcP[i];
            mcP += cP[i];
        }
        mrcP *= 0.25;
        mcP *= 0.25;

        t = mcP-mrcP;
        cv::Point3d cto(t.x, t.y, t.z);

        cv::Mat R3f;
        cv::Rodrigues(R, R3f);

        R3f.convertTo(Rvec, CV_32F);
        cv::Mat(cto).convertTo(Tvec, CV_32F);

        //Optimization using Opencv
        (*this).optimization_error = aruco_solve_pnp<float>(listP, (*this), camMatrix, xi, distCoeff, Rvec, Tvec);

        // rotate the X axis so that Y is perpendicular to the marker plane
        if (setYPerpendicular)
            rotateXAxis(Rvec);
        //std::cout<<(*this).id<<" Optimization_Error: "<<(*this).optimization_error<<std::endl;
    }

    template <typename T>
        double Marker::aruco_solve_pnp(const std::vector<cv::Point3f>& p3d, const std::vector<cv::Point2f>& p2d, cv::InputArray cam_matrix, double xi, const cv::Mat& dist, cv::Mat& r_io, cv::Mat& t_io)
        {
            assert(r_io.type() == CV_32F);
            assert(t_io.type() == CV_32F);
            assert(t_io.total() == r_io.total());
            assert(t_io.total() == 3);
            auto toSol = [](const cv::Mat& r, const cv::Mat& t) {
                typename LevMarq<T>::eVector sol(6);
                for (int i = 0; i < 3; i++)
                {
                    sol(i) = r.ptr<float>(0)[i];
                    sol(i + 3) = t.ptr<float>(0)[i];
                }
                return sol;
            };
            auto fromSol = [](const typename LevMarq<T>::eVector& sol, cv::Mat& r, cv::Mat& t) {
                r.create(1, 3, CV_32F);
                t.create(1, 3, CV_32F);
                for (int i = 0; i < 3; i++)
                {
                    r.ptr<float>(0)[i] = sol(i);
                    t.ptr<float>(0)[i] = sol(i + 3);
                }
            };

            cv::Mat Jacb;
            auto err_f = [&](const typename LevMarq<T>::eVector& sol, typename LevMarq<T>::eVector& err) {
                cv::Mat r, t;
                fromSol(sol, r, t);
                std::vector<cv::Point2f> p2d_rej{4};
                projectPointsOmni(p3d, p2d_rej, r, t, cam_matrix, xi, dist, Jacb);
                err.resize(p3d.size() * 2);
                int err_idx = 0;
                for (size_t i = 0; i < p3d.size(); i++)
                {
                    cv::Point2f  errP=p2d_rej[i] -p2d[i];

                    double SqErr=(errP.x*errP.x+ errP.y*errP.y);

                    float robuse_weight= getHubberMonoWeight(SqErr,1);
                    err(err_idx++) = robuse_weight* errP.x;//p2d_rej[i].x - p2d[i].x;
                    err(err_idx++) = robuse_weight* errP.y;//p2d_rej[i].y - p2d[i].y;
                }
            };
            auto jac_f = [&](const typename LevMarq<T>::eVector& sol, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& J) {
                (void)(sol);
                J.resize(p3d.size() * 2, 6);
                for (size_t i = 0; i < p3d.size() * 2; i++)
                {
                    double* jacb = Jacb.ptr<double>(i);
                    for (int j = 0; j < 6; j++)
                        J(i, j) = jacb[j];
                }
            };

            LevMarq<T> solver;
            solver.setParams(100, 0.01, 0.01);
            //  solver.verbose()=true;
            typename LevMarq<T>::eVector sol = toSol(r_io, t_io);
            auto err = solver.solve(sol, err_f, jac_f);

            fromSol(sol, r_io, t_io);
            return err;
        }

    void Marker::projectPointsOmni(cv::InputArray objectPoints, cv::OutputArray imagePoints,
            cv::InputArray rvec, cv::InputArray tvec, cv::InputArray K, double xi, cv::InputArray D, cv::OutputArray jacobian)
    {
        CV_Assert(objectPoints.type() == CV_64FC3 || objectPoints.type() == CV_32FC3);
        CV_Assert((rvec.depth() == CV_64F || rvec.depth() == CV_32F) && rvec.total() == 3);
        CV_Assert((tvec.depth() == CV_64F || tvec.depth() == CV_32F) && tvec.total() == 3);
        CV_Assert((K.type() == CV_64F || K.type() == CV_32F) && K.size() == cv::Size(3,3));
        CV_Assert((D.type() == CV_64F || D.type() == CV_32F) && D.total() == 4);

        imagePoints.create(objectPoints.size(), CV_MAKETYPE(objectPoints.depth(), 2));

        int n = (int)objectPoints.total();

        cv::Vec3d om = rvec.depth() == CV_32F ? (cv::Vec3d)*rvec.getMat().ptr<cv::Vec3f>() : *rvec.getMat().ptr<cv::Vec3d>();
        cv::Vec3d T  = tvec.depth() == CV_32F ? (cv::Vec3d)*tvec.getMat().ptr<cv::Vec3f>() : *tvec.getMat().ptr<cv::Vec3d>();

        cv::Vec2d f,c;
        double s;
        if (K.depth() == CV_32F)
        {
            cv::Matx33f Kc = K.getMat();
            f = cv::Vec2f(Kc(0,0), Kc(1,1));
            c = cv::Vec2f(Kc(0,2),Kc(1,2));
            s = (double)Kc(0,1);
        }
        else
        {
            cv::Matx33d Kc = K.getMat();
            f = cv::Vec2d(Kc(0,0), Kc(1,1));
            c = cv::Vec2d(Kc(0,2),Kc(1,2));
            s = Kc(0,1);
        }

        cv::Vec4d kp = D.depth() == CV_32F ? (cv::Vec4d)*D.getMat().ptr<cv::Vec4f>() : *D.getMat().ptr<cv::Vec4d>();
        //Vec<double, 4> kp= (Vec<double,4>)*D.getMat().ptr<Vec<double,4> >();

        const cv::Vec3d* Xw_alld = objectPoints.getMat().ptr<cv::Vec3d>();
        const cv::Vec3f* Xw_allf = objectPoints.getMat().ptr<cv::Vec3f>();
        cv::Vec2d* xpd = imagePoints.getMat().ptr<cv::Vec2d>();
        cv::Vec2f* xpf = imagePoints.getMat().ptr<cv::Vec2f>();

        cv::Matx33d R;
        cv::Matx<double, 3, 9> dRdom;
        cv::Rodrigues(om, R, dRdom);

        struct JacobianRow
        {
            cv::Matx13d dom,dT;
            cv::Matx12d df;
            double ds;
            cv::Matx12d dc;
            double dxi;
            cv::Matx14d dkp;    // distortion k1,k2,p1,p2
        };
        JacobianRow *Jn = 0;
        if (jacobian.needed())
        {
            int nvars = 2+2+1+4+3+3+1; // f,c,s,kp,om,T,xi
            jacobian.create(2*int(n), nvars, CV_64F);
            Jn = jacobian.getMat().ptr<JacobianRow>(0);
        }

        double k1=kp[0],k2=kp[1];
        double p1 = kp[2], p2 = kp[3];

        for (int i = 0; i < n; i++)
        {
            // convert to camera coordinate
            cv::Vec3d Xw = objectPoints.depth() == CV_32F ? (cv::Vec3d)Xw_allf[i] : Xw_alld[i];

            cv::Vec3d Xc = (cv::Vec3d)(R*Xw + T);

            // convert to unit sphere
            cv::Vec3d Xs = Xc/cv::norm(Xc);

            // convert to normalized image plane
            cv::Vec2d xu = cv::Vec2d(Xs[0]/(Xs[2]+xi), Xs[1]/(Xs[2]+xi));

            // add distortion
            cv::Vec2d xd;
            double r2 = xu[0]*xu[0]+xu[1]*xu[1];
            double r4 = r2*r2;

            xd[0] = xu[0]*(1+k1*r2+k2*r4) + 2*p1*xu[0]*xu[1] + p2*(r2+2*xu[0]*xu[0]);
            xd[1] = xu[1]*(1+k1*r2+k2*r4) + p1*(r2+2*xu[1]*xu[1]) + 2*p2*xu[0]*xu[1];

            // convert to pixel coordinate
            cv::Vec2d final;
            final[0] = f[0]*xd[0]+s*xd[1]+c[0];
            final[1] = f[1]*xd[1]+c[1];

            if (objectPoints.depth() == CV_32F)
            {
                xpf[i] = final;
            }
            else
            {
                xpd[i] = final;
            }
            /*xpd[i][0] = f[0]*xd[0]+s*xd[1]+c[0];
              xpd[i][1] = f[1]*xd[1]+c[1];*/

            if (jacobian.needed())
            {
                double dXcdR_a[] = {Xw[0],Xw[1],Xw[2],0,0,0,0,0,0,
                    0,0,0,Xw[0],Xw[1],Xw[2],0,0,0,
                    0,0,0,0,0,0,Xw[0],Xw[1],Xw[2]};
                cv::Matx<double,3, 9> dXcdR(dXcdR_a);
                cv::Matx33d dXcdom = dXcdR * dRdom.t();
                double r_1 = 1.0/norm(Xc);
                double r_3 = pow(r_1,3);
                cv::Matx33d dXsdXc(r_1-Xc[0]*Xc[0]*r_3, -(Xc[0]*Xc[1])*r_3, -(Xc[0]*Xc[2])*r_3,
                        -(Xc[0]*Xc[1])*r_3, r_1-Xc[1]*Xc[1]*r_3, -(Xc[1]*Xc[2])*r_3,
                        -(Xc[0]*Xc[2])*r_3, -(Xc[1]*Xc[2])*r_3, r_1-Xc[2]*Xc[2]*r_3);
                cv::Matx23d dxudXs(1/(Xs[2]+xi),    0,    -Xs[0]/(Xs[2]+xi)/(Xs[2]+xi),
                        0,    1/(Xs[2]+xi),    -Xs[1]/(Xs[2]+xi)/(Xs[2]+xi));
                // pre-compute some reusable things
                double temp1 = 2*k1*xu[0] + 4*k2*xu[0]*r2;
                double temp2 = 2*k1*xu[1] + 4*k2*xu[1]*r2;
                cv::Matx22d dxddxu(k2*r4+6*p2*xu[0]+2*p1*xu[1]+xu[0]*temp1+k1*r2+1,    2*p1*xu[0]+2*p2*xu[1]+xu[0]*temp2,
                        2*p1*xu[0]+2*p2*xu[1]+xu[1]*temp1,    k2*r4+2*p2*xu[0]+6*p1*xu[1]+xu[1]*temp2+k1*r2+1);
                cv::Matx22d dxpddxd(f[0], s,
                        0, f[1]);
                cv::Matx23d dxpddXc = dxpddxd * dxddxu * dxudXs * dXsdXc;

                // derivative of xpd respect to om
                cv::Matx23d dxpddom = dxpddXc * dXcdom;
                cv::Matx33d dXcdT(1.0,0.0,0.0,
                        0.0,1.0,0.0,
                        0.0,0.0,1.0);
                // derivative of xpd respect to T

                cv::Matx23d dxpddT = dxpddXc * dXcdT;
                cv::Matx21d dxudxi(-Xs[0]/(Xs[2]+xi)/(Xs[2]+xi),
                        -Xs[1]/(Xs[2]+xi)/(Xs[2]+xi));

                // derivative of xpd respect to xi
                cv::Matx21d dxpddxi = dxpddxd * dxddxu * dxudxi;
                cv::Matx<double,2,4> dxddkp(xu[0]*r2, xu[0]*r4, 2*xu[0]*xu[1], r2+2*xu[0]*xu[0],
                        xu[1]*r2, xu[1]*r4, r2+2*xu[1]*xu[1], 2*xu[0]*xu[1]);

                // derivative of xpd respect to kp
                cv::Matx<double,2,4> dxpddkp = dxpddxd * dxddkp;

                // derivative of xpd respect to f
                cv::Matx22d dxpddf(xd[0], 0,
                        0, xd[1]);

                // derivative of xpd respect to c
                cv::Matx22d dxpddc(1, 0,
                        0, 1);

                Jn[0].dom = dxpddom.row(0);
                Jn[1].dom = dxpddom.row(1);
                Jn[0].dT = dxpddT.row(0);
                Jn[1].dT = dxpddT.row(1);
                Jn[0].dkp = dxpddkp.row(0);
                Jn[1].dkp = dxpddkp.row(1);
                Jn[0].dxi = dxpddxi(0,0);
                Jn[1].dxi = dxpddxi(1,0);
                Jn[0].df = dxpddf.row(0);
                Jn[1].df = dxpddf.row(1);
                Jn[0].dc = dxpddc.row(0);
                Jn[1].dc = dxpddc.row(1);
                Jn[0].ds = xd[1];
                Jn[1].ds = 0;
                Jn += 2;
            }
        }
    }
    std::vector<cv::Point3f> Marker::get3DPoints(float msize)
    {
        float halfSize = msize / 2.f;
        //        std::vector<cv::Point3f>  res(4);
        //        res[0]=cv::Point3f(-halfSize, halfSize, 0);
        //        res[1]=cv::Point3f(halfSize, halfSize, 0);
        //        res[2]=cv::Point3f(halfSize,-halfSize, 0);
        //        res[3]=cv::Point3f(-halfSize,- halfSize, 0);
        //        return res;
        return {cv::Point3f(-halfSize, halfSize, 0),cv::Point3f(halfSize, halfSize, 0),cv::Point3f(halfSize,-halfSize, 0),cv::Point3f(-halfSize, -halfSize, 0)};

    }

    void Marker::rotateXAxis( cv::Mat& rotation)
    {
        cv::Mat R(3, 3, CV_32F);
        cv::Rodrigues(rotation, R);
        // create a rotation matrix for x axis
        cv::Mat RX = cv::Mat::eye(3, 3, CV_32F);
        const float angleRad = 3.14159265359f / 2.f;
        RX.at<float>(1, 1) = cos(angleRad);
        RX.at<float>(1, 2) = -sin(angleRad);
        RX.at<float>(2, 1) = sin(angleRad);
        RX.at<float>(2, 2) = cos(angleRad);
        // now multiply
        R = R * RX;
        // finally, the the rodrigues back
        Rodrigues(R, rotation);
    }

    /**
    */
    cv::Point2f Marker::getCenter() const
    {
        cv::Point2f cent(0, 0);
        for (size_t i = 0; i < size(); i++)
        {
            cent.x += (*this)[i].x;
            cent.y += (*this)[i].y;
        }
        cent.x /= float(size());
        cent.y /= float(size());
        return cent;
    }

    /**
    */
    float Marker::getArea() const
    {
        assert(size() == 4);
        // use the cross products
        cv::Point2f v01 = (*this)[1] - (*this)[0];
        cv::Point2f v03 = (*this)[3] - (*this)[0];
        float area1 = fabs(v01.x * v03.y - v01.y * v03.x);
        cv::Point2f v21 = (*this)[1] - (*this)[2];
        cv::Point2f v23 = (*this)[3] - (*this)[2];
        float area2 = fabs(v21.x * v23.y - v21.y * v23.x);
        return (area2 + area1) / 2.f;
    }
    /**
    */
    float Marker::getPerimeter() const
    {
        assert(size() == 4);
        float sum = 0;
        for (int i = 0; i < 4; i++)
            sum += static_cast<float>(norm((*this)[i] - (*this)[(i + 1) % 4]));
        return sum;
    }


    // saves to a binary stream
    void Marker::toStream(std::ostream& str) const
    {
        assert(Rvec.type() == CV_32F && Tvec.type() == CV_32F);
        str.write((char*)&id, sizeof(id));
        str.write((char*)&ssize, sizeof(ssize));
        str.write((char*)Rvec.ptr<float>(0), 3 * sizeof(float));
        str.write((char*)Tvec.ptr<float>(0), 3 * sizeof(float));
        // write the 2d points
        uint32_t np = static_cast<uint32_t> (size());
        str.write((char*)&np, sizeof(np));
        for (size_t i = 0; i < size(); i++)
            str.write((char*)&at(i), sizeof(cv::Point2f));
        //write the additional info
        uint32_t s=dict_info.size();
        str.write((char*)&s, sizeof(s));
        str.write((char*)&dict_info[0], dict_info.size());
        s=contourPoints.size();
        str.write((char*)&s, sizeof(s));
        str.write((char*)&contourPoints[0], contourPoints.size()*sizeof(contourPoints[0]));

    }
    // reads from a binary stream
    void Marker::fromStream(std::istream& str)
    {
        Rvec.create(1, 3, CV_32F);
        Tvec.create(1, 3, CV_32F);
        str.read((char*)&id, sizeof(id));
        str.read((char*)&ssize, sizeof(ssize));
        str.read((char*)Rvec.ptr<float>(0), 3 * sizeof(float));
        str.read((char*)Tvec.ptr<float>(0), 3 * sizeof(float));
        uint32_t np;
        str.read((char*)&np, sizeof(np));
        resize(np);
        for (size_t i = 0; i < size(); i++)
            str.read((char*)&(*this)[i], sizeof(cv::Point2f));
        //read the additional info
        uint32_t s;
        str.read((char*)&s, sizeof(s));
        dict_info.resize(s);
        str.read((char*)&dict_info[0], dict_info.size());
        str.read((char*)&s, sizeof(s));
        contourPoints.resize(s);
        str.read((char*)&contourPoints[0], contourPoints.size()*sizeof(contourPoints[0]));
    }


}
