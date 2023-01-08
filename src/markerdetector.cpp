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

#include "markerdetector.h"
#include "cameraparameters.h"
#include "markerlabeler.h"
#include "timers.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <fstream>
#include <iostream>
#include <valarray>
#include <chrono>
#include <thread>
#include "debug.h"

#ifdef _DEBUG
#include <opencv2/highgui/highgui.hpp>
#endif
using namespace std;
using namespace cv;

namespace aruco
{
/************************************
     *
     *
     *
     *
     ************************************/
MarkerDetector::MarkerDetector()
{
    markerIdDetector = aruco::MarkerLabeler::create(Dictionary::ALL_DICTS);
    setDetectionMode(DM_NORMAL);

    double vvf = 10;
    auto markerWarpSize=getMarkerWarpSize();
    vK = cv::Matx33d(vvf, 0, markerWarpSize/2, 0., vvf, markerWarpSize/2, 0., 0., 1.);
    vKinv = vK.inv();
    int cst = vK(0,2)*2;
    pointsRes[0] = Point2d(0, 0);
    pointsRes[1] = Point2d(2*vK(0,2), 0);
    pointsRes[2] = Point2d(2*vK(0,2), 2*vK(1,2));
    pointsRes[3] = Point2d(0, 2*vK(1,2));

    for(int s=0;s<4;s++)
        stereographique(pointsRes[s], xi0, vKinv, pointsRes3d[s]);

    //Pre computing the spherical coordinates of the destination image
    dstSphere.create(cst, cst, CV_64FC3);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < cst; ++i)
    {
        for(int j = 0; j < cst; ++j)
        {
            cv::Point3d temp;
            stereographique(cv::Point(i,j), xi0, vKinv, temp);
            dstSphere.at<cv::Vec3d>(i,j)[0] = temp.x;
            dstSphere.at<cv::Vec3d>(i,j)[1] = temp.y;
            dstSphere.at<cv::Vec3d>(i,j)[2] = temp.z;
        }
    }
}
/************************************
     *
     *
     *
     *
     ************************************/
MarkerDetector::MarkerDetector(int dict_type, float error_correction_rate ){
    setDictionary(dict_type,error_correction_rate);
    setDetectionMode(DM_NORMAL);

    double vvf = 10;
    auto markerWarpSize=getMarkerWarpSize();
    vK = cv::Matx33d(vvf, 0, markerWarpSize/2, 0., vvf, markerWarpSize/2, 0., 0., 1.);
    vKinv = vK.inv();
    int cst = vK(0,2)*2;
    pointsRes[0] = Point2d(0, 0);
    pointsRes[1] = Point2d(2*vK(0,2), 0);
    pointsRes[2] = Point2d(2*vK(0,2), 2*vK(1,2));
    pointsRes[3] = Point2d(0, 2*vK(1,2));

    for(int s=0;s<4;s++)
        stereographique(pointsRes[s], xi0, vKinv, pointsRes3d[s]);

    //Pre computing the spherical coordinates of the destination image
    dstSphere.create(cst, cst, CV_64FC3);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < cst; ++i)
    {
        for(int j = 0; j < cst; ++j)
        {
            cv::Point3d temp;
            stereographique(cv::Point(i,j), xi0, vKinv, temp);
            dstSphere.at<cv::Vec3d>(i,j)[0] = temp.x;
            dstSphere.at<cv::Vec3d>(i,j)[1] = temp.y;
            dstSphere.at<cv::Vec3d>(i,j)[2] = temp.z;
        }
    }
}
/************************************
     *
     *
     *
     *
     ************************************/
MarkerDetector::MarkerDetector(std::string dict_type, float error_correction_rate ){
    setDictionary(dict_type,error_correction_rate);
    setDetectionMode(DM_NORMAL);

    double vvf = 10;
    auto markerWarpSize=getMarkerWarpSize();
    vK = cv::Matx33d(vvf, 0, markerWarpSize/2, 0., vvf, markerWarpSize/2, 0., 0., 1.);
    vKinv = vK.inv();
    int cst = vK(0,2)*2;
    pointsRes[0] = Point2d(0, 0);
    pointsRes[1] = Point2d(2*vK(0,2), 0);
    pointsRes[2] = Point2d(2*vK(0,2), 2*vK(1,2));
    pointsRes[3] = Point2d(0, 2*vK(1,2));

    for(int s=0;s<4;s++)
        stereographique(pointsRes[s], xi0, vKinv, pointsRes3d[s]);

    //Pre computing the spherical coordinates of the destination image
    dstSphere.create(cst, cst, CV_64FC3);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < cst; ++i)
    {
        for(int j = 0; j < cst; ++j)
        {
            cv::Point3d temp;
            stereographique(cv::Point(i,j), xi0, vKinv, temp);
            dstSphere.at<cv::Vec3d>(i,j)[0] = temp.x;
            dstSphere.at<cv::Vec3d>(i,j)[1] = temp.y;
            dstSphere.at<cv::Vec3d>(i,j)[2] = temp.z;
        }
    }
}
/************************************
     *
     *
     *
     *
     ************************************/

MarkerDetector::~MarkerDetector()
{
}

void MarkerDetector::setParameters(const Params &params){
    _params=params;
    setDictionary(_params.dictionary,_params.error_correction_rate);
}

/************************************
     *
     *
     *
     *
     ************************************/
void MarkerDetector::setDetectionMode( DetectionMode dm,float minMarkerSize){
    _params.setDetectionMode(dm,minMarkerSize);
}

DetectionMode MarkerDetector::getDetectionMode( ){
return _params.detectMode;
}



/************************************
     *
     *
     *
     *
     ************************************/

std::vector<aruco::Marker> MarkerDetector::detect(const cv::Mat& input)
{
    std::vector<Marker> detectedMarkers;
    detect(input, detectedMarkers);
    return detectedMarkers;
}

std::vector<aruco::Marker> MarkerDetector::detect(const cv::Mat& input, const CameraParameters& camParams,
                                                  float markerSizeMeters,
                                                  bool setYPerperdicular)
{
    std::vector<Marker> detectedMarkers;
    detect(input, detectedMarkers, camParams, markerSizeMeters, setYPerperdicular);
    return detectedMarkers;
}

/************************************
     *
     *
     *
     *
     ************************************/
void MarkerDetector::detect(const cv::Mat& input, std::vector<Marker>& detectedMarkers, CameraParameters camParams,
                            float markerSizeMeters, bool setYPerpendicular)
{
    if (camParams.CamSize != input.size() && camParams.isValid() && markerSizeMeters > 0)
    {
        // must resize camera parameters if we want to compute properly marker poses
        CameraParameters cp_aux = camParams;
        cp_aux.resize(input.size());
        detect(input, detectedMarkers, cp_aux.xi, cp_aux.CameraMatrix, cp_aux.Distorsion, markerSizeMeters, setYPerpendicular);
    }
    else
    {
        detect(input, detectedMarkers, camParams.xi, camParams.CameraMatrix, camParams.Distorsion, markerSizeMeters,setYPerpendicular);
    }
}
int MarkerDetector::getMarkerWarpSize(){

    auto bis=markerIdDetector->getBestInputSize();
    if ( bis!=-1) return bis;

    int ndiv= markerIdDetector->getNSubdivisions();
    if (ndiv==-1) ndiv=7;//set any possible value(it is used for non dictionary based labelers)
    return _params.markerWarpPixSize*ndiv;//this is the minimum size that the smallest marker will have

}





void MarkerDetector::buildPyramid(vector<cv::Mat> &ImagePyramid,const cv::Mat &grey,int minSize){
    //determine numbre of pyramid images
    int npyrimg=1;
    cv::Size imgpsize=grey.size();
    while (imgpsize.width > minSize) { imgpsize=cv::Size(imgpsize.width/_params.pyrfactor,imgpsize.height/_params.pyrfactor) ;npyrimg++;}

    ImagePyramid.resize(npyrimg);
    imagePyramid[0]=grey;
    //now, create pyramid images
    imgpsize=grey.size();
    for(int i=1;i<npyrimg;i++){
            cv::Size nsize( ImagePyramid[i-1].cols/_params.pyrfactor,ImagePyramid[i-1].rows/_params.pyrfactor);
            cv::resize(ImagePyramid[i-1],ImagePyramid[i],nsize);
    }
}



/**************************************************
     *
     */

vector< MarkerDetector::MarkerCandidate> MarkerDetector::thresholdAndDetectRectangles(const cv::Mat  & input, int thres_param1,int thres_param2,bool enclosed,cv::Mat &auxThresImage){
     // ensure that _thresParam1%2==1
    ScopedTimerEvents tev("hafc ");
    if (thres_param1 < 3) thres_param1 = 3;
    else if (((int)thres_param1) % 2 != 1) thres_param1 = (int)(thres_param1 + 1);

    int enclosedOffset=-1;
    cv::Mat auxImage;
    //if ( !erode)
        auxImage=auxThresImage;
    if (_params.thresMethod==THRES_AUTO_FIXED){
            cv::threshold(input, auxImage, static_cast<int>(thres_param2),255,THRESH_BINARY_INV);
            if(enclosed){
               cv::Mat aux1;
               enclosedOffset=int(std::max(3.0, 3./1920. * float(auxImage.cols) ));
               if (enclosedOffset%2==0) enclosedOffset++;
               cv::erode(auxImage,aux1, getStructuringElement( MORPH_CROSS,cv::Size( enclosedOffset, enclosedOffset ),cv::Point( enclosedOffset/2,enclosedOffset/2 ) ));
               cv::bitwise_xor(aux1,auxImage,auxImage);
               tev.add("erode");
            }
    }
    else {
        cv::adaptiveThreshold(input, auxImage, 255., ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, static_cast<int>(thres_param1), static_cast<int>(thres_param2));
        enclosedOffset=thres_param1;
     }
    tev.add("thres");

//     //

    vector<MarkerCandidate> MarkerCanditates;
    // calcualte the min_max contour sizes
        int thisImageMinSize=int( 3.5*float(_params.lowResMarkerSize));
    //if image is eroded, minSize must be adapted
     std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
      cv::findContours(auxThresImage, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
       tev.add("find-cont");
     vector<Point> approxCurve;
//#define _aruco_debug_detectrectangles
//#ifdef _aruco_debug_detectrectangles
     cv::Mat simage;
     cv::cvtColor(input,simage,cv::COLOR_GRAY2BGR);
//#endif

    /// for each contour, analyze if it is a paralelepiped likely to be the marker
    for (unsigned int i = 0; i < contours.size(); i++)
    {
#ifdef _aruco_debug_detectrectangles
            drawContour(simage, contours[i], Scalar(125, 125, 255) );
#endif
        // check it is a possible element by first checking that is is large enough
        if (thisImageMinSize < int(contours[i].size())  )
        {
            // can approximate to a convex rect?
            cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * 0.05, true);

            if (approxCurve.size() == 4 && cv::isContourConvex(approxCurve))
            {
#ifdef _aruco_debug_detectrectangles
                drawApproxCurve(simage, approxCurve, Scalar(255, 0, 255),1);
#endif
                // ensure that the   distace between consecutive points is large enough
                float minDist = std::numeric_limits<float>::max();
                for (int j = 0; j < 4; j++)
                {
                    float d = cv::norm(approxCurve[j]-approxCurve[(j + 1) % 4] );
                    if (d < minDist) minDist = d;
                }

                    // add the points
                    MarkerCanditates.push_back(MarkerCandidate());
                    for (int j = 0; j < 4; j++)
                        MarkerCanditates.back().push_back( Point2f(static_cast<float>(approxCurve[j].x),static_cast<float>(approxCurve[j].y)));
                    //now, if it is eroded, must enalrge 1 bit the corners to go to the real location
                    //for each opposite pair, take the line joining them and move one pixel apart
                    //ideally, Bresenham's algorithm should be used
                    if (enclosed)  {
                            enlargeMarkerCandidate(MarkerCanditates.back(),float(enclosedOffset)/2.);
                    }
#ifdef _aruco_debug_detectrectangles
                    MarkerCanditates.back().draw(simage,Scalar(255, 255, 0),1,false);
#endif
                    MarkerCanditates.back().contour=contours[i];
            }

        }
    }
#ifdef _aruco_debug_detectrectangles
    cv::imshow("contours",simage);
#endif
    return MarkerCanditates;
}


void MarkerDetector::thresholdAndDetectRectangles_thread( ){
    while(true){
        //stringstream sstr;sstr<<"thread-"<<std::this_thread::get_id()<<" "<<  std::chrono::high_resolution_clock::now().time_since_epoch().count();
//        ScopedTimerEvents tev(sstr.str());
        bool enclosed=false;
        auto tad=_tasks.pop();
//        tev.add("pop");
        if (tad.task==EXIT_TASK) return;
        else   if (tad.task==ENCLOSE_TASK) enclosed=true;
        _vcandidates[tad.outIdx]=thresholdAndDetectRectangles(_thres_Images[tad.inIdx],tad.param1,tad.param2,enclosed,_thres_Images[tad.outIdx]);
  //      tev.add("thres param: "+to_string(tad.param1));

    };
}

vector<aruco::MarkerDetector::MarkerCandidate> MarkerDetector::thresholdAndDetectRectangles(const cv::Mat &image ){

    // compute the different values of param1

    int adaptiveWindowSize=_params.AdaptiveThresWindowSize;
    if (_params.AdaptiveThresWindowSize==-1)
        adaptiveWindowSize=max(int(3),int(15*float(image.cols)/1920.));
    if(adaptiveWindowSize%2==0)adaptiveWindowSize++;


    vector<int> p1_values;
    for (int i = static_cast<int>(std::max(3., adaptiveWindowSize- 2. * _params.AdaptiveThresWindowSize_range)); i <= adaptiveWindowSize + 2 * _params.AdaptiveThresWindowSize_range; i += 2)
        p1_values.push_back(i);

    size_t nimages=p1_values.size();
    _tooNearDistance=p1_values.back();
    _vcandidates.resize(nimages);
    _thres_Images.resize(nimages+1);
    _thres_Images.back()=image; //add at the end the original image
    //first, thresholded images
    ThresAndDetectRectTASK tad;
    vector<ThresAndDetectRectTASK> vtad;

    ThreadTasks task=THRESHOLD_TASK;
    if (_params.enclosedMarker) task=ENCLOSE_TASK;
    for (size_t i = 0; i < p1_values.size(); i++){
        tad.inIdx=int(_thres_Images.size()-1);
        tad.param1=p1_values[i];
        tad.param2=_params.ThresHold;
        tad.outIdx=i;
        tad.task=task;
        _tasks.push(tad);
        vtad.push_back(tad);
    }

    //reserve images
    for(size_t i=0;i<nimages;i++)
        _thres_Images[i].create( image.size(),CV_8UC1);


    //how many threads will be used?
    int nthreads=0;
    if (_params.maxThreads<=0)//if allowed to use all , take max()-1, since the buildpyramid must be working at this moment
        nthreads= std::thread::hardware_concurrency()-1;
    else nthreads= max(1,_params.maxThreads-1);

    tad.task=EXIT_TASK;
    for(int i=0;i<nthreads;i++) _tasks.push(tad);

    if (nthreads==1){//no threads
        ScopeTimer Timer("non-parallel");
        thresholdAndDetectRectangles_thread ( );
    }
    else{//parallell mode
        //add the final task END
        ScopeTimer Timer("parallel");

        //run the tasks (in parallel)
        vector<std::thread> threads;
        for(int i=0;i<nthreads;i++ )
            threads.push_back( std::thread(&MarkerDetector::thresholdAndDetectRectangles_thread, this));
        //wait for them to finish
        for(auto &th:threads) th.join();
    }
    vector<MarkerCandidate> joined;
    joinVectors( _vcandidates,joined);
    return joined;

}

vector< MarkerDetector::MarkerCandidate> MarkerDetector::prefilterCandidates(      vector<aruco::MarkerDetector::MarkerCandidate> &MarkerCanditates,cv::Size imgSize){
    /////////////////////////////////////////////////////////////////////////////////////
    /// CANDIDATE PREFILTERING- Merge and Remove candidates so that only reliable ones are returned
    //////////////////////////////////////////////////////////////////////////////////
    /// sort the points in anti-clockwise order
    valarray<bool> swapped(false, MarkerCanditates.size());  // used later
    for (unsigned int i = 0; i < MarkerCanditates.size(); i++)
    {
        // trace a line between the first and second point.
        // if the thrid point is at the right side, then the points are anti-clockwise
        double dx1 = MarkerCanditates[i][1].x - MarkerCanditates[i][0].x;
        double dy1 = MarkerCanditates[i][1].y - MarkerCanditates[i][0].y;
        double dx2 = MarkerCanditates[i][2].x - MarkerCanditates[i][0].x;
        double dy2 = MarkerCanditates[i][2].y - MarkerCanditates[i][0].y;
        double o = (dx1 * dy2) - (dy1 * dx2);

        if (o < 0.0)
        {  // if the third point is in the left side, then sort in anti-clockwise order
            swap(MarkerCanditates[i][1], MarkerCanditates[i][3]);
            swapped[i] = true;
        }
    }
    /// remove these elements which corners are too close to each other
    // first detect candidates to be removed
    vector<pair<int, int>> TooNearCandidates;
    for (unsigned int i = 0; i < MarkerCanditates.size(); i++)
    {
        // calculate the average distance of each corner to the nearest corner of the other marker candidate
        for (unsigned int j = i + 1; j < MarkerCanditates.size(); j++)
        {
            valarray<float> vdist(4);
            for (int c = 0; c < 4; c++)
                vdist[c] =cv::norm( MarkerCanditates[i][c] -MarkerCanditates[j][c]);
            // if distance is too small
            if (vdist[0] < _tooNearDistance && vdist[1] < _tooNearDistance && vdist[2] < _tooNearDistance && vdist[3] < _tooNearDistance)
                TooNearCandidates.push_back(pair<int, int>(i, j));
        }
    }

    /// mark for removal the element of  the pair with smaller perimeter
    vector<bool> toRemove( MarkerCanditates.size(),false);
    for (unsigned int i = 0; i < TooNearCandidates.size(); i++)
    {
        if (perimeter(MarkerCanditates[TooNearCandidates[i].first])  > perimeter(MarkerCanditates[TooNearCandidates[i].second]))
            toRemove[TooNearCandidates[i].second] = true;
        else
            toRemove[TooNearCandidates[i].first] = true;
    }

    /// find these too near borders  and remove them
    // remove markers with corners too near the image limits
    int borderDistThresX = static_cast<int>(_params.borderDistThres * float(imgSize.width));
    int borderDistThresY =  static_cast<int>(_params.borderDistThres * float(imgSize.height));
    for (size_t i = 0; i < MarkerCanditates.size(); i++)
    {
        // delete if any of the corners is too near image border
        for (size_t c = 0; c < MarkerCanditates[i].size(); c++)
        {
            if (MarkerCanditates[i][c].x < borderDistThresX || MarkerCanditates[i][c].y < borderDistThresY
                    || MarkerCanditates[i][c].x > imgSize.width - borderDistThresX
                    || MarkerCanditates[i][c].y > imgSize.height - borderDistThresY)
            {
                toRemove[i] = true;
            }
        }
    }

    //move to output only valid ones
    vector<MarkerCandidate> finalCandidates;
    finalCandidates.reserve(MarkerCanditates.size());
    for(size_t i=0;i<MarkerCanditates.size();i++)
        if (!toRemove[i])finalCandidates.push_back(MarkerCanditates[i]);
    return finalCandidates;
}


void addToImageHist(cv::Mat &im,std::vector<float>&hist)
{
    for(int y=0;y<im.rows;y++){
        uchar*ptr=im.ptr<uchar>(y);
        for(int x=0;x<im.cols;x++)
           hist[ptr[x]]++;
    }
}

int Otsu(std::vector<float> &hist){
    float sum=0,invsum;
    for(auto c:hist) sum+=c;
    invsum=1./sum;
    for(auto &c:hist) c*=invsum;

    float maxVar=0;
    int bestT=-1;
    for(int t=1;t<256;t++){
        float w0=0,w1=0,mean0=0,mean1=0;
        for(int v=0;v<t;v++) {
            w0+=hist[v];
            mean0+=float(v)*hist[v];
        }
        for(int v=t;v<256;v++) {
            w1+=hist[v];
            mean1+=hist[v]*float(v);
        }
        if (w0>1e-4 && w1>1e-4){
            mean0/=w0;
            mean1/=w1;
            float var=w0*w1*(mean0-mean1)*(mean0-mean1);
            //cout<<t<<" : "<<var<<"|"<<w0<<" "<<w1<<" "<<mean0<<" "<<mean1<<endl;
            if (var>maxVar){
                maxVar=var;
                bestT=t;
            }
        }

    }
    return bestT;
}
/************************************
     * Main detection function. Performs all steps
     ************************************/
void MarkerDetector::detect(const cv::Mat& input, vector<Marker>& detectedMarkers, double xi, Mat camMatrix, Mat distCoeff,
                            float markerSizeMeters, bool setYPerpendicular)
{
    // clear input data
    detectedMarkers.clear();
    _vcandidates.clear();
    _candidates.clear();
    ScopedTimerEvents Timer("detect");


    // it must be a 3 channel image
    if (input.type() == CV_8UC3)
        cv::cvtColor(input,grey,cv::COLOR_BGR2GRAY);
    //  convertToGray(input, grey);
    else grey = input;
    Timer.add("ConvertGrey");

    //////////////////////////////////////////////////////////////////////
    ///CREATE LOW RESOLUTION IMAGE IN WHICH MARKERS WILL BE DETECTED
    //////////////////////////////////////////////////////////////////////
    float ResizeFactor=1;
    //use the minimum and markerWarpSize to determine the optimal image size on which to do rectangle detection
    cv::Mat imgToBeThresHolded ;
    cv::Size maxImageSize=grey.size();
    auto minpixsize=getMinMarkerSizePix(input.size());//min pixel size of the marker in the original image
    if ( _params.lowResMarkerSize<minpixsize  ){
        ResizeFactor= float(_params.lowResMarkerSize)/float(minpixsize  ) ;
        if (ResizeFactor<0.9){//do not waste time if smaller than this
            _debug_msg("Scale factor="<<ResizeFactor,1);
            maxImageSize.width=float(grey.cols)*ResizeFactor+0.5;
            maxImageSize.height=float(grey.rows)*ResizeFactor+0.5;
            if ( maxImageSize.width%2!=0) maxImageSize.width++;
            if ( maxImageSize.height%2!=0) maxImageSize.height++;
            cv::resize(grey,imgToBeThresHolded,maxImageSize,0,0,cv::INTER_NEAREST);
            //                cv::resize(grey,imgToBeThresHolded,maxImageSize,0,0,cv::INTER_LINEAR);
        }
    }

    if(imgToBeThresHolded.empty())//if not set in previous step, add original now
        imgToBeThresHolded=grey;

    Timer.add("CreateImageToTheshold");
    bool needPyramid=  true;//ResizeFactor< 1/_params.pyrfactor;//only use pyramid if working on a big image.
    std::thread buildPyramidThread;
    if(needPyramid){
        if (_params.maxThreads>1)
            buildPyramidThread=std::thread([&]{buildPyramid(imagePyramid,grey ,2*getMarkerWarpSize());});
        else buildPyramid(imagePyramid,grey,2*getMarkerWarpSize());
        Timer.add("BuildPyramid");
    }
    else{
        imagePyramid.resize(1);
        imagePyramid[0]=grey;
    }


    int nAttemptsAutoFix=0;
    bool keepLookingFor=false;
    std::vector<float> hist(256,0);
    do{
        ///////////////////////////////////////////////////////////////////////////
        /// THRESHOLD IMAGES AND DETECT INITIAL RECTANGLES
        ///////////////////////////////////////////////////////////////////////////
        vector<MarkerCandidate> MarkerCanditates;
        MarkerCanditates=thresholdAndDetectRectangles(imgToBeThresHolded );
        thres    =_thres_Images[0];

        _debug_exec(10,
                {//only executes when compiled in DEBUG mode if debug level is at least 10
                //show the thresholded images
                for (size_t i = 0; i < _thres_Images.size(); i++) {
                stringstream sstr; sstr << "thres-" << i;
                cv::namedWindow(sstr.str(),cv::WINDOW_NORMAL);
                cv::imshow(sstr.str(),_thres_Images[i]);
                }});


        Timer.add("Threshold and Detect rectangles");
        //prefilter candidates
        _debug_exec(10,//only executes when compiled in DEBUG mode if debug level is at least 10
                //show the thresholded images
                cv::Mat imrect;
                cv::cvtColor(imgToBeThresHolded,imrect,cv::COLOR_GRAY2BGR);
                for(auto m: MarkerCanditates )
                m.draw(imrect,cv::Scalar(0,245,0));
                cv::imshow("rect-nofiltered",imrect);
                );

        MarkerCanditates=prefilterCandidates(MarkerCanditates,imgToBeThresHolded.size());

        Timer.add("prefilterCandidates");

        _debug_exec(10,//only executes when compiled in DEBUG mode if debug level is at least 10
                //show the thresholded images
                cv::Mat imrect;
                cv::cvtColor(imgToBeThresHolded,imrect,cv::COLOR_GRAY2BGR);
                for(auto m: MarkerCanditates)
                m.draw(imrect,cv::Scalar(0,245,0));
                cv::imshow("rect-filtered",imrect);
                );
        //before going on, make sure the piramid is built
        if (buildPyramidThread.joinable())
            buildPyramidThread.join();


        ///////////////////////////////////////////////////////////////////////////
        /// CANDIDATE CLASSIFICATION: Decide which candidates are really markers
        ///////////////////////////////////////////////////////////////////////////

        //  Debug::setLevel(10);
        auto markerWarpSize=getMarkerWarpSize();

        detectedMarkers.clear();
        _candidates.clear();
        for(auto &b:hist) b=0;
        float desiredarea = std::pow(static_cast<float>(markerWarpSize), 2.f);


        for (size_t i = 0; i < MarkerCanditates.size(); i++)
        {
            // Find proyective homography
            cv::Mat canonicalMarker,canonicalMarkerAux;
            cv::Mat inToWarp=imgToBeThresHolded;
            MarkerCandidate points2d_pyr = MarkerCanditates[i];
            size_t imgPyrIdx = 0;
            if (needPyramid){ //TODO disable that because we need to resize the cammatrix also for the projections
                // warping is one of the most time consuming operations, especially when the region is large.
                // To reduce computing time, let us find in the image pyramid, the best configuration to save time
                // indicates how much bigger observation is wrt to desired patch
                for (size_t p = 1; p < imagePyramid.size(); p++)
                {
                    if (MarkerCanditates[i].getArea() / pow(4, p) >= desiredarea) imgPyrIdx = p;
                    else break;
                }
                inToWarp=imagePyramid[imgPyrIdx];
                //move points to the image level p
                float ratio=float(inToWarp.cols)/float(imgToBeThresHolded.cols);
                for (auto& p : points2d_pyr)  p *= ratio;//1. / pow(2, imgPyrIdx);
            }

            //Warping using spherical projection
            if (camMatrix.rows != 0)
            {
                //If the image has been resized using imagePyramid above, the cammatrix needs to be resized as well
                cv::Mat camMatrixResized;
                camMatrix.copyTo(camMatrixResized);
                if (imgPyrIdx > 0) 
                {
                    camMatrixResized.at<float>(0,0) = camMatrix.at<float>(0,0)/(imgPyrIdx*_params.pyrfactor);
                    camMatrixResized.at<float>(1,1) = camMatrix.at<float>(1,1)/(imgPyrIdx*_params.pyrfactor);
                    camMatrixResized.at<float>(0,2) = camMatrix.at<float>(0,2)/(imgPyrIdx*_params.pyrfactor);
                    camMatrixResized.at<float>(1,2) = camMatrix.at<float>(1,2)/(imgPyrIdx*_params.pyrfactor);
                }

                warpSphere( inToWarp, canonicalMarker, points2d_pyr, pointsRes3d, xi, camMatrixResized, MarkerCanditates[i].sCorners);
            }

            //Warping using perspective projection
            //warp( inToWarp, canonicalMarker,  Size(markerWarpSize,markerWarpSize),points2d_pyr);
            int id, nRotations;
            double min,Max;
            cv::minMaxIdx(canonicalMarker,&min,&Max);
            canonicalMarker.copyTo(canonicalMarkerAux);
            string additionalInfo;
            _debug_exec(10,//only executes when compiled in DEBUG mode if debug level is at least 10
                    //show the thresholded images
                    stringstream sstr;sstr<<"test-"<<i;
                    cout  <<"test"<<i<<endl;
                    //cv::namedWindow(sstr.str(),cv::WINDOW_NORMAL);
                    std::cout<<canonicalMarkerAux.size()<<std::endl;
                    cv::imshow(sstr.str(),canonicalMarkerAux);
                    cv::waitKey(0);
                    );
            if (markerIdDetector->detect(canonicalMarkerAux, id, nRotations,additionalInfo))
            {
                detectedMarkers.push_back(MarkerCanditates[i]);
                detectedMarkers.back().id = id;
                detectedMarkers.back().dict_info=additionalInfo;
                detectedMarkers.back().contourPoints=MarkerCanditates[i].contour;
                for (int mi = 0; mi < 4; mi++)
                    detectedMarkers.back().sCorners[mi] = MarkerCanditates[i].sCorners[mi];
                // sort the points so that they are always in the same order no matter the camera orientation
                std::rotate(detectedMarkers.back().begin(),
                        detectedMarkers.back().begin() + 4 - nRotations,
                        detectedMarkers.back().end());
                _debug_exec(10,//only executes when compiled in DEBUG mode if debug level is at least 10
                        //show the thresholded images
                        stringstream sstr;sstr<<"can-"<<detectedMarkers.back().id;
                        cv::namedWindow(sstr.str(),cv::WINDOW_NORMAL);
                        cv::imshow(sstr.str(),canonicalMarker);
                        cout<<"ID="<<id<<" "<< detectedMarkers.back()<<endl;
                        );
                if (_params.thresMethod==THRES_AUTO_FIXED)
                    addToImageHist(canonicalMarker,hist);
            }
            else
                _candidates.push_back(MarkerCanditates[i]);

        }
        Timer.add("Marker classification. ");
        if (detectedMarkers.size()==0 &&  _params.thresMethod==THRES_AUTO_FIXED && ++nAttemptsAutoFix < _params.NAttemptsAutoThresFix){
            _params.ThresHold=  10+ rand()%230 ;
            keepLookingFor=true;
        }
        else keepLookingFor=false;
    }while(keepLookingFor);

    //  Debug::setLevel(5);


    if (_params.thresMethod== THRES_AUTO_FIXED){
        int newThres=Otsu(hist);;
        if(newThres>0)
            _params.ThresHold=  float(newThres) ;
    }

#ifdef debug_lines
    cv::imshow("image-lines",image);
    cv::waitKey(10);
#endif
    //now, move the points to the original image (upsample corners)
    if (input.cols!=imgToBeThresHolded.cols){
        cornerUpsample(detectedMarkers,imgToBeThresHolded.size());
        Timer.add("Corner Upsample");
    }



    //////////////////////////////////////////////////////////////////////
    /// REMOVAL OF DUPLICATED
    //////////////////////////////////////////////////////////////////////

    // sort by id
    std::sort(detectedMarkers.begin(), detectedMarkers.end());

    // there might be still the case that a marker is detected twice because of the double border indicated earlier,
    // detect and remove these cases
    vector<bool> toRemove(detectedMarkers.size(), false);

    for (int i = 0; i < int(detectedMarkers.size()) - 1; i++)
    {
        for (int j = i + 1; j < int(detectedMarkers.size()) && !toRemove[i]; j++)
        {
            if (detectedMarkers[i].id == detectedMarkers[j].id && detectedMarkers[i].dict_info == detectedMarkers[j].dict_info)
            {
                // deletes the one with smaller perimeter
                if (perimeter(detectedMarkers[i]) < perimeter(detectedMarkers[j]))
                    toRemove[i] = true;
                else
                    toRemove[j] = true;
            }
        }
    }

    removeElements(detectedMarkers,toRemove);


    //////////////////////////////////////////////////////////////////////
    /// CORNER REFINEMENT IF REQUIRED
    //////////////////////////////////////////////////////////////////////
    /// refine the corner location if enclosed markers and we did not do it via upsampling
    if (detectedMarkers.size() > 0  &&  input.size()==imgToBeThresHolded.size() ){

        if (_params.cornerRefinementM==CORNER_SUBPIX  ){

            int halfwsize= 4*float(input.cols)/float(imgToBeThresHolded.cols) +0.5 ;

            vector<Point2f> Corners;
            for (unsigned int i = 0; i < detectedMarkers.size(); i++)
                for (int c = 0; c < 4; c++)
                    Corners.push_back(detectedMarkers[i][c]);
            cornerSubPix(grey, Corners, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
            // copy back
            for (unsigned int i = 0; i < detectedMarkers.size(); i++)
                for (int c = 0; c < 4; c++)
                    detectedMarkers[i][c] = Corners[i * 4 + c];

        }
        else if ( _params.cornerRefinementM==CORNER_LINES){
            //use the lines method for estimation of the corners
            for (unsigned int i = 0; i < detectedMarkers.size(); i++)
                refineCornerWithContourLines(detectedMarkers[i]);
        }
    }
    Timer.add("Corner Refinement");

    //    auto setPrecision=[](double f, double prec){
    //        int x=roundf(f*prec);
    //        return  double(x)/prec;
    //    };

    //    auto discretizeFloat=[](float v){
    //        float f=v-float(int(v));
    //        if (f<0.25) return  float(int(v));
    //        else if (f<0.75) return  float(int(v))+0.5f;
    //        else return float(int(v+0.5f));

    //    };

    //for(auto &dm:detectedMarkers){
    //    for(auto &p:dm){
    ////        p.x=setPrecision(p.x,10);
    ////        p.y=setPrecision(p.y,10);
    //        p.x=discretizeFloat(p.x);
    //        p.y=discretizeFloat(p.y);

    //    }
    //}

    //////////////////////////////////////////////////////////////////////
    /// MARKER POSE ESTIMATION
    //////////////////////////////////////////////////////////////////////

    /// detect the position of detected markers if desired
    if (camMatrix.rows != 0 && markerSizeMeters > 0)
    {
        /*for (unsigned int i = 0; i < detectedMarkers.size(); i++)
            detectedMarkers[i].calculateExtrinsics(markerSizeMeters, camMatrix, distCoeff, setYPerpendicular);*/

        for (unsigned int i = 0; i < detectedMarkers.size(); i++)
            detectedMarkers[i].calculateExtrinsicsOmni(markerSizeMeters, camMatrix, xi, distCoeff, setYPerpendicular);

        Timer.add("Pose Estimation");
    }

    //compute _markerMinSize
    float mlength=std::numeric_limits<float>::max();
    for(const auto &marker:detectedMarkers){
        float l=0;
        for(int c=0;c<4;c++)
            l+=cv::norm(marker[c]-marker[(c+1)%4]);
        if (mlength>l)mlength=l;
    }
    float  markerMinSize;
    if (mlength!=std::numeric_limits<float>::max() )
        markerMinSize=mlength/(4*std::max(input.cols,input.rows));
    else markerMinSize=0;
    if (_params.autoSize){
        _params.minSize= markerMinSize*(1-_params.ts);
    }
 }
void MarkerDetector::refineCornerWithContourLines( aruco::Marker &marker,cv::Mat camMatrix,cv::Mat distCoeff ){
    // search corners on the contour vector

      std::vector<cv::Point> &contour=marker.contourPoints;
       vector< int > cornerIndex(4,-1);
       vector<float> dist(4,std::numeric_limits<float>::max());
       for (unsigned int j = 0; j < contour.size(); j++) {
           for (unsigned int k = 0; k < 4; k++) {
               float d=  (contour[j].x-marker[k].x)*(contour[j].x-marker[k].x) +
                       (contour[j].y-marker[k].y)*(contour[j].y-marker[k].y) ;
               if (d<dist[k]){
                   cornerIndex[k] = j;
                   dist[k]=d;
               }
           }
       }

       //if any cornerIndex


       // contour pixel in inverse order or not?
       bool inverse;
       if ((cornerIndex[1] > cornerIndex[0]) && (cornerIndex[2] > cornerIndex[1] || cornerIndex[2] < cornerIndex[0]))
           inverse = false;
       else if (cornerIndex[2] > cornerIndex[1] && cornerIndex[2] < cornerIndex[0])
           inverse = false;
       else
           inverse = true;


       // get pixel vector for each line of the marker
       int inc = 1;
       if (inverse)
           inc = -1;

       // undistort contour
       vector< Point2f > contour2f;
       if(!camMatrix.empty() && !distCoeff.empty()){
       for (unsigned int i = 0; i < contour.size(); i++)
           contour2f.push_back(cv::Point2f(contour[i].x, contour[i].y));
       if (!camMatrix.empty() && !distCoeff.empty())
           cv::undistortPoints(contour2f, contour2f, camMatrix, distCoeff, cv::Mat(), camMatrix);

       }
       else {
           contour2f.reserve(contour.size());
           for(auto p:contour)
               contour2f.push_back(cv::Point2f(p.x,p.y));
       }

       vector< std::vector< cv::Point2f > > contourLines;
       contourLines.resize(4);
       for (unsigned int l = 0; l < 4; l++) {
           for (int j = (int)cornerIndex[l]; j != (int)cornerIndex[(l + 1) % 4]; j += inc) {
               if (j == (int)contour.size() && !inverse)
                   j = 0;
               else if (j == 0 && inverse)
                   j = contour.size() - 1;
               contourLines[l].push_back(contour2f[j]);
               if (j == (int)cornerIndex[(l + 1) % 4])
                   break; // this has to be added because of the previous ifs
           }
       }

       // interpolate marker lines
       vector< Point3f > lines;
       lines.resize(4);
       for (unsigned int j = 0; j < lines.size(); j++)
           interpolate2Dline(contourLines[j], lines[j]);

       // get cross points of lines
       vector< Point2f > crossPoints;
       crossPoints.resize(4);
       for (unsigned int i = 0; i < 4; i++)
           crossPoints[i] = getCrossPoint(lines[(i - 1) % 4], lines[i]);

       // distort corners again if undistortion was performed
       if (!camMatrix.empty() && !distCoeff.empty())
           distortPoints(crossPoints, crossPoints, camMatrix, distCoeff);

       // reassing points
       for (unsigned int j = 0; j < 4; j++){
          // cout<<marker[j]<<" "<<crossPoints[j]<<endl;
           marker[j] = crossPoints[j];
       }
}




//expands the corners of the candidate to reach the real locations
//used in eroded images
void MarkerDetector::enlargeMarkerCandidate(MarkerCandidate &cand,int fact){
    for(int j=0;j<2;j++){
        auto startp=j;
        auto endp=(j + 2) % 4;
        //sort so that the nearest to x is first
        if (cand[startp].x>cand[endp].x){
            swap( startp,endp);
        }
        const float _180=3.14159f ;

        const float _22=3.14159/8.f;
        const float _3_22=3.*3.14159f/8.f;
        const float _5_22=5.f*3.14159f/8.f;
        const float _7_22=7.f*3.14159f/8.f;

        int incx=0,incy=0;
        //compute the angle
        auto v1=cand[endp]-cand[startp];
        float angle=atan2( v1.y,v1.x);
        if ( _22<angle && angle<3*_22){ //a
            incx=incy=fact;
        }
        else if ( -_22<angle && angle<_22 ){//b
            incx=fact;
            incy=0;
        }
        else if ( -_3_22<angle && angle<-_22 ){//c
            incx=fact;
            incy=-fact;
        }
        else if ( -_5_22<angle && angle<-_3_22 ){//D
            incx=0;
            incy=-fact;
        }
        else if ( -_7_22<angle && angle<-_5_22 ){//E
            incx=-fact;
            incy=-fact;
        }
        else if ( (-_180<angle && angle<-_7_22)||(_7_22<angle && angle<_180) ){//f
            incx=-fact;
            incy=0;
        }
        else if ( (_5_22<angle && angle<_7_22) ){//g
            incx=-fact;
            incy=fact;
        }
        else if ( (_3_22<angle && angle<_5_22) ){//h
            incx=fact;
            incy=fact;
        }
        cand[endp].x+=incx;
        cand[endp].y+=incy;
        cand[startp].x-=incx;
        cand[startp].y-=incy;

    }
}



int  MarkerDetector::getMinMarkerSizePix(cv::Size orginput_imageSize)const{
    if (_params.minSize==-1 && _params.minSize_pix==-1)return 0;
    // calcualte the min_max contour sizes
    int maxDim=std::max(orginput_imageSize.width, orginput_imageSize.height);
    int minSize =0;
    if (_params.minSize!=-1) minSize =static_cast<float>(_params.minSize)* static_cast<float>(maxDim) ;
    if (_params.minSize_pix!=-1) minSize=std::min(_params.minSize_pix, minSize );
    return minSize ;

}
/************************************
     *
     *
     *
     *
     ************************************/
bool MarkerDetector::warp(Mat& in, Mat& out, Size size, vector<Point2f> points)
{
    if (points.size() != 4)
        throw cv::Exception(9001, "point.size()!=4", "MarkerDetector::warp", __FILE__, __LINE__);
    // obtain the perspective transform
    Point2f pointsRes[4], pointsIn[4];
    for (int i = 0; i < 4; i++)
        pointsIn[i] = points[i];
    pointsRes[0] = (Point2f(0, 0));
    pointsRes[1] = Point2f(static_cast<float>(size.width - 1), 0.f);
    pointsRes[2] = Point2f(static_cast<float>(size.width - 1), static_cast<float>(size.height - 1));
    pointsRes[3] = Point2f(0.f, static_cast<float>(size.height - 1));

    Mat M = getPerspectiveTransform(pointsIn, pointsRes);
    cv::warpPerspective(in, out, M, size, cv::INTER_LINEAR);
  //  cv::warpPerspective(in, out, M, size, cv::INTER_NEAREST);
  return true;
}

void MarkerDetector::warpPerspectiveSphere(cv::Mat& in, cv::Mat& out, cv::Mat& Ms, double xi, cv::Matx33d rK, cv::Matx33d vK, int method)
{
    //cv::Matx33d vKinv = vK.inv();

    cv::Vec2d rf, rc;
    rf = cv::Vec2d(rK(0,0),rK(1,1));
    rc = cv::Vec2d(rK(0,2),rK(1,2));

    int cst = vK(0,2)*2;

    out = cv::Mat::zeros(cst, cst, CV_8UC1);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < cst; ++i)
    {
        for(int j = 0; j < cst; ++j)
        {
            cv::Mat a = Ms*cv::Mat(dstSphere.at<cv::Vec3d>(i,j));
            double rho = cv::norm(a);
            auto x = cv::Vec3d(a.at<double>(0,0)/(a.at<double>(2,0) + rho*xi), a.at<double>(1,0)/(a.at<double>(2,0) + rho*xi), 1);

            if (auto u = cv::Vec2d(rf[0]*x[0]+rc[0], rf[1]*x[1]+rc[1]); (u[0]>=0 && u[1]>=0 ) && (u[0]<in.cols && u[1]<in.rows))
            {
                //Bilinear Interpolation
                if( method == INTER_LINEAR )
                {
                    int x1 = std::ceil(u[0]), x2 = std::floor(u[0]);
                    int y1 = std::ceil(u[1]), y2 = std::floor(u[1]);

                    unsigned char q11 = in.at<uchar>(y1,x1);
                    unsigned char q12 = in.at<uchar>(y2,x1);
                    unsigned char q21 = in.at<uchar>(y1,x2);
                    unsigned char q22 = in.at<uchar>(y2,x2);

                    out.at<uchar>(j, i) = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, u[0], u[1]);
                }
                //Nearest neighbor Interpolation
                else if( method == INTER_NEAREST)
                {
                    u = cv::Vec2d(std::round(rf[0]*x[0]+rc[0]), std::round(rf[1]*x[1]+rc[1]));
                    out.at<uchar>(j, i) = in.at<uchar>(u[1],u[0]);
                }
            }
        }
    }
}

std::string MarkerDetector::type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

bool MarkerDetector::warpSphere(Mat& in, Mat& out, vector<Point2f> points4Corners, Point3d pointsRes3d[], double xi, cv::InputArray realCamMatrix, cv::Point3d pointsIn3d[])
{
    if (points4Corners.size() != 4)
        throw cv::Exception(9001, "point.size()!=4", "MarkerDetector::warp", __FILE__, __LINE__);

    Matx33d rK = realCamMatrix.getMat();
    Matx33d rKinv = rK.inv();

    //std::cout<<"points4Corners: "<<points4Corners<<std::endl;
    for(int i=0;i<4;i++)
        stereographique(points4Corners[i], xi, rKinv, pointsIn3d[i]);

    Mat M = getPerspectiveTransformSphere(pointsRes3d, pointsIn3d, DECOMP_LU);
    warpPerspectiveSphere(in, out, M, xi, rK, vK, INTER_LINEAR);

    return true;
}
void MarkerDetector::stereographique(const Point2d& src, double xi,  Matx33d Kinv, Point3d& dst)
{
    Vec3d p;
    p[0] = Kinv(0,0) * src.x + Kinv(0,2);
    p[1] = Kinv(1,1) * src.y + Kinv(1,2);
    p[2] = 1;
    double xi2 = 1-xi*xi;
    if (double r = p[0]*p[0] + p[1]*p[1], alpha = 1+xi2*r; alpha>0)
    {
        double beta = (xi+std::sqrt(alpha))/(r+1);
        dst = cv::Point3d(beta*p[0], beta*p[1], beta-xi);
    }
    /*else
    std::cout<<"alpha: "<<(1+xi2*(p[0]*p[0] + p[1]*p[1]))<<" "<<p[0]<<" "<<p[1]<<" "<<xi<<std::endl;*/
}

Mat MarkerDetector::getPerspectiveTransformSphere(const Point3d src[], const Point3d dst[], int solveMethod)
{
    Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.ptr());
    double a[8][8], b[8];
    Mat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);

    for(int i = 0; i < 4; ++i)
    {
        a[i+4][0] =  src[i].x*dst[i].z;
        a[i+4][1] =  src[i].y*dst[i].z;
        a[i+4][2] =  src[i].z*dst[i].z;

        a[i][3] =  -src[i].x*dst[i].z;
        a[i][4] =  -src[i].y*dst[i].z;
        a[i][5] =  -src[i].z*dst[i].z;

        a[i][6] = src[i].x*dst[i].y;
        a[i][7] = src[i].y*dst[i].y;

        a[i+4][6] = -src[i].x*dst[i].x;
        a[i+4][7] = -src[i].y*dst[i].x;

        a[i][0] = a[i][1] = a[i][2] = a[i+4][3] = a[i+4][4] = a[i+4][5] = 0;

        b[i+0] = -dst[i].y*src[i].z;
        b[i+4] =  dst[i].x*src[i].z;
    }

    /*A.convertTo(A, CV_32F);
    B.convertTo(B, CV_32F);
    solveLSQR(A, B, X, 0, 100);*/

    solve(A, B, X, solveMethod);
    M.ptr<double>()[8] = 1.;

    return M;
}

void MarkerDetector::undistortPoints( InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, InputArray xi, InputArray R)
{
    CV_Assert(distorted.type() == CV_64FC2 || distorted.type() == CV_32FC2);
    CV_Assert(R.empty() || (!R.empty() && (R.size() == Size(3, 3) || R.total() * R.channels() == 3)
        && (R.depth() == CV_64F || R.depth() == CV_32F)));
    CV_Assert((D.depth() == CV_64F || D.depth() == CV_32F) && D.total() == 4);
    CV_Assert(K.size() == Size(3, 3) && (K.depth() == CV_64F || K.depth() == CV_32F));
    CV_Assert(xi.total() == 1 && (xi.depth() == CV_64F || xi.depth() == CV_32F));

    undistorted.create(distorted.size(), distorted.type());

    cv::Vec2d f, c;
    double s = 0.0;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0,0), camMat(1,1));
        c = Vec2f(camMat(0,2), camMat(1,2));
        s = (double)camMat(0,1);
    }
    else if (K.depth() == CV_64F)
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0,0), camMat(1,1));
        c = Vec2d(camMat(0,2), camMat(1,2));
        s = camMat(0,1);
    }

    Vec4d kp = D.depth()==CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>():(Vec4d)*D.getMat().ptr<Vec4d>();
    Vec2d k = Vec2d(kp[0], kp[1]);
    Vec2d p = Vec2d(kp[2], kp[3]);

    //double _xi = xi.depth() == CV_32F ? (double)*xi.getMat().ptr<float>() : *xi.getMat().ptr<double>();
    cv::Matx33d RR = cv::Matx33d::eye();
    // R is om
    if(!R.empty() && R.total()*R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        cv::Rodrigues(rvec, RR);
    }
    else if (!R.empty() && R.size() == Size(3,3))
    {
        R.getMat().convertTo(RR, CV_64F);
    }

    const cv::Vec2d *srcd = distorted.getMat().ptr<cv::Vec2d>();
    const cv::Vec2f *srcf = distorted.getMat().ptr<cv::Vec2f>();

    cv::Vec2d *dstd = undistorted.getMat().ptr<cv::Vec2d>();
    cv::Vec2f *dstf = undistorted.getMat().ptr<cv::Vec2f>();

    int n = (int)distorted.total();
    for (int i = 0; i < n; i++)
    {
        Vec2d pi = distorted.depth() == CV_32F ? (Vec2d)srcf[i]:(Vec2d)srcd[i];    // image point
        Vec2d pp((pi[0]*f[1]-c[0]*f[1]-s*(pi[1]-c[1]))/(f[0]*f[1]), (pi[1]-c[1])/f[1]); //plane
        Vec2d pu = pp;    // points without distortion

        // remove distortion iteratively
        for (int j = 0; j < 20; j++)
        {
            double r2 = pu[0]*pu[0] + pu[1]*pu[1];
            double r4 = r2*r2;
            pu[0] = (pp[0] - 2*p[0]*pu[0]*pu[1] - p[1]*(r2+2*pu[0]*pu[0])) / (1 + k[0]*r2 + k[1]*r4);
            pu[1] = (pp[1] - 2*p[1]*pu[0]*pu[1] - p[0]*(r2+2*pu[1]*pu[1])) / (1 + k[0]*r2 + k[1]*r4);
        }

        // project to unit sphere
        /*double r2 = pu[0]*pu[0] + pu[1]*pu[1];
        double a = (r2 + 1);
        double b = 2*_xi*r2;
        double cc = r2*_xi*_xi-1;
        double Zs = (-b + sqrt(b*b - 4*a*cc))/(2*a);
        Vec3d Xw = Vec3d(pu[0]*(Zs + _xi), pu[1]*(Zs +_xi), Zs);*/


        // rotate
        //Xw = RR * Xw;

        // project back to sphere
        //Vec3d Xs = Xw / cv::norm(Xw);
        // reproject to camera plane
        //Vec3d ppu = Vec3d(Xs[0]/(Xs[2]+_xi), Xs[1]/(Xs[2]+_xi), 1.0);

        if (undistorted.depth() == CV_32F)
        {
            dstf[i] = Vec2f((float)(f[0]*pu[0]+c[0]), (float)f[1]*pu[1]+c[1]);
        }
        else if (undistorted.depth() == CV_64F)
        {
            dstd[i] = Vec2d(f[0]*pu[0]+c[0], f[1]*pu[1]+c[1]);
        }
    }
}
/************************************
     *
     *
     *
     *
     ************************************/
int MarkerDetector::perimeter(const vector<Point2f>& a)
{
    int sum = 0;
    for (unsigned int i = 0; i < a.size(); i++)
    {
        int i2 = (i + 1) % a.size();
        sum += static_cast<int>( sqrt((a[i].x - a[i2].x) * (a[i].x - a[i2].x) +
                                      (a[i].y - a[i2].y) * (a[i].y - a[i2].y)));
    }
    return sum;
}


/**
     */
void MarkerDetector::interpolate2Dline(const std::vector<Point2f>& inPoints, Point3f& outLine)
{
    float minX, maxX, minY, maxY;
    minX = maxX = inPoints[0].x;
    minY = maxY = inPoints[0].y;
    for (unsigned int i = 1; i < inPoints.size(); i++)
    {
        if (inPoints[i].x < minX)
            minX = inPoints[i].x;
        if (inPoints[i].x > maxX)
            maxX = inPoints[i].x;
        if (inPoints[i].y < minY)
            minY = inPoints[i].y;
        if (inPoints[i].y > maxY)
            maxY = inPoints[i].y;
    }

    // create matrices of equation system
    const int pointsCount = static_cast<int>(inPoints.size());
    Mat A(pointsCount, 2, CV_32FC1, Scalar(0));
    Mat B(pointsCount, 1, CV_32FC1, Scalar(0));
    Mat X;

    if (maxX - minX > maxY - minY)
    {
        // Ax + C = y
        for (int i = 0; i < pointsCount; i++)
        {
            A.at<float>(i, 0) = inPoints[i].x;
            A.at<float>(i, 1) = 1.;
            B.at<float>(i, 0) = inPoints[i].y;
        }

        // solve system
        solve(A, B, X, DECOMP_SVD);
        // return Ax + By + C
        outLine = Point3f(X.at<float>(0, 0), -1., X.at<float>(1, 0));
    }
    else
    {
        // By + C = x
        for (int i = 0; i < pointsCount; i++)
        {
            A.at<float>(i, 0) = inPoints[i].y;
            A.at<float>(i, 1) = 1.;
            B.at<float>(i, 0) = inPoints[i].x;
        }

        // solve system
        solve(A, B, X, DECOMP_SVD);
        // return Ax + By + C
        outLine = Point3f(-1., X.at<float>(0, 0), X.at<float>(1, 0));
    }
}

/**
     */
Point2f MarkerDetector::getCrossPoint(const cv::Point3f& line1, const cv::Point3f& line2)
{
    // create matrices of equation system
    Mat A(2, 2, CV_32FC1, Scalar(0));
    Mat B(2, 1, CV_32FC1, Scalar(0));
    Mat X;

    A.at<float>(0, 0) = line1.x;
    A.at<float>(0, 1) = line1.y;
    B.at<float>(0, 0) = -line1.z;

    A.at<float>(1, 0) = line2.x;
    A.at<float>(1, 1) = line2.y;
    B.at<float>(1, 0) = -line2.z;

    // solve system
    solve(A, B, X, DECOMP_SVD);
    return Point2f(X.at<float>(0, 0), X.at<float>(1, 0));
}

void  MarkerDetector::cornerUpsample(vector<Marker>& MarkerCanditates, cv::Size lowResImageSize ){
     cornerUpsample_SUBP(MarkerCanditates,lowResImageSize);
}

void MarkerDetector::cornerUpsample_SUBP(vector<Marker>& MarkerCanditates,   cv::Size lowResImageSize ){
    if (MarkerCanditates.size()==0)return;
    //first, determine the image in the pyramid nearest to this one
    int startPyrImg=0;

    for(size_t i=0;i<imagePyramid.size();i++){
        if ( lowResImageSize.width < imagePyramid[i].cols) startPyrImg=i;
        else break;
    }
//#define _aruco_marker_detector_fast

     cv::Size prevLowResSize=lowResImageSize;
    for(int curpyr=startPyrImg;curpyr>=0;curpyr--){
        float factor= float(imagePyramid[curpyr].cols)/float(prevLowResSize.width) ;
        //upsample corner locations
        for(auto &m:MarkerCanditates)
            for(auto &point:m) {point*=factor;}
        int halfwsize=  0.5+2.5*factor;
             vector<cv::Point2f> p2d;p2d.reserve(MarkerCanditates.size()*4);
            for(auto &m:MarkerCanditates)
                for(auto &point:m) { p2d.push_back(point);}
             cv::cornerSubPix( imagePyramid[curpyr],p2d,cv::Size(halfwsize,halfwsize),cv::Size(-1,-1),cv::TermCriteria(cv::TermCriteria::MAX_ITER , 4,0.5));
            int cidx=0;
            for(auto &m:MarkerCanditates)
                for(auto &point:m) {point =p2d[cidx++];}

        prevLowResSize=imagePyramid[curpyr].size();
    }


}


/************************************
     *
     *
     *
     *
     ************************************/
void MarkerDetector::drawAllContours(Mat input, std::vector<std::vector<cv::Point>>& contours)
{
    drawContours(input, contours, -1, Scalar(255, 0, 255));
}

/************************************
     *
     *
     *
     *
     ************************************/
void MarkerDetector::drawContour(Mat& in, vector<Point>& contour, Scalar color)
{
    for (unsigned int i = 0; i < contour.size(); i++)
    {
        cv::rectangle(in, contour[i], contour[i], color);
    }
}

void MarkerDetector::drawApproxCurve(Mat& in, vector<Point>& contour, Scalar color,int thickness)
{
    for (unsigned int i = 0; i < contour.size(); i++)
    {
        cv::line(in, contour[i], contour[(i + 1) % contour.size()], color,thickness);
    }
}
/************************************
     *
     *
     *
     *
     ************************************/

void MarkerDetector::draw(Mat out, const vector<Marker>& markers)
{
    for (unsigned int i = 0; i < markers.size(); i++)
    {
        cv::line(out, markers[i][0], markers[i][1], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        cv::line(out, markers[i][1], markers[i][2], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        cv::line(out, markers[i][2], markers[i][3], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        cv::line(out, markers[i][3], markers[i][0], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }
}


void MarkerDetector::setMarkerLabeler(cv::Ptr<MarkerLabeler> detector)
{
    markerIdDetector = detector;
}

void MarkerDetector::setDictionary(int dict_type,
                                   float error_correction_rate)
{
    markerIdDetector = MarkerLabeler::create((Dictionary::DICT_TYPES)dict_type, error_correction_rate);
    _params.error_correction_rate=error_correction_rate;
    _params.dictionary=aruco::Dictionary::getTypeString((Dictionary::DICT_TYPES)dict_type);
}




void MarkerDetector::setDictionary(string dict_type, float error_correction_rate)
{
		auto _to_string=[](float i){
			std::stringstream str;str<<i;return str.str();
			};
        _params.dictionary=dict_type;
    markerIdDetector = MarkerLabeler::create(dict_type, _to_string(error_correction_rate));
    _params.error_correction_rate=error_correction_rate;
}


cv::Mat MarkerDetector::getThresholdedImage(uint32_t idx)
{
    if (_thres_Images.size()==0)return cv::Mat();
    if (idx>=_thres_Images.size()) idx=_thres_Images.size()-1;//last one is the original image
    return _thres_Images[idx];
}



/**
 */
void MarkerDetector::distortPoints(vector< cv::Point2f > in, vector< cv::Point2f > &out, const Mat &camMatrix, const Mat &distCoeff) {
    // trivial extrinsics
    cv::Mat Rvec = cv::Mat(3, 1, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Tvec = Rvec.clone();
    // calculate 3d points and then reproject, so opencv makes the distortion internally
    vector< cv::Point3f > cornersPoints3d;
    for (unsigned int i = 0; i < in.size(); i++)
        cornersPoints3d.push_back(cv::Point3f((in[i].x - camMatrix.at< float >(0, 2)) / camMatrix.at< float >(0, 0), // x
                                              (in[i].y - camMatrix.at< float >(1, 2)) / camMatrix.at< float >(1, 1), // y
                                              1)); // z
    cv::projectPoints(cornersPoints3d, Rvec, Tvec, camMatrix, distCoeff, out);
}


void MarkerDetector::Params::save(cv::FileStorage &fs)const{
    fs<<"aruco-dictionary"<<dictionary;
    fs<<"aruco-detectMode"<<toString(detectMode);
    fs<<"aruco-cornerRefinementM"<<toString(cornerRefinementM);
    fs<<"aruco-thresMethod"<<toString(thresMethod);
    fs<<"aruco-maxThreads"<<maxThreads;
    fs<<"aruco-borderDistThres"<<borderDistThres;
    fs<<"aruco-lowResMarkerSize"<<lowResMarkerSize;
    fs<<"aruco-minSize"<<minSize;
    fs<<"aruco-minSize_pix"<<minSize_pix;
    fs<<"aruco-enclosedMarker"<<enclosedMarker;
    fs<<"aruco-NAttemptsAutoThresFix"<<NAttemptsAutoThresFix;
    fs<<"aruco-AdaptiveThresWindowSize"<<AdaptiveThresWindowSize;
    fs<<"aruco-ThresHold"<<ThresHold;
    fs<<"aruco-AdaptiveThresWindowSize_range"<<AdaptiveThresWindowSize_range;
    fs<<"aruco-markerWarpPixSize"<<markerWarpPixSize;
    fs<<"aruco-autoSize"<<autoSize;
    fs<<"aruco-ts"<<ts;
    fs<<"aruco-pyrfactor"<<pyrfactor;
    fs<<"aruco-error_correction_rate"<<error_correction_rate;
}

void MarkerDetector::Params::load(cv::FileStorage &fs){
     attemtpRead("aruco-maxThreads",maxThreads,fs);
    attemtpRead("aruco-borderDistThres",borderDistThres,fs);
    attemtpRead("aruco-lowResMarkerSize",lowResMarkerSize,fs);
    attemtpRead("aruco-minSize",minSize,fs);
    attemtpRead("aruco-minSize_pix",minSize_pix,fs);
    attemtpRead("aruco-enclosedMarker",enclosedMarker,fs);
    attemtpRead("aruco-NAttemptsAutoThresFix",NAttemptsAutoThresFix,fs);
    attemtpRead("aruco-AdaptiveThresWindowSize",AdaptiveThresWindowSize,fs);
    attemtpRead("aruco-ThresHold",ThresHold,fs);
    attemtpRead("aruco-AdaptiveThresWindowSize_range",AdaptiveThresWindowSize_range,fs);
    attemtpRead("aruco-markerWarpPixSize",markerWarpPixSize,fs);
    attemtpRead("aruco-autoSize",autoSize,fs);
    attemtpRead("aruco-ts",ts,fs);
    attemtpRead("aruco-pyrfactor",pyrfactor,fs);
    attemtpRead("aruco-error_correction_rate",error_correction_rate,fs);
    attemtpRead("aruco-dictionary",dictionary,fs);

    if ( fs["aruco-detectMode"].type()!=cv::FileNode::NONE){
        string aux;
        fs["aruco-detectMode"]>>aux;
        detectMode=getDetectionModeFromString(aux);
    }
    if ( fs["aruco-thresMethod"].type()!=cv::FileNode::NONE){
        string aux;
        fs["aruco-thresMethod"]>>aux;
        thresMethod=getCornerThresMethodFromString(aux);
    }
    if ( fs["aruco-cornerRefinementM"].type()!=cv::FileNode::NONE){
        string aux;
        fs["aruco-cornerRefinementM"]>>aux;
        cornerRefinementM=getCornerRefinementMethodFromString(aux);
    }

  }


void MarkerDetector::Params::toStream(std::ostream &str)const{
    str.write((char*)&detectMode,sizeof(detectMode));
    str.write((char*)&maxThreads,sizeof(maxThreads));
    str.write((char*)&borderDistThres,sizeof(borderDistThres));
    str.write((char*)&lowResMarkerSize,sizeof(lowResMarkerSize));
    str.write((char*)&minSize,sizeof(minSize));
    str.write((char*)&minSize_pix,sizeof(minSize_pix));
    str.write((char*)&enclosedMarker,sizeof(enclosedMarker));
    str.write((char*)&thresMethod,sizeof(thresMethod));
    str.write((char*)&NAttemptsAutoThresFix,sizeof(NAttemptsAutoThresFix));
    str.write((char*)&AdaptiveThresWindowSize,sizeof(AdaptiveThresWindowSize));
    str.write((char*)&ThresHold,sizeof(ThresHold));
    str.write((char*)&AdaptiveThresWindowSize_range,sizeof(AdaptiveThresWindowSize_range));
    str.write((char*)&markerWarpPixSize,sizeof(markerWarpPixSize));
    str.write((char*)&cornerRefinementM,sizeof(cornerRefinementM));
    str.write((char*)&autoSize,sizeof(autoSize));
    str.write((char*)&ts,sizeof(pyrfactor));
    str.write((char*)&error_correction_rate,sizeof(error_correction_rate));
    _toStream(dictionary,str);
}
void MarkerDetector::Params::fromStream(std::istream &str) {
    str.read((char*)&detectMode,sizeof(detectMode));
    str.read((char*)&maxThreads,sizeof(maxThreads));
    str.read((char*)&borderDistThres,sizeof(borderDistThres));
    str.read((char*)&lowResMarkerSize,sizeof(lowResMarkerSize));
    str.read((char*)&minSize,sizeof(minSize));
    str.read((char*)&minSize_pix,sizeof(minSize_pix));
    str.read((char*)&enclosedMarker,sizeof(enclosedMarker));
    str.read((char*)&thresMethod,sizeof(thresMethod));
    str.read((char*)&NAttemptsAutoThresFix,sizeof(NAttemptsAutoThresFix));
    str.read((char*)&AdaptiveThresWindowSize,sizeof(AdaptiveThresWindowSize));
    str.read((char*)&ThresHold,sizeof(ThresHold));
    str.read((char*)&AdaptiveThresWindowSize_range,sizeof(AdaptiveThresWindowSize_range));
    str.read((char*)&markerWarpPixSize,sizeof(markerWarpPixSize));
    str.read((char*)&cornerRefinementM,sizeof(cornerRefinementM));
    str.read((char*)&autoSize,sizeof(autoSize));
    str.read((char*)&ts,sizeof(pyrfactor));
    str.read((char*)&error_correction_rate,sizeof(error_correction_rate));
    _fromStream(dictionary,str);
}
/**Saves the configuration of the detector to a file
 */
void MarkerDetector::saveParamsToFile(const std::string &path) const{

    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if(!fs.isOpened())throw std::runtime_error("Could not open "+path);
    _params.save(fs);
}

/**Loads the configuration from a file
 */
void MarkerDetector::loadParamsFromFile(const std::string &path){
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if(!fs.isOpened())throw std::runtime_error("Could not open "+path);
    _params.load(fs);
    setDictionary(_params.dictionary,_params.error_correction_rate);
}

void MarkerDetector::toStream(std::ostream &str)const
{
    uint64_t sig=13213;
    str.write((char*)&sig,sizeof(sig));
    _params.toStream(str);
}

void MarkerDetector::fromStream(std::istream &str){
    uint64_t sig=13213;
    str.read((char*)&sig,sizeof(sig));
    if (sig!=13213) throw std::runtime_error("MarkerDetector::fromStream invalid signature");
    _params.fromStream(str);
    setDictionary(_params.dictionary,_params.error_correction_rate);

}

std::string MarkerDetector::Params::toString(DetectionMode dm){
    switch(dm){
    case DM_FAST:return "DM_FAST";
    case DM_NORMAL:return "DM_NORMAL";
    case DM_VIDEO_FAST:return "DM_VIDEO_FAST";
    };
    return "DM_NORMAL";
}

DetectionMode MarkerDetector::Params::getDetectionModeFromString(const std::string &str){
    if ( str=="DM_FAST")return DM_FAST;
    if ( str=="DM_NORMAL")return DM_NORMAL;
    if ( str=="DM_VIDEO_FAST")return DM_VIDEO_FAST;
    return DM_NORMAL;

}

std::string MarkerDetector::Params::toString(CornerRefinementMethod dm){
    switch(dm){
    case CORNER_LINES:return "CORNER_LINES";
    case CORNER_SUBPIX:return "CORNER_SUBPIX";
    case CORNER_NONE:return "CORNER_NONE";
    };
    return "CORNER_SUBPIX";
}
CornerRefinementMethod MarkerDetector::Params::getCornerRefinementMethodFromString(const std::string &str){
    if ( str=="CORNER_LINES")return CORNER_LINES;
    if ( str=="CORNER_SUBPIX")return CORNER_SUBPIX;
    if ( str=="CORNER_NONE")return CORNER_NONE;
    return CORNER_SUBPIX;
}
std::string MarkerDetector::Params::toString(MarkerDetector::ThresMethod dm){
    switch(dm){
    case THRES_ADAPTIVE:return "THRES_ADAPTIVE";
    case THRES_AUTO_FIXED:return "THRES_AUTO_FIXED";
     };
    return "THRES_ADAPTIVE";
}
MarkerDetector::ThresMethod MarkerDetector::Params::getCornerThresMethodFromString(const std::string &str){
    if ( str=="THRES_ADAPTIVE")return THRES_ADAPTIVE;
    if ( str=="THRES_AUTO_FIXED")return THRES_AUTO_FIXED;
     return THRES_ADAPTIVE;
}
void  MarkerDetector::Params::setThresholdMethod( MarkerDetector::ThresMethod  method,int thresHold,int wsize,int wsize_range ){
    AdaptiveThresWindowSize=wsize;
    thresMethod=method;
    if (thresHold==-1){
        if ( method==THRES_AUTO_FIXED ) ThresHold=100;
        else ThresHold=15;
    }
    else  ThresHold=thresHold;
    AdaptiveThresWindowSize_range=wsize_range;
}
void  MarkerDetector::Params::setDetectionMode( DetectionMode dm,float minMarkerSize){
    detectMode=dm;
    minSize=minMarkerSize;
    if(detectMode==DM_NORMAL){
        setAutoSizeSpeedUp(false);
        setThresholdMethod(THRES_ADAPTIVE);
    }
    else if (detectMode==DM_FAST ){
        setAutoSizeSpeedUp(false);
        setThresholdMethod(THRES_AUTO_FIXED);
    }
    else if(detectMode==DM_VIDEO_FAST){
        setThresholdMethod(THRES_AUTO_FIXED);
        setAutoSizeSpeedUp(true,0.3);
    }
}
void MarkerDetector::Params::setCornerRefinementMethod( CornerRefinementMethod method){
    cornerRefinementM=method;
    if(method!=CORNER_SUBPIX)  minSize=0;
}

};
