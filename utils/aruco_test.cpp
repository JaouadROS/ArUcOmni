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

 #include "aruco.h"
#include "cvdrawingutils.h"
#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <string>
#include <stdexcept>
#include <opencv2/core/utility.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace aruco;

MarkerDetector MDetector;
VideoCapture TheVideoCapturer;
vector<Marker> TheMarkers;
Mat TheInputImage,TheInputImageGrey, TheInputImageCopy;
CameraParameters TheCameraParameters;
void cvTackBarEvents(int pos, void*);
string dictionaryString;
int iDetectMode=0,iMinMarkerSize=0,iCorrectionRate=0,iShowAllCandidates=0,iEnclosed=0,iThreshold,iCornerMode,iDictionaryIndex=0;

int waitTime = 0;
bool showMennu=false,bPrintHelp=false,isVideo=false;
class CmdLineParser{int argc;char** argv;public:CmdLineParser(int _argc, char** _argv): argc(_argc), argv(_argv){}   bool operator[](string param)    {int idx = -1;  for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param)idx = i;return (idx != -1);}    string operator()(string param, string defvalue = "-1")    {int idx = -1;for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param)idx = i;if (idx == -1)return defvalue;else return (argv[idx + 1]);}};
struct   TimerAvrg{std::vector<double> times;size_t curr=0,n; 

std::chrono::high_resolution_clock::time_point begin,end;   

TimerAvrg(int _n=30){n=_n;times.reserve(n);   }

inline void start(){begin= std::chrono::high_resolution_clock::now();    }

inline void stop(){end= std::chrono::high_resolution_clock::now();
double duration=double(std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())*1e-6;
if ( times.size()<n) times.push_back(duration);else{ times[curr]=duration; curr++;if (curr>=times.size()) curr=0;}}

double getAvrg(){double sum=0;for(auto t:times) sum+=t;return sum/double(times.size());}};

TimerAvrg Fps;
cv::Mat resize(const cv::Mat& in, int width)
{
    if (in.size().width <= width)
        return in;
    float yf = float(width) / float(in.size().width);
    cv::Mat im2;
    cv::resize(in, im2, cv::Size(width, static_cast<int>(in.size().height * yf)));
    return im2;
}
/************************************
 *
 *
 *
 *
 ************************************/
void setParamsFromGlobalVariables(aruco::MarkerDetector &md){


    md.setDetectionMode((DetectionMode)iDetectMode,float(iMinMarkerSize)/1000.);
    md.getParameters().setCornerRefinementMethod( (aruco::CornerRefinementMethod) iCornerMode);

    md.getParameters().detectEnclosedMarkers(iEnclosed);
    md.getParameters().ThresHold=iThreshold;
    if ( aruco::Dictionary::getTypeFromString( md.getParameters().dictionary)!=Dictionary::CUSTOM)
            md.setDictionary((aruco::Dictionary::DICT_TYPES) iDictionaryIndex,float(iCorrectionRate)/10. );  // sets the dictionary to be employed (ARUCO,APRILTAGS,ARTOOLKIT,etc)
}

void createMenu(){
    cv::createTrackbar("Dictionary", "menu", &iDictionaryIndex, 13, cvTackBarEvents);
   cv::createTrackbar("DetectMode", "menu", &iDetectMode, 2, cvTackBarEvents);
   cv::createTrackbar("CornerMode", "menu", &iCornerMode, 2, cvTackBarEvents);

   cv::createTrackbar("MinMarkerSize", "menu", &iMinMarkerSize, 1000, cvTackBarEvents);
   cv::createTrackbar("Threshold", "menu", &iThreshold, 40, cvTackBarEvents);
   cv::createTrackbar("ErrorRate", "menu", &iCorrectionRate, 10, cvTackBarEvents);
   cv::createTrackbar("Enclosed", "menu", &iEnclosed, 1, cvTackBarEvents);
   cv::createTrackbar("ShowAll", "menu", &iShowAllCandidates, 1, cvTackBarEvents);
   iThreshold=MDetector.getParameters().ThresHold;
   iCornerMode= MDetector.getParameters().cornerRefinementM;
}

void putText(cv::Mat &im,string text,cv::Point p,float size){
    float fact=float(im.cols)/float(640);
    if (fact<1) fact=1;

    cv::putText(im,text,p,FONT_HERSHEY_SIMPLEX, size,cv::Scalar(0,0,0),3*fact);
    cv::putText(im,text,p,FONT_HERSHEY_SIMPLEX, size,cv::Scalar(125,255,255),1*fact);

}
void printHelp(cv::Mat &im)
{
    float fs=float(im.cols)/float(1200);

    putText(im,"'m': show/hide menu",cv::Point(10,fs*60),fs*0.5f);
    putText(im,"'s': start/stop video capture",cv::Point(10,fs*80),fs*0.5f);
    putText(im,"'w': write image to file",cv::Point(10,fs*100),fs*0.5f);
    putText(im,"'t': do a speed test",cv::Point(10,fs*120),fs*0.5f);
    putText(im,"'f': saves current configuration to file 'arucoConfig.yml'",cv::Point(10,fs*140),fs*0.5f);
}

void printInfo(cv::Mat &im){
    float fs=float(im.cols)/float(1200);
    putText(im,"fps="+to_string(1./Fps.getAvrg()),cv::Point(10,fs*20),fs*0.5f);
    putText(im,"'h': show/hide help",cv::Point(10,fs*40),fs*0.5f);
    if(bPrintHelp) printHelp(im);
}

void printMenuInfo(){
        cv::Mat image(200,400,CV_8UC3);
        image=cv::Scalar::all(255);
        string str="Dictionary="+aruco::Dictionary::getTypeString((aruco::Dictionary::DICT_TYPES) iDictionaryIndex) ;

        cv::putText(image,str,cv::Size(10,20),FONT_HERSHEY_SIMPLEX, 0.35,cv::Scalar(0,0,0),1);

        str="Detection Mode="+MarkerDetector::Params::toString(MDetector.getParameters().detectMode);
        cv::putText(image,str,cv::Size(10,40),FONT_HERSHEY_SIMPLEX, 0.35,cv::Scalar(0,0,0),1);
        str="Corner Mode="+MarkerDetector::Params::toString(MDetector.getParameters().cornerRefinementM);;
        cv::putText(image,str,cv::Size(10,60),FONT_HERSHEY_SIMPLEX, 0.35,cv::Scalar(0,0,0),1);
        cv::imshow("menu",image);
}

cv::Mat resizeImage(cv::Mat &in,float resizeFactor){
    if (fabs(1-resizeFactor)<1e-3 )return in;
    float nc=float(in.cols)*resizeFactor;
    float nr=float(in.rows)*resizeFactor;
    cv::Mat imres;
    cv::resize(in,imres,cv::Size(nc,nr));
    cout<<"Imagesize="<<imres.size()<<endl;
    return imres;
}
/************************************
 *
 *
 *
 *
 ************************************/
void inverseRT(const cv::Mat &M,cv::Mat &Minv)
{
     assert(Minv.total()==16 && M.type()==CV_32F);
     //transposed R first
     const float *m=M.ptr<float>(0);
     float *minv=Minv.ptr<float>(0);

     minv[0]=m[0]; minv[1]=m[4];minv[2]=m[8];
     minv[4]=m[1]; minv[5]=m[5];minv[6]=m[9];
     minv[8]=m[2]; minv[9]=m[6];minv[10]=m[10];

     //now, translation
     minv[3] =-( minv[0]*m[3]+minv[1]*m[7]+minv[2]*m[11]);
     minv[7] =-( minv[4]*m[3]+minv[5]*m[7]+minv[6]*m[11]);
     minv[11]=-( minv[8]*m[3]+minv[9]*m[7]+minv[10]*m[11]);

     minv[12]=minv[13]=minv[14]=0;
     minv[15]=1;
}

int main(int argc, char** argv)
{
    try
    {
        CmdLineParser cml(argc, argv);
        if (argc < 2 || cml["-h"])
        {
            cerr << "Invalid number of arguments" << endl;
            cerr << "Usage: (in.avi|live[:camera_index(e.g 0 or 1)]) [-c camera_params.yml] [-s  marker_size_in_meters] [-d "
                    "dictionary:ALL_DICTS by default] [-h]"
                 << endl;
            cerr << "\tDictionaries: ";
            for (auto dict : aruco::Dictionary::getDicTypes())
                cerr << dict << " ";
            cerr << endl;
            cerr << "\t Instead of these, you can directly indicate the path to a file with your own generated "
                    "dictionary"
                 << endl;
            return false;
        }

        ///////////  PARSE ARGUMENTS
        string TheInputVideo = argv[1];
        // read camera parameters if passed
        if (cml["-c"])
            TheCameraParameters.readFromXMLFile(cml("-c"));

        float TheMarkerSize = std::stof(cml("-s", "-1"));
        //resize factor
        float resizeFactor=stof(cml("-rf","1"));

        ///////////  OPEN VIDEO
        // read from camera or from  file
        if (TheInputVideo.find("live") != string::npos)
        {
            int vIdx = 0;
            // check if the :idx is here
            char cad[100];
            if (TheInputVideo.find(":") != string::npos)
            {
                std::replace(TheInputVideo.begin(), TheInputVideo.end(), ':', ' ');
                sscanf(TheInputVideo.c_str(), "%s %d", cad, &vIdx);
            }
            cout << "Opening camera index " << vIdx << endl;
            TheVideoCapturer.open(vIdx);
            waitTime = 10;
            isVideo=true;
        }
        else{
            TheVideoCapturer.open(TheInputVideo);
            if ( TheVideoCapturer.get(cv::CAP_PROP_FRAME_COUNT)>=2) isVideo=true;
        }
        // check video is open
        if (!TheVideoCapturer.isOpened())
            throw std::runtime_error("Could not open video");


        ///// CONFIGURE DATA
        // read first image to get the dimensions
        TheVideoCapturer >> TheInputImage;
        if (TheCameraParameters.isValid())
            TheCameraParameters.resize(TheInputImage.size());
        dictionaryString=cml("-d", "ALL_DICTS");
        iDictionaryIndex=(uint64_t)aruco::Dictionary::getTypeFromString(dictionaryString);
         MDetector.setDictionary(dictionaryString,float(iCorrectionRate)/10. );  // sets the dictionary to be employed (ARUCO,APRILTAGS,ARTOOLKIT,etc)
         iThreshold=MDetector.getParameters().ThresHold;
         iCornerMode= MDetector.getParameters().cornerRefinementM;

        cv::namedWindow("in",cv::WINDOW_NORMAL);
        cv::resizeWindow("in",640,480);

        setParamsFromGlobalVariables(MDetector);

        {
        float w=std::min(int(1920),int(TheInputImage.cols));
        float f=w/float(TheInputImage.cols);
        resizeWindow("in",w,float(TheInputImage.rows)*f);

        }
        // go!
        char key = 0;
        int index = 0,indexSave=0;
        // capture until press ESC or until the end of the video

         do
        {

             TheVideoCapturer.retrieve(TheInputImage);

             TheInputImage=resizeImage(TheInputImage,resizeFactor);
              // copy image
            Fps.start();
            TheMarkers = MDetector.detect(TheInputImage, TheCameraParameters, TheMarkerSize);
            Fps.stop();
            // chekc the speed by calculating the mean speed of all iterations
            //cout << "\rTime detection=" << Fps.getAvrg()*1000 << " milliseconds nmarkers=" << TheMarkers.size() << std::endl;//

            // print marker info and draw the markers in image
            TheInputImage.copyTo(TheInputImageCopy);

            if (iShowAllCandidates){
                auto candidates=MDetector.getCandidates();
                for(auto cand:candidates)
                    Marker(cand,-1).draw(TheInputImageCopy, Scalar(255, 0, 255));
            }

            for (unsigned int i = 0; i < TheMarkers.size(); i++)
            {
                //cout << TheMarkers[i] << endl;
                TheMarkers[i].draw(TheInputImageCopy, Scalar(0, 0, 255),2,true);
            }

            //cv::Matx44f cMo44d[3];
            cv::Mat cMo44d[3];
            cMo44d[0] = cv::Mat::eye(4, 4, CV_32FC1);
            cMo44d[1] = cv::Mat::eye(4, 4, CV_32FC1);
            cMo44d[2] = cv::Mat::eye(4, 4, CV_32FC1);                        

            for (unsigned int i = 0; i < TheMarkers.size(); i++)
            {
		        cv::Mat rotMat;
		        cv::Rodrigues(TheMarkers[i].Rvec, rotMat);
                //std::cout<<"Marker: "<<i<<" "<<TheMarkers[i].id<<std::endl;
	            rotMat.copyTo(cMo44d[i].rowRange(0, 3).colRange(0, 3)); // copies R into camPose
	            TheMarkers[i].Tvec = TheMarkers[i].Tvec.t();
                TheMarkers[i].Tvec.copyTo(cMo44d[i].rowRange(0, 3).colRange(3, 4)); // copies tvec into camPose
		        /*for(int k=0;k<3;k++)
		         for(int l=0;l<3;l++)
		             cMo44d[i].at<float>(k,l) = rotMat.at<float>(k,l);

				for(int k=0;k<3;k++)
		             cMo44d[i].at<float>(k,3) = TheMarkers[i].Tvec.ptr<float>(0)[k];

		             cMo44d[i].at<float>(3,3) = 1;     */
		             //std::cout<<"Marker: "<<i<<" "<<TheMarkers[i].id<<std::endl;
		        //std::cout<<cMo44d[i]<<std::endl;
		        //cout << TheMarkers[i].id<< " "<<cMo44d[i].row(0)<<" "<<cMo44d[i].row(1)<<" "<<cMo44d[i].row(2)<<" "<<TheMarkers[i].optimization_error<<std::endl;
            }
        
            /*int ma_i = 2;
            //inverseRT(cMo44d[ma_i],cMo44d[ma_i]);
            cv::Mat R;
	        cv::Rodrigues(TheMarkers[ma_i].Rvec, R);
	        R = R.t();
	        TheMarkers[ma_i].Tvec = -R*TheMarkers[ma_i].Tvec;
	        cv::Mat camPose = cv::Mat::eye(4, 4, R.type());
            R.copyTo(camPose.rowRange(0, 3).colRange(0, 3)); // copies R into camPose
            TheMarkers[ma_i].Tvec.copyTo(camPose.rowRange(0, 3).colRange(3, 4)); // copies tvec into camPose
	        //std::cout<<R*R.t()<<std::endl;
	        
	        cv::Mat cMo67;
			cMo67 = camPose*cMo44d[0];
			cMo67.convertTo(cMo67, CV_32S);
			std::cout<<cMo67<<std::endl;*/
			/*cv::Mat dst;
			Euler(R, dst, 0);
			std::cout<<dst<<std::endl;*/
	        
	        /*
	        //std::cout<<rotMat.size()<<" "<<TheMarkers[1].Tvec.t()<<std::endl;		    
	        TheMarkers[1].Tvec = -rotMat*TheMarkers[1].Tvec.t();
	        std::cout<<rotMat<<std::endl;
	        //std::cout<<cMo44d[0]<<" "<<cMo44d[1]<<std::endl;	
	        cv::Mat rvec3, tvec3;
	        cv::composeRT(TheMarkers[0].Rvec.t(),TheMarkers[0].Tvec.t(), TheMarkers[1].Rvec.t(),TheMarkers[1].Tvec, rvec3, tvec3);	    
	        std::cout<<rvec3<<" "<<tvec3<<std::endl;	        
	        cv::Rodrigues(rvec3, rotMat);

	        rotMat.copyTo(cMo44d[1].rowRange(0, 3).colRange(0, 3)); // copies R into camPose
            tvec3.copyTo(cMo44d[1].rowRange(0, 3).colRange(3, 4)); // copies tvec into camPose
	        //std::cout<<rotMat<<std::endl;
	        
			cv::Mat cMo67;
			cMo67 = cMo44d[0]*cMo44d[1];
			
			//std::cout<<cMo67<<std::endl;
			  */        
            
            // draw a 3d cube in each marker if there is 3d info
            if (TheCameraParameters.isValid() && TheMarkerSize > 0)
                for (unsigned int i = 0; i < TheMarkers.size(); i++)
                {
                    //CvDrawingUtils::draw3dCube(TheInputImageCopy, TheMarkers[i], TheCameraParameters);
                    CvDrawingUtils::draw3dAxis(TheInputImageCopy, TheMarkers[i], TheCameraParameters);
                }

            // DONE! Easy, right?
            // show input with augmented information and  the thresholded image
            printInfo(TheInputImageCopy);
            if(showMennu)printMenuInfo();
            //cv::imshow("thres", resize(MDetector.getThresholdedImage(), 1024));
            cv::imshow("in", TheInputImageCopy);

            key = cv::waitKey(waitTime);  // wait for key to be pressed
            if (key == 's')
                waitTime = waitTime == 0 ? 10 : 0;
            if (key == 'w'){//writes current input image
                string number=std::to_string(indexSave++);
                while(number.size()!=3)number="0"+number;
                string imname="arucoimage"+number+".png";
                cv::imwrite(imname,TheInputImageCopy);
                cout<<"saved "<<imname<<endl;
                imname="orgimage"+number+".png";
                cv::imwrite(imname,TheInputImage);
                cout<<"saved "<<imname<<endl;
                imname="thresimage"+number+".png";
                cv::imwrite(imname,MDetector.getThresholdedImage());

            }
             if (key=='m') {
                 if (showMennu)                     cv::destroyWindow("menu");
                 else {
                     cv::namedWindow("menu",cv::WINDOW_NORMAL);
                     cv::resizeWindow("menu",640,480);
                     createMenu();
                     printMenuInfo();
                 }
                showMennu=!showMennu;
            }
            if (key=='h')bPrintHelp=!bPrintHelp;

            if (key=='t'){//run a deeper speed test

                for(int t=0;t<30;t++){
                    // Detection of markers in the image passed
                    Fps.start();
                    TheMarkers = MDetector.detect(TheInputImage, TheCameraParameters, TheMarkerSize);
                    Fps.stop();
                    // chekc the speed by calculating the mean speed of all iterations
                }
                printInfo(TheInputImageCopy);
            }
            if(key=='f'){
                cerr<<"Configuration saved to arucoConfig.yml"<<endl;
                MDetector.saveParamsToFile("arucoConfig.yml");
            }
            index++;  // number of images captured

            if (isVideo)
                if ( TheVideoCapturer.grab()==false) key=27;
        } while (key != 27 );
    }
    catch (std::exception& ex)

    {
        cout << "Exception :" << ex.what() << endl;
    }
}


void cvTackBarEvents(int pos, void*)
{
    (void)(pos);


    setParamsFromGlobalVariables(MDetector);

    // recompute
        Fps.start();
        MDetector.detect(TheInputImage, TheMarkers, TheCameraParameters);
        Fps.stop();
    // chekc the speed by calculating the mean speed of all iterations
    TheInputImage.copyTo(TheInputImageCopy);
    if (iShowAllCandidates){
        auto candidates=MDetector.getCandidates();
        for(auto cand:candidates)
            Marker(cand,-1).draw(TheInputImageCopy, Scalar(255, 0, 255),1);
    }

    for (unsigned int i = 0; i < TheMarkers.size(); i++){
        cout << TheMarkers[i] << endl;
        TheMarkers[i].draw(TheInputImageCopy, Scalar(0, 0, 255),2);
    }

    // draw a 3d cube in each marker if there is 3d info
    if (TheCameraParameters.isValid())
        for (unsigned int i = 0; i < TheMarkers.size(); i++)
            CvDrawingUtils::draw3dCube(TheInputImageCopy, TheMarkers[i], TheCameraParameters);
    cv::putText(TheInputImageCopy,"fps="+to_string(1./Fps.getAvrg() ),cv::Point(10,20),FONT_HERSHEY_SIMPLEX, 0.5f,cv::Scalar(125,255,255),2,LINE_AA);

    cv::imshow("in",  TheInputImageCopy );
    cv::imshow("thres", resize(MDetector.getThresholdedImage(), 1024));
    if(showMennu)printMenuInfo();

}
