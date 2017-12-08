#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;

int main()
{
    VideoCapture cap("C:/Users/ambika.v/Desktop/data_from_matlab_before_bg_composite/test2.avi");
    VideoCapture cap2("C:/Users/ambika.v/Desktop/data_from_matlab_before_bg_composite/masks_test2.avi");

    int frames_num = cap.get(CV_CAP_PROP_FRAME_COUNT);
    int frames_num2 = cap2.get(CV_CAP_PROP_FRAME_COUNT);

    //cout<<frames_num<<endl;
    //cout<<frames_num2<<endl;

    double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    double dWidth2 = cap2.get(CV_CAP_PROP_FRAME_WIDTH);
    double dHeight2 = cap2.get(CV_CAP_PROP_FRAME_HEIGHT);
    int fps= cap.get(CV_CAP_PROP_FPS);

    cout<<dWidth<<" "<<dHeight<<endl;
    cout<<dWidth2<<" "<<dHeight2<<endl;

    //dWidth=dWidth+7;
    //dHeight=dHeight+5;

    Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

    VideoWriter result_out ("C:/Users/ambika.v/Desktop/testing/test2_new_test.avi", CV_FOURCC('M','J','P','G'), fps, frameSize, true);

    Mat bgImg;

    bgImg = imread("C:/Users/ambika.v/Desktop/testing/inpaint_test2_cpp.jpg", CV_LOAD_IMAGE_COLOR);
    //bgImg.convertTo(bgImg,CV_16SC3);
    bgImg.convertTo(bgImg,CV_64FC3,1.0/255.0);
    //namedWindow("background Image", WINDOW_AUTOSIZE);
    //imshow("background Image",bgImg);

    //waitKey(0);

    vector <Mat> G1;
    vector <Mat> G2;
    vector <Mat> G3;

    int levels = 4;

    Mat src_img, dst_img;
    Mat src_imgB, dst_imgB;
    Mat src_imgC, dst_imgC;
    src_img=bgImg;

    G1.push_back(src_img);

    for (int i=1;i<=levels-1;i++){
        pyrDown(src_img,dst_img);
        //cout<<dst_img.size();
        G1.push_back(dst_img);
        src_img=dst_img;
    }

    vector <Mat> L1;
    vector <Mat> L2;

    Mat lpA=G1[levels-1];
    Mat lp_temp,L;

    L1.push_back(G1[levels-1]);

    for (int i=levels-1;i>=1;i--){
        pyrUp(lpA,lp_temp,G1[i-1].size());
        //cout<<lpA.size()<<endl;
        //cout<<lp_temp.size()<<endl;
        //cout<<G1[i-1].size()<<endl;
        //resize(G1[i-1],G1[i-1],lp_temp.size());
        //subtract(G1[i-1],lp_temp,L);
        L=G1[i-1]-lp_temp;
        //cout<<L.size()<<endl;
        L1.push_back(L);
        lpA=G1[i-1];
    }

    int index=0;
    //frames_num=2;
    int mask_value;

    while(index<frames_num){
        G2.clear();
        G3.clear();
        L2.clear();
        Mat frame_in, mask_in;
        bool success = cap.read(frame_in);
        if (!success){
            cout<<"cannot read";
        }

        bool success2 = cap2.read(mask_in);
        if (!success2){
            cout<<"cannot read";
        }

        //cout<<frame_in.type()<<endl;
        //cout<<mask_in.type()<<endl;

        //frame_in.convertTo(frame_in,CV_16SC3);
        //mask_in.convertTo(mask_in,CV_16SC3);
        frame_in.convertTo(frame_in,CV_64FC3,1.0/255.0);
        mask_in.convertTo(mask_in,CV_64FC3,1.0/255.0);

        /*namedWindow("mask",CV_WINDOW_AUTOSIZE);
        mask_in.convertTo(mask_in,CV_8UC3,255);
        imshow("mask",mask_in);*/

        src_imgB=frame_in;
        G2.push_back(src_imgB);

        for (int i=1;i<=levels-1;i++){
                pyrDown(src_imgB,dst_imgB);
                G2.push_back(dst_imgB);
                src_imgB=dst_imgB;
        }

        src_imgC=mask_in;
        G3.push_back(src_imgC);

        for (int i=1;i<=levels-1;i++){
                pyrDown(src_imgC,dst_imgC);
                //cout<<dst_img.size();
                G3.push_back(dst_imgC);
                src_imgC=dst_imgC;
        }

        Mat lpB=G2[levels-1];
        L2.push_back(G2[levels-1]);

        for (int i=levels-1;i>=1;i--){
                pyrUp(lpB,lp_temp,G2[i-1].size());
                //resize(G2[i-1],G2[i-1],lp_temp.size());
                //subtract(G2[i-1],lp_temp,L);
                L=G2[i-1]-lp_temp;
                L2.push_back(L);
                lpB=G2[i-1];
        }

        vector <Mat> R;
        Mat r_temp;
        int temp_w,temp_h;

        for (int i=0;i<=levels-1;i++){
                temp_w=L1[i].cols;
                temp_h=L1[i].rows;

                cout<<temp_h<<" "<<temp_w<<endl;
                //r_temp=Mat::zeros(temp_h,temp_w,CV_16SC3);
                r_temp=Mat::zeros(temp_h,temp_w,CV_64FC3);
                resize(G3[i],G3[i],r_temp.size());

                for (int y=0;y<temp_h;y++){
                    for (int x=0;x<temp_w;x++){
                        //Vec3s value1=G3[i].at<Vec3s>(y,x);
                        //Vec3f value1=G3[i].at<Vec3f>(y,x);
                        Vec3d value1=G3[i].at<Vec3d>(y,x);
                        //Vec3b value1=mask_in.at<Vec3b>(y,x);
                        //cout<<value1<<endl;
                        if (value1[0]>0 && value1[1]>0 && value1[2]>0){
                            mask_value=1;
                            //cout<<"yes"<<endl;
                        }
                        else{
                            mask_value=0;
                        }
                        //Vec3s value2=L1[i].at<Vec3s>(y,x);
                        //Vec3s value3=L2[i].at<Vec3s>(y,x);
                        //Vec3s value4;
                        //Vec3f value2=L1[i].at<Vec3f>(y,x);
                        //Vec3f value3=L2[i].at<Vec3f>(y,x);
                        //Vec3f value4;
                        Vec3d value2=L1[i].at<Vec3d>(y,x);
                        Vec3d value3=L2[i].at<Vec3d>(y,x);
                        Vec3d value4;
                        value4[0]=mask_value*value3[0]+(1-mask_value)*value2[0];
                        value4[1]=mask_value*value3[1]+(1-mask_value)*value2[1];
                        value4[2]=mask_value*value3[2]+(1-mask_value)*value2[2];
                        //r_temp.at<Vec3s>(y,x)=value4;
                        //r_temp.at<Vec3f>(y,x)=value4;
                        r_temp.at<Vec3d>(y,x)=value4;
                        //r_temp.at<Vec3b>(y,x)=mask_in.at<Vec3b>(y,x)*L2[i].at<Vec3b>(y,x)+(1-mask_in.at<Vec3b>(y,x))*L1[i].at<Vec3b>(y,x);

                    }
                }
                R.push_back(r_temp);
                //cout<<R[i].size()<<endl;
        }

        Mat final_temp=R[0];
        Mat out,r;

        for (int i=1;i<=levels-1;i++){
            pyrUp(final_temp,r,R[i].size());
            //add(out,R[i],out);
            out=R[i]+r;
            final_temp=out;
        }

        //out.convertTo(out,CV_8UC3);
        out.convertTo(out,CV_8UC3,255);

        result_out.write(out);

        /*namedWindow("out",CV_WINDOW_AUTOSIZE);
        imshow("out",out);

        waitKey(0);*/
        index++;
    }

    return 0;
}
