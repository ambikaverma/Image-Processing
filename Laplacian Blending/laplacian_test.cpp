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
    Mat imgA,imgB,mask;
    imgB = imread("C:/Users/ambika.v/Downloads/laplacianBlend-master/laplacianBlend-master/white.jpg",CV_LOAD_IMAGE_COLOR);
    cout<<imgA.type()<<endl;
    imgA = imread("C:/Users/ambika.v/Downloads/laplacianBlend-master/laplacianBlend-master/black.jpg",CV_LOAD_IMAGE_COLOR);
    mask=imread("C:/Users/ambika.v/Downloads/laplacianBlend-master/laplacianBlend-master/mask.png",CV_LOAD_IMAGE_COLOR);
    //imgA.convertTo(imgA,CV_64FC3,1.0/255.0); works
    //imgA.convertTo(imgA,CV_32FC3,1.0/255.0); //works
    imgA.convertTo(imgA,CV_64FC3,1.0/255.0); // works
    imgB.convertTo(imgB,CV_64FC3,1.0/255.0);
    mask.convertTo(mask,CV_64FC3,1.0/255.0);
    // bad result with cv_8UC3 (lowest bit format, i.e. lowest memory).
    // same result as MATLAB with CV_64FC3 i.e. 64 bit floating point.
    // other options are - CV_8SC3, CV_16UC3 (same result as 8UC3), CV_16S, CV_32S, CV_32F.
    // does not work with 8S and 32S, not supported formats by pyrdown. same result as 8U with 16U. need signed due to presence of negative numbers.
    cout<<imgA.type()<<endl;
    //imgB = imread("white.jpg",CV_LOAD_IMAGE_COLOR);

    /*namedWindow("original 1",CV_WINDOW_AUTOSIZE);
    imshow("original 1",imgA);*/

    //namedWindow("original 2",CV_WINDOW_AUTOSIZE);
    //imshow("original 2",imgB);

    cout<<"imgA size"<<imgA.size()<<endl;
    //cout<<"imgB size"<<imgB.size()<<endl;

    int levels=4;

    vector <Mat> G1;

    Mat src_img, dst_img;
    src_img=imgA;

    G1.push_back(src_img);

    for (int i=1;i<=levels-1;i++){
        pyrDown(src_img,dst_img);
        G1.push_back(dst_img);
        src_img=dst_img;
    }

    //cout<<G1[0].size()<<endl;
    //cout<<G1[1].size()<<endl;
    //cout<<G1[2].size()<<endl;
    //cout<<G1[3].size()<<endl;

    vector <Mat> G2;

    Mat src_img2, dst_img2;
    src_img2=imgB;

    G2.push_back(src_img2);

    for (int i=1;i<=levels-1;i++){
        pyrDown(src_img2,dst_img2);
        G2.push_back(dst_img2);
        src_img2=dst_img2;
    }

    vector <Mat> G3;

    Mat src_img3, dst_img3;
    src_img3=mask;

    G3.push_back(src_img3);

    for (int i=1;i<=levels-1;i++){
        pyrDown(src_img3,dst_img3);
        G3.push_back(dst_img3);
        src_img3=dst_img3;
    }

    namedWindow("1",CV_WINDOW_AUTOSIZE);
    imshow("1",G3[0]);
    namedWindow("2",CV_WINDOW_AUTOSIZE);
    imshow("2",G3[1]);
    namedWindow("3",CV_WINDOW_AUTOSIZE);
    imshow("3",G3[2]);
    namedWindow("4",CV_WINDOW_AUTOSIZE);
    imshow("4",G3[3]);

    vector <Mat> L1;

    Mat lpA=G1[levels-1];
    Mat lp_temp,L;

    L1.push_back(G1[levels-1]);

    for (int i=levels-1;i>=1;i--){
        pyrUp(lpA,lp_temp,G1[i-1].size());
        //resize(G1[i-1],G1[i-1],lp_temp.size());
        //subtract(G1[i-1],lp_temp,L);
        L=G1[i-1]-lp_temp;
        L1.push_back(L);
        lpA=G1[i-1];
    }

    //cout<<L1.size()<<endl;
    //cout<<L1[0].size()<<endl;
    //cout<<L1[1].size()<<endl;
    //cout<<L1[2].size()<<endl;
    //cout<<L1[3].size()<<endl;

    vector <Mat> L2;

    Mat lpB=G2[levels-1];
    Mat lp_temp2,L_2;

    L2.push_back(G2[levels-1]);

    for (int i=levels-1;i>=1;i--){
        pyrUp(lpB,lp_temp2,G2[i-1].size());
        //resize(G2[i-1],G2[i-1],lp_temp2.size());
        //subtract(G1[i-1],lp_temp,L);
        L_2=G2[i-1]-lp_temp2;
        L2.push_back(L_2);
        lpB=G2[i-1];
    }

    vector <Mat> L3;
    Mat blend_temp;
    int temp_w,temp_h;
    int mask_value;

    for (int i=0;i<=levels-1;i++){
        temp_w=L1[i].cols;
        temp_h=L1[i].rows;
        cout<<temp_h<<" "<<temp_w<<endl;
        blend_temp=Mat::zeros(temp_h,temp_w,CV_64FC3);
        resize(G3[i],G3[i],blend_temp.size());

        for (int y=0;y<temp_h;y++){
            for (int x=0;x<temp_w;x++){
                Vec3d value1=G3[i].at<Vec3d>(y,x);
                //cout<<value1<<endl;
                //cout<<1-value1[0]<<" "<<1-value1[1]<<" "<<1-value1[2]<<endl;
                /*if (value1[0]>0 && value1[1]>0 && value1[2]>0){
                    mask_value=1;
                }
                else{
                    mask_value=0;
                }*/
                Vec3d value2=L1[i].at<Vec3d>(y,x);
                Vec3d value3=L2[i].at<Vec3d>(y,x);
                Vec3d value4;
                //value4[0]=mask_value*value3[0]+(1-mask_value)*value2[0];
                //value4[1]=mask_value*value3[1]+(1-mask_value)*value2[1];
                //value4[2]=mask_value*value3[2]+(1-mask_value)*value2[2];
                value4[0]=value1[0]*value3[0]+(1-value1[0])*value2[0];
                value4[1]=value1[1]*value3[1]+(1-value1[1])*value2[1];
                value4[2]=value1[2]*value3[2]+(1-value1[2])*value2[2];
                blend_temp.at<Vec3d>(y,x)=value4;
            }
        }
        L3.push_back(blend_temp);
    }

    Mat r,r_temp;
    Mat R;
    r_temp=L3[0];

    for (int i=0;i<levels-1;i++){
        pyrUp(r_temp,r,L3[i+1].size());
        //add(r,L1[i+1],R);
        R=L3[i+1]+r;
        r_temp=R;

    }

    cout<<R.size()<<endl;

    namedWindow("recon 1",CV_WINDOW_AUTOSIZE);
    R.convertTo(R,CV_8UC3,255);
    imshow("recon 1",R);

    imwrite("C:/Users/ambika.v/Desktop/testing/blend_test1_cpp.png",R);

    waitKey(0);


    return 0;
}
