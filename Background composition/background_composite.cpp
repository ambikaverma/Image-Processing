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
    VideoCapture cap("C:/Users/ambika.v/Desktop/data_from_matlab_before_bg_composite/test5.avi"); //to get VideoSrc
    VideoCapture cap2("C:/Users/ambika.v/Desktop/data_from_matlab_before_bg_composite/masks_test5.avi"); // to get alphas

    int frames_num = cap.get(CV_CAP_PROP_FRAME_COUNT);
    int fps= cap.get(CV_CAP_PROP_FPS);

    int index=0;
    int startFrame;
    int endFrame;

    startFrame=0;
    endFrame=98;

    Mat frame_in, mask_in;
    Mat mask_gray;
    vector <Mat> alpha_maps;
    vector <Mat> video_src;
    //vector <Mat> alphas;
    alpha_maps.clear();
    video_src.clear();
    //frames_num=2;
    while(index<frames_num){
        bool success = cap.read(frame_in);
        if (!success){
            cout<<"cannot read";
        }
        bool success2 = cap2.read(mask_in);
        if (!success2){
            cout<<"cannot read";
        }
        frame_in.convertTo(frame_in,CV_32FC3,1.0/255.0);
        mask_in.convertTo(mask_in,CV_32FC3,1.0/255.0);
        cvtColor(mask_in,mask_gray,CV_BGR2GRAY);
        alpha_maps.push_back(mask_gray.clone());
        video_src.push_back(frame_in.clone());
        index++;
    }
    cout<<alpha_maps.size()<<endl;
    vector<Mat>::const_iterator first = alpha_maps.begin()+startFrame;
    vector<Mat>::const_iterator last = alpha_maps.begin()+endFrame+1;
    vector<Mat> alphas(first, last);
    cout<<alphas.size()<<endl;

    int len=endFrame-startFrame+1;
    int w,h;
    w=frame_in.cols;
    h=frame_in.rows;
    Mat alpha_sum=Mat::zeros(h,w,CV_32FC1);
    double minval;
    double maxval;
    Point minloc;
    Point maxloc;

    for (int i=0;i<len;i++){
        for (int y=0;y<h;y++){
            for (int x=0;x<w;x++){
                alpha_sum.at<float>(y,x)=round(alphas[i].at<float>(y,x)+alpha_sum.at<float>(y,x));
            }
        }
        //alpha_sum=alphas[i]+alpha_sum;
        //add(alpha_sum,alphas[i],alpha_sum);
    }
    cout<<alpha_sum.type()<<endl;
    cout<<"w "<<w<<endl;
    cout<<"h "<<h<<endl;
    /*Mat blah;
    blah=alpha_sum;
    blah.convertTo(blah,CV_8UC1,255);
    imwrite("C:/Users/ambika.v/Desktop/data_from_matlab_before_bg_composite/alpha_sum.jpg",blah);*/
    minMaxLoc(alpha_sum,&minval,&maxval,&minloc,&maxloc);

    cout<<"min value"<<minval<<endl;
    cout<<"max value"<<maxval<<endl;

    namedWindow("test2",CV_WINDOW_AUTOSIZE);
    imshow("test2",alpha_sum);

    Mat mask=Mat::zeros(h,w,CV_32FC1);

    for (int y=0;y<h;y++){
        for (int x=0;x<w;x++){
            float value=alpha_sum.at<float>(y,x);
            if (value>0){
                mask.at<float>(y,x)=1;
            }
            else{
                mask.at<float>(y,x)=0;
            }
            //mask.at<float>(y,x)=round(alpha_sum.at<float>(y,x));
        }
    }

    cout<<mask.size()<<endl;
    //mask.convertTo(mask,CV_8UC3,255);
    namedWindow("mask",CV_WINDOW_AUTOSIZE);
    imshow("mask",mask);

    minMaxLoc(mask,&minval,&maxval,&minloc,&maxloc);
    cout<<"min value"<<minval<<endl;
    cout<<"max value"<<maxval<<endl;
    //mask=mask/maxval;
    //threshold(mask,mask,0,1,THRESH_BINARY);
    //minMaxLoc(mask,&minval,&maxval,&minloc,&maxloc);
    //cout<<"min value"<<minval<<endl;
    //cout<<"max value"<<maxval<<endl;
    Mat beta_sum, temp;
    temp=Mat::ones(h,w,CV_32FC1);
    beta_sum = (endFrame-startFrame+1)*temp - alpha_sum;

    minMaxLoc(beta_sum,&minval,&maxval,&minloc,&maxloc);
    //beta_sum=beta_sum/maxval;
    cout<<"min value"<<minval<<endl;
    cout<<"max value"<<maxval<<endl;

    namedWindow("betasum",CV_WINDOW_AUTOSIZE);
    imshow("betasum",beta_sum);

    vector <Mat> beta_maps;
    Mat betamap=Mat::zeros(h,w,CV_32FC1);
    Mat B1=Mat::zeros(h,w,CV_32FC3);

    for (int i=0;i<len;i++){
        for (int y=0;y<h;y++){
            for (int x=0;x<w;x++){
                if (beta_sum.at<float>(y,x)!=0){
                    betamap.at<float>(y,x)=(round(1-alphas[i].at<float>(y,x)))/beta_sum.at<float>(y,x);
                }
                else{
                    betamap.at<float>(y,x)=0;
                }
            //betamap.at<float>(y,x)=round((1-alphas[i].at<float>(y,x))/beta_sum.at<float>(y,x));
            }

        }
        //namedWindow("betamap",CV_WINDOW_AUTOSIZE);
        //imshow("betamap",betamap);
        //waitKey(0);
        minMaxLoc(betamap,&minval,&maxval,&minloc,&maxloc);
        cout<<"min value"<<minval<<endl;
        cout<<"max value"<<maxval<<endl;
        beta_maps.push_back(temp-alphas[i]); ///// ADD ROUND HERE
        //namedWindow("test beta",CV_WINDOW_AUTOSIZE);
        //imshow("test beta",beta_maps[i]);
        //waitKey(0);
        Mat betac3;
        Mat t[]={betamap,betamap,betamap};
        merge(t,3,betac3);
        B1=B1+betac3.mul(video_src[i+startFrame]);
    }
    /*Mat betac3;
    Mat t[]={beta_sum,beta_sum,beta_sum};
    merge(t,3,betac3);
    B1=betac3.mul(video_src[startFrame]);*/
    namedWindow("B1",CV_WINDOW_AUTOSIZE);
    imshow("B1",B1);
    waitKey(0);
    //B2 = bgImg.*(1-mask) + B1.*mask;
    Mat masklarge, masksmall;
    Mat element=getStructuringElement(MORPH_ELLIPSE,Size(25,25));

    dilate(mask,masklarge,element);
    erode(mask,masksmall,element);

    namedWindow("dilate",CV_WINDOW_AUTOSIZE);
    imshow("dilate",masklarge);

    minMaxLoc(masklarge,&minval,&maxval,&minloc,&maxloc);
    cout<<"min value"<<minval<<endl;
    cout<<"max value"<<maxval<<endl;

    minMaxLoc(masksmall,&minval,&maxval,&minloc,&maxloc);
    cout<<"min value"<<minval<<endl;
    cout<<"max value"<<maxval<<endl;

    namedWindow("erode",CV_WINDOW_AUTOSIZE);
    imshow("erode",masksmall);
    waitKey(0);

    Mat diff=Mat::zeros(h,w,CV_32FC1);

    subtract(masklarge,masksmall,diff);

    namedWindow("diff",CV_WINDOW_AUTOSIZE);
    imshow("diff",diff);
    waitKey(0);

    Mat colorImg;
    Mat t[]={Mat::zeros(h,w,CV_32FC1),Mat::ones(h,w,CV_32FC1),Mat::zeros(h,w,CV_32FC1)};
    merge(t,3,colorImg);

    namedWindow("colorimg",CV_WINDOW_AUTOSIZE);
    imshow("colorimg",colorImg);
    waitKey(0);

    namedWindow("bgimg",CV_WINDOW_AUTOSIZE);
    imshow("bgimg",video_src[0]);
    waitKey(0);

    Mat B1new=Mat::zeros(h,w,CV_32FC3);
    for (int y=0;y<h;y++){
        for (int x=0;x<w;x++){
            Vec3f temp_val;
            Vec3f temp_val1=video_src[0].at<Vec3f>(y,x);
            Vec3f temp_val2=B1.at<Vec3f>(y,x);
            temp_val[0]=temp_val1[0]*round(1-mask.at<float>(y,x))+round(mask.at<float>(y,x))*temp_val2[0];
            temp_val[1]=temp_val1[1]*round(1-mask.at<float>(y,x))+round(mask.at<float>(y,x))*temp_val2[1];
            temp_val[2]=temp_val1[2]*round(1-mask.at<float>(y,x))+round(mask.at<float>(y,x))*temp_val2[2];
            B1new.at<Vec3f>(y,x)=temp_val;
        }
    }
    Mat B2;
    B1new.copyTo(B2);
    namedWindow("new b2",CV_WINDOW_AUTOSIZE);
    imshow("new b2",B1new);
    waitKey(0);

    for (int y=0;y<h;y++){
            for (int x=0;x<w;x++){
                if (diff.at<float>(y,x)>0){
                    B2.at<Vec3f>(y,x)=colorImg.at<Vec3f>(y,x);
                }
            }
    }

    namedWindow("B2",CV_WINDOW_AUTOSIZE);
    imshow("B2",B2);
    waitKey(0);

    //////////////////////////////////////INPAINT////////////////////////////////////
    /*inputs are:
    B1
    B2
    vector [ 0 255 0] i.e. colorimg
    betamaps
    videosrc*/
    Mat fillRegion=Mat::zeros(h,w,CV_32FC1);
    Mat sourceRegion=Mat::ones(h,w,CV_32FC1);

    for (int y=0;y<h;y++){
            for (int x=0;x<w;x++){
                if (B2.at<Vec3f>(y,x)==colorImg.at<Vec3f>(y,x)){
                    fillRegion.at<float>(y,x)=1.0;
                    sourceRegion.at<float>(y,x)=0;
                }
            }
    }

    namedWindow("fillregion",CV_WINDOW_AUTOSIZE);
    imshow("fillregion",fillRegion);

    namedWindow("sourceregion",CV_WINDOW_AUTOSIZE);
    imshow("sourceregion",sourceRegion);
    waitKey(0);

    Mat origImg=B1new;
    Mat img[3];
    split(origImg,img);

    namedWindow("img0",CV_WINDOW_AUTOSIZE);
    imshow("img0",img[0]);
    namedWindow("img1",CV_WINDOW_AUTOSIZE);
    imshow("img1",img[1]);
    namedWindow("img2",CV_WINDOW_AUTOSIZE);
    imshow("img2",img[2]);
    cout<<"img0 type"<<img[0].type()<<endl;
    cout<<"img1 type"<<img[1].type()<<endl;
    cout<<"img2 type"<<img[2].type()<<endl;

    minMaxLoc(img[0],&minval,&maxval,&minloc,&maxloc);
    cout<<"min value"<<minval<<endl;
    cout<<"max value"<<maxval<<endl;
    minMaxLoc(img[1],&minval,&maxval,&minloc,&maxloc);
    cout<<"min value"<<minval<<endl;
    cout<<"max value"<<maxval<<endl;
    minMaxLoc(img[2],&minval,&maxval,&minloc,&maxloc);
    cout<<"min value"<<minval<<endl;
    cout<<"max value"<<maxval<<endl;

    waitKey(0);
    ////////////////gradient////////////////////////////////
    int spacing=1;
    int ind=0;
    Mat gx0=Mat::zeros(h,w,CV_32FC1);
    Mat gx1=Mat::zeros(h,w,CV_32FC1);
    Mat gx2=Mat::zeros(h,w,CV_32FC1);
    Mat gy0=Mat::zeros(h,w,CV_32FC1);
    Mat gy1=Mat::zeros(h,w,CV_32FC1);
    Mat gy2=Mat::zeros(h,w,CV_32FC1);

    for(int i = 0; i < h; i++){
        gx0.at<float>(i,0)=img[0].at<float>(i,1)-img[0].at<float>(i,0);
        gx1.at<float>(i,0)=img[1].at<float>(i,1)-img[1].at<float>(i,0);
        gx2.at<float>(i,0)=img[2].at<float>(i,1)-img[2].at<float>(i,0);
        gx0.at<float>(i,w-1)=img[0].at<float>(i,w-1)-img[0].at<float>(i,w-2);
        gx1.at<float>(i,w-1)=img[1].at<float>(i,w-1)-img[1].at<float>(i,w-2);
        gx2.at<float>(i,w-1)=img[2].at<float>(i,w-1)-img[2].at<float>(i,w-2);
    }

    for (int i=0;i<h;i++){
        for (int j=1;j<w-1;j++){
            gx0.at<float>(i,j)=(img[0].at<float>(i,j+1)-img[0].at<float>(i,j-1))/2.0;
            gx1.at<float>(i,j)=(img[1].at<float>(i,j+1)-img[1].at<float>(i,j-1))/2.0;
            gx2.at<float>(i,j)=(img[2].at<float>(i,j+1)-img[2].at<float>(i,j-1))/2.0;
        }
    }

    for(int j = 0; j < w; j++){
        gy0.at<float>(0,j)=img[0].at<float>(1,j)-img[0].at<float>(0,j);
        gy1.at<float>(0,j)=img[1].at<float>(1,j)-img[1].at<float>(0,j);
        gy2.at<float>(0,j)=img[2].at<float>(1,j)-img[2].at<float>(0,j);
        gy0.at<float>(h-1,j)=img[0].at<float>(h-1,j)-img[0].at<float>(h-2,j);
        gy1.at<float>(h-1,j)=img[1].at<float>(h-1,j)-img[1].at<float>(h-2,j);
        gy2.at<float>(h-1,j)=img[2].at<float>(h-1,j)-img[2].at<float>(h-2,j);
    }

    for (int i=1;i<h-1;i++){
        for (int j=0;j<w;j++){
            gy0.at<float>(i,j)=(img[0].at<float>(i+1,j)-img[0].at<float>(i-1,j))/2.0;
            gy1.at<float>(i,j)=(img[1].at<float>(i+1,j)-img[1].at<float>(i-1,j))/2.0;
            gy2.at<float>(i,j)=(img[2].at<float>(i+1,j)-img[2].at<float>(i-1,j))/2.0;
        }
    }

    Mat gx=Mat::zeros(h,w,CV_32FC1);
    Mat gy=Mat::zeros(h,w,CV_32FC1);

    for (int y=0;y<h;y++){
        for (int x=0;x<w;x++){
            gx.at<float>(y,x)=(gx0.at<float>(y,x)+gx1.at<float>(y,x)+gx2.at<float>(y,x))/3.0;
            gy.at<float>(y,x)=(gy0.at<float>(y,x)+gy1.at<float>(y,x)+gy2.at<float>(y,x))/3.0;
        }
    }

    namedWindow("gx",CV_WINDOW_AUTOSIZE);
    imshow("gx",gx);
    namedWindow("gy",CV_WINDOW_AUTOSIZE);
    imshow("gy",gy);
    waitKey(0);

    Mat temp_grad=Mat::zeros(h,w,CV_32FC1);
    temp_grad=gx;
    gx=-1*gy;
    gy=temp_grad;
    /////////////////////initialize confidence and data////////////////////////////
    Mat C=sourceRegion;
    Mat D=Mat::ones(h,w,CV_32FC1);
    D=(-0.1)*D;
    int iter=1;
    cv::RNG RNG(0);
    float flt[3][3]={{1.0,1.0,1.0},{1.0,-8.0,1.0},{1.0,1.0,1.0}};
    Mat f;
    f=Mat(3,3,CV_32FC1,&flt);
    //int n=countNonZero(fillRegion);
    //int n=1;
    /*float A[4][4]={{1.0,2.0,3.0,4.0},{1.0,1.0,1.0,1.0},{5.0,6.0,7.0,8.0},{2.0,2.0,2.0,2.0}};
    Mat Anew;
    Anew=Mat(4,4,CV_32FC1,&A);
    Mat test;
    filter2D(Anew,test,-1,f,Point(-1,-1),0,BORDER_CONSTANT);
    cout<<test<<endl;*/
    //int n=5;
    //minMaxLoc(fillRegion,&minval,&maxval,&minloc,&maxloc);
    //cout<<"min value"<<minval<<endl;
    //cout<<"max value"<<maxval<<endl;
    while(countNonZero(fillRegion)>0 && iter<500){

        Mat fillRegionD=fillRegion;
        Mat dR=Mat::zeros(h,w,CV_32FC1);
        filter2D(fillRegionD,dR,-1,f,Point(-1,-1),0,BORDER_CONSTANT);
        cout<<dR.size()<<endl;
        //transpose(dR,dR);
        //cout<<dR.type()<<endl;
        vector <float> dRind;
        vector <int> xind;
        vector <int> yind;

        for (int x=0;x<w;x++){
            for (int y=0;y<h;y++){
                if (dR.at<float>(y,x)>0.0){
                    dRind.push_back(x*h+y);
                    xind.push_back(x);
                    yind.push_back(y);
                }
            }
        }
        /*for (int i=0;i<xind.size();i++){
            cout<<"x "<<xind[i]<<endl;
            cout<<"y "<<yind[i]<<endl;
        }*/
        /*cout<<dRind.size()<<endl;
        cout<<dRind[1162]<<endl;
        cout<<xind.size()<<endl;
        cout<<yind.size()<<endl;*/
        Mat Nx=Mat::zeros(h,w,CV_32FC1);
        Mat Ny=Mat::zeros(h,w,CV_32FC1);
        Mat test=Mat::zeros(h,w,CV_32FC1);

        for (int u=0;u<h;u++){
            for (int v=0;v<w;v++){
                    test.at<float>(u,v)=round(1-fillRegion.at<float>(u,v));
                }
        }

        for(int i = 0; i < h; i++){
            Nx.at<float>(i,0)=test.at<float>(i,1)-test.at<float>(i,0);
            Nx.at<float>(i,w-1)=test.at<float>(i,w-1)-test.at<float>(i,w-2);
        }
        for (int i=0;i<h;i++){
            for (int j=1;j<w-1;j++){
                Nx.at<float>(i,j)=(test.at<float>(i,j+1)-test.at<float>(i,j-1))/2.0;
            }
        }
        for(int j = 0; j < w; j++){
            Ny.at<float>(0,j)=test.at<float>(1,j)-test.at<float>(0,j);
            Ny.at<float>(h-1,j)=test.at<float>(h-1,j)-test.at<float>(h-2,j);
        }
        for (int i=1;i<h-1;i++){
            for (int j=0;j<w;j++){
                Ny.at<float>(i,j)=(test.at<float>(i+1,j)-test.at<float>(i-1,j))/2.0;
            }
        }
        //cout<<Nx.rows<<endl;
        //cout<<Nx.cols<<endl;
        int s=dRind.size();
        Mat N=Mat::zeros(s,2,CV_32FC1);
        for (int z=0;z<s;z++){
            float nx=Nx.at<float>(yind[z],xind[z]);
            float ny=Ny.at<float>(yind[z],xind[z]);
            float ntemp=sqrt(pow(nx,2)+pow(ny,2));
            float ntemp1=nx/ntemp;
            float ntemp2=ny/ntemp;
            N.at<float>(z,0)=ntemp1;
            N.at<float>(z,1)=ntemp2;
        }
        //cout<<N.size()<<endl;
        minMaxLoc(N,&minval,&maxval,&minloc,&maxloc);
        //cout<<"min value"<<minval<<endl;
        //cout<<"max value"<<maxval<<endl;
        for (int i=0;i<dRind.size();i++){
            //int k=(int)dRind[i];
            int wtemp=8;
            //int p=k-1;
            //int y=floor(p/400)+1;
            //p=p % 400;
            //int x=floor(p)+1;
            int x=xind[i];
            int y=yind[i];
            int r1=max(y-wtemp,0);
            //cout<<"r1"<<r1<<endl;
            int r2=min(y+wtemp,h-1);
            int l1=r2-r1+1;
            //cout<<l1<<endl;
            int c1=max(x-wtemp,0);
            //cout<<"c1"<<c1<<endl;
            int c2=min(x+wtemp,w-1);
            int l2=c2-c1+1;
            //cout<<l2<<endl;
            Mat Xtemp=Mat::zeros(l1,1,CV_32FC1);
            Mat Ytemp=Mat::zeros(1,l2,CV_32FC1);
            Mat X=Mat::zeros(l1,l2,CV_32FC1);
            Mat Y=Mat::zeros(l1,l2,CV_32FC1);
            //Mat Hp=Mat::zeros(17,17,CV_32FC1);
            int temp_r=r1-1;
            for (int i=0;i<l1;i++){
                    Xtemp.at<float>(i,0)=++temp_r;
            }
            int temp_c=c1-1;
            for (int j=0;j<l2;j++){
                    Ytemp.at<float>(0,j)=++temp_c;
            }
            repeat(Xtemp,1,l2,X);
            repeat(Ytemp,l1,1,Y);

            //cout<<X<<endl;
            //cout<<Y<<endl;
            //Hp=X+(Y-1)*400;
            vector <int> x2;
            vector <int> y2;
            for (int u=0;u<l1;u++){
                for (int v=0;v<l2;v++){
                    int pos1=X.at<float>(u,v);
                    int pos2=Y.at<float>(u,v);
                    float val=fillRegion.at<float>(pos1,pos2);
                    if(val==0){
                        x2.push_back(pos1);
                        y2.push_back(pos2);
                    }
                }
            }
            //cout<<x2.size()<<endl;
            float sum_var=0;
            for (int i=0;i<x2.size();i++){
                sum_var=sum_var+C.at<float>(x2[i],y2[i]);
            }
            C.at<float>(y,x)=sum_var/(l1*l2);
        }

        Mat priorities=Mat::zeros(s,1,CV_32FC1);
        for (int i=0;i<s;i++){
            float D_var=abs(gx.at<float>(yind[i],xind[i])*N.at<float>(i,0)+gy.at<float>(yind[i],xind[i])*N.at<float>(i,1))+0.001;
            D.at<float>(yind[i],xind[i])=D_var;
            float prior_var=C.at<float>(yind[i],xind[i])*D.at<float>(yind[i],xind[i]);
            priorities.at<float>(i,0)=prior_var;
        }
        cout<<priorities.size()<<endl;
        minMaxLoc(priorities,&minval,&maxval,&minloc,&maxloc);
        //cout<<"min value"<<minval<<endl;
        //cout<<"max value"<<maxval<<endl;
        cout<<"max loc"<<maxloc<<endl;

        int loc=(int) maxloc.y; //// TODO
        cout<< "loc"<<loc<<endl;
        ////////////////get patch/////////////////////
        int wtemp=8;
        int xmax=xind[loc];
        int ymax=yind[loc];
        int r1=max(ymax-wtemp,0);
        cout<<"r1 "<<r1<<endl;
        int r2=min(ymax+wtemp,h-1);
        cout<<"r2 "<<r2<<endl;
        int l1=r2-r1+1;
        int c1=max(xmax-wtemp,0);
        cout<<"c1 "<<c1<<endl;
        int c2=min(xmax+wtemp,w-1);
        cout<<"c2 "<<c2<<endl;
        int l2=c2-c1+1;
        Mat Xtemp=Mat::zeros(l1,1,CV_32FC1);
        Mat Ytemp=Mat::zeros(1,l2,CV_32FC1);
        Mat X=Mat::zeros(l1,l2,CV_32FC1);
        Mat Y=Mat::zeros(l1,l2,CV_32FC1);
            //Mat Hp=Mat::zeros(17,17,CV_32FC1);
        int temp_r=r1-1;
        for (int i=0;i<l1;i++){
                Xtemp.at<float>(i,0)=++temp_r;
        }
        int temp_c=c1-1;
        for (int j=0;j<l2;j++){
                Ytemp.at<float>(0,j)=++temp_c;
        }
        repeat(Xtemp,1,l2,X);
        repeat(Ytemp,l1,1,Y);
        cout<<"X size "<<X.size()<<endl;
        cout<<"Y size "<<Y.size()<<endl;
        //cout<<X<<endl;
        //cout<<Y<<endl;
        Mat tofill=Mat::zeros(l1,l2,CV_32FC1);
        for (int i=0;i<l1;i++){
            for (int j=0;j<l2;j++){
                tofill.at<float>(i,j)=fillRegion.at<float>(X.at<float>(i,j),Y.at<float>(i,j));
            }
        }
        minMaxLoc(tofill,&minval,&maxval,&minloc,&maxloc);
        cout<<"min value"<<minval<<endl;
        cout<<"max value"<<maxval<<endl;
        //cout<<tofill<<endl;
        //cout<<X<<endl;
        //cout<<Y<<endl;

        if (countNonZero(tofill)==0){
            break;
        }
        /////////////////////best exemplar video///////////////////////////////////
        //cout<<video_src.size()<<endl;
        //int num=99;
        float bestErr = 1000000000.0;
        Mat probe=Mat::zeros(l1,l2,CV_32FC3);
        for (int i=0;i<l1;i++){
            for (int j=0;j<l2;j++){
                probe.at<Vec3f>(i,j)=B1new.at<Vec3f>(X.at<float>(i,j),Y.at<float>(i,j));
            }
        }
        Mat mask_var=Mat::zeros(l1,l2,CV_32FC3);
        Mat t[]={tofill,tofill,tofill};
        merge(t,3,mask_var);
        int bestpick=0;
        //cout<<"wow"<<endl;

        for (int i=startFrame;i<=endFrame;i++){
            Mat frame=video_src[i];
            Mat target=Mat::zeros(l1,l2,CV_32FC3);
            for (int i=0;i<l1;i++){
                for (int j=0;j<l2;j++){
                    target.at<Vec3f>(i,j)=frame.at<Vec3f>(X.at<float>(i,j),Y.at<float>(i,j));
                }
            }
            //cout<<"wow in"<<endl;

            Mat beta=beta_maps[i-startFrame];
            Mat segmentation=Mat::zeros(l1,l2,CV_32FC1);
            for (int i=0;i<l1;i++){
                for (int j=0;j<l2;j++){
                    segmentation.at<float>(i,j)=beta.at<float>(X.at<float>(i,j),Y.at<float>(i,j));
                }
            }
            minMaxLoc(segmentation,&minval,&maxval,&minloc,&maxloc);
            //cout<<"seg min value"<<minval<<endl;
            //cout<<"seg max value"<<maxval<<endl;
            if(!(minval==0)){
                //cout<<segmentation<<endl;
                float patcherr=0;
                for (int i=0;i<l1;i++){
                    for (int j=0;j<l2;j++){
                        Vec3f blah;
                        blah[0]=pow(((probe.at<Vec3f>(i,j)[0]-target.at<Vec3f>(i,j)[0])*mask_var.at<Vec3f>(i,j)[0]),2);
                        blah[1]=pow(((probe.at<Vec3f>(i,j)[1]-target.at<Vec3f>(i,j)[1])*mask_var.at<Vec3f>(i,j)[1]),2);
                        blah[2]=pow(((probe.at<Vec3f>(i,j)[2]-target.at<Vec3f>(i,j)[2])*mask_var.at<Vec3f>(i,j)[2]),2);
                        float blah2=blah[0]+blah[1]+blah[2];
                        patcherr=patcherr+blah2;
                    }
                }
                //cout<<patcherr<<endl;
                if (patcherr<bestErr){
                    bestErr=patcherr;
                    bestpick=i;
                }
            }

        }
        cout<<"bestpick"<<bestpick<<endl;

        for (int i=0;i<l1;i++){
            for (int j=0;j<l2;j++){
                fillRegion.at<float>(X.at<float>(i,j),Y.at<float>(i,j))=0;
                if (tofill.at<float>(i,j)>0){
                    //fillRegion.at<float>(X.at<float>(i,j),Y.at<float>(i,j))=0;
                    C.at<float>(X.at<float>(i,j),Y.at<float>(i,j))=C.at<float>(ymax,xmax);

                }
            }
        }
        cout<<countNonZero(fillRegion)<<endl;
        /////////////////////////gradient///////////////////////////
        Mat video_best=video_src[bestpick];
        /*Mat vidbest=Mat::zeros(17,17,CV_32FC3);
        for (int i=0;i<17;i++){
            for (int j=0;j<17;j++){
                vidbest.at<Vec3f>(i,j)=video_best.at<Vec3f>(X.at<float>(i,j),Y.at<float>(i,j));
            }
        }*/
        Mat vidbest_split[3];
        split(video_best,vidbest_split);
        //int st=17;
        Mat tx0=Mat::zeros(h,w,CV_32FC1);
        Mat tx1=Mat::zeros(h,w,CV_32FC1);
        Mat tx2=Mat::zeros(h,w,CV_32FC1);
        Mat ty0=Mat::zeros(h,w,CV_32FC1);
        Mat ty1=Mat::zeros(h,w,CV_32FC1);
        Mat ty2=Mat::zeros(h,w,CV_32FC1);

        for(int i = 0; i < h; i++){
            tx0.at<float>(i,0)=vidbest_split[0].at<float>(i,1)-vidbest_split[0].at<float>(i,0);
            tx1.at<float>(i,0)=vidbest_split[1].at<float>(i,1)-vidbest_split[1].at<float>(i,0);
            tx2.at<float>(i,0)=vidbest_split[2].at<float>(i,1)-vidbest_split[2].at<float>(i,0);
            tx0.at<float>(i,w-1)=vidbest_split[0].at<float>(i,w-1)-vidbest_split[0].at<float>(i,w-2);
            tx1.at<float>(i,w-1)=vidbest_split[1].at<float>(i,w-1)-vidbest_split[1].at<float>(i,w-2);
            tx2.at<float>(i,w-1)=vidbest_split[2].at<float>(i,w-1)-vidbest_split[2].at<float>(i,w-2);
        }

        for (int i=0;i<h;i++){
            for (int j=1;j<w-1;j++){
                tx0.at<float>(i,j)=(vidbest_split[0].at<float>(i,j+1)-vidbest_split[0].at<float>(i,j-1))/2.0;
                tx1.at<float>(i,j)=(vidbest_split[1].at<float>(i,j+1)-vidbest_split[1].at<float>(i,j-1))/2.0;
                tx2.at<float>(i,j)=(vidbest_split[2].at<float>(i,j+1)-vidbest_split[2].at<float>(i,j-1))/2.0;
            }
        }

        for(int j = 0; j < w; j++){
            ty0.at<float>(0,j)=vidbest_split[0].at<float>(1,j)-vidbest_split[0].at<float>(0,j);
            ty1.at<float>(0,j)=vidbest_split[1].at<float>(1,j)-vidbest_split[1].at<float>(0,j);
            ty2.at<float>(0,j)=vidbest_split[2].at<float>(1,j)-vidbest_split[2].at<float>(0,j);
            ty0.at<float>(h-1,j)=vidbest_split[0].at<float>(h-1,j)-vidbest_split[0].at<float>(h-2,j);
            ty1.at<float>(h-1,j)=vidbest_split[1].at<float>(h-1,j)-vidbest_split[1].at<float>(h-2,j);
            ty2.at<float>(h-1,j)=vidbest_split[2].at<float>(h-1,j)-vidbest_split[2].at<float>(h-2,j);
        }

        for (int i=1;i<h-1;i++){
            for (int j=0;j<w;j++){
                ty0.at<float>(i,j)=(vidbest_split[0].at<float>(i+1,j)-vidbest_split[0].at<float>(i-1,j))/2.0;
                ty1.at<float>(i,j)=(vidbest_split[1].at<float>(i+1,j)-vidbest_split[1].at<float>(i-1,j))/2.0;
                ty2.at<float>(i,j)=(vidbest_split[2].at<float>(i+1,j)-vidbest_split[2].at<float>(i-1,j))/2.0;
            }
        }
        Mat tx=Mat::zeros(h,w,CV_32FC1);
        Mat ty=Mat::zeros(h,w,CV_32FC1);

        for (int y=0;y<h;y++){
            for (int x=0;x<w;x++){
                tx.at<float>(y,x)=(tx0.at<float>(y,x)+tx1.at<float>(y,x)+tx2.at<float>(y,x))/3.0;
                ty.at<float>(y,x)=(ty0.at<float>(y,x)+ty1.at<float>(y,x)+ty2.at<float>(y,x))/3.0;
            }
        }
        tx=-1*tx;
        for (int i=0;i<l1;i++){
            for (int j=0;j<l2;j++){
                    if (tofill.at<float>(i,j)>0){
                        gx.at<float>(X.at<float>(i,j),Y.at<float>(i,j))=tx.at<float>(X.at<float>(i,j),Y.at<float>(i,j));
                        gy.at<float>(X.at<float>(i,j),Y.at<float>(i,j))=ty.at<float>(X.at<float>(i,j),Y.at<float>(i,j));
                    }
            }
        }

        for (int i=0;i<l1;i++){
            for (int j=0;j<l2;j++){
                origImg.at<Vec3f>(X.at<float>(i,j),Y.at<float>(i,j))=video_src[bestpick].at<Vec3f>(X.at<float>(i,j),Y.at<float>(i,j));
            }
        }
        iter++;
        cout<<iter<<endl;
        //minMaxLoc(fillRegion,&minval,&maxval,&minloc,&maxloc);
        //cout<<"min value"<<minval<<endl;
        //cout<<"max value"<<maxval<<endl;
        //n--;
    }

    namedWindow("orgimg",CV_WINDOW_AUTOSIZE);
    imshow("orgimg",origImg);
    origImg.convertTo(origImg,CV_8UC3,255);
    imwrite("C:/Users/ambika.v/Desktop/testing/inpaint_test5_cpp.jpg",origImg);

    waitKey(0);
    return 0;
}
