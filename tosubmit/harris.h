#ifndef HARRIS_H
#define HARRIS_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <set>
using namespace std;
using namespace cv;
#define debugFlag 0
#define debug(x) cerr << #x << ": " << x << endl;
#define debugv(v) {cerr << #v << ": ";for(int qwe = 0; qwe < (int)v.size();qwe++) cerr << v[qwe] << ", "; cerr << endl;}
int dIx[3][3]=
{
		{-1, 0, 1},
		{-1, 0, 1},
		{-1, 0, 1}
};
int dIy[3][3]=
{
		{-1, -1, -1},
		{ 0,  0,  0},
		{ 1,  1,  1}
};
int dr[3] ={-1,0,1};
int dc[3] ={-1,0,1};
//void calcIx(Mat &, Mat &);
//void calcIy(Mat &, Mat &);
//void calcIx2(Mat &,Mat &);
//void calcIy2(Mat &,Mat &);
//void calcIxIy(Mat &,Mat &,Mat &);
//void calcH(Mat &,Mat &,
//		Mat &,  vector<vector<Mat> > &, int, int);
//// This function is responsible for calculating R matrix and the direction at each point
//void calcRAndDir(vector<vector<Mat> > &, Mat &, Mat &, double &, double &);
//// This function is responsible for setting the R value of the border
//// of the image to 0.
//void cleanse(Mat &R, int pixels);
//void nonMaximumSuppression(Mat&,int,int,double);
//void calcCorners(Mat &, vector<vector<bool> > &, double, double, double);
//// This function is responsible for placing a red circle over the corners.
//void overlayCircle(Mat &, vector<vector<bool> > &);
//// This function displays the Mat as an image.
//void display(Mat, string, bool toScale=false, int mn=0, int mx=0, bool isAbs=0);
void display(Mat img, string msg, bool toScale=false, int mn=0, int mx=0, bool isAbs=0){
	if(toScale){
		Mat tmp;
		tmp=Mat(img.rows,img.cols,0);
		int diff = mx-mn;
		int highest = 0;
		for(int r = 0; r < img.rows; r++){
			for(int c = 0; c < img.cols; c++){
				int t = img.at<int>(r,c);
				if(isAbs)t=abs(t);
				tmp.at<uchar>(r,c) = 1.0*(t-mn)/diff * 255;
				highest = max(highest,(int)tmp.at<uchar>(r,c));
			}
		}
		namedWindow( "Display " + msg, CV_WINDOW_AUTOSIZE );
		imshow( "Display " + msg, tmp );

		waitKey(0);
	} else {
		namedWindow( "Display " + msg, CV_WINDOW_AUTOSIZE );
		imshow( "Display " + msg, img);

		waitKey(0);
	}

}
inline int calcSingleIx(Mat &img, int r, int c){
	int sum = 0;
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			int nr = r+dr[i], nc = c+dc[j];
			if(nr < 0 || nr >= img.rows || nc < 0 || nc >= img.cols) continue;
			sum += dIx[i][j]*(int)img.at<uchar>(nr,nc);
		}
	}
	return sum;
}
void calcIx(Mat &img, Mat &Ix){
	Ix=Mat(img.rows,img.cols,CV_32S);
	for(int r = 0; r < img.rows; r++){
		for(int c = 0; c < img.cols; c++){
			Ix.at<int>(r,c) =  calcSingleIx(img,r,c);
		}
	}
	if(debugFlag)
		display(Ix, "Ix", 1,0,255,1);
}
inline int calcSingleIy(Mat &img, int r, int c){
	int sum = 0;
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			int nr = r+dr[i], nc = c+dc[j];
			if(nr < 0 || nr >= img.rows || nc < 0 || nc >= img.cols) continue;
			sum += dIy[i][j]*(int)img.at<uchar>(nr,nc);
		}
	}
	return sum;
}
void calcIy(Mat &img, Mat &Iy){
	Iy=Mat(img.rows,img.cols,CV_32S);

	for(int r = 0; r < img.rows; r++){
		for(int c = 0; c < img.cols; c++){
			Iy.at<int>(r,c) = calcSingleIy(img,r,c);
		}
	}
	if(debugFlag)
		display(Iy, "Iy", 1,0,255,1);
}
void calcIx2(Mat &Ix,Mat &Ix2){
	Ix2 = Mat(Ix.rows,Ix.cols,CV_32S);
	int mx = 0;
	for(int r = 0; r < Ix2.rows; r++){
		for(int c = 0; c < Ix2.cols; c++){
			Ix2.at<int>(r,c) = (int)Ix.at<int>(r,c)*(int)Ix.at<int>(r,c);
			mx = max((int)Ix2.at<int>(r,c),mx);
		}
	}
	if(debugFlag)
		display(Ix2, "Ix2",true,0,255*3*255*3);
}
void calcIy2(Mat &Iy,Mat &Iy2){
	Iy2 = Mat(Iy.rows,Iy.cols,CV_32S);
	int mx = 0;
	for(int r = 0; r < Iy2.rows; r++){
		for(int c = 0; c < Iy2.cols; c++){
			Iy2.at<int>(r,c) = (int)Iy.at<int>(r,c)*(int)Iy.at<int>(r,c);
			mx = max((int)Iy2.at<int>(r,c),mx);
		}
	}
	if(debugFlag)
		display(Iy2, "Iy2",true,0,255*3*255*3);
}
void calcIxIy(Mat &Ix,Mat &Iy,Mat &IxIy){
	assert(Ix.rows==Iy.rows);
	assert(Ix.cols==Iy.cols);
	IxIy = Mat(Ix.rows,Ix.cols,CV_32S);
	for(int r = 0; r < IxIy.rows; r++){
		for(int c = 0; c < IxIy.cols; c++){
			IxIy.at<int>(r,c) = (int)Ix.at<int>(r,c)*(int)Iy.at<int>(r,c);
		}
	}
	if(debugFlag)
		display(IxIy, "IxIy", 1,0,2*255*2*255,1);
}
void calcH(Mat &Ix2,Mat &Iy2,
		Mat &IxIy,  vector<vector<Mat> > &H, int rows, int cols){
	assert(Ix2.rows==Iy2.rows);
	assert(Ix2.cols==Iy2.cols);
	assert(rows==cols);
	assert(rows%2);
	H.clear();
	H = vector<vector<Mat> >(Ix2.rows, vector<Mat>(Ix2.cols));

	for(int r = 0; r < (int)H.size(); r++){
		for(int c = 0; c < (int)H[0].size(); c++){
			H[r][c] = Mat(2,2,CV_64F);
			H[r][c].at<double>(0,0) = 0;
			H[r][c].at<double>(1,1) = 0;
			H[r][c].at<double>(0,1) = 0;
			H[r][c].at<double>(1,0) = 0;
			for(int i = max(-r,-(rows/2)); i <= rows/2 && r+i < (int)H.size(); i++){
				for(int j = max(-c,-(cols/2)); j <= cols/2 && c+j < (int)H[0].size(); j++){
					H[r][c].at<double>(0,0) += Ix2.at<int>(r+i,c+j);
					H[r][c].at<double>(1,1) += Iy2.at<int>(r+i,c+j);
					H[r][c].at<double>(0,1) += IxIy.at<int>(r+i,c+j);
					H[r][c].at<double>(1,0) += IxIy.at<int>(r+i,c+j);
				}
			}

		}
	}
}
void calcRAndDir(vector<vector<Mat> > &H, Mat &R, Mat &dir, double &minR, double &maxR){

	assert(H.size());
	assert(H[0].size());

	double alpha = 0.05;
	minR = DBL_MAX;
	maxR = -DBL_MAX;
	R = Mat(H.size(),H[0].size(),CV_64F);
	dir = Mat(H.size(),H[0].size(),CV_64F);
	for(int r = 0; r < (int)H.size(); r++){
		for(int c = 0; c < (int)H[0].size(); c++){
			R.at<double>(r,c) = determinant(H[r][c]) - alpha * trace(H[r][c])[0];

			maxR = max(maxR, R.at<double>(r,c));
			minR = min(minR, R.at<double>(r,c));
			Mat eigenValues, eigenVectors;
			eigen(H[r][c],eigenValues,eigenVectors);

			if(eigenValues.at<double>(0,0) > eigenValues.at<double>(1,0)){
				dir.at<double>(r,c) = atan2(eigenVectors.at<double>(0,1),eigenVectors.at<double>(0,0));
			} else {
				dir.at<double>(r,c) = atan2(eigenVectors.at<double>(1,1),eigenVectors.at<double>(1,0));
			}
		}
	}
	if(debugFlag)
		display(R, "R");
}
void cleanse(Mat &R, int pixels){
	assert(R.rows>=pixels);
	assert(R.cols>=pixels);
	for(int r = 0; r < R.rows; r++)
		for(int i = 0; i < pixels; i++)
			R.at<double>(r,i)=R.at<double>(r,R.cols-1-i)=0;
	for(int c = 0; c < R.cols; c++)
		for(int i = 0; i < pixels; i++)
			R.at<double>(i,c)=R.at<double>(R.rows-1-i,c)=0;
}

void nonMaximumSuppression(Mat& R, int rows, int cols, double minR){
	Mat tmp = R.clone();
	for(int r = 0; r < R.rows; r++){
		for(int c = 0; c < R.cols; c++){
			for(int i = max(-r,-(rows/2)); i <= rows/2 && r+i < R.rows; i++){
				for(int j = max(-c,-(cols/2)); j <= cols/2 && c+j < R.cols; j++){
					if(i==0&&j==0)continue;
					tmp.at<double>(r,c)+=R.at<double>(i+r,j+c);
				}
			}
		}
	}
	for(int r = 0; r < R.rows; r++){
		for(int c = 0; c < R.cols; c++){
			for(int i = max(-r,-(rows/2)); i <= rows/2 && r+i < R.rows; i++){
				for(int j = max(-c,-(cols/2)); j <= cols/2 && c+j < R.cols; j++){
					if(tmp.at<double>(r,c) > tmp.at<double>(i+r,j+c))
						 R.at<double>(i+r,j+c) = minR;
				}
			}
		}
	}
	for(int r = 0; r < R.rows; r++){
		for(int c = 0; c < R.cols; c++){
			for(int i = 0; i < rows && r+i < R.rows; i++){
				for(int j = 0; j < cols && c+j < R.cols; j++){
					if(i==0&&j==0)continue;
					if(tmp.at<double>(r,c) >= tmp.at<double>(i+r,j+c))
						 R.at<double>(i+r,j+c) = minR;
				}
			}
		}
	}
}
void calcCorners(Mat &R, vector<vector<bool> > &isCorner, double threshold, double minR, double maxR){
	threshold = (maxR-minR)*threshold/100.0+minR;

	isCorner = vector<vector<bool> >(R.rows,vector<bool>(R.cols));
	double mxR = -DBL_MAX;
	for(int r = 0; r < R.rows; r++){
		for(int c = 0; c < R.cols; c++){
			isCorner[r][c]=(R.at<double>(r,c)>threshold);
			mxR = max(mxR,R.at<double>(r,c));
		}
	}
	if(debugFlag){
		Mat tmp(R.rows, R.cols, CV_8U);
		for(int r = 0; r < R.rows; r++)
			for(int c = 0; c < R.cols; c++){
				tmp.at<uchar>(r,c)=255*isCorner[r][c];
			}
		display(tmp, "isCorner");
	}
}
void overlayCircle(Mat &img, vector<vector<bool> > &isCorner){
	int thickness = -1;
	int lineType = 8;
	int rad=3;
	int cnt = 0;
	for(int r = 0; r < img.rows; r++){
		for(int c = 0; c < img.cols; c++){
			if(isCorner[r][c]){
				cnt++;
				 circle( img,
				         Point(c,r),
				         rad,
				         Scalar( 0, 0, 255 ),
				         thickness,
				         lineType );
			}
		}
	}

	if(debugFlag)
		display(img,"overlay");
}

inline string int2str(int n){
	string s = "";
	while(n)s+=char(n%10+'0'),n/=10;
	reverse(s.begin(),s.end());
	return s;
}
void applyHarris(char *in, vector<vector<bool> > &isCorner, Mat &orientation, double threshold = 5,
		int winSide = 5)
{
  Mat grayscale_img, img;
//  char in[]=filename;
//  char out[]="/media/ghooo/01CBEED4B2009090/Coursework/Computer Vision/Assignment#1b/TestImages/shapesCorners.jpg";
  img = imread( in);
  cvtColor( img, grayscale_img, CV_BGR2GRAY );
//  display(img, "IMG");
  Mat Ix, Iy, Ix2, Iy2, IxIy;
  vector<vector<Mat> > H;
  Mat R;

  calcIx(grayscale_img,Ix);
  calcIy(grayscale_img,Iy);
  calcIx2(Ix,Ix2);
  calcIy2(Iy,Iy2);
  calcIxIy(Ix,Iy,IxIy);
  calcH(Ix2,Iy2,IxIy,H,winSide,winSide);

  double maxR, minR;
  calcRAndDir(H,R,orientation,minR, maxR);
  cleanse(R,winSide);
  nonMaximumSuppression(R,winSide,winSide,minR);
  calcCorners(R,isCorner,threshold,minR,maxR);
//  overlayCircle(img,isCorner);
//  display(img,"IMG Corners"+int2str(winSide));
//  imwrite(out,img);
//  return 0;
}
void applyHarris(Mat &img, vector<vector<bool> > &isCorner, Mat &orientation, double threshold = 5,
		int winSide = 5)
{
  Mat grayscale_img;
//  char in[]=filename;
//  char out[]="/media/ghooo/01CBEED4B2009090/Coursework/Computer Vision/Assignment#1b/TestImages/shapesCorners.jpg";
//  img = imread( in);
  cvtColor( img, grayscale_img, CV_BGR2GRAY );
//  display(img, "IMG");
  Mat Ix, Iy, Ix2, Iy2, IxIy;
  vector<vector<Mat> > H;
  Mat R;

  calcIx(grayscale_img,Ix);
  calcIy(grayscale_img,Iy);
  calcIx2(Ix,Ix2);
  calcIy2(Iy,Iy2);
  calcIxIy(Ix,Iy,IxIy);
  calcH(Ix2,Iy2,IxIy,H,winSide,winSide);

  double maxR, minR;
  calcRAndDir(H,R,orientation,minR, maxR);
  cleanse(R,winSide);
  nonMaximumSuppression(R,winSide,winSide,minR);
  calcCorners(R,isCorner,threshold,minR,maxR);
//  overlayCircle(img,isCorner);
//  display(img,"IMG Corners"+int2str(winSide));
//  imwrite(out,img);
//  return 0;
}
#endif //HARRIS_H
