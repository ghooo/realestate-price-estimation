#ifndef SIFT_H
#define SIFT_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "harris.h"
using namespace cv;

/*
 * 1. constructing a scale space
 * 2. Laplacian of Gaussian approximation
 * 3. Finding keypoints
 * 4. Elimnate edges and low contrast regions
 * 5. Assign an orientation to the key point
 * 6. Generate SIFT features
 */
Point calcPointInLine(const Point &p, double theta, double dist){
	double x, y;
	if(fabs(theta-M_PI/2.0)<1e-9) x=0, y=1;
	else if(fabs(theta-M_PI*3.0/2.0)<1e-9) x=0, y=-1;
	else {
		double t = tan(theta);
		double b = sqrt(t*t+1);
		x = 1/b, y=t/b;
	}
	x*=dist,y*=dist;
	x+=p.x, y+=p.y;
	return Point(x,y);
}
void overlayArrow(Mat &img, vector<vector<bool> > &isCorner, Mat &orientation, vector<vector<int> > scale){
	int thickness = 2;
	int lineType = 8;
	for(int r = 0; r < img.rows; r++){
		for(int c = 0; c < img.cols; c++){
			if(isCorner[r][c]){
				line(img,
					 Point(c,r),
					 calcPointInLine(Point(c,r),orientation.at<double>(r,c),scale[r][c]),
				     Scalar( 255, 0, 0 ),
				     thickness,
				     lineType );
			}
		}
	}
	char *out1 = "/media/ghooo/01CBEED4B2009090/Coursework/Computer Vision/Assignment#2/TestImages/jimOut1.jpg";
	char *out2 = "/media/ghooo/01CBEED4B2009090/Coursework/Computer Vision/Assignment#2/TestImages/jimOut2.jpg";

	static int idx = 0;
	if(idx==0) imwrite(out1,img);
	else imwrite(out2,img);
	idx++;

//	if(debugFlag)
		display(img,"Arrows Overlay"+rand()%128);
}

void rotate(Mat& src, Mat &dst, double angle)
{
    Point2f pt(src.cols/2., src.rows/2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, Size(src.cols, src.rows));
}
void rotate(int i, int j, int &ni, int &nj, int w, int h, double theta){
	i-=h/2;
	j-=w/2;
	ni = round(i*cos(theta)-j*sin(theta)), nj= round(i*sin(theta)+j*cos(theta));
	ni+=h/2;
	nj+=w/2;
}
void calcSIFTFeature(Mat &img, int r, int c, double theta, int scale, vector<double>&res){
	Mat rotated_img;
	rotate(img,rotated_img,theta*180/M_PI);
	int nr, nc;
	rotate(r,c,nr,nc,img.cols,img.rows,theta);
	r = nr, c = nc;

	int rows = scale*4, cols = scale*4;
	double oneNormalized = 1.0/(scale*scale*4*4);
	double sift[4][4][8]={};

	for(int i = -(rows/2); i < rows/2; i++){
		for(int j = -(cols/2); j < cols/2; j++){
			int ni = i, nj = j;
			if(ni+r < 0 || ni+r >= rotated_img.rows || nj+c < 0 || nj+c >= rotated_img.cols)continue;
			double sumX = 0, sumY = 0;
			for(int ii = -3; ii <= 3; ii++){
				for(int jj = -3; jj <= 3; jj++){
					sumX+=calcSingleIx(rotated_img,r+ni+ii,c+nj+jj);
					sumY+=calcSingleIy(rotated_img,r+ni+ii,c+nj+jj);
				}
			}
			double a = atan2(sumY,sumX);
			if(a < 0) a+=2*M_PI;

			int subPatchR = (i+rows/2)/scale;
			int subPatchC = (j+cols/2)/scale;
			sift[subPatchR][subPatchC][min(7,(int)(a/(0.25*M_PI)))]+=oneNormalized;
		}
	}


	res.clear();
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			for(int k = 0; k < 8; k++){
				res.push_back(sift[i][j][k]);
			}
		}
	}
}
void applySIFT(char *in, vector<vector<double> >& features, vector<pair<int,int> >& featureCenter)
{
	vector<vector<bool> > isCorner[4],finalCorners;
	Mat orientation[4], finalOrientation;
	vector<vector<int> > finalScale;
	const int WINSIDECNT=1;
	int winSide[WINSIDECNT]={5};
	for(int i = 0; i < WINSIDECNT; i++){
		applyHarris(in,isCorner[i],orientation[i],2,winSide[i]);
		if(i==0){
			finalCorners=isCorner[i];
			finalOrientation=orientation[i].clone();
			finalScale = vector<vector<int> > (finalCorners.size(),vector<int>(finalCorners[0].size(),winSide[0]));
		} else {
			for(int j = 0; j < (int)finalCorners.size(); j++){
				for(int k = 0; k < (int)finalCorners[0].size(); k++){
					if(isCorner[i][j][k]){
						finalCorners[j][k]=1;
						finalScale[j][k]=winSide[i];
						finalOrientation.at<double>(j,k)=orientation[i].at<double>(j,k);
					}
				}
			}
		}
	}
	Mat img = imread(in), grayscale_img;

	cvtColor( img, grayscale_img, CV_BGR2GRAY );
//	vector<vector<double> > features;
	features.clear();
	featureCenter.clear();
	for(int r = 0; r < grayscale_img.rows; r++){
		for(int c = 0; c < grayscale_img.cols; c++){
			if(finalCorners[r][c]){
				featureCenter.push_back(make_pair(r,c));
				features.push_back(vector<double>());
				calcSIFTFeature(grayscale_img,r,c,finalOrientation.at<double>(r,c),finalScale[r][c],features.back());
			}
		}
	}
	overlayArrow(img,finalCorners,finalOrientation,finalScale);
}
#endif //SIFT_H
