/*
 * Vision-Assignment3-main.cpp
 *
 *  Created on: May 13, 2015
 *      Author: ghooo
 */

#include <iomanip>
#include "harris.h"
#include "sift.h"
#include "svm.h"
#include <dirent.h>
#include <map>

enum keyPointType{SIFTDETECTOR,HARRIS,DENSE};
enum featureType{SIFTDESCRIPTOR,HOG};
enum clusteringType{KMEANS,SOM};

#define LOG 0
#define funcStarted() if(LOG) cerr << __FUNCTION__ << " STARTED" << endl;
#define funcEnded() if(LOG) cerr << __FUNCTION__ << " ENDED" << endl;
const int IMAGESCNTPERCLASS = 15;
const double CANONICALHEIGHT = 128;
const int TESTIMAGESCOUNT = 20;
const int SCALE = 5;
const int CLASSCOUNT = 101;
const int TRAININGCLASSCNT = 4;
const int TESTINGCLASSCNT = 4;
const keyPointType KPT = HARRIS;

map<double,string> classes;
//const string CALTECH101="/media/ghooo/01CBEED4B2009090/Coursework/Computer Vision/Assignment#3/101_ObjectCategories/101_ObjectCategories/";
const string CALTECH101="/media/ghooo/01CBEED4B2009090/Coursework/Computer Vision/Project/D2/training/";


struct dataPoint{
	string filePath;
	string className;
	double classId;
	Mat img;
	vector<vector<bool> > isKeyPoint;
	Mat orientation;
	vector<vector<double> > features;
	vector<double> histogram;
};
void saveData(vector<dataPoint> &data, string filename){
	// TODO: write this function
	assert(false);
}
void loadData(vector<dataPoint> &data, string filename){
	// TODO: implement this function
	assert(false);
}
void getDataSize(vector<dataPoint> &data){
	cerr << endl;
	cerr << "=============================================" << endl;
	long long ret=0;
	for(int i = 0; i < data.size(); i++){
		dataPoint &dp = data[i];
		ret += dp.filePath.size();
		ret += dp.className.size();
		ret += 8; //classId
		if(dp.img.rows) ret += 1*dp.img.rows*dp.img.cols;
		if(dp.isKeyPoint.size()) ret += dp.isKeyPoint.size()*dp.isKeyPoint[0].size();
		if(dp.orientation.rows) ret += 8*dp.orientation.rows*dp.orientation.cols;
		if(dp.features.size()) ret += 8*dp.features.size()*dp.features[0].size();
		if(dp.histogram.size()) ret += 8*dp.histogram.size();
		if(ret == 236650){
			cerr << ret << endl;
		}
		cerr << ret << endl;
	}
	cerr << "DATA SIZE: " << endl;
	cerr << ret << " BYTES" << endl;
	cerr << ret/1024 << " KILO BYTES" << endl;
	cerr << ret/1024/1024 << " MEGA BYTES" << endl;
	cerr << "=============================================" << endl;
	cerr << endl;

}
ostream&operator<<(ostream &out, const dataPoint &dp){
	out << "{" << dp.filePath << ", " << dp.className << ", " << dp.classId << "}";
	return out;
}
void getListOfFolders(vector<string>&folders, string parent=CALTECH101){
	funcStarted();
    const char* PATH = parent.c_str();

    DIR *dir = opendir(PATH);

    struct dirent *entry = readdir(dir);

    while (entry != NULL)
    {
    	if(entry->d_type == DT_DIR && entry->d_name[0]!='.')
    		folders.push_back(entry->d_name);
        entry = readdir(dir);
    }

    closedir(dir);
    funcEnded();
}
void getListOfFiles(vector<string>&files, string folder){
	funcStarted();
    const char* PATH = folder.c_str();

    DIR *dir = opendir(PATH);

    struct dirent *entry = readdir(dir);

    while (entry != NULL)
    {
    	if(entry->d_type != DT_DIR && entry->d_name[0]!='.')
    		files.push_back(entry->d_name);
        entry = readdir(dir);
    }
    sort(files.begin(), files.end());

    closedir(dir);
	funcEnded();
}
void getKeyPoints(Mat &img, vector<vector<bool> > &isKeyPoint, Mat &orientation,
		int scale = SCALE, keyPointType kpt = HARRIS){
	funcStarted();
	if(kpt == HARRIS){
		applyHarris(img,isKeyPoint,orientation,0,scale);
	} else if(kpt == DENSE){
		int side = scale*4;
		isKeyPoint = vector<vector<bool> > (img.rows,vector<bool>(img.cols,0));
		orientation = Mat(img.rows,img.cols,CV_64F);
		for(int r = 0; r < img.rows; r++)
			for(int c = 0; c < img.cols; c++)
				orientation.at<double>(r,c) = 0.0;

		for(int r = side/2; r < img.rows; r+=side){
			for(int c = side/2; c < img.cols; c+=side){
				isKeyPoint[r][c]=true;
			}
		}
	} else if(kpt == SIFTDETECTOR){
		assert(kpt!=SIFTDETECTOR);
	}
	funcEnded();
}
void getAllImagesKeyPoints(vector<dataPoint>& data){
	funcStarted();
	for(int i = 0; i < (int)data.size(); i++){
		cerr << i << "/" << data.size()<<endl;
		getKeyPoints(data[i].img, data[i].isKeyPoint, data[i].orientation, SCALE, KPT);
	}
	funcEnded();
}
void getFeatures(Mat &img, vector<vector<bool> > &isKeyPoint,
		Mat &orientation, int scale, vector<vector<double> > &features,
		featureType ft = SIFTDESCRIPTOR){
	funcStarted();

	assert(img.rows == (int)isKeyPoint.size());
	assert(img.cols == (int)isKeyPoint[0].size());
	assert(img.rows == orientation.rows);
	assert(img.cols == orientation.cols);
	features.clear();
	if(ft == SIFTDESCRIPTOR){
		for(int r = 0; r < img.rows; r++){
			for(int c = 0; c < img.cols; c++){
				if(isKeyPoint[r][c]){
					features.push_back(vector<double>());
					calcSIFTFeature(img,r,c,orientation.at<double>(r,c),scale,features.back());
				}
			}
		}
	} else if(ft == HOG){
		assert(ft != HOG);
	}
	funcEnded();
}
void getAllImagesFeatures(vector<dataPoint> &data){
	funcStarted();
	for(int i = 0; i < (int)data.size(); i++){
		cerr << i << "/" << data.size()<<endl;
		getFeatures(data[i].img, data[i].isKeyPoint, data[i].orientation, SCALE, data[i].features);
	}
	funcEnded();
}
void groupAllFeatures(vector<dataPoint> &data, vector<vector<double> > &features){//might be named getAllFeatures
	funcStarted();
	features.clear();
	for(int i = 0; i < (int)data.size(); i++){
		features.insert(features.end(),data[i].features.begin(),data[i].features.end());
	}
	funcEnded();

}
void getRandomKinNElements(int k, int n, vector<int>&res){
	funcStarted();
	assert(k<=n);
	res.resize(n);
	for(int i = 0; i < n; i++)res[i]=i;
	random_shuffle(res.begin(),res.end());
	res.erase(res.begin()+k,res.end());
	funcEnded();

}
double dist2(vector<double> &v1, vector<double> &v2){
	double ret = 0.0;
	for(int i = 0; i < (int)v1.size(); i++)
		ret+=(v1[i]-v2[i])*(v1[i]-v2[i]);
	return ret;
}
void getClusters(vector<vector<double> > &points, vector<vector<double> > &clusters, int clusterCnt, clusteringType ct = KMEANS){
	//TODO: MAKE PARALLEL
	funcStarted();
	if(ct == KMEANS){
		assert(clusterCnt);
		int k = clusterCnt;
		if(k > points.size()) k = points.size();
		vector<int> chosenPoints;
		getRandomKinNElements(k,(int)points.size(),chosenPoints);
		clusters = vector<vector<double> > (k);
		for(int i = 0; i < k; i++){
			clusters[i]=points[chosenPoints[i]];
		}
		vector<int> assignedCluster(points.size());
		double minChange = 1, change = 0, lastChange;
		do{
			vector<int> pointsInCluster(k,0);
			for(int pointIdx = 0; pointIdx < (int)points.size(); pointIdx++){
				int closestCluster = 0;
				double closestDist = dist2(points[pointIdx],clusters[0]);
				for(int clusterIdx = 1; clusterIdx < (int)clusters.size(); clusterIdx++){
					double d = dist2(points[pointIdx],clusters[clusterIdx]);
					if(d < closestDist) {
						closestCluster = clusterIdx;
						closestDist = d;
					}
				}
				assignedCluster[pointIdx] = closestCluster;
				pointsInCluster[closestCluster]++;
			}
			vector<vector<double> > tmpClusters(clusters.size(), vector<double>(clusters[0].size(),0.0) );
//			for(int clusterIdx = 0; clusterIdx < (int)clusters.size(); clusterIdx++)
//				for(int i = 0; i < (int)clusters[i].size(); i++)
//					clusters[clusterIdx][i]=0.0;

			for(int pointIdx = 0; pointIdx < (int)points.size(); pointIdx++){
				int clusterIdx = assignedCluster[pointIdx];
				double pointsCnt = pointsInCluster[clusterIdx];
				for(int i = 0; i < (int)points[pointIdx].size(); i++){
					tmpClusters[clusterIdx][i] += points[pointIdx][i]/pointsCnt;
				}
			}
			lastChange = change;
			change = 0;
			for(int i = 0; i < (int)clusters.size(); i++)
				change += sqrt(dist2(clusters[i],tmpClusters[i]));
			swap(clusters,tmpClusters);
			debug(change);
			debug(lastChange);

		}while(fabs(change-lastChange) > minChange);
	} else if(ct == SOM){
		assert(ct != SOM);
	}
	funcEnded();
}
void getImgHistogram(vector<vector<double> > &features, vector<vector<double> > &clusters, vector<double> &histogram){
	funcStarted();
	//TODO: MAKE PARALLEL
	assert(clusters.size());

	debug(features.size());
	histogram = vector<double>(clusters.size(),0.);
	double oneNormalized = 1.0/features.size();

	for(int featureIdx = 0; featureIdx < (int)features.size(); featureIdx++){
		int closestCluster = 0;
		double closestDist = dist2(features[featureIdx], clusters[0]);
		for(int clusterIdx = 1; clusterIdx < (int)clusters.size(); clusterIdx++){
			double d = dist2(features[featureIdx], clusters[clusterIdx]);
			if(d < closestDist){
				closestCluster = clusterIdx;
				closestDist = d;
			}
		}
		histogram[closestCluster]+=oneNormalized;
	}
	funcEnded();
}
void getAllImagesHistogram(vector<dataPoint> &data, vector<vector<double> > &clusters){
	funcStarted();
	for(int i = 0; i < (int)data.size(); i++){
		cerr << i << "/" << data.size()<<endl;
		getImgHistogram(data[i].features, clusters, data[i].histogram);
	}
	funcEnded();
}
void getAllClasses(){
	funcStarted();
	vector<string> folders;
	getListOfFolders(folders,CALTECH101);
	classes.clear();
	for(double i = 0; i < folders.size(); i++)
		classes[i]=folders[i];
	funcEnded();
}
void getTrainingData(vector<dataPoint> &data, int classCnt=CLASSCOUNT){
	funcStarted();
	data.clear();
	vector<string> folders;
	getListOfFolders(folders,CALTECH101);
	if(LOG) debugv(folders);
	for(int i = 0; (classCnt==-1||i < classCnt) && i < (int)folders.size(); i++){
		string currentFolder = CALTECH101 + "/" + folders[i];
		vector<string> files;
		getListOfFiles(files,currentFolder);
		for(int j = 0; j < IMAGESCNTPERCLASS && j < (int)files.size(); j++){
			data.push_back({currentFolder+"/"+files[j],folders[i],(double)i});
		}
	}
	funcEnded();
}
void getTestingData(vector<dataPoint> &data, int fromClassCnt = -1){
	funcStarted();
	data.clear();
	vector<string> folders;
	getListOfFolders(folders,CALTECH101);
	if(LOG) debugv(folders);
	if(fromClassCnt < 0) fromClassCnt = (int)folders.size();
	if(fromClassCnt > (int)folders.size()) fromClassCnt = (int)folders.size();
	for(int i = 0; i < TESTIMAGESCOUNT; i++){
		int curClass = rand() % fromClassCnt;
		vector<string> files;
		getListOfFiles(files, CALTECH101+"/"+folders[curClass]);
		// TODO: FIX THIS
		data.push_back({CALTECH101+"/"+folders[curClass]+"/"+files[rand()%((int)files.size()-IMAGESCNTPERCLASS)+IMAGESCNTPERCLASS],
		                                                           folders[curClass],curClass});
	}
	funcEnded();
}
void getImage(Mat&img, string& filePath){
	img = imread(filePath);
	double ratio = CANONICALHEIGHT/img.rows;
	Size s(ratio*img.cols,CANONICALHEIGHT);
	resize(img, img,s);

}
void getImages(vector<dataPoint> &data){
	funcStarted();
	for(int i = 0; i < (int)data.size(); i++){
		getImage(data[i].img, data[i].filePath);
//		data[i].img = imread(data[i].filePath);
//		double ratio = CANONICALHEIGHT/data[i].img.rows;
////		display(data[i].img, data[i].className);
//		Size s(ratio*data[i].img.cols,CANONICALHEIGHT);
//		resize(data[i].img, data[i].img,s);
////		display(data[i].img, data[i].className);
	}
	funcEnded();
}
void convHistogramToSVMNode(vector<double> &histogram, svm_node *&nodes){
	funcStarted();
//	// TODO: REMOVE THIS
//	static int xx = 0;
//	xx++;
//	nodes = new svm_node[2];
//	nodes[0].index = 1, nodes[0].value = xx;
//	nodes[1].index = -1, nodes[1].value = -1;
//	return ;
//	// TODO: TILL HERE;
	int cnt = 0;
	for(int histogramIdx = 0; histogramIdx < (int)histogram.size(); histogramIdx++){
		if(histogram[histogramIdx] != 0.0) cnt++;
	}
	nodes = new svm_node[cnt+1];
	for(int histogramIdx = 0, nodeIdx = 0; histogramIdx < (int)histogram.size(); histogramIdx++){
		if(histogram[histogramIdx] != 0.0){
			nodes[nodeIdx] = svm_node();
			nodes[nodeIdx].index = histogramIdx+1;
			nodes[nodeIdx].value = histogram[histogramIdx]*100;
			nodeIdx++;
		}
	}
	nodes[cnt].index = -1;
	nodes[cnt].value = 0;

	funcEnded();
}
void convToSVMFormat(vector<dataPoint> &data, svm_problem &prob){
	funcStarted();
//	// TODO: REMOVE THIS
//	prob.l = 10;
//	prob.y =  new double[10];
//	prob.x = new svm_node*[10];
//	for(int i = 0; i < 5; i++) prob.y[i] = 1, prob.y[i+5] = 2;
//	for(int i = 0; i < 5; i++){
//		prob.x[i] = new svm_node[2];
//		prob.x[i][0].index = 1, prob.x[i][0].value = rand()%11/5.0 - 1.0;
//		prob.x[i][1].index = -1, prob.x[i][1].value = rand()%11/5.0 - 1.0;
//	}
//	for(int i = 5; i < 10; i++){
//		prob.x[i] = new svm_node[2];
//		prob.x[i][0].index = 1, prob.x[i][0].value = rand()%11/5.0 - 1.0+10;
//		prob.x[i][1].index = -1, prob.x[i][1].value = rand()%11/5.0 - 1.0+10;
//	}
//	return;
//	// TODO: TILL HERE
	prob.l = data.size();
	prob.y = new double[data.size()];
	prob.x = new svm_node*[data.size()];
	for(int dataIdx = 0; dataIdx < (int)data.size(); dataIdx++){
		prob.y[dataIdx] = data[dataIdx].classId;
		convHistogramToSVMNode(data[dataIdx].histogram,prob.x[dataIdx]);
	}
	funcEnded();
}
void getSVMPrameter(svm_parameter &param){
	funcStarted();
	// DEAFAULTS
	param = svm_parameter();
	// default values
	//	-s svm_type : set type of SVM (default 0)
	//		0 -- C-SVC		(multi-class classification)
	//		1 -- nu-SVC		(multi-class classification)
	//		2 -- one-class SVM
	//		3 -- epsilon-SVR	(regression)
	//		4 -- nu-SVR		(regression)
	param.svm_type = C_SVC;
	//	-t kernel_type : set type of kernel function (default 2)
	//		0 -- linear: u'*v
	//		1 -- polynomial: (gamma*u'*v + coef0)^degree
	//		2 -- radial basis function: exp(-gamma*|u-v|^2)
	//		3 -- sigmoid: tanh(gamma*u'*v + coef0)
	//		4 -- precomputed kernel (kernel values in training_set_file)
	param.kernel_type = RBF;
	//	-d degree : set degree in kernel function (default 3)
	param.degree = 3;
	//	-g gamma : set gamma in kernel function (default 1/num_features)
	param.gamma = 0.00000001;	// 1/num_features
	//	-r coef0 : set coef0 in kernel function (default 0)
	param.coef0 = 0;
	//	-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
	param.C = 10000;
	//	-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
	param.nu = 0.5;
	//	-m cachesize : set cache memory size in MB (default 100)
	param.cache_size = 100;
	//	-e epsilon : set tolerance of termination criterion (default 0.001)
	param.eps = 1e-6;
	//	-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
	param.p = 0.1;
	//	-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
	param.shrinking = 1;
	//	-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
	param.probability = 0;
	//	-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
//	-v n: n-fold cross validation mode
//	-q : quiet mode (no outputs)	param.
	funcEnded();
}
void test(){
	funcStarted();
	svm_parameter param;
	getSVMPrameter(param);
	funcEnded();
	exit(0);
}
#include <fstream>
void saveClusters(vector<vector<double> > &clusters, string filename){
	ofstream cout(filename.c_str());
	cout << fixed;
//	sort(clusters.begin(),clusters.end());
	for(int i = 0; i < clusters.size(); i++){
		for(int j = 0; j < clusters[i].size(); j++){
			cout << clusters[i][j] << " \n"[j==clusters[i].size()-1];
		}
	}
}
void loadClusters(vector<vector<double> > &clusters, string filename){
	ifstream cin(filename.c_str());
	string s;
	clusters.clear();
	while(getline(cin,s)){
		stringstream ss(s);
		clusters.push_back(vector<double>());
		double d;
		while(ss >> d) clusters.back().push_back(d);
	}
}
int main2(){
//	freopen("log.txt", "wt", stderr)
	funcStarted();

	// Get Training set and testing set.
	vector<dataPoint> trainingData, testingData;
	vector<Mat> trainingImages, testingImages;
	getTrainingData(trainingData,TRAININGCLASSCNT);
	debugv(trainingData);
//	getTestingData(testingData,TESTINGCLASSCNT);
	getImages(trainingData);
//	getImages(testingData);
	getAllClasses();

	// Get Image Key Points
	getAllImagesKeyPoints(trainingData);
//	getAllImagesKeyPoints(testingData);

	// get images' features
	getAllImagesFeatures(trainingData);
//	getAllImagesFeatures(testingData);

	// cluster all features.
	vector<vector<double> > features, clusters;
	groupAllFeatures(trainingData,features);
	getClusters(features,clusters,TRAININGCLASSCNT*20);
	saveClusters(clusters, "clusters.txt");
	// Get Images' Histograms
	getAllImagesHistogram(trainingData, clusters);
//	getAllImagesHistogram(testingData, clusters);

	// SVM Training
	svm_parameter param;		// set by parse_command_line
	svm_problem prob;		// set by read_problem
	svm_model *model;
	const char *error_msg;
	char *model_file_name = "svmmodel.model";

	convToSVMFormat(trainingData,prob);
	getSVMPrameter(param);
	error_msg = svm_check_parameter(&prob,&param);
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

	model = svm_train(&prob,&param);
	if(svm_save_model(model_file_name,model))
	{
		fprintf(stderr, "can't save model to file %s\n", model_file_name);
		exit(1);
	}

//	// SVM Testing
//	double correct = 0, wrong = 0;
//	for(int i = 0; i < testingData.size(); i++){
//		svm_node *x;
//		convHistogramToSVMNode(testingData[i].histogram,x);
//		double correct_label = testingData[i].classId;
//		double predicted_label = svm_predict(model,x);
//		debug(predicted_label);
//		if(predicted_label == correct_label){
//			cerr << "CORRECT: ";
//			correct++;
//		} else {
//			cerr << "WRONG  : ";
//			wrong++;
//		}
//
//		cerr << "Image of type: " << setw(20) << classes[correct_label]
//		     << " is classified as " <<  setw(20) << classes[predicted_label]
//		     << endl;
//
//		free(x);
//	}
//
//	cerr << "Accuracy is " << correct*100 / (correct+wrong)
//	     << "% (" << correct << " / " << wrong+correct << ")" << endl;

	svm_free_and_destroy_model(&model);

	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);

	funcEnded();
}
struct projectDataPoint{
	double bedroomsNumber;
	double bathroomsNumber;
	double area;
	int zipcode;
	double price;
	Mat images[4];
	vector<double> histograms[4];
};
projectDataPoint loadSingleProjectDataPoint(int id){
	string folderPath = "/media/ghooo/01CBEED4B2009090/Coursework/Computer Vision/Project/Dataset/";
	string dataFilePath = "/media/ghooo/01CBEED4B2009090/Coursework/Computer Vision/Project/HousesInfo.txt";
	projectDataPoint pdp;
	ifstream cin(dataFilePath.c_str());
	string s;
	for(int i = 1; i <= id; i++) getline(cin,s);
//	debug(s);
	stringstream ss(s);
	ss >> pdp.bedroomsNumber >> pdp.bathroomsNumber >> pdp.area >> pdp.zipcode >> pdp.price;

	string types[] ={"bathroom", "bedroom", "frontal", "kitchen"};
	for(int i = 0; i < 4; i++){
		string s = folderPath + types[i] + "_" + int2str(id) + ".jpg";
		getImage(pdp.images[i],s);
	}
	cin.clear();
	cin.close();
	return pdp;

}
void loadProjectDataPoints(vector<projectDataPoint> &pdpv, int from, int to){
	pdpv.clear();
	for(int i = from; i <= to; i++){
		pdpv.push_back(loadSingleProjectDataPoint(i));
	}
}
void loadProjectDataPointsHistograms(vector<projectDataPoint> &data, vector<vector<double> > &clusters){
	for(int i = 0; i < data.size(); i++){
		projectDataPoint &pdp = data[i];
		for(int j = 0; j < 4; j++){
			vector<vector<bool> > isKeyPoint;
			Mat orientation;
			vector<vector<double> > features;
			getKeyPoints(pdp.images[j],isKeyPoint,orientation,SCALE,KPT);
			getFeatures(pdp.images[j],isKeyPoint,orientation,SCALE,features);
			getImgHistogram(features,clusters,pdp.histograms[j]);
		}
	}
}

const double MAXPRICE = 1e8;
const double MAXBEDROOMS = 8;
const double MAXBATHROOMS = 8;
const double MAXAREA = 1e5;
void convSingleProjectDataPointToSVMFormat(projectDataPoint &pdp, svm_node *&nodes){
	int cnt = 3;
	for(int i = 0; i < 0; i++)
	for(int histogramIdx = 0; histogramIdx < (int)pdp.histograms[i].size(); histogramIdx++){
		if(pdp.histograms[i][histogramIdx] != 0.0) cnt++;
	}
	nodes = new svm_node[cnt+1];
	nodes[0].index = 1, nodes[0].value = pdp.bedroomsNumber/MAXBEDROOMS;
	nodes[1].index = 2, nodes[1].value = pdp.bathroomsNumber/MAXBATHROOMS;
	nodes[2].index = 3, nodes[2].value = pdp.area/MAXAREA;
	for(int i = 0; i < 0; i++)
	for(int histogramIdx = 0, nodeIdx = 3; histogramIdx < (int)pdp.histograms[i].size(); histogramIdx++){
		if(pdp.histograms[i][histogramIdx] != 0.0){
			nodes[nodeIdx].index = 3+i*pdp.histograms[0].size()+histogramIdx;
			nodes[nodeIdx].value = pdp.histograms[i][histogramIdx];
			nodeIdx++;
		}
	}
//	for(int histogramIdx = 0, nodeIdx = 0; histogramIdx < (int)histogram.size(); histogramIdx++){
//		if(histogram[histogramIdx] != 0.0){
//			nodes[nodeIdx] = svm_node();
//			nodes[nodeIdx].index = histogramIdx+1;
//			nodes[nodeIdx].value = histogram[histogramIdx]*100;
//			nodeIdx++;
//		}
//	}
	nodes[cnt].index = -1;
	nodes[cnt].value = 0;
}
void convProjectDataPointToSVMFormat(vector<projectDataPoint> &data, svm_problem &prob){
	prob.l = data.size();
	prob.y = new double[data.size()];
	prob.x = new svm_node*[data.size()];
	for(int dataIdx = 0; dataIdx < (int)data.size(); dataIdx++){
		prob.y[dataIdx] = data[dataIdx].price/MAXPRICE;
		convSingleProjectDataPointToSVMFormat(data[dataIdx],prob.x[dataIdx]);
	}
}
void getProjectSVMPrameter(svm_parameter &param){
	funcStarted();
	// DEAFAULTS
	param = svm_parameter();
	// default values
	//	-s svm_type : set type of SVM (default 0)
	//		0 -- C-SVC		(multi-class classification)
	//		1 -- nu-SVC		(multi-class classification)
	//		2 -- one-class SVM
	//		3 -- epsilon-SVR	(regression)
	//		4 -- nu-SVR		(regression)
	param.svm_type = EPSILON_SVR;
	//	-t kernel_type : set type of kernel function (default 2)
	//		0 -- linear: u'*v
	//		1 -- polynomial: (gamma*u'*v + coef0)^degree
	//		2 -- radial basis function: exp(-gamma*|u-v|^2)
	//		3 -- sigmoid: tanh(gamma*u'*v + coef0)
	//		4 -- precomputed kernel (kernel values in training_set_file)
	param.kernel_type = RBF;
	//	-d degree : set degree in kernel function (default 3)
	param.degree = 3;
	//	-g gamma : set gamma in kernel function (default 1/num_features)
	param.gamma = 0.000001;	// 1/num_features
	//	-r coef0 : set coef0 in kernel function (default 0)
	param.coef0 = 0;
	//	-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
	param.C = 0.00001;
	//	-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
	param.nu = 0.5;
	//	-m cachesize : set cache memory size in MB (default 100)
	param.cache_size = 100;
	//	-e epsilon : set tolerance of termination criterion (default 0.001)
	param.eps = 1e-6;
	//	-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
	param.p = 0.1;
	//	-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
	param.shrinking = 1;
	//	-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
	param.probability = 0;
	//	-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
//	-v n: n-fold cross validation mode
//	-q : quiet mode (no outputs)	param.
	funcEnded();
}
int main(){
	vector<vector<double> > clusters;
	loadClusters(clusters,"project.clusters");

	vector<projectDataPoint> training, testing;

	// read training data
	loadProjectDataPoints(training,1,2);
	loadProjectDataPointsHistograms(training,clusters);
	// read testing data
	loadProjectDataPoints(testing,6,7);
	loadProjectDataPointsHistograms(testing,clusters);

//	svm_model *modelRoomType;
//	char *modelRooType_file_name = "project.model";
//	modelRoomType = svm_load_model(modelRooType_file_name);


	// SVM Training
	svm_parameter param;		// set by parse_command_line
	svm_problem prob;		// set by read_problem
	svm_model *model;
	const char *error_msg;
	char *model_file_name = "svmmodel.model";

	convProjectDataPointToSVMFormat(training,prob);
	getProjectSVMPrameter(param);
	error_msg = svm_check_parameter(&prob,&param);
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
	while(true){
		cerr << "ENTER PARAM.C: ";
		cin >> param.C;
		cerr << "ENTER PARAM.gamma: ";
		cin >> param.gamma;
		model = svm_train(&prob,&param);
		if(svm_save_model(model_file_name,model))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			exit(1);
		}
		// SVM Testing
		double correct = 0, wrong = 0;
		for(int i = 0; i < testing.size(); i++){
			svm_node *x;
			convSingleProjectDataPointToSVMFormat(testing[i],x);
			double predicted_price = svm_predict(model,x);
			cerr << "Actual Price: " << testing[i].price << ", Expected Price: " << predicted_price*MAXPRICE << endl;

			free(x);
		}

	}
	svm_free_and_destroy_model(&model);
//	svm_free_and_destroy_model(&modelRoomType);

}
