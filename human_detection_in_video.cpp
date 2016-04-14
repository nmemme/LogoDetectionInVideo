#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <fstream>

//#pragma comment(lib,"ws2_32.lib")


using namespace cv;
using namespace std;




class Mysvm: public CvSVM
{
public:
	int get_alpha_count()
	{
		return this->sv_total;
	}

	int get_sv_dim()
	{
		return this->var_all;
	}

	int get_sv_count()
	{
		return this->decision_func->sv_count;
	}

	double* get_alpha()
	{
		return this->decision_func->alpha;
	}

	float** get_sv()
	{
		return this->sv;
	}

	float get_rho()
	{
		return this->decision_func->rho;
	}
};

void Train()
{
	char classifierSavePath[256] = "D:/INRIAPerson/pedestrianDetect-peopleFlow.txt";

	string positivePath = "D:\\INRIAPerson\\train_64x128_H96\\pos2\\";
	string negativePath = "D:\\INRIAPerson\\train_64x128_H96\\neg2\\";

	int positiveSampleCount = 2000;
	int negativeSampleCount = 800;
	int totalSampleCount = positiveSampleCount + negativeSampleCount;

	cout<<"//////////////////////////////////////////////////////////////////"<<endl;
	cout<<"totalSampleCount: "<<totalSampleCount<<endl;
	cout<<"positiveSampleCount: "<<positiveSampleCount<<endl;
	cout<<"negativeSampleCount: "<<negativeSampleCount<<endl;

	CvMat *sampleFeaturesMat = cvCreateMat(totalSampleCount , 1764, CV_32FC1);//114660
	//64*128的训练样本，该矩阵将是totalSample*3780,64*64的训练样本，该矩阵将是totalSample*1764
	cvSetZero(sampleFeaturesMat);  
	CvMat *sampleLabelMat = cvCreateMat(totalSampleCount, 1, CV_32FC1);//样本标识  
	cvSetZero(sampleLabelMat);  

	cout<<"************************************************************"<<endl;
	cout<<"start to training positive samples..."<<endl;

	char positiveImgName[256];
	string path;
	for(int i=1; i<positiveSampleCount; i++)  
	{  
		memset(positiveImgName, '\0', 256*sizeof(char));
		sprintf(positiveImgName, "%d.png", i);
		int len = strlen(positiveImgName);
		string tempStr = positiveImgName;
		path = positivePath + tempStr;

		cv::Mat img = cv::imread(path);
		if( img.data == NULL )
		{
			cout<<"positive image sample load error: "<<i<<" "<<path<<endl;
			system("pause");
			continue;
		}

		cv::HOGDescriptor hog(cv::Size(64,64), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
		vector<float> featureVec; 

		hog.compute(img, featureVec, cv::Size(8,8));  
		int featureVecSize = featureVec.size();

		for (int j=0; j<featureVecSize; j++)  
		{  		
			CV_MAT_ELEM( *sampleFeaturesMat, float, i, j ) = featureVec[j]; //cout<<featureVecSize<<','<<j<<endl;
		}  
		sampleLabelMat->data.fl[i] = 1;
	}
	cout<<"end of training for positive samples..."<<endl;

	cout<<"*********************************************************"<<endl;
	cout<<"start to train negative samples..."<<endl;

	char negativeImgName[256];
	for (int i=1; i<negativeSampleCount; i++)
	{  
		memset(negativeImgName, '\0', 256*sizeof(char));
		sprintf(negativeImgName, "%d.png", i);
		path = negativePath + negativeImgName;
		cv::Mat img = cv::imread(path);
		if(img.data == NULL)
		{
			cout<<"negative image sample load error: "<<path<<endl;
			system("pause");
			continue;
		}
		
		cv::HOGDescriptor hog(cv::Size(64,64), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);  
		vector<float> featureVec; 

		hog.compute(img,featureVec,cv::Size(8,8));//计算HOG特征
		int featureVecSize = featureVec.size();  

		for ( int j=0; j<featureVecSize; j ++)  
		{  
			CV_MAT_ELEM( *sampleFeaturesMat, float, i + positiveSampleCount, j ) = featureVec[ j ];
		}  

		sampleLabelMat->data.fl[ i + positiveSampleCount ] = -1;
	}  

	cout<<"end of training for negative samples..."<<endl;
	cout<<"********************************************************"<<endl;
	cout<<"start to train for SVM classifier..."<<endl;

	CvSVMParams params;  
	params.svm_type = CvSVM::C_SVC;  
	params.kernel_type = CvSVM::LINEAR;  
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);
	params.C = 0.01;

	Mysvm svm;
	svm.train( sampleFeaturesMat, sampleLabelMat, NULL, NULL, params ); //用SVM线性分类器训练
	svm.save(classifierSavePath);

	cvReleaseMat(&sampleFeaturesMat);
	cvReleaseMat(&sampleLabelMat);

	int supportVectorSize = svm.get_support_vector_count();
	cout<<"support vector size of SVM："<<supportVectorSize<<endl;
	cout<<"************************ end of training for SVM ******************"<<endl;

	CvMat *sv,*alp,*re;//所有样本特征向量 
	sv  = cvCreateMat(supportVectorSize , 1764, CV_32FC1);
	alp = cvCreateMat(1 , supportVectorSize, CV_32FC1);
	re  = cvCreateMat(1 , 1764, CV_32FC1);
	CvMat *res  = cvCreateMat(1 , 1, CV_32FC1);

	cvSetZero(sv);
	cvSetZero(re);
  
	for(int i=0; i<supportVectorSize; i++)
	{
		memcpy( (float*)(sv->data.fl+i*1764), svm.get_support_vector(i), 1764*sizeof(float));	
	}

	double* alphaArr = svm.get_alpha();
	int alphaCount = svm.get_alpha_count();

	for(int i=0; i<supportVectorSize; i++)
	{
        alp->data.fl[i] = alphaArr[i];
	}
	cvMatMul(alp, sv, re);

	int posCount = 0;
	for (int i=0; i<1764; i++)
	{
		re->data.fl[i] *= -1;
	}

	FILE* fp = fopen("D:/INRIAPerson/hogSVMDetector-peopleFlow.txt","wb");
	if( NULL == fp )
	{
		//return 1;
	}
	for(int i=0; i<1764; i++)
	{
		fprintf(fp,"%f \n",re->data.fl[i]);
	}
	float rho = svm.get_rho();
	fprintf(fp, "%f", rho);
	cout<<"D:/INRIAPerson/hogSVMDetector.txt 保存完毕"<<endl;//保存HOG能识别的分类器
	fclose(fp);
	cvWaitKey(-1);
//	return 1;
}

void Detect()
{
	IplImage* SizeImg=cvCreateImage(cvSize(320,240),8,3);
	CvCapture* cap = cvCreateFileCapture("shipin2.avi");
	if (!cap)
	{
		cout<<"avi file load error..."<<endl;
		system("pause");
		exit(-1);
	}

	vector<float> x;
	ifstream fileIn("hogSVMDetector-peopleFlow.txt", ios::in);
	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn>>val;
		x.push_back(val);
	}
	fileIn.close();

	vector<cv::Rect>  found;
	cv::HOGDescriptor hog(cv::Size(64,128), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	IplImage* img = NULL;
	cvNamedWindow("img", CV_WINDOW_AUTOSIZE);
	while(img=cvQueryFrame(cap))
	{
		cvZero(SizeImg);
		cvResize(img,SizeImg);
		hog.detectMultiScale(SizeImg, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
		if (found.size() > 0)
		{

			for (int i=0; i<found.size(); i++)
			{
				CvRect tempRect = cvRect(found[i].x, found[i].y, found[i].width, found[i].height);

				cvRectangle(SizeImg, cvPoint(tempRect.x,tempRect.y),
					cvPoint(tempRect.x+tempRect.width,tempRect.y+tempRect.height),CV_RGB(255,0,0), 2);
			}
		}
		cvShowImage("img",SizeImg);
		cvWaitKey(1);
	}
	cvReleaseCapture(&cap);
}


int main()
{
	//Train();
	Detect();
	return 0;
}

/*
int main(){
    Mat img;
    vector<Rect> found;
	vector<float> x;
	ifstream fileIn("hogSVMDetector-peopleFlow.txt", ios::in);
	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn>>val;
		x.push_back(val);
	}
	fileIn.close();

	img = imread( "111.jpg");

    //if(argc != 2 || !img.data){
    //    printf("没有图片\n");
    //    return -1;
    //}

    HOGDescriptor defaultHog(cv::Size(64,64), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
    defaultHog.setSVMDetector(x);


    //进行检测
    defaultHog.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);

    //画长方形，框出行人
    for(int i = 0; i < found.size(); i++){
        Rect r = found[i];
        rectangle(img, r.tl(), r.br(), Scalar(0, 0, 255), 3);
    }

    
    namedWindow("检测行人", CV_WINDOW_AUTOSIZE);
    imshow("检测行人", img);

    waitKey(0);

    return 0;
}
*/
