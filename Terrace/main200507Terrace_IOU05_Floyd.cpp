#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/opencv.hpp> 
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <cvaux.h>
#include "cameraModel.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>

#include<vector>
#include "petrick.h"
#include<string>
#include <cassert>

#include "camera.h"
#include "BackgroundSubtractorSuBSENSE.h"


using namespace std;

using namespace cv;


#define MAXVEX 100
#define INFINITY 65535

typedef int Patharc[MAXVEX][MAXVEX];
typedef float ShortPathTable[MAXVEX][MAXVEX];

typedef struct {
   int vex[MAXVEX];
   float arc[MAXVEX][MAXVEX];
   int numVertexes;

} MGraph;


#define density 6
#define scl 20
#define sclh 0.6

#define PHeight 1700
#define pWidth 680
#define METHOD 0 //0 For QM, 1 For Petrick

const int N = 100;
int visit[N];
int mark[N];
int match[N][N];
int nd, ng;
int ansH = 0;

float dijmatch[N][N];

int shorttable = 0;
int countcolumn[400];

Scalar colorful[40] = { CV_RGB(255, 128, 128), CV_RGB(255, 255, 128), CV_RGB(128, 0, 64), CV_RGB(128, 0, 255), CV_RGB(128, 255, 255), CV_RGB(0, 128, 255), CV_RGB(255, 0, 128), CV_RGB(255, 128, 255),
CV_RGB(128, 255, 128), CV_RGB(128, 128, 255), CV_RGB(0, 64, 128), CV_RGB(0, 128, 128), CV_RGB(0, 255, 0), CV_RGB(255, 128, 64), CV_RGB(128, 64, 64), CV_RGB(255, 0, 0), CV_RGB(255, 255, 0),
CV_RGB(0,255,255), CV_RGB(0, 255, 0), CV_RGB(255, 255, 255),
CV_RGB(255, 128, 128), CV_RGB(255, 255, 128), CV_RGB(128, 0, 64), CV_RGB(128, 0, 255), CV_RGB(128, 255, 255), CV_RGB(0, 128, 255), CV_RGB(255, 0, 128), CV_RGB(255, 128, 255),
CV_RGB(128, 255, 128), CV_RGB(128, 128, 255), CV_RGB(0, 64, 128), CV_RGB(0, 128, 128), CV_RGB(0, 255, 0), CV_RGB(255, 128, 64), CV_RGB(128, 64, 64), CV_RGB(255, 0, 0), CV_RGB(255, 255, 0),
CV_RGB(64, 0, 64), CV_RGB(0, 255, 128), CV_RGB(255, 255, 255) };


double  table[3][16] = { { 0.995, 0.99, 0.975, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0005, 0.0001, },
{ 0.0000393, 0.000157, 0.000982, 0.00293, 0.0158, 0.102, 0.455, 1.32, 2.71, 3.84, 5.02, 6.63, 7.88, 10.8, 12.1, 15.1 },
{ 0.0100, 0.0201, 0.0506, 0.103, 0.211, 0.575, 1.39, 2.77, 4.61, 5.99, 7.38, 9.21, 10.6, 13.8, 15.2, 18.4 } };

double  tableG[2][16] = { { 0.5, 0.5793, 0.6554, 0.7257, 0.7881, 0.8413, 0.8849, 0.9192, 0.9452, 0.9641, 0.9772, 0.9861, 0.9918, 0.9953, 0.9974, 0.9987 },
{ 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0 } };

double  tableG2[2][16] = { { 0.5, 0.5080, 0.5160, 0.5239, 0.5319, 0.5398,0.5478,0.5557, 0.5636,0.5714,0.5793, 0.5871,0.5948 , 0.6026,0.6103, 0.9987},
{ 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2,0.22, 0.24, 0.26, 0.28, 3.0 } };

double  tableG3[2][16] = { { 0.5, 0.5398,0.5793,0.6179, 0.6554,0.6915, 0.7257,0.7580, 0.7881,0.8159, 0.8413,0.8643, 0.8849,0.9032, 0.9192,0.9332 },
{ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 } };

double Gaussian1D(double dis); 
double Gaussian1D2(double dis);
double MahaDistance1D(double x, double y, double var);
void QMmethod(vector<Camera*> camera, Mat* st, int num, vector<int>& result);
void QMmap(CvRect rect, Mat* pFrame, int i);
int RemoveDuplates(Mat* A, Mat*B, Mat* fmask);
int removeRow(bool a[], bool b[], int num);
void showtable(int totalNum, int num, bool state[], bool** t, vector<Camera*> camera);
void drawDashRect(CvArr* img, int linelength, int dashlength, CvBlob* blob, CvScalar color, int thickness);
float bbOverlap(Rect box1, Rect box2);
int dfs(int i);
int Hungary();
void CreateMGraph(MGraph *G);
void ShortPath_Floyd(MGraph G, Patharc P, ShortPathTable D);
void PrintShortPath(MGraph G, Patharc P, ShortPathTable D, vector < vector<int> >  &lines, vector <Point>  &points);
void InitialG(MGraph *G);

typedef struct Measure
{
	double prob;
	CvRect R1;
	CvRect R2;
	CvRect R3;
	IplImage* cmask[3];
	int state = 0;
	Point p;
};

Point GetFoot(const Point &pt, const Point &begin, const Point &end);

Point findLineCross(vector < vector<int> >  lines)
{
	if (lines.size() == 1)
		return Point(0, 0);

	vector<Point> pt;
	Point pta;
	vector<Point> pts;
	vector<double>A;
	vector<double>B;
	vector<double>C;
	double A1, B1, C1, A2, B2, C2, AB;

	A1 = 0; B1 = 0; C1 = 0; A2 = 0; B2 = 0; C2 = 0;

	for (int kk = 0; kk < lines.size(); kk++)
	{
		A.push_back(lines[kk][3] - lines[kk][1]);
		B.push_back(lines[kk][0] - lines[kk][2]);
		C.push_back(lines[kk][2] * lines[kk][1] - lines[kk][0] * lines[kk][3]);
	}

	for (int kk = 0; kk < lines.size(); kk++)
	{
		//对12号flag位进行判断，对长线和短线加权重处理
		if (lines[kk][12] != -1)
		{
			AB = A[kk] * A[kk] + B[kk] * B[kk];
			A1 += 2 * A[kk] * A[kk] / AB;
			B1 += 2 * A[kk] * B[kk] / AB;
			C1 += 2 * A[kk] * C[kk] / AB;
			A2 += 2 * A[kk] * B[kk] / AB;
			B2 += 2 * B[kk] * B[kk] / AB;
			C2 += 2 * B[kk] * C[kk] / AB;
		}
		else
		{
			AB = A[kk] * A[kk] + B[kk] * B[kk];
			A1 += 0.125 * A[kk] * A[kk] / AB;
			B1 += 0.125 * A[kk] * B[kk] / AB;
			C1 += 0.125 * A[kk] * C[kk] / AB;
			A2 += 0.125 * A[kk] * B[kk] / AB;
			B2 += 0.125 * B[kk] * B[kk] / AB;
			C2 += 0.125 * B[kk] * C[kk] / AB;
		}
	}
	pta.x = (C2*B1 - B2 * C1) / (B2*A1 - B1 * A2);
	pta.y = (C2*A1 - A2 * C1) / (B1*A2 - B2 * A1);

	return pta;
}

bool isLineSegmentCross(const Point &P1, const Point &P2, const Point &Q1, const Point &Q2)
{
	if (
		((Q1.x - P1.x)*(Q1.y - Q2.y) - (Q1.y - P1.y)*(Q1.x - Q2.x)) * ((Q1.x - P2.x)*(Q1.y - Q2.y) - (Q1.y - P2.y)*(Q1.x - Q2.x)) < 0 &&
		((P1.x - Q1.x)*(P1.y - P2.y) - (P1.y - Q1.y)*(P1.x - P2.x)) * ((P1.x - Q2.x)*(P1.y - P2.y) - (P1.y - Q2.y)*(P1.x - P2.x)) < 0
		)
		return true;
	else
		return false;
}


void pushLine(vector < vector<int> > & lines, vector <Point>&  points, vector<Camera*>& camera, int c, int I)
{
	//if(isLineSegmentCross(const Point &P1, const Point &P2, const Point &Q1, const Point &Q2))
	//cout << "I" << I << " c" << c << " size:" << camera[c]->line[I].size() << endl;
	if (camera[c]->line[I].size() != 0)
		for (int i = 0; i < camera[c]->line[I].size(); i++)
		{
			//把线两两分组，之后分别送findlinecross找交点
			lines.push_back(camera[c]->line[I][i]);
			//points.push_back(findLineCross(lines));
			if (c != camera.size() - 1)
			{
				pushLine(lines, points, camera, c + 1, I);
			}
			else
			{
				points.push_back(findLineCross(lines));
				//cout << findLineCross(lines) << endl;
			}
			lines.pop_back();

		}
	else
	{
		if (c != camera.size() - 1)
		{
			pushLine(lines, points, camera, c + 1, I);
		}
		else
		{
			points.push_back(findLineCross(lines));
			//cout << findLineCross(lines) << endl;
		}
	}
}


int main(int argc, const char * argv[])
{
	//读取ground truth box文件
	//FILE * fpbox;
	//if ((fpbox = fopen("E:\\Terrace\\gtbox3.txt", "wb")) == NULL) {
	//	printf("cant open the file");
	//	exit(0);
	//}
	
	cv::Rect select_ROI;
	cv::Rect select_ROI2;
	cv::Rect select_ROI3;
	cv::Rect select_ROI4;

	ifstream infile;
	infile.open("F:\\Open CV\\opencv program\\Terrace-4v\\ConsoleApplication6\\gt_terrace.txt");   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 

	string input_s;
	getline(infile, input_s);
	getline(infile, input_s);
	getline(infile, input_s);

	int input_int;
	vector<int> gt_n;
	vector<Rect> gt;

	//读mask boundingbox位置信息
	fstream file1;//创建文件流对象
	file1.open("F:\\Open CV\\opencv program\\Terrace-4v\\ConsoleApplication6\\allmasks.txt");
	Mat bboxData = Mat::zeros(3166, 10, CV_32FC1);//创建Mat类矩阵，定义初始化值全部是0，矩阵大小和txt一致



												  //将txt文件数据写入到Data矩阵中
	for (int i = 0; i < 3166; i++)
	{
		for (int j = 0; j < 12; j++)
		{
			file1 >> bboxData.at<float>(i, j);

		}
	}

	//cout << "矩阵的数据输出为：" << endl;
	//cout << bboxData << endl;
	//cout << endl;




	char ch;
	int fstep = 0;

	vector<Camera*> camera(4);

	camera[0] = new Camera("F:\\Open CV\\opencv program\\video\\terrace1-c0.avi");
	camera[0]->cam->setExtrinsic(-4.8441913843e+03, 5.5109448682e+02, 4.9667438357e+03, 1.9007833770e+00, 4.9730769727e-01, 1.8415452559e-01);
	camera[0]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[0]->cam->setIntrinsic(20.161920, 5.720865e-04, 366.514507, 305.832552, 1);
	camera[0]->cam->internalInit();

	camera[1] = new Camera("F:\\Open CV\\opencv program\\video\\terrace1-c1.avi");
	camera[1]->cam->setExtrinsic(-65.433635, 1594.811988, 2113.640844, 1.9347282363e+00, -7.0418616982e-01, -2.3783238362e-01);
	camera[1]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[1]->cam->setIntrinsic(19.529144, 5.184242e-04, 360.228130, 255.166919, 1);
	camera[1]->cam->internalInit();

	//试出来的内参数据 255.166919改为262.566919
	//camera[1]->cam->setIntrinsic(19.529144, 5.184242e-04, 360.228130, 262.566919, 1);


	camera[2] = new Camera("F:\\Open CV\\opencv program\\video\\terrace1-c2.avi");
	camera[2]->cam->setExtrinsic(1.9782813424e+03, -9.4027627332e+02, 1.2397750058e+04, -1.8289537286e+00, 3.7748154985e-01, 3.0218614321e+00);
	camera[2]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[2]->cam->setIntrinsic(19.903218, 3.511557e-04, 355.506436, 241.205640, 1.0000000000e+00);
	camera[2]->cam->internalInit();

	camera[3] = new Camera("F:\\Open CV\\opencv program\\video\\terrace1-c3.avi");
	camera[3]->cam->setExtrinsic(4.6737509054e+03, -2.5743341287e+01, 8.4155952460e+03, -1.8418460467e+00, -4.6728290805e-01, -3.0205552749e+00);   //view 3
	camera[3]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[3]->cam->setIntrinsic(20.047015, 4.347668e-04, 349.154019, 245.786168, 1);
	camera[3]->cam->internalInit();

#pragma omp parallel for
	for (int i = 0; i < camera.size(); i++)
	{
		camera[i]->initialBG(1);
		camera[i]->mapToTop();//单映与反单映顶视图与各摄像头view坐标
	}

	int nFrmNum = 0;

	cv::namedWindow("C0", cv::WINDOW_NORMAL);
	cv::resizeWindow("C0", 400, 300);
	cv::moveWindow("C0", 0, 0);
	cv::namedWindow("C1", cv::WINDOW_NORMAL);
	cv::resizeWindow("C1", 400, 300);
	cv::moveWindow("C1", 450, 0);
	cv::namedWindow("C2", cv::WINDOW_NORMAL);
	cv::resizeWindow("C2", 400, 300);
	cv::moveWindow("C2", 0, 400);
	cv::namedWindow("C3", cv::WINDOW_NORMAL);
	cv::resizeWindow("C3", 400, 300);
	cv::moveWindow("C3", 450, 400);
	cv::namedWindow("top", cv::WINDOW_NORMAL);
	//cv::namedWindow("top2", cv::WINDOW_NORMAL);
	//cv::namedWindow("top3", cv::WINDOW_NORMAL);
	//cv::namedWindow("top4", cv::WINDOW_NORMAL);
	cv::resizeWindow("top", 1000, 1000);
	//cv::resizeWindow("top2", 500, 500);
	//cv::resizeWindow("top3", 500, 500);
	//cv::resizeWindow("top4", 500, 500);
	cv::moveWindow("top", 850, 0);
	//cv::moveWindow("top2", 1300, 0);
	//cv::moveWindow("top3", 850, 500);
	//cv::moveWindow("top4", 1300, 500);

	//cv::namedWindow("foreground1", cv::WINDOW_NORMAL);
	//cv::resizeWindow("foreground1", 400, 300);
	//cv::moveWindow("foreground1", 800, 0);
	//cv::namedWindow("foreground2", cv::WINDOW_NORMAL);
	//cv::resizeWindow("foreground2", 400, 300);
	//cv::moveWindow("foreground2", 1200, 0);
	//cv::namedWindow("foreground3", cv::WINDOW_NORMAL);
	//cv::resizeWindow("foreground3", 400, 300);
	//cv::moveWindow("foreground3", 800, 400);
	//cv::namedWindow("foreground4", cv::WINDOW_NORMAL);
	//cv::resizeWindow("foreground4", 400, 300);
	//cv::moveWindow("foreground4", 1200, 400);

	//初始化占空比图
	static Mat map(1000, 1000, CV_32F, Scalar(0));
	static Mat mapTop(1000, 1000, CV_8UC1, Scalar(0));

	//
	int num = 0;
	static Mat st(1, 400, CV_8U, Scalar(0));
	static Measure mea[400];
	int ex = 0;

	Mat topCross(1000, 1000, CV_8UC3);
	Mat topCross2(1000, 1000, CV_8UC3);
	Mat topCross3(1000, 1000, CV_8UC3);
	Mat topCross4(1000, 1000, CV_8UC3);

	vector<int> result;

	float countHit = 0;
	int countdet = 0;
	int countGT = 0;
	float MODP = 0;
	float NMODP = 0;
	float NMODA = 0;
	float RECALL, PRECISION, TER, FSCORE = 0;
	int Nframe = 0;

	double X, Y, Z = 0;
	double x, y, xt, yt;
	Point pl[4];
	//Point LT(290, 285), RB(460, 450);
	Point LT(285, 290), RB(450, 460);
	//扩大
	//Point LT(280, 285), RB(455, 465);
	//缩小
	//Point LT(290, 295), RB(445, 455);

	Mat topback = imread("F:\\Open CV\\opencv program\\video\\t4.png");//topback存储预制好的顶视图

	//Mat bdata = imread("C:\\pyhon program\\Mask_RCNN-master\\allmasks.txt");

	while (1)
	{

		for (int i = 0; i < camera.size(); i++)
			camera[i]->readNextFrame();

		nFrmNum = camera[0]->FrameNumber;
		gt_n.clear();
		for (int i_in = 0; i_in < 9; i_in++)
		{
			infile >> input_int;
			//cout << input_int << " ";
			if (input_int >= 0)
				gt_n.push_back(input_int);
		}

		if (nFrmNum > 5000)
			break;

		cout << nFrmNum << endl;
		//每n帧显示图像
		//if (nFrmNum < 50|| (nFrmNum % 5 == 0 && nFrmNum % 25 != 0))
		//if (nFrmNum < 50)
		{
			//改camera和main中的内容即可
//#pragma omp parallel for
			//for (int i = 0; i < camera.size(); i++)
				//if (nFrmNum >= 125)
				//	if (i == 0)
				//		camera[i]->maskfore0();
				//	else if (i == 1)
				//		camera[i]->maskfore1();
				//	else if (i == 2)
				//		camera[i]->maskfore2();
				//	else
				//		camera[i]->maskfore3();
				//else
				//camera[i]->updateMog();

		}
		if (nFrmNum % 25 == 0)
		{
			nd = 0;
			ng = 0;

			for (int i = 0; i < 400; i++)
			{
				camera[0]->p[i] = Point(0, 0);
			}
#pragma omp parallel for
			for (int i = 0; i < camera.size(); i++)
				if (nFrmNum >= 25)
					if (i == 0)
						camera[i]->maskfore0();
					else if (i == 1)
						camera[i]->maskfore1();
					else if (i == 2)
						camera[i]->maskfore2();
					else
						camera[i]->maskfore3();
				else
					camera[i]->updateMog();

			//之后的内容不用改
			//积分图添加
#pragma omp parallel for
			for (int i = 0; i < camera.size(); i++)
				integral(camera[i]->foreground / 255.0, camera[i]->iiimage, CV_32S);


			//计算每一个网格的占空比
			//int countt = 0;
			int flag;
			int flag1;
			int flag2;
			map.setTo(1);

			topCross.setTo(0);
			//topCross2.setTo(0);
			//topCross3.setTo(0);
			//topCross4.setTo(0);
			topCross += topback;//将顶视图赋给topcross(1000,1000)这个初始化好的数组中
			//topCross2 += topback;
			//topCross3 += topback;
			//topCross4 += topback;

			//Area of Interest (AOT) in topview
			rectangle(topCross, Rect(Point(290, 285), Point(460, 450)), CV_RGB(255, 0, 0), 2);
			//rectangle(topCross2, Rect(Point(290, 285), Point(460, 450)), CV_RGB(255, 0, 0), 2);
			//rectangle(topCross3, Rect(Point(290, 285), Point(460, 450)), CV_RGB(255, 0, 0), 2);
			//rectangle(topCross4, Rect(Point(290, 285), Point(460, 450)), CV_RGB(255, 0, 0), 2);
			
			transpose(topCross, topCross);
			//transpose(topCross2, topCross2);
			//transpose(topCross3, topCross3);
			//transpose(topCross4, topCross4);

			//Area of Interest (AOI) in camera views
			//Point a(320, 300), b(320, 430), c(450, 430), d(450, 300);
			Point a(285, 290), b(450, 290), c(450, 460), d(285, 460);

			//扩大5 AOI
			//Point a(280, 285), b(455, 285), c(455, 465), d(280, 465);
			//缩小5 AOI
			//Point a(290, 295), b(445, 295), c(445, 455), d(290, 455);
		
			for (int k = 0; k < camera.size(); k++)
			{
				X = (a.x - 250) * 30;
				Y = (a.y - 250) * 30;
				Z = 0;
				camera[k]->cam->worldToImage(X, Y, Z, x, y);
				pl[0].x = x / 2;
				pl[0].y = y / 2;
				
				//circle(camera[k]->frame, Point(x/2, y/2), 4, CV_RGB(255, 0, 0), -1);

				X = (b.x - 250) * 30;
				Y = (b.y - 250) * 30;
				Z = 0;
				camera[k]->cam->worldToImage(X, Y, Z, x, y);
				pl[1].x = x / 2;
				pl[1].y = y / 2;
				//circle(camera[k]->frame, Point(x / 2, y / 2), 4, CV_RGB(255, 0, 0), -1);

				X = (c.x - 250) * 30;
				Y = (c.y - 250) * 30;
				Z = 0;
				camera[k]->cam->worldToImage(X, Y, Z, x, y);
				pl[2].x = x / 2;
				pl[2].y = y / 2;
				//circle(camera[k]->frame, Point(x / 2, y / 2), 4, CV_RGB(255, 0, 0), -1);

				X = (d.x - 250) * 30;
				Y = (d.y - 250) * 30;
				Z = 0;
				camera[k]->cam->worldToImage(X, Y, Z, x, y);
				pl[3].x = x / 2;
				pl[3].y = y / 2;
				//circle(camera[k]->frame, Point(x / 2, y / 2), 4, CV_RGB(255, 0, 0), -1);

				line(camera[k]->frame2, pl[0], pl[1], CV_RGB(255, 0, 0), 2);
				line(camera[k]->frame2, pl[1], pl[2], CV_RGB(255, 0, 0), 2);
				line(camera[k]->frame2, pl[2], pl[3], CV_RGB(255, 0, 0), 2);
				line(camera[k]->frame2, pl[3], pl[0], CV_RGB(255, 0, 0), 2);

				//cout <<"C"<<k<<" "<< "pl[0]: " << pl[0] << " " << "pl[1]: " << pl[1] << " " << "pl[2]: " << pl[2] 
				//	<< " " << "pl[3]: " << pl[3] << endl;
				
			}

			//图中行人人高度估算
			//double hratio;
			//Mat test = imread("F:\\Open CV\\opencv program\\Terrace-4v\\ConsoleApplication6\\C0 saved image V5\\175.jpg");
			//Mat test2 = imread("F:\\Open CV\\opencv program\\Terrace-4v\\ConsoleApplication6\\C2 saved image V5\\175.jpg");
			////cout << "empty: " << test.empty() << endl;
			//camera[0]->cam->imageToWorld(170 * 2, 200 * 2, 0, X, Y); 
			//camera[0]->cam->worldToImage(X, Y, 2000, x, y);
			//x = x / 2;
			//y = y / 2;
			//hratio = (200-33) / (200 - y);
			////circle(test, Point(x, y), 4, CV_RGB(255, 0, 0), -1);
			////line(test, Point(x-50, y), Point(x + 50, y), CV_RGB(255, 0, 0), 2);

			//cout << "2000mm y: " << y << endl;
			//cout << "hratio: " << hratio << endl;
			//cout << "World height: " << 2000 * hratio << endl;
			//camera[0]->cam->imageToWorld(170 * 2, 200 * 2, 0, X, Y);
			//camera[0]->cam->worldToImage(X, Y, 2000 * hratio, x, y);
			//x = x / 2;
			//y = y / 2;
			////circle(test, Point(x, y), 4, CV_RGB(255, 0, 0), -1);
			//line(test, Point(x - 50, y), Point(x + 50, y), CV_RGB(255, 0, 0), 2);
			//cout << "x, y: " << x << " " << y << endl;

			////投影到其他视图
			//camera[0]->cam->imageToWorld((x - 50)*2, y * 2, 0, X, Y);
			//camera[1]->cam->worldToImage(X, Y, 2000 * hratio, x, y);
			//x = x / 2;
			//y = y / 2;
			//
			//camera[0]->cam->imageToWorld((x + 50) * 2, y * 2, 0, X, Y);
			//camera[1]->cam->worldToImage(X, Y, 2000 * hratio, xt, yt);
			//xt = xt / 2;
			//yt = yt / 2;

			//line(test2, Point(x, y), Point(xt, yt), CV_RGB(255, 0, 0), 2);
			//imshow("hello", test);
			//imshow("hi", test2);

			//消影点
			//vector < vector<int> >  linestest;
			//Point ptest;
			//linestest.push_back(vector<int>(4, 0));
			//X = (365 - 250) * 30;
			//Y = (373 - 250) * 30;
			//Z = 0;
			//camera[0]->cam->worldToImage(X, Y, Z, x, y);
			//x = x / 2;
			//y = y / 2;
			//Z = 1700;
			//camera[0]->cam->worldToImage(X, Y, Z, xt, yt);
			//xt = xt / 2;
			//yt = yt / 2;
			////line(camera[0]->frame2, Point(x,y), Point(xt,yt), CV_RGB(255, 0, 0), 2);
			//linestest.push_back(vector<int>(4, 0));
			//linestest[0][0] = x;
			//linestest[0][1] = y;
			//linestest[0][2] = xt;
			//linestest[0][3] = yt;
			//X = (408 - 250) * 30;
			//Y = (375 - 250) * 30;
			//Z = 0;
			//camera[0]->cam->worldToImage(X, Y, Z, x, y);
			//x = x / 2;
			//y = y / 2;
			//Z = 1700;
			//camera[0]->cam->worldToImage(X, Y, Z, xt, yt);
			//xt = xt / 2;
			//yt = yt / 2;
			////line(camera[0]->frame2, Point(x, y), Point(xt, yt), CV_RGB(255, 0, 0), 2);
			//
			//linestest[1][0] = x;
			//linestest[1][1] = y;
			//linestest[1][2] = xt;
			//linestest[1][3] = yt;

			//linestest.push_back(vector<int>(4, 0));
			//X = (306 - 250) * 30;
			//Y = (370 - 250) * 30;
			//Z = 0;
			//camera[0]->cam->worldToImage(X, Y, Z, x, y);
			//x = x / 2;
			//y = y / 2;
			//Z = 1700;
			//camera[0]->cam->worldToImage(X, Y, Z, xt, yt);
			//xt = xt / 2;
			//yt = yt / 2;
			////line(camera[0]->frame2, Point(x, y), Point(xt, yt), CV_RGB(255, 0, 0), 2);
			//
			//linestest[2][0] = x;
			//linestest[2][1] = y;
			//linestest[2][2] = xt;
			//linestest[2][3] = yt;
			//ptest = findLineCross(linestest);
			//line(camera[0]->frame2, ptest, Point(xt, yt), CV_RGB(255, 0, 0), 2);
			////line(camera[0]->frame2, Point(x, y), Point(xt, yt), CV_RGB(255, 0, 0), 2);
			//line(camera[0]->frame2, Point(xt, 288), Point(xt, yt), CV_RGB(0, 0, 255), 2);

			//double am, bm, cm, dm, em, fm;
			//X = (273 - 250) * 30; //将毫米转化为多少个像素
			//Y = (288 - 250) * 30;
			//Z = 0;//将高度设置为0，矩形框底部高度
			//camera[1]->cam->worldToImage(X, Y, Z, x, y);//world coordinates to image coordinates
			//x = x / 2;//2倍下采样
			//y = y / 2;
			//Z = 2000; //2000毫米高度，矩形框顶部高度
			//camera[1]->cam->worldToImage(X, Y, Z, xt, yt);//world coordinates to image coordinates
			//xt = xt / 2;
			//yt = yt / 2;
			//am = x - abs(y - yt)*0.35*0.5;
			//bm = yt;
			//cm = abs(y - yt)*0.35;
			//dm = y - yt;
			//em = am + cm;//x + abs(y - yt)*0.35*0.5;
			//fm = bm + dm;//y

			//rectangle(camera[1]->frame2, Point(x - abs(y - yt)*0.35*0.5, yt), Point(x + abs(y - yt)*0.35*0.5, y), CV_RGB(155,0,0), 2);
			//circle(topCross, Point(273,288), 4, CV_RGB(255, 0, 0), -1);
			//flip(topCross, topCross, 0);
			//imshow("test1", topCross);
			//imshow("test", camera[1]->frame2);
			////cout << "ptest: " << ptest << endl;
			//waitKey(0);

			vector < vector<int> >  lines;
			vector < vector<int> >  lines2;
			vector < vector<int> >  lines3;
			vector < vector<int> >  lines4;
			vector < vector<int> >  lines5;
			lines2.clear();
			vector <Point>  points;
			vector <Point>  points2;
			vector <Point>  points3;
			int numI = 0;
			float tempc0, tempc1, tempc2, tempc3, numt;

			for (int k = 0; k < camera.size(); k++)
			{
				camera[k]->line.clear();
				numI += camera[k]->lineti(nFrmNum, bboxData, k, topCross, topCross2, topCross3);
				//cout << k <<""<< camera[k]->line.size() << endl;
				//cout << k <<"" << camera[k]->line[0].size() << endl;

				//line(topCross, camera[k]->top1, camera[k]->top2, CV_RGB(255, 0, 0), 2);

			}


			points.clear();


			//cout << "lines2.size: " << lines2.size() << endl;
			//所有points点显示
			//cout << "points.size: " << points.size() << endl;
			//for (int i = 0; i < points.size(); i++)
			//{
			//	//circle(topCross, points[i], 8, Scalar(0, 0, 0), 1);


			//	circle(topCross, points[i], 2, Scalar(0, 0, 0), -1);
			//}
			memset(dijmatch, -1, sizeof(dijmatch));
			int disc = 15;
			int disc2 = 20;
			int disc3 = 10;
			points3.clear();
			lines3.clear();
			for (int k = 0; k < camera.size(); k++)
			{

				if (camera[k]->line.size() > 0)
				{
					for (int i = 0; i < camera[k]->line[0].size(); i++)
					{
						lines3.push_back(camera[k]->line[0][i]);
					}
				}
			}
			//cout << "lines3.size: " << lines3.size() << endl;

			MGraph G;
			Patharc P;
			ShortPathTable D;

			InitialG(&G);

			for (int k = 0; k < lines3.size(); k++)
			{
				tempc0 = 0;
				tempc1 = 0;
				tempc2 = 0;
				tempc3 = 0;
				int num = 0;
				double dis = 0;
				lines4.clear();
				lines.clear();
				if (lines3[k][10] != -1 && lines3[k][11] != -1 && lines3[k][12] != -1)
				{
					if (lines3[k][9] == 0)
					{
						//lines4.push_back(lines3[k]);
						//lines3[k][10] = -1;
						lines.push_back(lines3[k]);
						for (int k1 = 0; k1 < lines3.size(); k1++)
						{
							tempc0 = 0;
							tempc1 = 0;
							points2.clear();
							//cout << "lines3.size:  " << lines3.size() << endl;
							//cout << "lines3[k1][9]:  " << lines3[k1][9] << endl;
							if (lines3[k1][9] > 0 && lines3[k1][10] != -1 && lines3[k1][11] != -1)
							{

								lines.push_back(lines3[k1]);
								//if (lines3[k1][9] == 3)
								//{
								//	cout << "dis: " << isLineSegmentCross(Point(lines3[k][0], lines3[k][1]), Point(lines3[k][2], lines3[k][3]),
								//		Point(lines3[k1][0], lines3[k1][1]), Point(lines3[k1][2], lines3[k1][3])) << endl;
								//}


								if (isLineSegmentCross(Point(lines3[k][0], lines3[k][1]), Point(lines3[k][2], lines3[k][3]),
									Point(lines3[k1][0], lines3[k1][1]), Point(lines3[k1][2], lines3[k1][3])) && lines3[k1][10] != -1
									&& tempc1 == 0)
								{
									

									points2.push_back(findLineCross(lines));

									tempc0 = (abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
										+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y))) + 
										abs(sqrt((lines3[k1][4] - points2[0].x)*(lines3[k1][4] - points2[0].x)
											+ (lines3[k1][5] - points2[0].y)*(lines3[k1][5] - points2[0].y)))) / 2;
									if (tempc0 <= disc)
									{
										if (k < k1)
										{
											if (G.arc[k][k1] == INFINITY)
											{
												G.arc[k][k1] = tempc0;
											}
											else
											{
												G.arc[k][k1] = min(G.arc[k][k1], tempc0);
											}

											//cout << "G.arc[k][k1]: " << G.arc[k][k1] << endl;
											//waitKey(0);
										}
										else
										{
											if (G.arc[k1][k] == INFINITY)
											{
												G.arc[k1][k] = tempc0;
											}
											else
											{
												G.arc[k1][k] = min(G.arc[k1][k], tempc0);
											}
										}
										tempc1 += 1;
									}
									//cout << "points2: " << points2[0] << endl;
									//cout <<"dis: "<< abs(sqrt((lines3[k][4] - lines3[k1][4])*(lines3[k][4] - lines3[k1][4])
										//+ (lines3[k][5] - lines3[k1][5])*(lines3[k][5] - lines3[k1][5]))) << endl;
									//if (abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
									//	+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y))) <= disc)
									//{
									//	lines3[k1][7] = abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
									//		+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y)));
									//	lines4.push_back(lines3[k1]);


									//	
									//	//这里尝试将长线不做标志位处理
									//	if (lines3[k1][12] != -1)
									//	{
									//		lines3[k1][10] = -1;
									//	}

									//	num += 1;
									//}

								}

								//替换为点到线垂足的距离
								if (abs(sqrt((lines3[k][4] - lines3[k1][4])*(lines3[k][4] - lines3[k1][4])
									+ (lines3[k][5] - lines3[k1][5])*(lines3[k][5] - lines3[k1][5]))) <= disc2 && 
									lines3[k1][10] != -1 && tempc1 == 0)
								{								
									//得到垂足坐标点
									Point testp = GetFoot(Point(lines3[k][4], lines3[k][5]), Point(lines3[k1][0], lines3[k1][1]),
										Point(lines3[k1][2], lines3[k1][3]));
									//点到垂足连线
									
									double distest = abs(sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
										+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y)));
									
									tempc0 = (abs(sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
										+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y))) +
										abs(sqrt((lines3[k1][4] - testp.x)*(lines3[k1][4] - testp.x)
											+ (lines3[k1][5] - testp.y)*(lines3[k1][5] - testp.y)))) / 2;
									if (k < k1)
									{
										if (G.arc[k][k1] == INFINITY)
										{
											G.arc[k][k1] = tempc0;
										}
										else
										{
											G.arc[k][k1] = min(G.arc[k][k1], tempc0);
										}
									}
									else
									{
										if (G.arc[k1][k] == INFINITY)
										{
											G.arc[k1][k] = tempc0;
										}
										else
										{
											G.arc[k1][k] = min(G.arc[k1][k], tempc0);
										}
									}
									tempc1 += 1;

									//if (distest < disc3)
									//{
									//lines3[k1][7] = distest;
									//lines4.push_back(lines3[k1]);
									////cout << "lines4.size1: " << lines4.size() << endl;

									////这里尝试将长线不做标志位处理
									//if (lines3[k1][12] != -1)
									//{
									//	lines3[k1][10] = -1;
									//}
									//num += 1;
									//cv::line(topCross, testp, Point(lines3[k][4], lines3[k][5]), Scalar(0, 0, 0), 2);
									//}
								}

								lines.pop_back();

							}

						}
						//if (num == 0)
						//{
						//	lines4.pop_back();
						//	lines3[k][10] = 0;
						//}
						//else
						//{
						//	lines3[k][10] = -1;
						//}
					}

					if (lines3[k][9] == 1)
					{
						//lines4.push_back(lines3[k]);
						//lines3[k][10] = -1;
						lines.push_back(lines3[k]);
						for (int k1 = 0; k1 < lines3.size(); k1++)
						{
							tempc0 = 0;
							tempc1 = 0;
							points2.clear();
							//cout << "lines3.size:  " << lines3.size() << endl;
							//cout << "lines3[k1][9]:  " << lines3[k1][9] << endl;
							if (lines3[k1][9] != 1 && lines3[k1][10] != -1 && lines3[k1][11] != -1)
							{

								lines.push_back(lines3[k1]);

								if (isLineSegmentCross(Point(lines3[k][0], lines3[k][1]), Point(lines3[k][2], lines3[k][3]),
									Point(lines3[k1][0], lines3[k1][1]), Point(lines3[k1][2], lines3[k1][3])) && lines3[k1][10] != -1
									&& tempc1 == 0)
								{

									points2.push_back(findLineCross(lines));

									tempc0 = (abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
										+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y))) +
										abs(sqrt((lines3[k1][4] - points2[0].x)*(lines3[k1][4] - points2[0].x)
											+ (lines3[k1][5] - points2[0].y)*(lines3[k1][5] - points2[0].y)))) / 2;
									if (tempc0 <= disc)
									{
										if (k < k1)
										{
											if (G.arc[k][k1] == INFINITY)
											{
												G.arc[k][k1] = tempc0;
											}
											else
											{
												G.arc[k][k1] = min(G.arc[k][k1], tempc0);
											}
										}
										else
										{
											if (G.arc[k1][k] == INFINITY)
											{
												G.arc[k1][k] = tempc0;
											}
											else
											{
												G.arc[k1][k] = min(G.arc[k1][k], tempc0);
											}
										}
										tempc1 += 1;
									}
									//cout << "points2: " << points2[0] << endl;
									//cout <<"dis: "<< abs(sqrt((lines3[k][4] - lines3[k1][4])*(lines3[k][4] - lines3[k1][4])
										//+ (lines3[k][5] - lines3[k1][5])*(lines3[k][5] - lines3[k1][5]))) << endl;
									//if (abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
									//	+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y))) <= disc)
									//{
									//	lines3[k1][7] = abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
									//		+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y)));
									//	lines4.push_back(lines3[k1]);


									//	
									//	//这里尝试将长线不做标志位处理
									//	if (lines3[k1][12] != -1)
									//	{
									//		lines3[k1][10] = -1;
									//	}

									//	num += 1;
									//}

								}

								//替换为点到线垂足的距离
								if (abs(sqrt((lines3[k][4] - lines3[k1][4])*(lines3[k][4] - lines3[k1][4])
									+ (lines3[k][5] - lines3[k1][5])*(lines3[k][5] - lines3[k1][5]))) <= disc2 &&
									lines3[k1][10] != -1 && tempc1 == 0)
								{
									//得到垂足坐标点
									Point testp = GetFoot(Point(lines3[k][4], lines3[k][5]), Point(lines3[k1][0], lines3[k1][1]),
										Point(lines3[k1][2], lines3[k1][3]));
									//点到垂足连线

									double distest = abs(sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
										+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y)));

									tempc0 = (abs(sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
										+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y))) +
										abs(sqrt((lines3[k1][4] - testp.x)*(lines3[k1][4] - testp.x)
											+ (lines3[k1][5] - testp.y)*(lines3[k1][5] - testp.y)))) / 2;
									if (k < k1)
									{
										if (G.arc[k][k1] == INFINITY)
										{
											G.arc[k][k1] = tempc0;
										}
										else
										{
											G.arc[k][k1] = min(G.arc[k][k1], tempc0);
										}
									}
									else
									{
										if (G.arc[k1][k] == INFINITY)
										{
											G.arc[k1][k] = tempc0;
										}
										else
										{
											G.arc[k1][k] = min(G.arc[k1][k], tempc0);
										}
									}
									tempc1 += 1;

									//if (distest < disc3)
									//{
									//lines3[k1][7] = distest;
									//lines4.push_back(lines3[k1]);
									////cout << "lines4.size1: " << lines4.size() << endl;

									////这里尝试将长线不做标志位处理
									//if (lines3[k1][12] != -1)
									//{
									//	lines3[k1][10] = -1;
									//}
									//num += 1;
									//cv::line(topCross, testp, Point(lines3[k][4], lines3[k][5]), Scalar(0, 0, 0), 2);
									//}
								}

								lines.pop_back();


							}

						}
						//if (num == 0)
						//{
						//	lines4.pop_back();
						//	lines3[k][10] = 0;
						//}
						//else
						//{
						//	lines3[k][10] = -1;
						//}

					}


					if (lines3[k][9] == 2)
					{
						//lines4.push_back(lines3[k]);
						//lines3[k][10] = -1;
						lines.push_back(lines3[k]);
						for (int k1 = 0; k1 < lines3.size(); k1++)
						{
							tempc0 = 0;
							tempc1 = 0;
							points2.clear();
							//cout << "lines3.size:  " << lines3.size() << endl;
							//cout << "lines3[k1][9]:  " << lines3[k1][9] << endl;
							if (lines3[k1][9] != 2 && lines3[k1][10] != -1 && lines3[k1][11] != -1)
							{

								lines.push_back(lines3[k1]);

								if (isLineSegmentCross(Point(lines3[k][0], lines3[k][1]), Point(lines3[k][2], lines3[k][3]),
									Point(lines3[k1][0], lines3[k1][1]), Point(lines3[k1][2], lines3[k1][3])) && lines3[k1][10] != -1
									&& tempc1 == 0)
								{

									points2.push_back(findLineCross(lines));

									tempc0 = (abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
										+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y))) +
										abs(sqrt((lines3[k1][4] - points2[0].x)*(lines3[k1][4] - points2[0].x)
											+ (lines3[k1][5] - points2[0].y)*(lines3[k1][5] - points2[0].y)))) / 2;
									if (tempc0 <= disc)
									{
										if (k < k1)
										{
											if (G.arc[k][k1] == INFINITY)
											{
												G.arc[k][k1] = tempc0;
											}
											else
											{
												G.arc[k][k1] = min(G.arc[k][k1], tempc0);
											}
										}
										else
										{
											if (G.arc[k1][k] == INFINITY)
											{
												G.arc[k1][k] = tempc0;
											}
											else
											{
												G.arc[k1][k] = min(G.arc[k1][k], tempc0);
											}
										}
										tempc1 += 1;
									}
									//cout << "points2: " << points2[0] << endl;
									//cout <<"dis: "<< abs(sqrt((lines3[k][4] - lines3[k1][4])*(lines3[k][4] - lines3[k1][4])
										//+ (lines3[k][5] - lines3[k1][5])*(lines3[k][5] - lines3[k1][5]))) << endl;
									//if (abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
									//	+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y))) <= disc)
									//{
									//	lines3[k1][7] = abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
									//		+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y)));
									//	lines4.push_back(lines3[k1]);


									//	
									//	//这里尝试将长线不做标志位处理
									//	if (lines3[k1][12] != -1)
									//	{
									//		lines3[k1][10] = -1;
									//	}

									//	num += 1;
									//}

								}

								//替换为点到线垂足的距离
								if (abs(sqrt((lines3[k][4] - lines3[k1][4])*(lines3[k][4] - lines3[k1][4])
									+ (lines3[k][5] - lines3[k1][5])*(lines3[k][5] - lines3[k1][5]))) <= disc2 &&
									lines3[k1][10] != -1 && tempc1 == 0)
								{
									//得到垂足坐标点
									Point testp = GetFoot(Point(lines3[k][4], lines3[k][5]), Point(lines3[k1][0], lines3[k1][1]),
										Point(lines3[k1][2], lines3[k1][3]));
									//点到垂足连线

									double distest = abs(sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
										+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y)));

									tempc0 = (abs(sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
										+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y))) +
										abs(sqrt((lines3[k1][4] - testp.x)*(lines3[k1][4] - testp.x)
											+ (lines3[k1][5] - testp.y)*(lines3[k1][5] - testp.y)))) / 2;
									if (k < k1)
									{
										if (G.arc[k][k1] == INFINITY)
										{
											G.arc[k][k1] = tempc0;
										}
										else
										{
											G.arc[k][k1] = min(G.arc[k][k1], tempc0);
										}
									}
									else
									{
										if (G.arc[k1][k] == INFINITY)
										{
											G.arc[k1][k] = tempc0;
										}
										else
										{
											G.arc[k1][k] = min(G.arc[k1][k], tempc0);
										}
									}
									tempc1 += 1;

									//if (distest < disc3)
									//{
									//lines3[k1][7] = distest;
									//lines4.push_back(lines3[k1]);
									////cout << "lines4.size1: " << lines4.size() << endl;

									////这里尝试将长线不做标志位处理
									//if (lines3[k1][12] != -1)
									//{
									//	lines3[k1][10] = -1;
									//}
									//num += 1;
									//cv::line(topCross, testp, Point(lines3[k][4], lines3[k][5]), Scalar(0, 0, 0), 2);
									//}
								}

								lines.pop_back();

							}

						}
						//if (num == 0)
						//{
						//	lines4.pop_back();
						//	lines3[k][10] = 0;
						//}
						//else
						//{
						//	lines3[k][10] = -1;
						//}

					}


					if (lines3[k][9] == 3)
					{
						//lines4.push_back(lines3[k]);
						//lines3[k][10] = -1;
						lines.push_back(lines3[k]);
						for (int k1 = 0; k1 < lines3.size(); k1++)
						{
							tempc0 = 0;
							tempc1 = 0;
							points2.clear();
							//cout << "lines3.size:  " << lines3.size() << endl;
							//cout << "lines3[k1][9]:  " << lines3[k1][9] << endl;
							if (lines3[k1][9] != 3 && lines3[k1][10] != -1 && lines3[k1][11] != -1)
							{

								lines.push_back(lines3[k1]);

								if (isLineSegmentCross(Point(lines3[k][0], lines3[k][1]), Point(lines3[k][2], lines3[k][3]),
									Point(lines3[k1][0], lines3[k1][1]), Point(lines3[k1][2], lines3[k1][3])) && lines3[k1][10] != -1
									&& tempc1 == 0)
								{

									points2.push_back(findLineCross(lines));

									tempc0 = (abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
										+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y))) +
										abs(sqrt((lines3[k1][4] - points2[0].x)*(lines3[k1][4] - points2[0].x)
											+ (lines3[k1][5] - points2[0].y)*(lines3[k1][5] - points2[0].y)))) / 2;
									if (tempc0 <= disc)
									{
										if (k < k1)
										{
											if (G.arc[k][k1] == INFINITY)
											{
												G.arc[k][k1] = tempc0;
											}
											else
											{
												G.arc[k][k1] = min(G.arc[k][k1], tempc0);
											}
										}
										else
										{
											if (G.arc[k1][k] == INFINITY)
											{
												G.arc[k1][k] = tempc0;
											}
											else
											{
												G.arc[k1][k] = min(G.arc[k1][k], tempc0);
											}
										}
										tempc1 += 1;
									}
									//cout << "points2: " << points2[0] << endl;
									//cout <<"dis: "<< abs(sqrt((lines3[k][4] - lines3[k1][4])*(lines3[k][4] - lines3[k1][4])
										//+ (lines3[k][5] - lines3[k1][5])*(lines3[k][5] - lines3[k1][5]))) << endl;
									//if (abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
									//	+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y))) <= disc)
									//{
									//	lines3[k1][7] = abs(sqrt((lines3[k][4] - points2[0].x)*(lines3[k][4] - points2[0].x)
									//		+ (lines3[k][5] - points2[0].y)*(lines3[k][5] - points2[0].y)));
									//	lines4.push_back(lines3[k1]);


									//	
									//	//这里尝试将长线不做标志位处理
									//	if (lines3[k1][12] != -1)
									//	{
									//		lines3[k1][10] = -1;
									//	}

									//	num += 1;
									//}

								}

								//替换为点到线垂足的距离
								if (abs(sqrt((lines3[k][4] - lines3[k1][4])*(lines3[k][4] - lines3[k1][4])
									+ (lines3[k][5] - lines3[k1][5])*(lines3[k][5] - lines3[k1][5]))) <= disc2 &&
									lines3[k1][10] != -1 && tempc1 == 0)
								{
									//得到垂足坐标点
									Point testp = GetFoot(Point(lines3[k][4], lines3[k][5]), Point(lines3[k1][0], lines3[k1][1]),
										Point(lines3[k1][2], lines3[k1][3]));
									//点到垂足连线

									double distest = abs(sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
										+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y)));

									tempc0 = (abs(sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
										+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y))) +
										abs(sqrt((lines3[k1][4] - testp.x)*(lines3[k1][4] - testp.x)
											+ (lines3[k1][5] - testp.y)*(lines3[k1][5] - testp.y)))) / 2;
									if (k < k1)
									{
										if (G.arc[k][k1] == INFINITY)
										{
											G.arc[k][k1] = tempc0;
										}
										else
										{
											G.arc[k][k1] = min(G.arc[k][k1], tempc0);
										}
									}
									else
									{
										if (G.arc[k1][k] == INFINITY)
										{
											G.arc[k1][k] = tempc0;
										}
										else
										{
											G.arc[k1][k] = min(G.arc[k1][k], tempc0);
										}
									}
									tempc1 += 1;

									//if (distest < disc3)
									//{
									//lines3[k1][7] = distest;
									//lines4.push_back(lines3[k1]);
									////cout << "lines4.size1: " << lines4.size() << endl;

									////这里尝试将长线不做标志位处理
									//if (lines3[k1][12] != -1)
									//{
									//	lines3[k1][10] = -1;
									//}
									//num += 1;
									//cv::line(topCross, testp, Point(lines3[k][4], lines3[k][5]), Scalar(0, 0, 0), 2);
									//}
								}

								lines.pop_back();

							}

						}
						//if (num == 0)
						//{
						//	lines4.pop_back();
						//	lines3[k][10] = 0;
						//}
						//else
						//{
						//	lines3[k][10] = -1;
						//}

					}

				}

				//在此处筛选同一组中重复的投影线，将重复的投影线10号标志位重新置0进行重新搜索
				//匈牙利算法加在循环之后 标志位替换为match[N][N]即可 这一段屏蔽 不再置标志位 所有线都可以一直搜索
				//lines5.clear();
				//if (lines4.size() > 2)
				//{
				//	for (int kk = 1; kk < lines4.size() - 1; kk++)
				//	{
				//		if (lines4[kk][9] == lines4[kk + 1][9] && lines4[kk][7] != lines4[kk + 1][7] && lines4[kk][8] != -1)
				//		{
				//			if (lines4[kk][7] < lines4[kk + 1][7])
				//			{
				//				lines4[kk + 1][8] = -1;
				//				for (int kk1 = 0; kk1 < lines3.size(); kk1++)
				//				{
				//					if (lines3[kk1][7] == lines4[kk + 1][7])
				//					{
				//						lines3[kk1][10] == 0;
				//					}
				//				}
				//			}
				//			else
				//			{
				//				lines4[kk][8] = -1;
				//				for (int kk1 = 0; kk1 < lines3.size(); kk1++)
				//				{
				//					if (lines3[kk1][7] == lines4[kk][7])
				//					{
				//						lines3[kk1][10] == 0;
				//					}
				//				}
				//			}
				//		}
				//	}
				//	for (int kk2 = 0; kk2 < lines4.size(); kk2++)
				//	{
				//		if (lines4[kk2][8] != -1)
				//		{
				//			lines5.push_back(lines4[kk2]);
				//		}
				//	}

				//}
				//if (lines4.size() == 2)
				//{
				//	Point tempdis;
				//	Point temppoi;
				//	for (int kk3 = 0; kk3 < lines4.size(); kk3++)
				//	{
				//		lines5.push_back(lines4[kk3]);
				//	}

				//	//判断两线交点是否过远（当两条线平行时或接近平行时的问题）
				//	tempdis = findLineCross(lines5);
				//	if (abs(sqrt((lines3[k][4] - tempdis.x)*(lines3[k][4] - tempdis.x)
				//		+ (lines3[k][5] - tempdis.y)*(lines3[k][5] - tempdis.y))) > 30)
				//	{
				//		lines5.clear();
				//		//若交点和底边中点过远则取两条线底边中点距离的中值
				//		temppoi = Point((lines4[0][4] + lines4[1][4]) / 2, (lines4[0][5] + lines4[1][5]) / 2);
				//		points3.push_back(temppoi);
				//		//cout << "This frame !!!!!!!!!!!!!!!!!!!!!" << endl;
				//	}
				//}


				//if (lines5.size() > 1)
				//{
				//	points3.push_back(findLineCross(lines5));
				//}

			}

			
			CreateMGraph(&G);
			ShortPath_Floyd(G, P, D);
			PrintShortPath(G, P, D,lines3,points3);

			//waitKey(0);

			//检测是否所有投影线的10号标志位均为-1，如果不是则表明剩余的单独的投影线
			for (int k = 0; k < lines3.size(); k++)
			{
				//if (lines3[k][9] == 1) 
				//{
				//	cout << "lines3[k][10]: " << lines3[k][10] << endl;
				//}
				if (lines3[k][10] == 0 && lines3[k][11] != -1 && lines3[k][12] != -1)
				{
					points3.push_back(Point(lines3[k][4], lines3[k][5]));
					//将单独的短线投影到top2中
					//line(topCross2, Point(lines3[k][0], lines3[k][1]), Point(lines3[k][2], lines3[k][3]), colorful[lines3[k][9] + 15], 2);

				}
			}

			st.setTo(0);
			int num2 = 0;
			//cout << "lines2.size: " << lines2.size() << endl;
			//cout << "points2.size: " << points2.size() << endl;
			//所有points2点显示
			for (int i = 0; i < points3.size(); i++)
			{
				//画点
				//circle(topCross, points3[i], 2, colorful[i], 2);

				if (points3[i].x != 0 || points3[i].y != 0)
				{
					for (int k = 0; k < camera.size(); k++)
					{
						X = (points3[i].x - 250) * 30;
						Y = (points3[i].y - 250) * 30;
						Z = 0;
						camera[k]->cam->worldToImage(X, Y, Z, x, y);
						x = x / 2;//2倍下采样
						y = y / 2;
						Z = 2000; //2000毫米高度，矩形框顶部高度
						camera[k]->cam->worldToImage(X, Y, Z, xt, yt);//world coordinates to image coordinates
						xt = xt / 2;
						yt = yt / 2;
						//rectangle(camera[k]->frame2, Point(x - abs(y - yt)*0.35*0.5, yt), Point(x + abs(y - yt)*0.35*0.5, y), colorful[i], 2);

						//st可观测记录
						if (camera[k]->top.at<uchar>(points3[i].x, points3[i].y) == 1)
						{
							st.at<uchar>(0, i) += pow(2, k);
							camera[k]->r[i] = Rect(camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[0],
								camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[1],
								camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[2],
								camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[3]);

							camera[k]->probM[i] = 1;

							camera[k]->prob[i] = camera[k]->dutyCycle2(points3[i].x, points3[i].y);
							if (camera[k]->prob[i] < 0.1)
							{
								if (camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[0] <10
									|| camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[0] + camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[2] > camera[k]->frame2.cols - 10
									|| camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[2] / camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[3] < 0.34
									|| camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[1] < 10
									|| camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[1] + camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[3] > camera[k]->frame2.rows - 10)
								{
									camera[k]->prob[i] = 1;
								}
							}
							camera[k]->probB[i] = 1;
							camera[k]->probT[i] = 1;
						}
						else
						{
							camera[k]->probM[i] = 1;
							camera[k]->prob[i] = 1;
							camera[k]->probB[i] = 1;
							camera[k]->probT[i] = 1;
						}

					}
					num2++;
				}
				//circle(topCross, points2[i], 2, Scalar(0, 0, 0), -1);
			}
			
			//为评估做准备 向camera类中传点，存于p中
			for (int k = 0; k < camera.size(); k++)
			{
				for (int i = 0; i < points3.size(); i++)
				{
					if (points3[i].x != 0 || points3[i].y != 0)
					{
						camera[k]->p[i] = points3[i];
					}
				}
			}
			



			////for (int i = 0; i < numI; i++)
			////{
			////	lines.clear();
			////	points.clear();
			////	pushLine(lines, points, camera, 0, i);
			////}

			////for (int j = 0; j < points.size(); j++)
			////{
			////	circle(topCross, points[j], 2, Scalar(255, 255, 255), -1);
			////}

			//double tempv;
			//double tempvb;
			//double tempvth;

			//for (int i = 0; i < 1000; i = i + density)
			//	for (int j = 0; j < 1000; j = j + density)
			//	{
			//	circle(topCross, Point(i, j), 0, CV_RGB(50, 50, 50), 1);
			//	flag = 0;
			//	flag1 = 0;
			//	flag2 = 0;

			//	int flagtemp = 0;


			//	for (int k = 0; k < camera.size(); k++)
			//	{
			//		if (camera[k]->topshow.at<uchar>(i, j) == 1)
			//		{
			//			flag1++;
			//		}
			//		//if (camera[k]->dutyCycle2(i, j) >= 0.6)
			//		//{
			//		//	flag2++;
			//		//}
			//		if (camera[k]->top.at<uchar>(i, j) == 1)
			//		{
			//			flag2++;
			//			if (camera[k]->foreground.at<uchar>(camera[k]->maph.at<Point2f>(i, j).y, camera[k]->maph.at<Point2f>(i, j).x) > 0
			//				|| camera[k]->foreground.at<uchar>(camera[k]->maph.at<Point2f>(i, j).y, camera[k]->maph.at<Point2f>(i, j).x + 3) > 0
			//				|| camera[k]->foreground.at<uchar>(camera[k]->maph.at<Point2f>(i, j).y, camera[k]->maph.at<Point2f>(i, j).x - 3) > 0)
			//				flag++;
			//		}
			//			
			//	}

			//	//if (flag1>=1 && flag>=1)
			//	if (flag1 >= 2)
			//	{
			//		for (int k = 0; k < camera.size(); k++)
			//		{
			//			//再单独对视野外的框判断？
			//			//if (camera[k]->topshow.at<uchar>(i, j) == 1)
			//			//{



			//				//tempv = abs(camera[k]->boxlikelihoood(nFrmNum, bboxData, k, i, j));
			//				//tempv = camera[k]->dutyCycle2(i, j);
			//				
			//				tempv = camera[k]->dutyCycle(i, j);
			//				//if (i >= LT.x+5 && i <= RB.x-5 && j >= LT.y+5  && j <= RB.y-5)
			//				//{
			//				//	if (tempv == 1)
			//				//	{
			//				//		tempv = 0;
			//				//	}
			//				//}

			//				//if (125 < i < 500 && 125 < j < 500)
			//				//{

			//				//	if (tempv == 1)
			//				//	{
			//				//		flagtemp += 1;
			//				//	}

			//				//}


			//				if (tempv >= 0.4)
			//				{
			//					map.at<float>(i, j) *= tempv;
			//					//camera[k]->camerarect(i, j);
			//					//break;
			//				}


			//				else
			//				{
			//					//if (flag1 >= 1 && flag2 >=3)
			//					//{
			//					//	map.at<float>(i, j) *= 1;
			//					//	//break;
			//					//}
			//					//else
			//					//{
			//						map.at<float>(i, j) = 0;
			//						break;
			//					//}
			//				}
			//			//}
			//		}
			//	}
			//	else if (flag1 >= 1 && flag >= 0)
			//	{
			//		//if (i >= LT.x - 30 && i <= RB.x + 30 && j >= LT.y - 30 && j <= RB.y + 30)
			//		if (i >= LT.x && i <= RB.x && j >= LT.y  && j <= RB.y)
			//		{
			//			for (int k = 0; k < camera.size(); k++)
			//			{
			//				if (camera[k]->topshow.at<uchar>(i, j) == 1)
			//				{

			//					//tempv = camera[k]->dutyCycle2(i, j);
			//					//tempv = abs(camera[k]->boxlikelihoood(nFrmNum, bboxData, k, i, j));
			//					//if (nFrmNum == 100 && tempv >= 0.7)
			//					//{
			//					//	cout << "tempv at 100:  " << tempv << endl;
			//					//}
			//					//if (i >= LT.x + 5 && i <= RB.x - 5 && j >= LT.y + 5 && j <= RB.y - 5)
			//					//{
			//					//	if (tempv == 1)
			//					//	{
			//					//		tempv = 0;
			//					//	}
			//					//}
			//					tempv = camera[k]->dutyCycle(i, j);


			//					//if (125 < i < 500 && 125 < j < 500)
			//					//{

			//					//	if (tempv == 1)
			//					//	{
			//					//		flagtemp += 1;
			//					//	}

			//					//}


			//					if (tempv >= 0.4)
			//					{
			//						map.at<float>(i, j) *= tempv;
			//						//camera[k]->camerarect(i, j);
			//					}
			//					else
			//					{
			//						map.at<float>(i, j) = 0;
			//						break;
			//					}
			//				}

			//				


			//			}
			//		}
			//	}


			//	//if (flagtemp == 3)
			//	//{
			//	//	map.at<float>(i, j) = 0;
			//	//}

			//	if (map.at<float>(i, j) == 1)
			//	{
			//		map.at<float>(i, j) = 0;
			//	}


			//	//if (map.at<float>(i, j) > 0)
			//	//{
			//	//	for (int k = 0; k < camera.size(); k++)
			//	//	{
			//	//		if (camera[k]->topshow.at<uchar>(i, j) == 1)
			//	//		{
			//	//			camera[k]->camerarect(i, j);
			//	//		}
			//	//		
			//	//	}

			//	//}

			//	}


			////找到占空比峰值并且框出来，存在mea中
			//st.setTo(0);
			//flag2 = 0;
			//num = 0;
			//int thNum;
			//int flagv;
			//for (int i = density; i < 1000 - density; i = i + density)
			//	for (int j = density; j < 1000 - density; j = j + density)
			//	{
			//	ex = 0;
			//	thNum = 0;
			//	for (int ii = i - density; ii <= i + density; ii = ii + density)
			//		for (int jj = j - density; jj <= j + density; jj = jj + density)
			//		{
			//		if (map.at<float>(i, j) > map.at<float>(ii, jj))
			//			ex++;
			//		}

			//	for (int k = 0; k < camera.size(); k++)
			//	{
			//		thNum += camera[k]->topshow.at<uchar>(i, j);
			//		if (camera[k]->dutyCycle2(i, j) >= 0.6)
			//		{
			//			flag2++;
			//		}
			//	}

			//	if ((ex == 8) && (map.at<float>(i, j) > pow(0.55, flag2)))
			//		if (i >= LT.x - 100 && i <= RB.x + 100 && j >= LT.y - 100 && j <= RB.y + 100)
			//		{
			//		cout << "No." << num << " " << map.at<float>(i, j) << " " << thNum << " " << pow(0.5, thNum) << endl;

			//		flag = 0;
			//		flag1 = 0;
			//		flag2 = 0;

			//		for (int k = 0; k < camera.size(); k++)
			//		{
			//			if (camera[k]->topshow.at<uchar>(i, j) == 1)
			//				flag1++;
			//			if (camera[k]->top.at<uchar>(i, j) == 1)
			//			{
			//				flag2++;
			//				if (camera[k]->foreground.at<uchar>(camera[k]->maph.at<Point2f>(i, j).y, camera[k]->maph.at<Point2f>(i, j).x) > 0)
			//					flag++;
			//			}

			//		}
			//		cout << "topshow." << flag1 << " top." << flag2 << " mid." << flag << "map "<< map.at<float>(i, j) << endl;


			//		mea[num].prob = map.at<float>(i, j);
			//		mea[num].p.x = i;
			//		mea[num].p.y = j;
			//		num++;
			//		}







			//	}

			////RSS
			//int num2 = 0;
			//bool flag_delete = 1;
			//double max_temp;
			//Point max_point(-1, -1);
			//bool flag_count;
			//while (flag_delete)
			//{
			//	flag_delete = 0;
			//	max_temp = 0;
			//	max_point = Point(-1, -1);
			//	//cout << endl;



			//	for (int i = 0; i < num; i++)
			//	{
			//		//cout << mea[i].prob << mea[i].p << endl;
			//		if (mea[i].prob > max_temp)
			//		{
			//			max_temp = mea[i].prob;
			//			max_point.x = mea[i].p.x;
			//			max_point.y = mea[i].p.y;
			//		}
			//	}
			//	//cout << "win "<<max_temp << max_point << endl;
			//	//waitKey(-1);

			//	//for (int i = 0; i < num; i++)
			//	//{
			//	//	if (mea[i].p.x != -1)
			//	//	{
			//	//		//cout << sqrt((mea[i].p.x - max_point.x)*(mea[i].p.x - max_point.x) + (mea[i].p.y - max_point.y)*(mea[i].p.y - max_point.y)) << endl;
			//	//		if (sqrt((mea[i].p.x - max_point.x)*(mea[i].p.x - max_point.x) + (mea[i].p.y - max_point.y)*(mea[i].p.y - max_point.y)) <= 8)
			//	//		{
			//	//			mea[i].prob = -1;
			//	//			mea[i].p.x = -1;
			//	//			mea[i].p.y = -1;
			//	//			flag_delete = 1;
			//	//		}
			//	//	}
			//	//}

			//	//for (int i = 0; i < num; i++)
			//	//{
			//	//	//cout << mea[i].prob << mea[i].p << endl;
			//	//	if (mea[i].prob > max_temp)
			//	//	{
			//	//		max_temp = mea[i].prob;
			//	//		max_point.x = mea[i].p.x;
			//	//		max_point.y = mea[i].p.y;
			//	//	}
			//	//}

			//	flag_count = 0;
			//	if (max_point.x != -1)
			//	{
			//		for (int k = 0; k < camera.size(); k++)
			//		{
			//			//st.at<uchar>(0, num2) += pow(2, k);
			//			//camera[k]->r[num2] = Rect(camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[0],
			//			//camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[1],
			//			//camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[2],
			//			//camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[3]);

			//			//向camera类中传最大值点，存在p中
			//			//camera[k]->p[num2] = max_point;

			//			//if (camera[k]->topshow.at<uchar>(max_point.x, max_point.y) == 1)
			//			//{
			//			//	camera[k]->camerarect(max_point.x, max_point.y);
			//			//}


			//			//camera[k]->prob[num2] = camera[k]->boxlikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y);
			//			//camera[k]->prob[num2] = camera[k]->dutyCycle2(max_point.x, max_point.y);

			//			//camera[k]->probB[num2] = Gaussian1D(camera[k]->bottomlikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y) / (camera[k]->r[num2].height / 9)) * 2;
			//			//camera[k]->probB[num2] = 1;
			//			//cout << "(camera[k]->r[num2].height / 10):   " 
			//			//	<< camera[k]->r[num2].height / 10 << endl;

			//			//camera[k]->probT[num2] = Gaussian1D(camera[k]->toplikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y) / (2 * camera[k]->r[num2].height / 5)) * 2;
			//			//camera[k]->probT[num2] = 1;


			//			//camera[k]->probM[num2] = Gaussian1D(camera[k]->midlikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y) / (4 * camera[k]->r[num2].height / 5)) * 2;
			//			camera[k]->probM[num2] = 1;

			//			//if (camera[k]->topshow.at<uchar>(max_point.x, max_point.y) == 1)
			//			if (camera[k]->top.at<uchar>(max_point.x, max_point.y) == 1)
			//			{
			//				//candidate 在多少个视角中出现，以二进制方式存储
			//				st.at<uchar>(0, num2) += pow(2, k);

			//				//camera[k]->r[num2] = Rect(camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[0],
			//				//	camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[1],
			//				//	camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[2],
			//				//	camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[3]);

			//				//rectangle(camera[k]->frame, camera[k]->r[num2], colorful[num2 % 40], 2);
			//				//cvPutText(&IplImage(camera[k]->frame), text, cvPoint(17 * num2, 250), &font, colorful[num2]);
			//				camera[k]->prob[num2] = camera[k]->dutyCycle(max_point.x, max_point.y);
			//				//camera[k]->prob[num2] = camera[k]->boxlikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y);
			//				//cout << "prob[num2]:  " << camera[k]->prob[num2] << endl;
			//				//int numI = 0;
			//				//int column[20];
			//				//double tempv1 = 0;
			//				//double tempv2 = 0;
			//				//for (int bbp = 0; bbp < 3166; bbp++)
			//				//{
			//				//	if (bboxData.at<float>(bbp, 0) == nFrmNum)
			//				//		if(bboxData.at<float>(bbp, 1)==k)
			//				//			{
			//				//				column[numI] = bbp;
			//				//				numI += 1;
			//				//				
			//				//			}
			//				//}
			//				//if (numI > 0)
			//				//{
			//				//	if (numI == 1)
			//				//	{
			//				//		int x1 = bboxData.at<float>(nFrmNum, 3);
			//				//		int y1 = bboxData.at<float>(nFrmNum, 4);
			//				//		int width = bboxData.at<float>(nFrmNum, 5);
			//				//		int height = bboxData.at<float>(nFrmNum, 6);
			//				//		tempv1 = camera[k]->bbOverlap2(Rect(x1, y1, width, height), Rect(camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[0], camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[1],
			//				//			camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[2], camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[3]));
			//				//		//if (tempv == 1.0)

			//				//		camera[k]->probI[num2] = tempv1;
			//				//	}
			//				//	else
			//				//	{
			//				//		
			//				//		for (int numI2 = numI; numI2 >= 0; numI2--)
			//				//		{
			//				//			int bbrow = column[numI2];
			//				//			int x1 = bboxData.at<float>(bbrow, 3);
			//				//			int y1 = bboxData.at<float>(bbrow, 4);
			//				//			int width = bboxData.at<float>(bbrow, 5);
			//				//			int height = bboxData.at<float>(bbrow, 6);
			//				//			tempv1 = camera[k]->bbOverlap2(Rect(x1, y1, width, height), Rect(camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[0], camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[1],
			//				//				camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[2], camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[3]));
			//				//		
			//				//			if (tempv1 > tempv2)
			//				//				tempv2 = tempv1;
			//				//		}
			//				//		if (tempv2 != 0)
			//				//			camera[k]->probI[num2] = tempv2;
			//				//		else
			//				//			camera[k]->probI[num2] = 1;
			//				//	}



			//				//}
			//				//else
			//				//{
			//				//	camera[k]->probI[num2] = 1;
			//				//}




			//				//if (camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[2] / camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[3]>=0.3)
			//				//	camera[k]->prob[num2] = camera[k]->dutyCycle(max_point.x, max_point.y);
			//				//else
			//				//{
			//				//	camera[k]->prob[num2] = 1;
			//				//	camera[k]->probB[num2] = 1;
			//				//	camera[k]->probT[num2] = 1;
			//				//	continue;
			//				//}

			//				//侧面投影
			//				Mat rst(1, camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[3], CV_64FC1, Scalar(0));
			//				Mat roim = camera[k]->foreground(camera[k]->r[num2]);
			//				Mat roi3 = camera[k]->frame(camera[k]->r[num2]);
			//				Mat roidoublem;
			//				roim.convertTo(roidoublem, CV_64FC1, 1 / 255.0);
			//				reduce(roidoublem, rst, 1, CV_REDUCE_SUM);

			//				double B;
			//				double T;

			//				flagv = 0;
			//				int vcounter = 0;
			//				if (camera[k]->r[num2].width > 0 && ((double)camera[k]->r[num2].width / (double)camera[k]->r[num2].height > 0.2) && (double)camera[k]->r[num2].width / (double)camera[k]->r[num2].height < 0.5)
			//					for (int m = camera[k]->r[num2].height - 1; m >= 0; m--)
			//					{
			//					if (flagv == 0)
			//					{
			//						if (rst.at<double>(0, m) <= camera[k]->r[num2].width / 10.0)
			//							vcounter++;
			//						else
			//							flagv++;
			//					}
			//					else
			//					{
			//						if (rst.at<double>(0, m) <= camera[k]->r[num2].width / 10.0)
			//						{
			//							vcounter = vcounter + flagv;
			//							flagv = 0;
			//						}
			//						else
			//							flagv++;
			//					}
			//					if (flagv >= 5)
			//					{
			//						B = m + 4;
			//						break;
			//					}
			//					}
			//				else
			//				{
			//					B = camera[k]->r[num2].height - 1;
			//				}
			//				//cout << "GSB: " << num2 << " " << B+1 << " "<<camera[k]->r[num2].height<<" ";
			//				//camera[k]->probB[num2] = MahaDistance1D(camera[k]->r[num2].height, B + 1, camera[k]->r[num2].height * 4.0 / 20);
			//				//camera[k]->probB[num2] = 1;
			//				//camera[k]->probB[num2] = Gaussian1D((camera[k]->r[num2].height - (B + 1)) / (camera[k]->r[num2].height * 1 / 10)) * 2;
			//				camera[k]->probB[num2] = Gaussian1D(camera[k]->bottomlikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y) / (camera[k]->r[num2].height / 9)) * 2;
			//				//if (camera[k]->probB[num2] < 0)
			//				//{
			//				//	camera[k]->probB[num2] = 0;
			//				//}
			//				//camera[k]->probB[num2] = 1;
			//				//camera[k]->probB[num2] = Gaussian1D(camera[k]->blikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y)) * 2;
			//				//camera[k]->probB[num2] = Gaussian1D(camera[k]->bottomlikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y) / (camera[k]->r[num2].height / 10)) * 2;
			//				//cout << "probB[num2]" << camera[k]->probB[num2] << endl;
			//				//cout<<"camera[k]->blikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y)"<< camera[k]->blikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y)<<endl;
			//				//cout << camera[k]->probB[num2] << endl;

			//				flagv = 0;
			//				vcounter = 0;
			//				if (camera[k]->r[num2].width > 0 && ((double)camera[k]->r[num2].width / (double)camera[k]->r[num2].height > 0.2) && (double)camera[k]->r[num2].width / (double)camera[k]->r[num2].height < 0.5)
			//					for (int m = 0; m < camera[k]->r[num2].height; m++)
			//					{
			//					if (flagv == 0)
			//					{
			//						if (rst.at<double>(0, m) <= camera[k]->r[num2].width / 10.0)
			//							vcounter++;
			//						else
			//							flagv++;
			//					}
			//					else
			//					{
			//						if (rst.at<double>(0, m) <= camera[k]->r[num2].width / 10.0)
			//						{
			//							vcounter = vcounter + flagv;
			//							flagv = 0;
			//						}
			//						else
			//							flagv++;
			//					}
			//					if (flagv >= 5)
			//					{
			//						T = m - 4;
			//						break;
			//					}
			//					}
			//				else
			//				{
			//					T = 0;
			//				}
			//				//cout << endl;
			//				//cout << "GST: " << num2 << " " << T << " " << endl;
			//				//camera[k]->probT[num2] = MahaDistance1D(0, T, camera[k]->r[num2].height * 8.0 / 20);

			//				//camera[k]->probT[num2] = Gaussian1D(T / (camera[k]->r[num2].height * 2 / 5)) * 2;
			//				camera[k]->probT[num2] = Gaussian1D(camera[k]->toplikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y) / (3 * camera[k]->r[num2].height / 5)) * 2;

			//				//if (camera[k]->probT[num2] < 0)
			//				//{
			//				//	camera[k]->probT[num2] = 0;
			//				//}
			//				//camera[k]->probT[num2] = 1;
			//				//camera[k]->probT[num2] = Gaussian1D(camera[k]->tlikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y))*2
			//				//	* Gaussian1D2(camera[k]->tlikelihoood2(nFrmNum, bboxData, k, max_point.x, max_point.y)) * 2;

			//				//这里还需要修改
			//				//camera[k]->probT[num2] = Gaussian1D(camera[k]->bottomlikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y) / (2 * camera[k]->r[num2].height / 10)) * 2;
			//				
			//				//camera[k]->probT[num2] = (abs(1 - camera[k]->tlikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y)))
			//				//	* Gaussian1D(camera[k]->tlikelihoood2(nFrmNum, bboxData, k, max_point.x, max_point.y) / (2 * camera[k]->r[num2].height / 5))*2;
			//				//cout << "t1:  " << Gaussian1D(camera[k]->tlikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y) / (camera[k]->r[num2].height / 5)) * 2 << endl;
			//				//cout << "t2:  " << Gaussian1D(camera[k]->tlikelihoood2(nFrmNum, bboxData, k, max_point.x, max_point.y) / (3 * camera[k]->r[num2].height / 5)) * 2 << endl;
			//				//cout << "camera[k]->tlikelihoood: " << camera[k]->tlikelihoood(nFrmNum, bboxData, k, max_point.x, max_point.y) << endl;
			//				//cout << "camera[k]->probT[num2]: " << camera[k]->probT[num2] << endl;
			//				//cout << camera[k]->probT[num2] << endl;

			//				//for (int m = 0; m < roim.rows - 1; m++)
			//				//	line(roi3, Point(rst.at<double>(0, m), m), Point(rst.at<double>(0, m + 1), m + 1), CV_RGB(255, 255, 255), 2);
			//				//line(roi3, Point(0, 0), Point(0, camera[k]->r[num2].height), CV_RGB(255, 255, 255), 2);
			//			}
			//			else
			//			{
			//				camera[k]->prob[num2] = 1;
			//				camera[k]->probB[num2] = 1;
			//				camera[k]->probT[num2] = 1;
			//				camera[k]->probM[num2] = 1;
			//			}

			//			//if (camera[k]->topshow.at<uchar>(max_point.x, max_point.y) == 1)
			//			//{
			//			//	if (camera[k]->probB[num2] > 0.6)
			//			//	{
			//			//		camera[k]->camerarect(max_point.x, max_point.y);
			//			//	}
			//			//}
			//		}

			//		num2++;

			//		//for (int k = 0; k < camera.size(); k++)
			//		//{
			//		//	if (camera[k]->topshow.at<uchar>(max_point.x, max_point.y) == 1)
			//		//	{
			//		//		camera[k]->camerarect(max_point.x, max_point.y);
			//		//	}
			//		//	
			//		//}



			//		//RSS
			//		for (int i = 0; i < num; i++)
			//		{
			//			if (mea[i].p.x != -1)
			//			{
			//				//cout << sqrt((mea[i].p.x - max_point.x)*(mea[i].p.x - max_point.x) + (mea[i].p.y - max_point.y)*(mea[i].p.y - max_point.y)) << endl;
			//				if (sqrt((mea[i].p.x - max_point.x)*(mea[i].p.x - max_point.x) + (mea[i].p.y - max_point.y)*(mea[i].p.y - max_point.y)) <= 8)
			//				{
			//					mea[i].prob = -1;
			//					mea[i].p.x = -1;
			//					mea[i].p.y = -1;
			//					flag_delete = 1;
			//				}
			//			}
			//		}

			//	}
			//}
				



			//找轮廓，画轮廓
			//vector<vector<Point>> contour;

			//for (int k = 0; k < camera.size(); k++)
			//{
			//	findContours(camera[k]->foreground, contour, RETR_LIST, CHAIN_APPROX_NONE);
			//	drawContours(camera[k]->frame, contour, -1, CV_RGB(0, 255, 0), 1);
			//}

			result.clear();
			//for (int i = 0; i < points3.size(); i++)
			//{
			//	result.push_back(i);
			//}

			//使用QM method
			//camera 摄像机类，包含了前景图和candidate
			//st 储存了每一个candidate是在几个视角中可见的，二进制方式储存
			//num2 candidate 的数目
			//
			QMmethod(camera, &st, num2, result);

			//for (int i = 0; i < result.size(); i++)
			//{
			//	cout << "result[" << i << "]" << ":   " << result[i] << endl;
			//}

			rectangle(topCross, Rect(125, 125, 500, 500), CV_RGB(255, 255, 255), 1);
			//rectangle(topCross2, Rect(125, 125, 500, 500), CV_RGB(255, 255, 255), 1);
			//rectangle(topCross3, Rect(125, 125, 500, 500), CV_RGB(255, 255, 255), 1);
			//rectangle(topCross4, Rect(125, 125, 500, 500), CV_RGB(255, 255, 255), 1);

			bool flagrst = 0;
			for (int i = 0; i < num2; i++)
			{
				flagrst = 0;
				for (int j = 0; j < result.size(); j++)
				{
					if (i == result[j])
						flagrst = 1;
				}
				//顶视图画点
				if (flagrst == 1)
				{
					//if(i!=5 && i!=8 && i!=9 && i!=11)
					//circle(topCross4, camera[0]->p[i], 4, colorful[i], -1);
					circle(topCross, camera[0]->p[i], 4, colorful[i], -1);

					//circle(topCross, camera[0]->p[i], 3, colorful[i], 2);
				}
				else
				{
					//if (i != 5 && i != 8 && i != 9 && i != 11)
					//circle(topCross4, camera[0]->p[i], 3, colorful[i], 2);
					circle(topCross, camera[0]->p[i], 3, colorful[i], 2);
				}

			}

			//}
			flip(topCross, topCross, 0);
			//flip(topCross2, topCross2, 0);
			//flip(topCross3, topCross3, 0);
			//flip(topCross4, topCross4, 0);

			putText(topCross, "C0", Point(camera[0]->cam->mCposx / 30.0 + 250, 1000 - (camera[0]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 2, CV_AA);
			putText(topCross, "C1", Point(camera[1]->cam->mCposx / 30.0 + 250, 1000 - (camera[1]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 255, 0), 2, CV_AA);
			putText(topCross, "C2", Point(camera[2]->cam->mCposx / 30.0 + 250 - 25, 1000 - (camera[2]->cam->mCposy / 30.0 + 250 - 25)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 255), 2, CV_AA);
			putText(topCross, "C3", Point(camera[3]->cam->mCposx / 30.0 + 250, 1000 - (camera[3]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 0), 2, CV_AA);

			//在topview上添加显示帧号
			char sz[10];//字符串
			itoa(nFrmNum, sz, 10);
			putText(topCross, sz, Point(camera[3]->cam->mCposx / 30.0 + 50, 1000 - (camera[3]->cam->mCposy / 30.0 + 350)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 255, 255), 2, CV_AA);

			//putText(topCross2, "C0", Point(camera[0]->cam->mCposx / 30.0 + 250, 1000 - (camera[0]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 2, CV_AA);
			//putText(topCross2, "C1", Point(camera[1]->cam->mCposx / 30.0 + 250, 1000 - (camera[1]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 255, 0), 2, CV_AA);
			//putText(topCross2, "C2", Point(camera[2]->cam->mCposx / 30.0 + 250 - 25, 1000 - (camera[2]->cam->mCposy / 30.0 + 250 - 25)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 255), 2, CV_AA);
			//putText(topCross2, "C3", Point(camera[3]->cam->mCposx / 30.0 + 250, 1000 - (camera[3]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 0), 2, CV_AA);


			//putText(topCross3, "C0", Point(camera[0]->cam->mCposx / 30.0 + 250, 1000 - (camera[0]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 2, CV_AA);
			//putText(topCross3, "C1", Point(camera[1]->cam->mCposx / 30.0 + 250, 1000 - (camera[1]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 255, 0), 2, CV_AA);
			//putText(topCross3, "C2", Point(camera[2]->cam->mCposx / 30.0 + 250 - 25, 1000 - (camera[2]->cam->mCposy / 30.0 + 250 - 25)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 255), 2, CV_AA);
			//putText(topCross3, "C3", Point(camera[3]->cam->mCposx / 30.0 + 250, 1000 - (camera[3]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 0), 2, CV_AA);

			//putText(topCross4, "C0", Point(camera[0]->cam->mCposx / 30.0 + 250, 1000 - (camera[0]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 2, CV_AA);
			//putText(topCross4, "C1", Point(camera[1]->cam->mCposx / 30.0 + 250, 1000 - (camera[1]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 255, 0), 2, CV_AA);
			//putText(topCross4, "C2", Point(camera[2]->cam->mCposx / 30.0 + 250 - 25, 1000 - (camera[2]->cam->mCposy / 30.0 + 250 - 25)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 255), 2, CV_AA);
			//putText(topCross4, "C3", Point(camera[3]->cam->mCposx / 30.0 + 250, 1000 - (camera[3]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 0), 2, CV_AA);


			imshow("C0", camera[0]->frame2); waitKey(1);
			imshow("C1", camera[1]->frame2); waitKey(1);
			imshow("C2", camera[2]->frame2); waitKey(1);
			imshow("C3", camera[3]->frame2); waitKey(1);
			//imshow("C0", camera[0]->foreground); waitKey(1);
			//imshow("C1", camera[1]->foreground); waitKey(1);
			//imshow("C2", camera[2]->foreground); waitKey(1);
			//imshow("C3", camera[3]->foreground); waitKey(1);
			//imwrite("C:\\Users\\Administrator\\Desktop\\Terrace-4view-POM\\ConsoleApplication6\\foregroundt.bmp", camera[0]->foreground);

			//imshow("top", mapTop); waitKey(1);
			//imshow("top2", mapTop); waitKey(1);
			//imshow("top3", mapTop); waitKey(1);
			
			select_ROI = Rect(50, 300, 650, 650);
			Mat ROI = topCross(select_ROI);
			imshow("top", ROI); waitKey(1);

			//select_ROI2 = Rect(50, 300, 650, 650);
			//Mat ROI2 = topCross2(select_ROI2);
			//imshow("top2", ROI2); waitKey(1);

			//select_ROI3 = Rect(50, 300, 650, 650);
			//Mat ROI3 = topCross3(select_ROI3);
			//imshow("top3", ROI3); waitKey(1);

			//select_ROI4 = Rect(50, 300, 650, 650);
			//Mat ROI4 = topCross4(select_ROI4);
			//imshow("top4", ROI4); waitKey(1);

			cout << "Current Frame Number:" << "" << nFrmNum << endl;

			//保存指定帧
			//string sname;
			//if (nFrmNum <= 5000)
			//{
			//	for (int k = 0; k < camera.size(); k++)
			//	{
			//		sname = "E:\\Terrace\\DR\\"+to_string(nFrmNum) + "_" + to_string(k) + ".png";
			//		imwrite(sname, camera[k]->frame2);
			//	}
			//	sname = "E:\\Terrace\\DR\\" +to_string(nFrmNum) + "_top.png";
			//	imwrite(sname, ROI);
			//}





			//在二分图里，我们定义左集合i 代表ground truth, 右集合j 代表detection results

			float overlap;

			bool flagdraw;
			bool flagmiss = 0;

			int miss_count = 0;
			int miss_countk = 0;

			float gdIOU = 0;
			float frIOU = 0;
			float totIOU = 0;
			int tNframe = 0;

			int flaga, flagb, flagaf, flagbf = 0;

			int canum, frHit = 0;
			float tfrIOU = 0;

			//countdet = countdet + result.size();

			//for (int j = 0; j < num2; j++)
			//{
			//	for (int i = 0; i < result.size(); i++)
			//	{
			//		if (j == result[i])
			//		{
			//			if (!(camera[0]->p[j].x >= LT.x - 5 && camera[0]->p[j].x <= RB.x + 5 &&
			//				camera[0]->p[j].y >= LT.y - 5 && camera[0]->p[j].y <= RB.y + 5))
			//			{
			//				countdet--;
			//			}
			//		}
			//	}
			//}

			int ga, gb, gn;
			if (gt_n.size() > 0)
			{
				for (int gt_in = 0; gt_in < gt_n.size(); gt_in++)
				{

					flagaf = 0; //表示当前ground truth的可观测的视图数量C
					flagbf = 0; //表示当前ground truth符合条件的视图数量N

					flagdraw = 0;
					ga = gt_n[gt_in] % 30 * 250 + 125 - 500;
					gb = gt_n[gt_in] / 30 * 250 + 125 - 1500;

					ga = ga / 30 + 250;
					gb = gb / 30 + 250;


					flag1 = 0;
					for (int k = 0; k < camera.size(); k++)
					{
						if (camera[k]->topshow.at<uchar>(ga, gb) == 1)
							flag1++;
					}

					if (flag1 >= 1)
					{
						//tNframe++;
						int tnum = 0;
						float maxIOU = 0;

						for (int i = 0; i < result.size(); i++)
						{
							flaga = 0; //表示当前检测目标的可观测的视图数量C
							flagb = 0; //表示当前检测目标符合条件的视图数量N
							if (result[i] != -1)
							{
								//flagdraw = 0;
								for (int k = 0; k < camera.size(); k++)
								{
									if (camera[k]->topshow.at<uchar>(ga, gb) == 1)
									{
										//在ground truth中中的点直接在各视图中和所有candidate box计算重合率
										overlap = bbOverlap(Rect(camera[k]->rctMap.at<Vec4f>(ga, gb)[0], camera[k]->rctMap.at<Vec4f>(ga, gb)[1],
											camera[k]->rctMap.at<Vec4f>(ga, gb)[2], camera[k]->rctMap.at<Vec4f>(ga, gb)[3]),
											Rect(camera[k]->rctMap.at<Vec4f>(int(camera[0]->p[result[i]].x), int(camera[0]->p[result[i]].y))[0], camera[k]->rctMap.at<Vec4f>(int(camera[0]->p[result[i]].x), int(camera[0]->p[result[i]].y))[1],
												camera[k]->rctMap.at<Vec4f>(int(camera[0]->p[result[i]].x), int(camera[0]->p[result[i]].y))[2], camera[k]->rctMap.at<Vec4f>(int(camera[0]->p[result[i]].x), int(camera[0]->p[result[i]].y))[3]));

										gdIOU += overlap;
										canum += 1;
										flaga += 1;
										//IOU小于0.5的情况
										if (overlap < 0.5)
										{
											//如果在ga gb点立的框宽高比不对且靠近视图边缘处
											if ((camera[k]->rctMap.at<Vec4f>(ga, gb)[0] <= 2
												|| (camera[k]->rctMap.at<Vec4f>(ga, gb)[0] + camera[k]->rctMap.at<Vec4f>(ga, gb)[2]) >= camera[k]->frame.cols - 2)
												&& camera[k]->rctMap.at<Vec4f>(ga, gb)[2] / camera[k]->rctMap.at<Vec4f>(ga, gb)[3] < 0.34)
												//&& camera[k]->rctMap.at<Vec4f>(ga, gb)[2] / camera[k]->rctMap.at<Vec4f>(ga, gb)[3] <= 0.36)
											{
												//flagdet = 1;
												//continue;
												flagb += 1;
											}
											else
											{
												//flagdraw = 1;
												//gdIOU = 0;
												//canum = 0;
												//break;												
											}
										}
										else
										{
											flagb += 1;
										}
									}
								}
								//如果flagb不等于0,说明至少有一个camera view满足条件
								if (flagb != 0)
								{
									tfrIOU = gdIOU / canum;
									//countHit++;
									if (tfrIOU > maxIOU)
									{
										tnum = i;
										maxIOU = tfrIOU;
										flagaf = flaga;
										flagbf = flagb;
									}
									tfrIOU = 0;
									gdIOU = 0;
									canum = 0;
								}
								else
								{
									gdIOU = 0;
									canum = 0;
								}
							}
						}

						//maxIOU不等于0说明result[tnum]是所有框里跟当前ground truth得到最大IOU的框
						if (maxIOU)
						{
							//gt和检测结果任意一个在buffer zone里，gt det hit 都+1
							if ((ga >= LT.x - 5 && ga <= RB.x + 5 && gb >= LT.y - 5 && gb <= RB.y + 5) || (camera[0]->p[result[tnum]].x >= LT.x - 5 &&
								camera[0]->p[result[tnum]].x <= RB.x + 5 && camera[0]->p[result[tnum]].y >= LT.y - 5 &&
								camera[0]->p[result[tnum]].y <= RB.y + 5))
							{
								countGT++;  //GT 计数
								countdet++; //检测结果计数+1
								//cout << "flagaf: " << flagaf <<" " << "flagbf: " << flagbf << endl;
								//匹配数需要看是否在所有可观测视图内都满足条件 满足就+1 不满足+N/C N是几个视图符合要求 C是可观测视图数
								if (flagaf == flagbf)
								{
									countHit++; //匹配数计数
								}
								else
								{
									countHit += float(flagbf) / float(flagaf);
								}
								frHit++; //每帧匹配数 用于计算MODP							
								tNframe++; //记录当前帧有没有匹配
								result[tnum] = -1;
								totIOU += maxIOU;
							}
						}
						else
						{
							//如果在检测范围-5里，则gt +1，也就是一个漏检
							if (ga >= LT.x + 5 && ga <= RB.x - 5 && gb >= LT.y + 5 && gb <= RB.y - 5)
							{
								countGT++;  //GT 计数
							}
						}

					}

				}

				//跑完所有的ground truth后如果仍然有没有被匹配的检测结果，如果在检测范围-5里，就算 det +1
				for (int i = 0; i < result.size(); i++)
				{
					if (result[i] != -1)
					{
						if (camera[0]->p[result[i]].x >= LT.x + 5 && camera[0]->p[result[i]].x <= RB.x - 5
							&& camera[0]->p[result[i]].y >= LT.y + 5 && camera[0]->p[result[i]].y <= RB.y - 5)
						{
							countdet++; //检测结果计数+1
						}
					}
				}

				if (tNframe > 0)
				{
					Nframe++;
				}

				if (totIOU > 0)
				{
					MODP += totIOU / frHit;
				}
				totIOU = 0;
				frHit = 0;
			}

			//if (gt_n.size() > 0)
			//{
			//	gt.clear();
			//	for (int gt_in = 0; gt_in < gt_n.size(); gt_in++)
			//	{
			//		flagdraw = 0;
			//		ga = gt_n[gt_in] % 30 * 250 + 125 - 500;
			//		gb = gt_n[gt_in] / 30 * 250 + 125 - 1500;

			//		wga = gt_n[gt_in] % 30 * 250 + 125 - 500;
			//		wgb = gt_n[gt_in] / 30 * 250 + 125 - 1500;

			//		ga = ga / 30 + 250;
			//		gb = gb / 30 + 250;

			//		//画r=0.5m时极限情况图

			//		//flip(topCross, topCross, 0);
			//		//circle(topCross, Point(ga, gb), 16, CV_RGB(255, 0, 0), 1);
			//		//circle(topCross, Point(ga, gb), 3, CV_RGB(255, 0, 0), -1);
			//		//circle(topCross, Point(ga+16, gb), 3, CV_RGB(0, 0, 255), -1);
			//		//rectangle(camera[0]->frame2, Rect(camera[0]->rctMap.at<Vec4f>(ga, gb)[0], camera[0]->rctMap.at<Vec4f>(ga, gb)[1], 
			//		//	camera[0]->rctMap.at<Vec4f>(ga, gb)[2], camera[0]->rctMap.at<Vec4f>(ga, gb)[3]), CV_RGB(255, 0, 0), 2);
			//		//rectangle(camera[0]->frame2, Rect(camera[0]->rctMap.at<Vec4f>(ga+16, gb)[0], camera[0]->rctMap.at<Vec4f>(ga+16, gb)[1],
			//		//	camera[0]->rctMap.at<Vec4f>(ga+16, gb)[2], camera[0]->rctMap.at<Vec4f>(ga+16, gb)[3]), CV_RGB(0, 0, 255), 2);
			//		//flip(topCross, topCross, 0);
			//		//imshow("topgt",topCross);
			//		//imshow("C0", camera[0]->frame2);
			//		//imwrite("E:\\Wildtrack\\r05top.jpg",topCross);
			//		//imwrite("E:\\Wildtrack\\r05C0.jpg", camera[0]->frame2);
			//		//overlap = bbOverlap(Rect(camera[0]->rctMap.at<Vec4f>(ga, gb)[0], camera[0]->rctMap.at<Vec4f>(ga, gb)[1],
			//		//	camera[0]->rctMap.at<Vec4f>(ga, gb)[2], camera[0]->rctMap.at<Vec4f>(ga, gb)[3]),
			//		//	Rect(camera[0]->rctMap.at<Vec4f>(ga + 16, gb)[0], camera[0]->rctMap.at<Vec4f>(ga + 16, gb)[1],
			//		//		camera[0]->rctMap.at<Vec4f>(ga + 16, gb)[2], camera[0]->rctMap.at<Vec4f>(ga + 16, gb)[3]));
			//		//cout << "overlap ratio: " << overlap << endl;
			//		//waitKey(0);

			//		for (int k = 0; k < camera.size(); k++)
			//		{
			//			gt.push_back(Rect(camera[k]->rctMap.at<Vec4f>(ga, gb)[0], camera[k]->rctMap.at<Vec4f>(ga, gb)[1], camera[k]->rctMap.at<Vec4f>(ga, gb)[2], camera[k]->rctMap.at<Vec4f>(ga, gb)[3]));
			//			
			//			//将ground truth在顶视图上的点反单应回各个视图立框，存这些框
			//			//if (ga >= LT.x - 5 && ga <= RB.x + 5 && gb >= LT.y - 5 && gb <= RB.y + 5)
			//			//{
			//			//	fprintf(fpbox, "%-5d", nFrmNum);
			//			//	fprintf(fpbox, "%-5d", k);
			//			//	fprintf(fpbox, "%5.2f ", camera[k]->rctMap.at<Vec4f>(ga, gb)[0]);
			//			//	fprintf(fpbox, "%5.2f ", camera[k]->rctMap.at<Vec4f>(ga, gb)[1]);
			//			//	fprintf(fpbox, "%5.2f ", camera[k]->rctMap.at<Vec4f>(ga, gb)[2]);
			//			//	fprintf(fpbox, "%5.2f ", camera[k]->rctMap.at<Vec4f>(ga, gb)[3]);
			//			//	fprintf(fpbox, "\n");
			//			//}
			//		}
			//		//fprintf(fpbox, "\n");
			//		

			//		flag1 = 0;
			//		for (int k = 0; k < camera.size(); k++)
			//		{
			//			if (camera[k]->topshow.at<uchar>(ga, gb) == 1)
			//				flag1++;
			//		}

			//		if (flag1 >= 1)
			//			if (ga >= LT.x - 5 && ga <= RB.x + 5 && gb >= LT.y - 5 && gb <= RB.y + 5)
			//			{
			//				countGT++;
			//				tNframe++;
			//				int tnum = 0;
			//				float maxIOU = 0;
			//				float mindistance = 600;

			//				for (int i = 0; i < result.size(); i++)
			//				{
			//					if (result[i] != -1)
			//					{ 
			//						wdx = (camera[0]->p[result[i]].x - 250) * 30;
			//						wdy = (camera[0]->p[result[i]].y - 250) * 30;
			//						flagdraw = 0;

			//						distance = abs(sqrt((wdx - wga)*(wdx - wga)
			//							+ (wdy - wgb)*(wdy - wgb)));

			//						
			//						for (int k = 0; k < camera.size(); k++)
			//						{
			//							if (camera[k]->topshow.at<uchar>(ga, gb) == 1)
			//							{
			//								//在ground truth中中的点直接在各视图中和所有candidate box计算重合率
			//								overlap = bbOverlap(Rect(camera[k]->rctMap.at<Vec4f>(ga, gb)[0], camera[k]->rctMap.at<Vec4f>(ga, gb)[1],
			//									camera[k]->rctMap.at<Vec4f>(ga, gb)[2], camera[k]->rctMap.at<Vec4f>(ga, gb)[3]),
			//									Rect(camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[i]].x), int(camera[k]->p[result[i]].y))[0], camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[i]].x), int(camera[k]->p[result[i]].y))[1],
			//										camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[i]].x), int(camera[k]->p[result[i]].y))[2], camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[i]].x), int(camera[k]->p[result[i]].y))[3]));

			//								gdIOU += overlap;
			//								canum += 1;

			//								//小于0.3的情况
			//								//if (overlap < 0.3)
			//								//{
			//								//	//rectangle(camera[k]->frame2, Rect(camera[k]->rctMap.at<Vec4f>(ga, gb)[0], camera[k]->rctMap.at<Vec4f>(ga, gb)[1], camera[k]->rctMap.at<Vec4f>(ga, gb)[2], camera[k]->rctMap.at<Vec4f>(ga, gb)[3]), CV_RGB(255, 255, 255), 2);
			//								//	//如果在ga gb点立的框宽高比不对且靠近视图边缘处
			//								//	if ((camera[k]->rctMap.at<Vec4f>(ga, gb)[0] <= 2
			//								//		|| (camera[k]->rctMap.at<Vec4f>(ga, gb)[0] + camera[k]->rctMap.at<Vec4f>(ga, gb)[2]) >= camera[k]->frame.cols - 2)
			//								//		&& camera[k]->rctMap.at<Vec4f>(ga, gb)[2] / camera[k]->rctMap.at<Vec4f>(ga, gb)[3] < 0.34)
			//								//		//&& camera[k]->rctMap.at<Vec4f>(ga, gb)[2] / camera[k]->rctMap.at<Vec4f>(ga, gb)[3] <= 0.36)
			//								//	{
			//								//		//flagdet = 1;
			//								//		//continue;
			//								//	}
			//								//	else
			//								//	{
			//								//		flagdraw = 1;
			//								//		//rectangle(camera[k]->frame2, Rect(camera[k]->rctMap.at<Vec4f>(ga, gb)[0], camera[k]->rctMap.at<Vec4f>(ga, gb)[1], camera[k]->rctMap.at<Vec4f>(ga, gb)[2], camera[k]->rctMap.at<Vec4f>(ga, gb)[3]), CV_RGB(255, 255, 255), 2);
			//								//		//countHit++;





			//								//		//
			//								//		//cout <<"rctMap(ga,gb)[0]"<<""<< camera[k]->rctMap.at<Vec4f>(ga, gb)[0] << endl;
			//								//		//cout << "rctMap(ga,gb)[1]" << "" << camera[k]->rctMap.at<Vec4f>(ga, gb)[1] << endl;
			//								//		//cout << "rctMap(ga,gb)[2]" << "" << camera[k]->rctMap.at<Vec4f>(ga, gb)[2] << endl;
			//								//		//cout << "rctMap(ga,gb)[3]" << "" << camera[k]->rctMap.at<Vec4f>(ga, gb)[3] << endl;
			//								//		gdIOU = 0;
			//								//		canum = 0;
			//								//		break;
			//								//	}
			//								//}
			//							}
			//						}
			//						if (distance <= 500)
			//						{
			//							tempdistance = distance;
			//						}
			//						else
			//						{
			//							distance = 1000;
			//							flagdraw = 1;
			//							gdIOU = 0;
			//							canum = 0;
			//							//break;
			//						}
			//						if (!flagdraw)
			//						{
			//							tfrIOU = gdIOU / canum;
			//							//countHit++;
			//							for (int k = 0; k < camera.size(); k++)
			//							{
			//								if (camera[k]->topshow.at<uchar>(ga, gb) == 1)
			//								{
			//									// rectangle(camera[k]->frame, Rect(camera[k]->rctMap.at<Vec4f>(ga, gb)[0], camera[k]->rctMap.at<Vec4f>(ga, gb)[1], camera[k]->rctMap.at<Vec4f>(ga, gb)[2], camera[k]->rctMap.at<Vec4f>(ga, gb)[3]), CV_RGB(255, 0, 0), 2);
			//								}
			//							}
			//							//break;
			//							if (tempdistance < mindistance)
			//							{
			//								mindistance = tempdistance;
			//								tnum = i;
			//								maxIOU = tfrIOU;
			//							}
			//							tfrIOU = 0;
			//							gdIOU = 0;
			//							canum = 0;
			//							distance = 1000;
			//						}
			//					}
			//				}

			//				if (mindistance != 600)
			//				{
			//					frHit++;
			//					countHit++;
			//					result[tnum] = -1;
			//					totIOU += maxIOU;
			//					//cout << "totIOU: " << totIOU << " " << "frHit: " <<frHit<< endl;
			//				}

			//				if (mindistance == 600)
			//				{
			//					for (int k = 0; k < camera.size(); k++)
			//					{
			//						if (camera[k]->topshow.at<uchar>(ga, gb) == 1)
			//						{
			//							//在AOI边界处设立缓冲区 点范围内5个像素找
			//							if (abs(ga - LT.x) < 5 || abs(gb - LT.y) < 5 || abs(ga - RB.x) < 5 || abs(gb - RB.y) < 5)
			//							{
			//								//countGT--;
			//								tNframe--;
			//								flagmiss = 1;
			//								break;
			//							}
			//						}
			//					}

			//					if (flagmiss == 0)
			//					{
			//						for (int k = 0; k < camera.size(); k++)
			//						{
			//							if (camera[k]->topshow.at<uchar>(ga, gb) == 1)
			//							{
			//								// rectangle(camera[k]->frame, Rect(camera[k]->rctMap.at<Vec4f>(ga, gb)[0], camera[k]->rctMap.at<Vec4f>(ga, gb)[1], camera[k]->rctMap.at<Vec4f>(ga, gb)[2], camera[k]->rctMap.at<Vec4f>(ga, gb)[3]), CV_RGB(0, 255, 0), 2);
			//							}
			//						}
			//					}
			//					else
			//						flagmiss = 0;

			//				}

			//			}


			//	}
			//	if (tNframe > 0)
			//	{
			//		Nframe++;
			//	}

			//	if (totIOU > 0)
			//	{
			//		MODP += totIOU / frHit;
			//		//cout << "MODP: " << MODP << endl;
			//		//cout << "totIOU; " << totIOU << endl;
			//	}
			//	totIOU = 0;
			//	frHit = 0;
			//}

			


			//cout << endl << endl << endl;
			////cout << countGT << " " << countdet << " " << countHit << endl;
			//cout << "FN " << countGT - countHit << endl;
			////cout << "FN rate" << (countGT - countHit) / double(countGT) << endl;
			//cout << "FP " << countdet - countHit << endl;
			////cout << "FP rate" << (countdet - countHit) / double(countdet) << endl;
			//cout << "MDR " << (countGT - countHit) / double(countGT) << endl;
			//cout << "FDR " << (countdet - countHit) / double(countGT) << endl;
			//cout << "TER " << ((countGT - countHit) + (countdet - countHit)) / double(countGT) << endl;
			//cout << "PRE " << countHit / double(countdet) << endl;
			//cout << "REC " << countHit / double(countGT) << endl;








		}

		if (fstep == 1 && nFrmNum % 25 == 0) {
			ch = getch();
			if (ch == 13) fstep = 0;
			if (ch == 27) break;
		}
		else {
			if (kbhit()) {
				ch = getch();
				if (ch == 32) fstep = 1;
				else break;
			}
		}


	}








	NMODP = MODP / Nframe;
	TER = ((countGT - countHit) + (countdet - countHit)) / double(countGT);
	PRECISION = countHit / double(countdet);
	RECALL = countHit / double(countGT);
	NMODA = 1 - TER;
	FSCORE = 2 * PRECISION*RECALL / (PRECISION + RECALL);

	cout << endl << endl << endl;
	cout << "count GT: " << countGT << " " << "count detection: " << countdet << " " << "count matched: " << countHit << endl;
	cout << "FN " << countGT - countHit << endl;
	cout << "FN rate" << (countGT - countHit) / double(countGT) << endl;
	cout << "FP " << countdet - countHit << endl;
	cout << "FP rate" << (countdet - countHit) / double(countdet) << endl;
	cout << "MDR " << (countGT - countHit) / double(countGT) << endl;
	cout << "FDR " << (countdet - countHit) / double(countGT) << endl;
	cout << "TER " << ((countGT - countHit) + (countdet - countHit)) / double(countGT) << endl;
	cout << "PRE " << countHit / double(countdet) << endl;
	cout << "REC " << countHit / double(countGT) << endl;
	cout << "F-Score: " << FSCORE << endl;
	cout << "N-MODA: " << NMODA << endl;
	cout << "N-MODP: " << NMODP << endl;







	//fclose(fpbox);
	return 0;
}

double MahaDistance1D(double x, double y, double var)
{
	double dis = pow(x - y, 2) / pow(var, 2);

	if (dis < table[1][0])
		return 1;
	else if (dis == table[1][0])
		return table[0][0];
	else if (dis > table[1][15])
		return table[0][15];
	else
		for (int i = 0; i < 15; i++)
			if ((dis>table[1][i]) && (dis <= table[1][i + 1]))
				return table[0][i] - (dis - table[1][i]) / (table[1][i + 1] - table[1][i]) * (table[0][i] - table[0][i + 1]);

}

void QMmethod(vector<Camera*> camera, Mat* st, int num, vector<int>& result)
{
	int n = 0;
	CvFont font;
	char text[10];
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.5, 0.5, 0, 2);

	bool state[400];
	double prob[400];
	int camNum[400];

	shorttable = 0;

	//显示所有candidate的概率
	cout << "\t\b\b\b\b" << "  ";

	for (int k = 0; k < camera.size(); k++)
	{
		cout << k + 1 << "F    " << k + 1 << "T    " << k + 1 << "B    " << k + 1 << "M    ";
	}
	cout << " JL" << endl;

	for (int i = 0; i < num; i++)
	{
		cout << "I" << i << "\t\b\b\b\b";
		prob[i] = 1;
		camNum[i] = camera.size();
		for (int k = 0; k < camera.size(); k++)
		{
			//if (camera[k]->probI[i] == 0)
			//{
			//	camera[k]->probI[i] = 1;
			//}

			cout << fixed << setw(5) << setprecision(3) << camera[k]->prob[i] << " " << fixed << setw(5) << setprecision(3) << 
				camera[k]->probT[i] << " " << fixed << setw(5) << setprecision(3) << camera[k]->probB[i] << " " 
				<< fixed << setw(5) << setprecision(3) << camera[k]->probM[i] << " ";
			//累乘所有概率
			prob[i] *= camera[k]->prob[i] * camera[k]->probT[i] * camera[k]->probB[i] * camera[k]->probM[i];
			if (camera[k]->prob[i] == 1)
				camNum[i]--;
		}
		prob[i] = pow(prob[i], 1.0 / camNum[i]);
		cout << " " << prob[i] << endl;
	}

	//删除小概率的candidate，重新整理结果
	int inx = 0;
	while (inx < num)
	{
		if (prob[inx] < 0.1)
		{
			for (int j = inx; j <= num - 1; j++)
			{
				for (int m = 0; m < camera.size(); m++)
				{
					camera[m]->r[j] = camera[m]->r[j + 1];
					camera[m]->prob[j] = camera[m]->prob[j + 1];
					camera[m]->probB[j] = camera[m]->probB[j + 1];
					camera[m]->probT[j] = camera[m]->probT[j + 1];
					camera[m]->probM[j] = camera[m]->probM[j + 1];
					//camera[m]->probI[j] = camera[m]->probI[j + 1];
					camera[m]->p[j] = camera[m]->p[j + 1];
					prob[j] = prob[j + 1];
					st->at<uchar>(0, j) = st->at<uchar>(0, j + 1);
				}
			}
			num--;
			inx--;
		}
		inx++;
	}

	//显示删除后的candidate的概率
	cout << endl;
	cout << "Modified Table" << endl;
	cout << "\t\b\b\b\b" << "  ";

	for (int k = 0; k < camera.size(); k++)
	{
		cout << k + 1 << "F    " << k + 1 << "T    " << k + 1 << "B    " << k + 1 << "M    ";
	}
	cout << " JL" << endl;

	for (int i = 0; i < num; i++)
	{
		cout << "I" << i << "\t\b\b\b\b";
		prob[i] = 1;
		camNum[i] = camera.size();
		for (int k = 0; k < camera.size(); k++)
		{
			cout << fixed << setw(5) << setprecision(3) << camera[k]->prob[i] << " " << fixed << setw(5) << setprecision(3) 
				<< camera[k]->probT[i] << " " << fixed << setw(5) << setprecision(3) << camera[k]->probB[i] << " " 
				<< fixed << setw(5) << setprecision(3) << camera[k]->probM[i] << " ";
			prob[i] *= camera[k]->prob[i] * camera[k]->probT[i] * camera[k]->probB[i] * camera[k]->probM[i];//multiply
			if (camera[k]->prob[i] == 1)
				camNum[i]--;

		}
		prob[i] = pow(prob[i], 1.0 / camNum[i]);
		cout << " " << prob[i] << endl;
	}

	//初始化decomposition图
	for (int i = 0; i < camera.size(); i++)
	{
		camera[i]->pF.setTo(0);
		camera[i]->T.setTo(0);
	}

	//将同一视角的矩形框根据其编号生成decomposition图
	for (int i = 0; i < num; i++)
	{
		state[i] = 0;
		for (int k = 0; k < camera.size(); k++)
			//0x01
			if (st->at<uchar>(0, i) >> k & 0x01)
				QMmap(camera[k]->r[i], &camera[k]->pF, i);
	}

	//根据decomposition图和前景统计每一个subregion的面积和其中的前景像素数
	//Tnum是每一个摄像机包含的子区域个数
	for (int k = 0; k < camera.size(); k++)
		camera[k]->Tnum = RemoveDuplates(&camera[k]->pF, &camera[k]->T, &camera[k]->foreground);


	bool** t = new bool*[num];

	//整理结果
	//camera[k]->T.at<int>(i, 0) 第i个子区域被矩形覆盖关系的二进制编码
	//camera[k]->T.at<int>(i, 1) 第i个子区域中的像素总个数
	//camera[k]->T.at<int>(i, 2) 第i个子区域中包含的前景像素个数
	for (int k = 0; k < camera.size(); k++)
	{
		for (int j = 0; j < camera[k]->Tnum; j++)
		{
			if (camera[k]->T.at<int>(j, 0) == 0)
			{
				for (int i = j; i < camera[k]->Tnum - 1; i++)
				{
					camera[k]->T.at<int>(i, 0) = camera[k]->T.at<int>(i + 1, 0);
					camera[k]->T.at<int>(i, 1) = camera[k]->T.at<int>(i + 1, 1);
					camera[k]->T.at<int>(i, 2) = camera[k]->T.at<int>(i + 1, 2);
				}
				break;
			}
		}
	}

	int totalNum = 0;
	for (int k = 0; k < camera.size(); k++)
	{
		camera[k]->Tnum--;
		totalNum += camera[k]->Tnum;
	}

	//初始化 QM table
	for (int i = 0; i < num; i++)
		t[i] = new bool[totalNum];

	//生成 QM table
	int idx;
	for (int i = 0; i < num; i++)
	{
		idx = 0;
		for (int k = 0; k < camera.size(); k++)
			for (int j = 0; j < camera[k]->Tnum; j++, idx++)
				t[i][idx] = camera[k]->T.at<int>(j, 0) >> i & 0x01;
	}
	showtable(totalNum, num, state, t, camera);

	//删除区域过小或包含像素少于150的box
	cout << "***********delete small sunregion*********" << endl << endl;
	idx = 0;
	for (int k = 0; k < camera.size(); k++)
	{
		for (int i = 0; i < camera[k]->Tnum; i++, idx++)
			if ((double)camera[k]->T.at<int>(i, 2) / camera[k]->T.at<int>(i, 1) < 0.20 || (double)camera[k]->T.at<int>(i, 2) < 150)
				for (int j = 0; j < num; j++)
					t[j][idx] = 0;
	}

	//统计哪些column需要被显示
	for (int j = 0; j < totalNum; j++)
	{
		countcolumn[j] = 0;
		for (int i = 0; i < num; i++)
		{
			if (t[i][j] == 1)
			{
				countcolumn[j] = 1;
				continue;
			}
		}
	}

	shorttable = 1;

	showtable(totalNum, num, state, t, camera);

	if (METHOD == 0)
	{
		//QM-Method
		int kmarker = 0;
		int imarker = 0;

		int sum;
		int* jmarker = new int[num];

		int tempnum;
		int* iIndex = new int[num];
		for (int i = 0; i < num; i++)
			iIndex[i] = 0;

		bool* ch = new bool[num];
		bool* nch = new bool[num];

		bool loop = 1;
		while (loop)
		{
			loop = 0;
			for (int i = 0; i < num; i++)
			{
				tempnum = 0;
				for (int j = 0; j < totalNum; j++)
				{
					tempnum += t[i][j];
				}
				if (iIndex[i] != tempnum)
				{
					iIndex[i] = tempnum;
					loop = 1;
				}
			}
			if (loop == 0)
				break;

			////化简QM第一步
			cout << "*************化简第一步**************" << endl;
			cout << "找出有唯一对应前景区域的box，标记为必选" << endl;

			for (int j = 0; j < num; j++)
				jmarker[j] = 0;

			for (int i = 0; i < totalNum; i++)
			{
				sum = 0;
				for (int j = 0; j < num; j++)
					sum += t[j][i];
				if (sum == 1)
				{
					for (int j = 0; j < num; j++)
						if (t[j][i] == 1)
							jmarker[j] = 1;
				}
			}

			for (int j = 0; j < num; j++)
			{
				if (jmarker[j] == 1)
				{
					for (int k = 0; k < totalNum; k++)
					{
						if (t[j][k] == 1)
							for (int n = 0; n < num; n++)
								t[n][k] = 0;
					}
					state[j] = 1;
				}
			}

			showtable(totalNum, num, state, t, camera);

			loop = 0;
			for (int i = 0; i < num; i++)
			{
				tempnum = 0;
				for (int j = 0; j < totalNum; j++)
				{
					tempnum += t[i][j];
				}
				if (iIndex[i] != tempnum)
				{
					iIndex[i] = tempnum;
					loop = 1;
				}
			}
			tempnum = 0;
			for (int i = 0; i < num; i++)
			{
				tempnum += iIndex[i];
			}
			if (loop == 0 || tempnum == 0)
				break;

			cout << "*************化简第二步**************" << endl;
			cout << "去掉被其他box所完全包含的box" << endl;
			for (int i = 0; i < num; i++)
				for (int j = 0; j < num; j++)
				{
				if (i != j)
				{
					if (removeRow(t[i], t[j], totalNum) == 1) //部分包含，保留区域多的
					{
						for (int k = 0; k < totalNum; k++)
							t[i][k] = 0;
					}
					else if (removeRow(t[i], t[j], totalNum) == 2)  //全部包含
					{
						if (prob[i] < prob[j]) //删掉概率小的
						{
							for (int k = 0; k < totalNum; k++)
								t[i][k] = 0;
						}
					}
				}
				}
			showtable(totalNum, num, state, t, camera);

			cout << "*************化简第三步**************" << endl;
			cout << "合并后的box，如果对应的数个前景区域都是唯一的，设为必选" << endl;

			for (int j = 0; j < num; j++)
				jmarker[j] = 0;

			for (int i = 0; i < totalNum; i++)
			{
				sum = 0;
				for (int j = 0; j < num; j++)
					sum += t[j][i];
				if (sum == 1)
				{
					for (int j = 0; j < num; j++)
						if (t[j][i] == 1)
							jmarker[j] = 1;
				}
			}

			for (int j = 0; j < num; j++)
			{
				if (jmarker[j] == 1)
				{
					for (int k = 0; k < totalNum; k++)
					{
						if (t[j][k] == 1)
							for (int n = 0; n < num; n++)
								t[n][k] = 0;
					}
					state[j] = 1;
				}
			}

			showtable(totalNum, num, state, t, camera);

			cout << "*************化简第四步**************" << endl;
			cout << "再次去掉被其他box所完全包含的box" << endl;
			for (int i = 0; i < num; i++)
				for (int j = 0; j < num; j++)
				{
				if (i != j)
				{
					if (removeRow(t[i], t[j], totalNum) == 1) //部分包含，保留区域多的
					{
						for (int k = 0; k < totalNum; k++)
							t[i][k] = 0;
					}
					else if (removeRow(t[i], t[j], totalNum) == 2)  //全部包含
					{
						if (prob[i] < prob[j]) //删掉概率小的
						{
							for (int k = 0; k < totalNum; k++)
								t[i][k] = 0;
						}
					}
				}
				}
			showtable(totalNum, num, state, t, camera);

			loop = 0;
			for (int i = 0; i < num; i++)
			{
				tempnum = 0;
				for (int j = 0; j < totalNum; j++)
				{
					tempnum += t[i][j];
				}
				if (iIndex[i] != tempnum)
				{
					iIndex[i] = tempnum;
					loop = 1;
				}
			}
			tempnum = 0;
			for (int i = 0; i < num; i++)
			{
				tempnum += iIndex[i];
			}
			if (loop == 0 || tempnum == 0)
				break;
		}

		//loopCount[iloopCount]++;
		cout << "如果有： " << endl;

		double chP = 1;
		double nchP = 1;

		//判断是否有剩余的
		int lgLoc, lgNum;
		lgNum = 0;
		//cout << endl;
		for (int i = 0; i < num; i++)
		{
			ch[i] = 0;
			nch[i] = 0;
			//cout << iIndex[i] << " ";
			if (iIndex[i] > lgNum)
			{
				lgLoc = i;
				lgNum = iIndex[i];
			}
		}
		showtable(totalNum, num, state, t, camera);
		//如果有
		if (lgNum > 0)
		{
			loop = 1;
			while (loop)
			{
				//loopCount[iloopCount]++;

				//选择剩余中，占前景最多的
				ch[lgLoc] = 1;
				for (int k = 0; k < totalNum; k++)
				{
					if (t[lgLoc][k] == 1)
						for (int n = 0; n < num; n++)
							t[n][k] = 0;
				}

				showtable(totalNum, num, state, t, camera);

				//同第二部，进一步化简
				for (int i = 0; i < num; i++)
					for (int j = 0; j < num; j++)
					{
					if (i != j)
					{
						if (removeRow(t[i], t[j], totalNum) == 1) //部分包含，保留区域多的
						{
							nch[i] = 1;
							for (int k = 0; k < totalNum; k++)
								t[i][k] = 0;
						}
						else if (removeRow(t[i], t[j], totalNum) == 2)  //全部包含
						{
							if (prob[i] < prob[j]) //删掉概率小的
							{
								nch[i] = 1;
								for (int k = 0; k < totalNum; k++)
									t[i][k] = 0;
							}
						}
					}
					}

				showtable(totalNum, num, state, t, camera);

				//同第三步
				for (int i = 0; i < num; i++)
				{
					kmarker = 0;
					imarker = 0;
					for (int j = 0; j < totalNum; j++)
					{
						if (t[i][j] == 1)
						{
							imarker = 1;
							for (int k = 0; k < num; k++)
								if ((t[k][j] == 1) && (k != i))
								{
								kmarker = 1;
								break;
								}
							if (kmarker == 1)
								break;
						}
					}
					if (kmarker == 0 && imarker == 1)
					{
						for (int j = 0; j < totalNum; j++)
							t[i][j] = 0;
						ch[i] = 1;
					}
				}

				for (int i = 0; i < num; i++)
					cout << ch[i] << " ";
				cout << endl;
				for (int i = 0; i < num; i++)
					cout << nch[i] << " ";
				cout << endl;

				cout << "选择联合likelihood最大的组合" << endl;
				chP = 1;
				nchP = 1;

				for (int i = 0; i < num; i++)
				{
					if (ch[i] == 1)
					{
						cout << "No." << i << ": " << prob[i] << endl;
						chP = chP * prob[i];
					}
				}
				cout << "total: " << chP << endl;

				cout << "VS" << endl;

				for (int i = 0; i < num; i++)
				{
					if (nch[i] == 1)
					{
						cout << "No." << i << ": " << prob[i] << endl;
						nchP = nchP * prob[i];
					}
				}
				cout << "total: " << nchP << endl;


				if (nchP == 1 || nchP < chP)
				{
					for (int i = 0; i < num; i++)
						if (ch[i] == 1)
							state[i] = 1;
				}
				else
				{
					for (int i = 0; i < num; i++)
						if (nch[i] == 1)
							state[i] = 1;
				}

				showtable(totalNum, num, state, t, camera);


				//同第一步，选唯一对应前景区域的box，设为必选

				for (int j = 0; j < num; j++)
					jmarker[j] = 0;

				for (int i = 0; i < totalNum; i++)
				{
					sum = 0;
					for (int j = 0; j < num; j++)
						sum += t[j][i];
					if (sum == 1)
					{
						for (int j = 0; j < num; j++)
							if (t[j][i] == 1)
								jmarker[j] = 1;
					}
				}

				for (int j = 0; j < num; j++)
				{
					if (jmarker[j] == 1)
					{
						for (int k = 0; k < totalNum; k++)
						{
							if (t[j][k] == 1)
								for (int n = 0; n < num; n++)
									t[n][k] = 0;
						}
						state[j] = 1;
					}
				}

				//统计剩余
				for (int i = 0; i < num; i++)
				{
					tempnum = 0;
					for (int j = 0; j < totalNum; j++)
					{
						tempnum += t[i][j];
					}
					if (iIndex[i] != tempnum)
					{
						iIndex[i] = tempnum;
					}
				}

				//判断是否有剩余的
				lgNum = 0;
				//cout << endl;
				for (int i = 0; i < num; i++)
				{
					ch[i] = 0;
					nch[i] = 0;
					//cout << iIndex[i] << " ";
					if (iIndex[i] > lgNum)
					{
						lgLoc = i;
						lgNum = iIndex[i];
					}
				}
				//cout << endl;

				loop = 0;
				if (lgNum > 0)
				{
					loop = 1;
				}

				//cout << lgLoc << " " << lgNum << endl;
				//getch();
			}
		}

		//iloopCount++;

		delete[] iIndex;
		delete[] jmarker;
		//delete[] idx;

	}
	//if METHOD=1, 选择petrick's method
	else
	{
		//start petrick's method
		vector<string> v;
		string tempstring;

		//将QM table转化为 petrick's method 所用的字符串形式
		for (int i = 0; i < totalNum; i++)
		{
			for (int j = 0; j < num; j++)
			{
				if (t[j][i] == 1)
					tempstring = tempstring + (char)(j);
			}
			if (tempstring.size() != 0)
				v.push_back(tempstring);
			tempstring.clear();
		}

		if (v.size() > 0)
		{
			petrick a(v, prob);
			v = a.run();

			double maxprob = 0;;
			double tempprob = 1;
			int idx;

			if (v.size() == 1)
			{
				for (int i = 0; i < v[0].size(); i++)
				{
					state[(int)v[0][i]] = 1;
				}
			}
			else
			{
				for (int k = 0; k < v.size(); k++)
				{
					tempprob = 1;
					for (int i = 0; i < v[k].size(); i++)
					{
						tempprob *= prob[(int)v[k][i]];
					}
					if (tempprob > maxprob)
					{
						maxprob = tempprob;
						idx = k;
					}
				}

				for (int i = 0; i < v[idx].size(); i++)
				{
					state[(int)v[idx][i]] = 1;
				}
			}

		}
	}

	for (int i = 0; i < num; i++)
		delete[] t[i];
	delete[] t;

	//显示结果
	for (int i = 0; i < num; i++)
	{
		itoa(i, text, 10);

		if (state[i] == 1)
			cout << i << " ";
		if (i < 16)
		{
			//在各摄像头视图中给各个candidate加颜色标号于图像下方
			//for (int k = 0; k < camera.size(); k++)

			//		cvPutText(&IplImage(camera[k]->frame2), text, cvPoint(20 * i, 250), &font, colorful[i]);

				

			if (state[i] != 1)
			{
				for (int k = 0; k < camera.size(); k++)
				{
					CvBlob rectb;




					if (st->at<uchar>(0, i) >> k & 0x01)
					{
						//rectangle(camera1.frame,mea[i].R1, colorful[i]);
						rectb.x = camera[k]->r[i].x + camera[k]->r[i].width / 2;
						rectb.y = camera[k]->r[i].y + camera[k]->r[i].height / 2;
						rectb.w = camera[k]->r[i].width;
						rectb.h = camera[k]->r[i].height;



						//画虚框
						drawDashRect(&IplImage(camera[k]->frame2), 1, 4, &rectb, colorful[i], 1);
						//cvPutText(&IplImage(camera[k]->frame2), text, cvPoint(camera[k]->r[i].x, camera[k]->r[i].y), &font, colorful[i]);
					}
				}
			}

			

			else
			{
				CvBlob rectb;
				result.push_back(i);
				for (int k = 0; k < camera.size(); k++)
				if (st->at<uchar>(0, i) >> k & 0x01)
					//{
						//画框
				{
					rectangle(camera[k]->frame2, camera[k]->r[i], colorful[i], 2);
					//cvPutText(&IplImage(camera[k]->frame2), text, cvPoint(camera[k]->r[i].x, camera[k]->r[i].y), &font, colorful[i]);
				}



						/*rectb.x = camera[k]->r[i].x + camera[k]->r[i].width / 2;
						rectb.y = camera[k]->r[i].y + camera[k]->r[i].height / 2;
						rectb.w = camera[k]->r[i].width;
						rectb.h = camera[k]->r[i].height;
						cout << "r[i]:" << rectb.x << rectb.y << rectb.w << rectb.h << endl;*/
					//}

			}
		}
		else
		{
			//for (int k = 0; k < camera.size(); k++)
				//cvPutText(&IplImage(camera[k]->frame), text, cvPoint(20 * (i - 20), 270), &font, colorful[i]);

			if (state[i] != 1)
			{
				for (int k = 0; k < camera.size(); k++)
				{
					CvBlob rectb;
					if (st->at<uchar>(0, i) >> k & 0x01)
					{
						//rectangle(camera1.frame,mea[i].R1, colorful[i]);
						rectb.x = camera[k]->r[i].x + camera[k]->r[i].width / 2;
						rectb.y = camera[k]->r[i].y + camera[k]->r[i].height / 2;
						rectb.w = camera[k]->r[i].width;
						rectb.h = camera[k]->r[i].height;

						//drawDashRect(&IplImage(camera[k]->frame2), 1, 4, &rectb, colorful[i], 1);
					}
				}
			}
			else
			{
				result.push_back(i);
				for (int k = 0; k < camera.size(); k++)
				if (st->at<uchar>(0, i) >> k & 0x01)
				//画框
				{
					rectangle(camera[k]->frame2, camera[k]->r[i], colorful[i], 2);
					//cvPutText(&IplImage(camera[k]->frame2), text, cvPoint(camera[k]->r[i].x, camera[k]->r[i].y), &font, colorful[i]);
				}
			}
		}
	}
	cout << endl;
}

//将矩形框转化为二进制图，如果要适用于更大的场景 可以考虑不要用pow(2, i)进行赋值
void QMmap(CvRect rect, Mat* pFrame, int i)
{
	int js = rect.y;
	int ks = rect.x;
	int je = rect.y + rect.height;
	int ke = rect.x + rect.width;
	int add = pow(2, i);

	for (int k = ks; k < ke; k++)
	{
		for (int j = js; j < je; j++)
		{
			//把框内的像素都赋值为编号
			pFrame->at<int>(j, k) = pFrame->at<int>(j, k) + add;
		}
	}
}


int RemoveDuplates(Mat* A, Mat*B, Mat* fmask)
{
	int k = 0;
	int lastNumber = 0;
	int mark = 0;
	int count = 0;
	int kk;

	for (int i = 0; i < A->rows; i++)
	{
		for (int j = 0; j < A->cols; j++)
		{
			if (lastNumber != A->at<int>(i, j))
			{
				mark = 0;
				for (int n = 0; n <= k; n++)
				{
					if (B->at<int>(n, 0) == lastNumber)
					{
						B->at<int>(n, 1) = B->at<int>(n, 1) + count;
						count = 0;
						mark = 1;
						break;
					}
				}
				if (mark == 0)
				{
					k++;
					B->at<int>(k, 0) = lastNumber;
					B->at<int>(k, 1) = B->at<int>(k, 1) + count;
					count = 0;
				}
			}
			lastNumber = A->at<int>(i, j);
			count++;
		}
	}

	for (int n = 0; n <= k; n++)
		if (B->at<int>(n, 0) == lastNumber)
			B->at<int>(n, 1) = B->at<int>(n, 1) + count;

	kk = k;
	lastNumber = 0;
	mark = 0;
	count = 0;
	for (int i = 0; i < A->rows; i++)
	{
		for (int j = 0; j < A->cols; j++)
		{
			if (fmask->at<uchar>(i, j) > 0)
			{
				if (lastNumber != A->at<int>(i, j))
				{
					for (int n = 0; n <= k; n++)
					{
						if (B->at<int>(n, 0) == lastNumber)
						{
							B->at<int>(n, 2) = B->at<int>(n, 2) + count;
							count = 0;
							break;
						}
					}
				}
				lastNumber = A->at<int>(i, j);
				count++;
			}
		}
	}

	for (int n = 0; n <= k; n++)
		if (B->at<int>(n, 0) == lastNumber)
			B->at<int>(n, 2) = B->at<int>(n, 2) + count;
	return kk + 1;
}

int removeRow(bool a[], bool b[], int num)  //返回2，a可以被b全完包含，返回1，a可以被b包含。返回0，a无法被b包含
{
	int zeroNum = 0;

	int oneNum = 0;
	int aOneNum = 0;

	for (int i = 0; i < num; i++)
	{
		switch (a[i])
		{
		case 1:
			aOneNum++;
			switch (b[i])
			{
			case 0:
				return 0;
				break;
			case 1:
				oneNum++;
				break;
			default:
				break;
			}
			break;
		case 0:
			zeroNum++;
			switch (b[i])
			{
			case 1:
				oneNum++;
				break;
			}
			break;
		}
	}
	if (zeroNum == num)
		return 0;
	else if (oneNum == aOneNum)
		return 2;
	else
		return 1;

}

void showtable(int totalNum, int num, bool state[], bool** t, vector<Camera*> camera)
{
	if (shorttable == 0)
	{
		//显示table
		cout << endl;
		cout << "\t\b\b\b";
		for (int k = 0; k < camera.size(); k++)
		{
			for (int j = 0; j < camera[k]->Tnum; j++)
				cout << k + 1;
		}
		cout << " ";
		cout << endl;


		for (int i = 0; i < num; i++)
		{
			cout << "I" << i << ":";
			if (state[i] == 1)
				cout << "o";
			//if (i < 10)
			//	cout << "\t\t";
			//else
			cout << "\t\b\b\b";

			for (int j = 0; j < totalNum; j++)
				if (t[i][j] == 0)
					cout << "+";
				else
					cout << "X";
			cout << endl;
		}
		cout << endl;
		//显示table结
	}
	else
	{
		//显示table
		cout << endl;
		cout << "\t\b\b\b";
		int n = 0;
		for (int k = 0; k < camera.size(); k++)
		{
			for (int j = 0; j < camera[k]->Tnum; j++)
			{
				if (countcolumn[n] == 1)
					cout << k + 1;
				n++;
			}
		}
		cout << " ";
		cout << endl;


		for (int i = 0; i < num; i++)
		{
			cout << "I" << i << ":";
			if (state[i] == 1)
				cout << "o";
			//if (i < 10)
			//	cout << "\t\t";
			//else
			cout << "\t\b\b\b";

			for (int j = 0; j < totalNum; j++)
			{
				if (countcolumn[j] == 1)
				{
					if (t[i][j] == 0)
						cout << "+";
					else
						cout << "X";
				}
			}
			cout << endl;
		}
		cout << endl;
		//显示table结
	}
}

void drawDashRect(CvArr* img, int linelength, int dashlength, CvBlob* blob, CvScalar color, int thickness)
{
	int w = cvRound(blob->w);//width
	int h = cvRound(blob->h);//height

	int tl_x = cvRound(blob->x - blob->w / 2);//top left x
	int tl_y = cvRound(blob->y - blob->h / 2);//top  left y

	int totallength = dashlength + linelength;
	int nCountX = w / totallength;//
	int nCountY = h / totallength;//

	CvPoint start, end;//start and end point of each dash

	//draw the horizontal lines
	start.y = tl_y;
	start.x = tl_x;

	end.x = tl_x;
	end.y = tl_y;

	for (int i = 0; i < nCountX; i++)
	{
		end.x = tl_x + (i + 1)*totallength - dashlength;//draw top dash line
		end.y = tl_y;
		start.x = tl_x + i*totallength;
		start.y = tl_y;
		cvLine(img, start, end, color, thickness);
	}
	for (int i = 0; i < nCountX; i++)
	{
		start.x = tl_x + i*totallength;
		start.y = tl_y + h;
		end.x = tl_x + (i + 1)*totallength - dashlength;//draw bottom dash line
		end.y = tl_y + h;
		cvLine(img, start, end, color, thickness);
	}

	for (int i = 0; i < nCountY; i++)
	{
		start.x = tl_x;
		start.y = tl_y + i*totallength;
		end.y = tl_y + (i + 1)*totallength - dashlength;//draw left dash line
		end.x = tl_x;
		cvLine(img, start, end, color, thickness);
	}

	for (int i = 0; i < nCountY; i++)
	{
		start.x = tl_x + w;
		start.y = tl_y + i*totallength;
		end.y = tl_y + (i + 1)*totallength - dashlength;//draw right dash line
		end.x = tl_x + w;
		cvLine(img, start, end, color, thickness);
	}
	start.x = tl_x + w;
	start.y = tl_y + h;
	end.x = tl_x + w;
	end.y = tl_y + h - 2;
	cvLine(img, start, end, color, thickness);

	end.x = tl_x + w - 2;
	end.y = tl_y + h;
	cvLine(img, start, end, color, thickness);
}

double Gaussian1D(double dis)
{
	if (dis >= 0)
	{
		if (dis <= tableG[1][0])
			return 1 - tableG[0][0];
		else if (dis > tableG[1][15])
		//else if (dis > tableG[1][1])
			return 1 - tableG[0][15];
		else
			for (int i = 0; i < 15; i++)
				if ((dis>tableG[1][i]) && (dis <= tableG[1][i + 1]))
					return 1 - (tableG[0][i] - (dis - tableG[1][i]) / (tableG[1][i + 1] - tableG[1][i]) * (tableG[0][i] - tableG[0][i + 1]));
	}
	else
	{
		return -1;
	}
}


double Gaussian1D2(double dis)
{
	if (dis >= 0)
	{
		if (dis <= tableG3[1][0])
			return 1 - tableG3[0][0];
		else if (dis > tableG3[1][15])
			//else if (dis > tableG[1][1])
			return 1 - tableG3[0][15];
		else
			for (int i = 0; i < 15; i++)
				if ((dis > tableG3[1][i]) && (dis <= tableG3[1][i + 1]))
					return 1 - (tableG3[0][i] - (dis - tableG3[1][i]) / (tableG3[1][i + 1] - tableG3[1][i]) * (tableG3[0][i] - tableG3[0][i + 1]));
	}
	else
	{
		return -1;
	}
}


float bbOverlap(Rect box1, Rect box2)
{
	if (box1.x > box2.x + box2.width) { return 0.0; }
	if (box1.y > box2.y + box2.height) { return 0.0; }
	if (box1.x + box1.width < box2.x) { return 0.0; }
	if (box1.y + box1.height < box2.y) { return 0.0; }
	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}

bool IsRectCross(const Point &p1, const Point &p2, const Point &q1, const Point &q2)
{
	bool ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
		min(q1.x, q2.x) <= max(p1.x, p2.x) &&
		min(p1.y, p2.y) <= max(q1.y, q2.y) &&
		min(q1.y, q2.y) <= max(p1.y, p2.y);
	return ret;
}

Point GetFoot(
	const Point &pt,     // 直线外一点
	const Point &begin,  // 直线开始点
	const Point &end)   // 直线结束点
{
	Point retVal;

	double dx = begin.x - end.x;
	double dy = begin.y - end.y;
	if (abs(dx) < 0.00000001 && abs(dy) < 0.00000001)
	{
		retVal = begin;
		return retVal;
	}

	double u = (pt.x - begin.x)*(begin.x - end.x) +
		(pt.y - begin.y)*(begin.y - end.y);
	u = u / ((dx*dx) + (dy*dy));

	retVal.x = begin.x + u * dx;
	retVal.y = begin.y + u * dy;

	return retVal;
}

int dfs(int i)
{
	for (int j = 0; j < nd; j++)//对左边的节点i 与右边的节点j 进行逐一检查
	{
		if (!visit[j] && match[i][j])
		{
			visit[j] = 1;//标记检查过的点
			if (mark[j] == -1 || dfs(mark[j]))//如果右边的点j 没有被匹配或者匹配了但是存在交错路的情况
			{
				mark[j] = i;//修改匹配关系
				return 1;
			}
		}
	}
	return 0;
}

int Hungary()
{
	int ans = 0;
	memset(mark, -1, sizeof(mark));
	for (int i = 0; i < ng; i++)
	{
		memset(visit, 0, sizeof(visit));
		if (dfs(i))
		{
			ans++;
		}
	}
	return ans;
}

//初始化图
void InitialG(MGraph *G)
{
	G->numVertexes = MAXVEX;
	for (int i = 0; i < G->numVertexes; ++i) {
		G->vex[i] = i;
	}
	for (int i = 0; i < G->numVertexes; ++i) {
		for (int j = 0; j < G->numVertexes; ++j) {
			if (i == j)
				G->arc[i][j] = 0;
			else
				G->arc[i][j] = G->arc[j][i] = INFINITY; //每个位置的值预设为无穷大
		}
	}
}

 // 构建图
void CreateMGraph(MGraph *G) {
	int i, j, k;

	// 设置对称位置元素值
	for (i = 0; i < G->numVertexes; ++i) 
	{
		for (j = i; j < G->numVertexes; ++j) 
		{
			G->arc[j][i] = G->arc[i][j]; //变成无向图 保证对称位置的值是一致的
		}
	}
}

// Floyd algorithm
void ShortPath_Floyd(MGraph G, Patharc P, ShortPathTable D) {
	int i, j, k;
	// 二重循环，初始化P, D
	for (i = 0; i < G.numVertexes; ++i) {
		for (j = 0; j < G.numVertexes; ++j) {
			D[i][j] = G.arc[i][j];
			P[i][j] = j;
		}
	}
	// 三重循环, Floyd algorithm
	for (k = 0; k < G.numVertexes; ++k) 
	{
		for (i = 0; i < G.numVertexes; ++i) 
		{
			for (j = 0; j < G.numVertexes; ++j) 
			{
				if (D[i][j] > D[i][k] + D[k][j] && D[i][k] + D[k][j] < 10) //寻找新路径时，路径总长度被限制
				{
					D[i][j] = D[i][k] + D[k][j];
					P[i][j] = P[i][k];
				}
			}
		}
	}
}

// 打印最短路径
void PrintShortPath(MGraph G, Patharc P, ShortPathTable D, vector < vector<int> >  &lines, vector <Point>  &points)
{
	cout << "lines size: " << lines.size() << endl;
	vector < vector<int> >  linec;
	int C0, C1, C2, C3;
	int i, j, k, num;
	//cout << "各顶点之间的最短路径如下: " << endl;
	for (i = 0; i < G.numVertexes; ++i) 
	{
		num = 0;
		C0 = 0;
		C1 = 0;
		C2 = 0;
		C3 = 0;
		linec.clear();

		if (i < lines.size() && lines[i][10] != -1)
		{
			linec.push_back(lines[i]);
			lines[i][10] = -1;

			for (j = i + 1; j < G.numVertexes; ++j)
			{
				if (j < lines.size())
				{
					if (D[i][j] != INFINITY  && lines[j][10] != -1 && lines[i][9] != lines[j][9] && lines[i][12] != -1)
					{

						//cout << "v" << i << "--" << "v" << j << " " << "weight: " << D[i][j] << "  Path: " << i << " -> ";
						//k = P[i][j];
						//while (k != j)
						//{
						//	cout << k << " -> ";
						//	k = P[k][j];
						//}
						//cout << j << endl;
						if (lines[i][9] == 0)
						{
							C0 += 1;
						}
						if (lines[i][9] == 1)
						{
							C1 += 1;
						}
						if (lines[i][9] == 2)
						{
							C2 += 1;
						}
						if (lines[i][9] == 3)
						{
							C3 += 1;
						}
						if (lines[j][9] == 0 && C0 == 0)
						{
							linec.push_back(lines[j]);
							if (lines[j][12] != -1)
							{
								lines[j][10] = -1;
							}
							C0 += 1;
							num++;
						}
						if (lines[j][9] == 1 && C1 == 0)
						{
							linec.push_back(lines[j]);
							if (lines[j][12] != -1)
							{
								lines[j][10] = -1;
							}
							C1 += 1;
							num++;
						}
						if (lines[j][9] == 2 && C2 == 0)
						{
							linec.push_back(lines[j]);
							if (lines[j][12] != -1)
							{
								lines[j][10] = -1;
							}
							C2 += 1;
							num++;
						}
						if (lines[j][9] == 3 && C3 == 0)
						{
							linec.push_back(lines[j]);
							if (lines[j][12] != -1)
							{
								lines[j][10] = -1;
							}
							C3 += 1;
							num++;
						}

						if (C1 != 0 && C2 != 0 && C3 != 0 && C0 != 0)
						{
							break;
						}
					}
				}
			}
			if (num == 0)
			{
				lines[i][10] = 0;
			}
			else
			{
				if (linec.size() == 2)
				{
					Point tempdis;
					Point temppoi;

					//判断两线交点是否过远（当两条线平行时或接近平行时的问题）
					tempdis = findLineCross(linec);
					if (abs(sqrt((linec[0][4] - tempdis.x)*(linec[0][4] - tempdis.x)
						+ (linec[0][5] - tempdis.y)*(linec[0][5] - tempdis.y))) > 30)
					{
						//若交点和底边中点过远则取两条线底边中点距离的中值
						temppoi = Point((linec[0][4] + linec[1][4]) / 2, (linec[0][5] + linec[1][5]) / 2);
						points.push_back(temppoi);
					}
					else
					{
						points.push_back(findLineCross(linec));
					}
				}
				else
				{
					points.push_back(findLineCross(linec));
				}
			}
			//cout << endl;
		}
		//cout << "num: " << num << endl;
	}
}