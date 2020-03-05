#pragma once
#include <vector>
#include "structs.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
using namespace std;
using namespace pcl;
class Markings
{
public:
	Markings();
	~Markings();

	double ave_point_interv; //平均点间距

	// 从原始的强度生成强度图
	vector<Point3DI> ccltSaliencyPoint(const vector<Point3DI> &refine_points);

	// 从强度图提取标线点，根据阈值，DBSCAN
	void extractMarkingPointsDBSCAN(const vector<Point3DI> &saliency, const double &threshold, vector<Point3DI> &markint_p, vector<Point3DI> &cluster);

	// 从强度图提取标线点，根据阈值，欧式聚类
	vector<PointCloud<PointXYZ> > extractMarkingPointsEUDistance(const vector<Point3DI> &saliency, const double &threshold, vector<Point3DI> &markint_p, vector<Point3DI> &cluster);

	// 循环ransac干掉outliers
	void filterOutliersByRansac(vector<PointCloud<PointXYZ> > &clusters, vector<Point3DI> &Poutliers, vector<Point3DI> &Pinliers);

private:
	vector<Point3DI> copyPC(PointCloud<PointXYZ> &pc);
};

class Point {
public:
	double x, y;
	int ptsCnt, cluster;
	double getDis(const Point & ot);
};

class DBCAN {
public:
	int minPts;
	double eps;
	vector<Point> points;
	int size;
	vector<vector<int> > adjPoints;
	vector<bool> visited;
	vector<vector<int> > cluster;
	int clusterIdx;

	DBCAN(double eps, int minPts, vector<Point> points);
	void run();
	void dfs(int now, int c);
	void checkNearPoints();
	bool isCoreObject(int idx);
	vector<vector<int> > getCluster();
};