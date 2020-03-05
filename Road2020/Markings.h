#pragma once
#include <vector>
#include "structs.h"

using namespace std;
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
	void extractMarkingPointsEUDistance(const vector<Point3DI> &saliency, const double &threshold, vector<Point3DI> &markint_p, vector<Point3DI> &cluster);
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