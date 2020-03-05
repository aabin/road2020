#pragma once
#include <vector>
#include "structs.h"

using namespace std;
class Markings
{
public:
	Markings();
	~Markings();

	double ave_point_interv; //ƽ������

	// ��ԭʼ��ǿ������ǿ��ͼ
	vector<Point3DI> ccltSaliencyPoint(const vector<Point3DI> &refine_points);

	// ��ǿ��ͼ��ȡ���ߵ㣬������ֵ��DBSCAN
	void extractMarkingPointsDBSCAN(const vector<Point3DI> &saliency, const double &threshold, vector<Point3DI> &markint_p, vector<Point3DI> &cluster);

	// ��ǿ��ͼ��ȡ���ߵ㣬������ֵ��ŷʽ����
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