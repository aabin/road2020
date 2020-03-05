#include "Markings.h"
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <cmath>
#include <map>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
using namespace pcl;
using namespace Eigen;

Markings::Markings()
{
}


Markings::~Markings()
{
}

vector<Point3DI> Markings::ccltSaliencyPoint(const vector<Point3DI> &refine_points)
{
	vector<Point3DI> salient = refine_points;

	VectorXd global_inten;
	vector<double> point_interv;
	global_inten.resize(refine_points.size());

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Generate pointcloud data
	cloud->width = refine_points.size();
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);

	for (size_t i = 0; i < cloud->points.size(); ++i)
	{
		cloud->points[i].x = refine_points[i].x;
		cloud->points[i].y = refine_points[i].y;
		cloud->points[i].z = refine_points[i].z;
		global_inten(i) = refine_points[i].inten;
	}

	double mean_intem = global_inten.mean();

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

	kdtree.setInputCloud(cloud);

	// 对每个点计算一个强度显著性
	for (size_t i = 0; i < cloud->points.size(); ++i)
	{
		pcl::PointXYZ searchPoint = cloud->points[i];

		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;

		float radius = 1.5;

		// 把邻域内的点的强度放入vector
		if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
		{
			VectorXd intens;
			intens.resize(pointIdxRadiusSearch.size());
			for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j)
				intens(j) = refine_points[pointIdxRadiusSearch[j]].inten;
			double sali = (salient[i].inten - intens.minCoeff());
			if (sali < 0)
				sali = 0;
			sali = log10(sali / mean_intem + 1) * 1000;
			salient[i].inten = sali;

			//平均点间距计算
			if (pointRadiusSquaredDistance.size() > 5)
			{
				Vector3d tmp(sqrt(pointRadiusSquaredDistance[1]), sqrt(pointRadiusSquaredDistance[2]), sqrt(pointRadiusSquaredDistance[3]));
				point_interv.push_back(tmp.mean());
			}
		}
		// 如果邻域内没有点，显著性为0
		else
		{
			salient[i].inten = 0;
		}
	}

	double sum = accumulate(std::begin(point_interv), std::end(point_interv), 0.0);
	this->ave_point_interv = sum / point_interv.size();
	cout << "平均点间距： " << ave_point_interv << endl;
	return salient;
}

void Markings::extractMarkingPointsDBSCAN(const vector<Point3DI> &saliency, const double &threshold, vector<Point3DI> &markint_p, vector<Point3DI> &cluster)
{
	vector<Point> points;
	for (int i = 0; i < saliency.size(); i++)
	{
		if (saliency[i].inten > threshold)
		{
			Point3DI pt = saliency[i];
			markint_p.push_back(pt);
			pt.inten = 0;
			cluster.push_back(pt);
			Point c_point;
			c_point.x = saliency[i].x;
			c_point.y = saliency[i].y;
			c_point.ptsCnt = 0;
			c_point.cluster = -1;
			points.push_back(c_point);
		}
	}
	double eps = 2.0;
	int minPts = 6;

	DBCAN dbScan(eps, minPts, points);
	dbScan.run();

	vector<vector<int> > rst = dbScan.getCluster();
	sort(rst.begin(), rst.end(), [&](const vector<int> i, const vector<int> j) {
		return (int)i.size() > (int)j.size();
	});

	srand((int)time(0));
	for (int i = 0; i < rst.size(); i++)
	{
		int t = rand() % 99 + 1;
		for (int j = 0; j < rst[i].size(); j++)
		{
			cluster[rst[i][j]].inten = t;
		}
	}
}

vector<PointCloud<PointXYZ> > Markings::extractMarkingPointsEUDistance(const vector<Point3DI> &saliency, const double &threshold, vector<Point3DI> &markint_p, vector<Point3DI> &cluster)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	vector<PointCloud<PointXYZ> > clusters_pcl;

	//提取标线点
	for (int i = 0; i < saliency.size(); i++)
	{
		if (saliency[i].inten > threshold)
		{
			Point3DI pt = saliency[i];
			markint_p.push_back(pt);
			PointXYZ point;
			point.x = pt.x;
			point.y = pt.y;
			point.z = pt.z;
			cloud->points.push_back(point);
		}
	}
	// Generate pointcloud data
	cloud->width = cloud->points.size();
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);

	// 欧式距离聚类
	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(1.5); // 欧式距离聚类间隔阈值1.5m
	ec.setMinClusterSize(3);     // 最少三个点
	ec.setMaxClusterSize(25000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud);
	ec.extract(cluster_indices);

	srand((int)time(0));
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZ> cluster_pcl;
		int t = rand() % 99 + 1;
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			Point3DI cpt;
			cpt.x = cloud->points[*pit].x;
			cpt.y = cloud->points[*pit].y;
			cpt.z = cloud->points[*pit].z;
			cpt.inten = t;
			cluster.push_back(cpt);

			cluster_pcl.points.push_back(cloud->points[*pit]);
		}
		cluster_pcl.width = cluster_pcl.points.size();
		cluster_pcl.height = 1;
		cluster_pcl.points.resize(cluster_pcl.width * cluster_pcl.height);

		clusters_pcl.push_back(cluster_pcl);
	}
	return clusters_pcl;
}

void Markings::filterOutliersByRansac(vector<PointCloud<PointXYZ> > &clusters, vector<Point3DI> &Poutliers, vector<Point3DI> &Pinliers)
{
	srand((int)time(0));
	for (size_t cnt = 0; cnt < clusters.size(); ++cnt)
	{
		PointCloud<PointXYZ>::Ptr cluster = clusters.at(cnt).makeShared(); // 一个cluster内的标线点

		pcl::SACSegmentation<pcl::PointXYZ> seg;
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_line(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>());

		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_LINE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setMaxIterations(100);
		seg.setDistanceThreshold(0.2);

		int nr_points = (int)cluster->points.size();
		while (cluster->points.size() > 4)
		{
			// Segment the largest planar component from the remaining cloud
			seg.setInputCloud(cluster);
			seg.segment(*inliers, *coefficients);
			if (inliers->indices.size() == 0)
			{
				std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
				break; // 如果剩下的点中没有一个直线,则剩下的点都是outliers
			}

			// Extract the planar inliers from the input cloud
			pcl::ExtractIndices<pcl::PointXYZ> extract;
			extract.setInputCloud(cluster);
			extract.setIndices(inliers);
			extract.setNegative(false);

			// Get the points associated with the planar surface
			extract.filter(*cloud_line);
			vector<Point3DI > line = copyPC(*cloud_line);
			int t = rand() % 99 + 1;
			for (int i = 0; i < line.size(); i++)
			{
				line[i].inten = t;
				Pinliers.push_back(line[i]);
			}

			// Remove the planar inliers, extract the rest
			extract.setNegative(true);
			extract.filter(*cloud_f);
			*cluster = *cloud_f;
		}

		if (cluster->points.size() > 0)
		{
			for (int i = 0; i < cluster->points.size(); ++i)
			{
				Point3DI otl(cluster->points.at(i));
				Poutliers.push_back(otl);
			}
		}
	}
}

vector<Point3DI> Markings::copyPC(PointCloud<PointXYZ> &pc)
{
	vector<Point3DI> rst;
	for (int i = 0; i < pc.points.size(); i++)
	{
		rst.push_back(Point3DI(pc.points.at(i)));
	}
	return rst;
}

DBCAN::DBCAN(double eps, int minPts, vector<Point> points) 
{
	this->eps = eps;
	this->minPts = minPts;
	this->points = points;
	this->size = (int)points.size();
	adjPoints.resize(size);
	this->clusterIdx = -1;
}

void DBCAN::run() {
	checkNearPoints();

	for (int i = 0; i < size; i++) {
		if (points[i].cluster != -1) continue;

		if (isCoreObject(i)) {
			dfs(i, ++clusterIdx);
		}
		else {
			points[i].cluster = -2;
		}
	}

	cluster.resize(clusterIdx + 1);
	for (int i = 0; i < size; i++) {
		if (points[i].cluster != -2) {
			cluster[points[i].cluster].push_back(i);
		}
	}
}

void DBCAN::dfs(int now, int c) {
	points[now].cluster = c;
	if (!isCoreObject(now)) return;

	for (auto&next : adjPoints[now]) {
		if (points[next].cluster != -1) continue;
		dfs(next, c);
	}
}

void DBCAN::checkNearPoints() {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (i == j) continue;
			if (points[i].getDis(points[j]) <= eps) {
				points[i].ptsCnt++;
				adjPoints[i].push_back(j);
			}
		}
	}
}

bool DBCAN::isCoreObject(int idx) {
	return points[idx].ptsCnt >= minPts;
}

vector<vector<int> > DBCAN::getCluster() {
	return cluster;
}

double Point::getDis(const Point & ot)
{
	return sqrt((x - ot.x)*(x - ot.x) + (y - ot.y)*(y - ot.y));
}