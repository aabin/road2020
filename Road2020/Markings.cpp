#include "Markings.h"
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
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
		}
		// 如果邻域内没有点，显著性为0
		else
		{
			salient[i].inten = 0;
		}
	}
	return salient;
}
