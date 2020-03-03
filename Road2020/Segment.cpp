//
// Created by joey on 2019/10/6.
//

#include "Segment.h"
#include <iostream>
#include <vector>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <algorithm>
#include <queue>
#include <Eigen/Dense>
#include <map>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Projection_traits_xy_3.h>

typedef K::Point_2  Point2;
typedef K::Point_3  Point3;

using namespace std;
using namespace Eigen;

void
Segment::segmentToGrid(const vector<Point3DI> &localPts, const Point3D &center, const float &size, const Point3D &minB, const Point3D &maxB,  vector<vector<Grid> > &output)
{
    if ( size<0.05 && size > 1000)
        return;

    cout << "#grid size:    " << size << endl;

    int col_num = (maxB.b_x - minB.b_x) / size + 1;
    int row_num = (maxB.b_y - minB.b_y) / size + 1;

    output.resize(row_num);
    for (int i = 0; i < row_num; ++i) {
        output[i].resize(col_num);
        for (int j = 0; j < col_num; ++j) {
            output[i][j].row_pos = i;
            output[i][j].col_pos = j;
            output[i][j].size = size;
        }
    }

    long pNum = localPts.size();
    for (long i = 0; i < pNum; ++i) {
        int row_pos = (localPts[i].y - minB.b_y) / size;
        int col_pos = (localPts[i].x - minB.b_x) / size;
        output[row_pos][col_pos].vPoints.push_back(localPts[i]);
        output[row_pos][col_pos].point_number++;
    }

    cout << "*points has been segmented into grids" << endl;
}

void Segment::ccltGridInfo(vector<vector<Grid> > &chess)
{
    for (int i = 0; i < chess.size(); ++i) {
        for (int j = 0; j < chess[i].size(); ++j) {
            Grid &grid = chess[i][j];

            if (grid.point_number == 0) {
                grid.info.isEmpty = true;
                grid.info.isValid = false;
                continue;
            }
            else if (grid.point_number < 5) {
                grid.info.isEmpty = false;
                grid.info.isValid = false;

                double inten_sum = 0.0;
                double z_sum = 0.0;
                for (int k = 0; k < grid.point_number; ++k) {
                    inten_sum += grid.vPoints[k].inten;
                    z_sum += grid.vPoints[k].z;
                }
                grid.info.average.z = z_sum / grid.point_number;
                grid.info.average.inten = inten_sum / grid.point_number;

                double accu1 = 0.0;
                double accu2 = 0.0;
                for (int k = 0; k < grid.point_number; ++k) {
                    accu1 += (grid.vPoints[k].z - grid.info.average.z) * (grid.vPoints[k].z - grid.info.average.z);
                    accu2 += (grid.vPoints[k].inten - grid.info.average.inten) *
                             (grid.vPoints[k].inten - grid.info.average.inten);
                }

                grid.info.stv_h = accu1 / (grid.point_number - 1);
                grid.info.stv_i = accu2 / (grid.point_number - 1);
                continue;
            }
            else//if point number >= 5 features are valid
            {
                grid.info.isEmpty = false;
                grid.info.isValid = true;

                CvMat *pData = cvCreateMat(grid.point_number, 3, CV_32FC1);
                CvMat *pMean = cvCreateMat(1, 3, CV_32FC1);
                CvMat *pEigVals = cvCreateMat(1, 3, CV_32FC1);
                CvMat *pEigVecs = cvCreateMat(3, 3, CV_32FC1);

                long intensity_sum = 0;
                for (int k = 0; k < grid.point_number; ++k) {
                    cvmSet(pData, k, 0, grid.vPoints[k].x);
                    cvmSet(pData, k, 1, grid.vPoints[k].y);
                    cvmSet(pData, k, 2, grid.vPoints[k].z);

                    intensity_sum += grid.vPoints[k].inten;
                }
                cvCalcPCA(pData, pMean, pEigVals, pEigVecs, CV_PCA_DATA_AS_ROW);

                grid.info.average.inten = 1.0 * intensity_sum / grid.point_number;

                grid.info.normalDirection.x() = cvmGet(pEigVecs, 2, 0);
                grid.info.normalDirection.y() = cvmGet(pEigVecs, 2, 1);
                grid.info.normalDirection.z() = cvmGet(pEigVecs, 2, 2);

                if (grid.info.normalDirection.z() < 0)
                {
                    grid.info.normalDirection.x() = 0.0 - grid.info.normalDirection.x();
                    grid.info.normalDirection.y() = 0.0 - grid.info.normalDirection.y();
                    grid.info.normalDirection.z() = 0.0 - grid.info.normalDirection.z();
                }

                grid.info.principleDirection.x() = cvmGet(pEigVecs, 0, 0);
                grid.info.principleDirection.y() = cvmGet(pEigVecs, 0, 1);
                grid.info.principleDirection.z() = cvmGet(pEigVecs, 0, 2);

                grid.info.lamda1 = cvmGet(pEigVals, 0, 0);
                grid.info.lamda2 = cvmGet(pEigVals, 0, 1);
                grid.info.lamda3 = cvmGet(pEigVals, 0, 2);

                grid.info.residual = sqrt(grid.info.lamda3 / grid.point_number);

                grid.info.average.x = cvmGet(pMean, 0, 0);
                grid.info.average.y = cvmGet(pMean, 0, 1);
                grid.info.average.z = cvmGet(pMean, 0, 2);

                double accu1 = 0.0;
                double accu2 = 0.0;
                for (int k = 0; k < grid.point_number; ++k) {
                    accu1 += (grid.vPoints[k].z - grid.info.average.z) * (grid.vPoints[k].z - grid.info.average.z);
                    accu2 += (grid.vPoints[k].inten - grid.info.average.inten) *
                             (grid.vPoints[k].inten - grid.info.average.inten);
                }

                grid.info.stv_h = accu1 / (grid.point_number - 1);
                grid.info.stv_i = accu2 / (grid.point_number - 1);

                grid.info.curvature = grid.info.lamda3 / (grid.info.lamda1 + grid.info.lamda2 + grid.info.lamda3);

                cvReleaseMat(&pData);
                cvReleaseMat(&pMean);
                cvReleaseMat(&pEigVals);
                cvReleaseMat(&pEigVecs);
            }
        }
    }

    cout << "*grid info calculated successfully" << endl;


}


void Segment::regionGrow(const vector<vector<Grid> > &chess, const float &planar_threshold, const float &angle_threshold, const float &height_threshold, vector<vector<uint> > &regions)
{
    cout << "#planar threshold is:  " << planar_threshold << endl;
    cout << "#angle threshold is:   " << angle_threshold << " degree" << endl;

    int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };

    //grids sort by lamda3 as candidates
    vector<SimpleGridInfo> cddtGrids;

    regions.resize(chess.size());
    for (int i = 0; i < regions.size(); ++i) {
        regions[i].resize(chess[i].size());
        for (int j = 0; j < regions[i].size(); ++j) {
            regions[i][j] = 0;

            //satisfy 2 conditions can be selected as candidate seeding points
            //1. points_num >= 5
            //2. smooth value <= planar threshold
            if (chess[i][j].info.isValid && chess[i][j].info.residual <= planar_threshold)
            {
                cddtGrids.push_back(SimpleGridInfo(i, j, chess[i][j].info.residual));
            }
        }
    }
    sort(cddtGrids.begin(), cddtGrids.end(), compGridByInfo);

    uint region_flag = 0;//region tag
    uint p = 0; //point to pos when searching cddtGrids
    queue<SimpleGridInfo> seed;

	while (p < cddtGrids.size())
	{
		if (regions[cddtGrids[p].row_pos][cddtGrids[p].col_pos] < 1)
		{
			region_flag++;
			seed.push(cddtGrids[p]);
			regions[cddtGrids[p].row_pos][cddtGrids[p].col_pos] = region_flag;
			while (!seed.empty())
			{
				Grid seedG = chess[seed.front().row_pos][seed.front().col_pos];
				for (int i = 0; i < 8; ++i)
				{
					if (seed.front().row_pos == 0 || seed.front().col_pos == 0
						|| seed.front().row_pos == chess.size() - 1 || seed.front().col_pos == chess[0].size() - 1)
						continue;
					if (!chess[seed.front().row_pos + DIR[i][0]][seed.front().col_pos + DIR[i][1]].info.isValid
						|| regions[seed.front().row_pos + DIR[i][0]][seed.front().col_pos + DIR[i][1]] > 0)
						continue;
					if (chess[seed.front().row_pos + DIR[i][0]][seed.front().col_pos + DIR[i][1]].info.residual > planar_threshold)
						continue;
					Grid tgtG = chess[seed.front().row_pos + DIR[i][0]][seed.front().col_pos + DIR[i][1]];

					Eigen::Vector3d src(seedG.info.average.x, seedG.info.average.y, seedG.info.average.z);
					Eigen::Vector3d tgt(tgtG.info.average.x, tgtG.info.average.y, tgtG.info.average.z);
					Eigen::Vector3d diff = tgt - src;

					Eigen::Vector3d nSrc = seedG.info.normalDirection;
					Eigen::Vector3d nTgt = tgtG.info.normalDirection;

					double radian_angle_n = atan2(nSrc.cross(nTgt).norm(), nSrc.transpose()*nTgt);
					double angle_n = radian_angle_n * 180.0 / M_PI;

					double upp;
					upp = fabs(diff.transpose() * nSrc);
					double btm;
					btm = sqrt(nSrc.transpose()*nSrc);

					double diss = upp / btm;

					//1. angle between normal directions
					//2. length of projection line when projecting diff_of_grids to src_normal
					if (angle_n < angle_threshold && diss < 0.05)
					{
						regions[seed.front().row_pos + DIR[i][0]][seed.front().col_pos + DIR[i][1]] = region_flag;
						seed.push(SimpleGridInfo(seed.front().row_pos + DIR[i][0], seed.front().col_pos + DIR[i][1], region_flag));
					}
				}
				seed.pop();
			}
			p++;
		}
		else
		{
			p++;
		}
	}
    cout << "*region growed" << endl;
}

void Segment::adjustRegions(vector<vector<uint> > &regions, const vector<vector<Grid> > &chess)
{
    int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };

    vector<vector<uint> > dst = regions;

    vector<ushort > labels(8);

    int largest_label = 0;

    for (int i = 1; i < regions.size() - 1; ++i) {
        for (int j = 1; j < regions[0].size() - 1; ++j) {
            if (regions[i][j] > largest_label)
                largest_label = regions[i][j];
            for (int k = 0; k < 8; ++k) {
                if (chess[i][j].info.isEmpty)
                    continue;
                labels[k] = regions[i + DIR[k][0]][j + DIR[k][1]];
            }
            sort(labels.begin(), labels.end());
            if(labels[0] == labels[6] && labels[1] == labels[7])
                dst[i][j] = labels[5];
        }
    }
    regions = dst;

    vector<SimpleGridInfo> sizes;
    sizes.resize(largest_label + 1);

    for (int i = 0; i < sizes.size(); ++i) {
        sizes[i].row_pos = i;
    }

    for (int i = 0; i < regions.size(); ++i) {
        for (int j = 0; j < regions[0].size(); ++j) {
            sizes.at(regions[i][j]).col_pos++;
        }
    }

    sort(sizes.begin() + 1, sizes.end(), compGridByColP);

    map<int, int> transform;
    transform[0] = 0;
    for (int i = 1; i < sizes.size(); ++i) {
        transform[sizes.at(i).row_pos] = i;
        if (i > 5)
            transform[sizes.at(i).row_pos] = 0;
    }

    for (int i = 0; i < regions.size(); ++i) {
        for (int j = 0; j < regions[0].size(); ++j) {
            regions[i][j] = transform.at(regions[i][j]);
        }
    }

    cout << "*adjusted regions" << endl;
}

void Segment::chooseRegions(const vector<vector<uint> > &regions, const ushort &largest_label, vector<vector<uint> > &alpha)
{
    alpha.resize(regions.size());
    for (int i = 0; i < regions.size(); ++i) {
        alpha[i].resize(regions[0].size());
        for (int j = 0; j < regions[0].size(); ++j) {
            if (regions[i][j] <= largest_label && regions[i][j] > 0)
                alpha[i][j] = 1;
            else
                alpha[i][j] = 0;
        }
    }
}

void Segment::refineRoad(const vector<vector<Grid> > &chess, vector<vector<uint> > &alpha, const float &distance_threshold, vector<Point3DI> &refine_points)
{
    int DIR8[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	int DIR4[4][2] = { { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } };

    for (int i = 1; i < chess.size() - 1; ++i) {
        for (int j = 1; j < chess[0].size() - 1; ++j) {
            if (alpha[i][j] == 1)
            {
                for (int k = 0; k < chess[i][j].point_number; ++k) {
                    refine_points.push_back(chess[i][j].vPoints[k]);
                }
				continue;
            }

            vector<Grid::gridInfo> infos;

            for (int k = 0; k < 4; ++k) {
                uint rpos = i + DIR4[k][0];
                uint cpos = j + DIR4[k][1];

                if (alpha[rpos][cpos] == 0)
                    continue;
                infos.push_back(chess[rpos][cpos].info);
            }

            //consider every point
            for (int k = 0; k < chess[i][j].point_number; ++k) {
                Point3DI cPt = chess[i][j].vPoints[k];
                for (int m = 0; m < infos.size(); ++m) {
                    Eigen::Vector3d diff(cPt.x - infos[m].average.x, cPt.y - infos[m].average.y, cPt.z - infos[m].average.z);
                    double upp = fabs(diff.transpose() * infos[m].normalDirection);
                    double btm = sqrt(infos[m].normalDirection.transpose()*infos[m].normalDirection);
                    double distance = upp / btm;
                    //if (distance <= distance_threshold && (cPt.inten - infos[m].average.inten) < infos[m].stv_i)
					if (distance <= distance_threshold && (cPt.inten - infos[m].average.inten) < 0)
                    {
                        refine_points.push_back(cPt);
                        break;//jump out and consider next point
                    }
                }
            }
        }
    }
}

void Segment::extractBoundary(const float &alpha, const vector<Point3DI> &cloud, const vector<vector<Grid> > &chess, vector<zBoundary> &boundarys, Point3D minB, vector<Point3DI> &edges)
{
    std::list<Point2> points;
    for (int i = 0; i < cloud.size(); i++) {
        Point2 p(int(100 * cloud[i].x), int(100 * cloud[i].y));
        points.push_back(p);
    }

    uint radius = alpha * alpha * 10000;
    Alpha_shape_2 AS(points.begin(), points.end(), FT(radius), Alpha_shape_2::GENERAL);

    vector<vector<Segment_CGAL> > cboundarys;
    std::deque<Segment_CGAL> allsegments;

    Alpha_shape_edges_iterator e_it;
    for (e_it = AS.alpha_shape_edges_begin(); e_it != AS.alpha_shape_edges_end(); e_it++)
    {
        Segment_CGAL line = AS.segment(*e_it);
        allsegments.push_back(line);
    }

    Segment_CGAL header, current;
    std::vector<Segment_CGAL> segment_line;
    while (allsegments.size() > 0)
    {
        header = allsegments.front();
        allsegments.pop_front();
        current = header;
        segment_line.push_back(current);

        bool one_segment = false;
        int near_header_num = 0;
        bool co_point;
        do
        {
            co_point = false;
            for (int k = 0; k < allsegments.size(); k++)
            {
                Segment_CGAL temp = allsegments[k];
                if (temp.source() == current.target())
                {
                    segment_line.push_back(temp);
                    current = temp;
                    co_point = true;
                    allsegments.erase(allsegments.begin() + k);
                    if (current.target() == header.source())
                        one_segment = true;
                    break;
                }
            }
        } while (!one_segment&&co_point);

        cboundarys.push_back(segment_line);
        segment_line.clear();
    };

    boundarys.resize(cboundarys.size());
    for (int i = 0; i < cboundarys.size(); ++i) {
        boundarys[i].vertices.resize(cboundarys[i].size());
        for (int j = 0; j < cboundarys[i].size(); ++j) {
            Point3DI cPt;
            cPt.x = cboundarys[i][j].source().x() / 100.0;
            cPt.y = cboundarys[i][j].source().y() / 100.0;
            cPt.z = 0.0;
            cPt.inten = i;

            uint rPos = (cPt.y - minB.b_y) / chess[0][0].size;
            uint cPos = (cPt.x - minB.b_x) / chess[0][0].size;
            if (chess[rPos][cPos].point_number < 1)
            {
                cPt.z = 0.0;
                boundarys[i].vertices[j] = cPt;
            }
            else
            {
                cPt.z = chess[rPos][cPos].vPoints[0].z;
                boundarys[i].vertices[j] = cPt;
                for (int k = 0; k < chess[rPos][cPos].point_number; ++k) {
                    cPt.z = chess[rPos][cPos].vPoints[0].z;
                    boundarys[i].vertices[j] = cPt;
                    if(fabs(cPt.x - chess[rPos][cPos].vPoints[k].x) <= 0.018 && fabs(cPt.y - chess[rPos][cPos].vPoints[k].y) <= 0.018)
                    {
                        cPt.z = chess[rPos][cPos].vPoints[k].z;
                        boundarys[i].vertices[j] = cPt;
                        break;
                    }
                }
            }
        }
    }

    for (int i = 0; i < boundarys.size(); ++i) {
        for (int j = 0; j < boundarys[i].vertices.size(); ++j) {
            Point3DI pt;
            pt.x = boundarys[i].vertices[j].x;
            pt.y = boundarys[i].vertices[j].y;
            pt.z = boundarys[i].vertices[j].z;
            if (fabs(pt.z) < 1e-5 && (j - 1) >= 0)
                pt.z = boundarys[i].vertices[j-1].z;
            if (fabs(pt.z) < 1e-5)
                continue;
            pt.inten= i;

            edges.push_back(pt);
        }
    }

    std::list<Point2>().swap(points);
    std::vector<std::vector<Segment_CGAL> >().swap(cboundarys);
    std::deque<Segment_CGAL>().swap(allsegments);
}
