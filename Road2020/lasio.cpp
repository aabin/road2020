//
// Created by joey on 2019/10/5.
//

#include "lasio.h"
#include <fstream>
#include <liblas/liblas.hpp>
using namespace std;

void Lasio::readLasFile(const string &filepath, vector<Point3DI> &points, Point3D &g_offset, Point3D &local_minB, Point3D &local_maxB)
{
    cout << "#input file path:    " << filepath << endl;
    ifstream ifs;
    ifs.open(filepath.c_str(), ios::in|ios::binary);

    liblas::ReaderFactory f;
    liblas::Reader reader = f.CreateWithStream(ifs);
    liblas::Header const& header = reader.GetHeader();

    minB.b_x = header.GetMinX();
    minB.b_y = header.GetMinY();
    minB.b_z = header.GetMinZ();
    maxB.b_x = header.GetMaxX();
    maxB.b_y = header.GetMaxY();
    maxB.b_z = header.GetMaxZ();

    //define the value of global_offset
    global_offset.b_x = int(50*(minB.b_x + maxB.b_x)) / 100.0;
    global_offset.b_y = int(50*(minB.b_y + maxB.b_y)) / 100.0;
    global_offset.b_z = 0.0;

    g_offset = global_offset;

    long p_number = header.GetPointRecordsCount();

    cout << "*points number:  " << p_number << endl;

    points.clear();
    reader.Seek(0);

    while(reader.ReadNextPoint())
    {
        const liblas::Point &curPt = reader.GetPoint();
        double l_x = curPt.GetX() - global_offset.b_x;
        double l_y = curPt.GetY() - global_offset.b_y;
        double l_z = curPt.GetZ() - global_offset.b_z;
        int inten = curPt.GetIntensity();
        Point3DI cPt(l_x, l_y, l_z, inten);

        points.push_back(cPt);
    }

    local_minB.b_x = minB.b_x - g_offset.b_x;
    local_minB.b_y = minB.b_y - g_offset.b_y;
    local_minB.b_z = minB.b_z - g_offset.b_z;

    local_maxB.b_x = maxB.b_x - g_offset.b_x;
    local_maxB.b_y = maxB.b_y - g_offset.b_y;
    local_maxB.b_z = maxB.b_z - g_offset.b_z;
}

void Lasio::writeLasFile(const string &filepath, const vector<Point3DI> &points, const Point3D &g_offset)
{
    cout << "#output file path:    " << filepath << endl;

    //get the boundary of the point cloud, local coord
    Point3D minB, maxB;
    this->ccltPcBound(points, minB, maxB);

    std::ofstream ofs;
    ofs.open(filepath, std::ios::out | std::ios::binary);

    if (ofs.is_open()) {
        liblas::Header header;
        header.SetDataFormatId(liblas::ePointFormat2);
        header.SetVersionMajor(1);
        header.SetVersionMinor(2);
        header.SetMin(minB.b_x + g_offset.b_x, minB.b_y + g_offset.b_y, minB.b_z + g_offset.b_z);
        header.SetMax(maxB.b_x + g_offset.b_x, maxB.b_y + g_offset.b_y, maxB.b_z + g_offset.b_z);
        header.SetOffset(g_offset.b_x, g_offset.b_y, g_offset.b_z);
        header.SetScale(0.001, 0.001, 0.0001);
        header.SetPointRecordsCount(points.size());

        liblas::Writer writer(ofs, header);
        liblas::Point pt(&header);

        for (int i = 0; i < points.size(); i++) {
            pt.SetCoordinates(double(points[i].x) + g_offset.b_x, double(points[i].y) + g_offset.b_y,
                              double(points[i].z) + g_offset.b_z);
            pt.SetIntensity(points[i].inten);
            writer.WritePoint(pt);
        }
        ofs.flush();
        ofs.close();

        cout << "*writed points number: " << points.size() << endl;
    }
}

void Lasio::writeLasFile(const string &filepath, const vector<vector<Grid> > &grids, const Point3D &g_offset)
{
    cout << "#output file path:    " << filepath << endl;

    //get the boundary of the point cloud, local coord
    Point3D minB, maxB;
    this->ccltPcBound(grids, minB, maxB);

    std::ofstream ofs;
    ofs.open(filepath, std::ios::out | std::ios::binary);

    long point_number = 0;
    for (int i = 0; i < grids.size(); ++i) {
        for (int j = 0; j < grids[i].size(); ++j) {
            point_number += grids[i][j].point_number;
        }
    }

    if (ofs.is_open()) {
        liblas::Header header;
        header.SetDataFormatId(liblas::ePointFormat2);
        header.SetVersionMajor(1);
        header.SetVersionMinor(2);
        header.SetMin(minB.b_x + g_offset.b_x, minB.b_y + g_offset.b_y, minB.b_z + g_offset.b_z);
        header.SetMax(maxB.b_x + g_offset.b_x, maxB.b_y + g_offset.b_y, maxB.b_z + g_offset.b_z);
        header.SetOffset(g_offset.b_x, g_offset.b_y, g_offset.b_z);
        header.SetScale(0.001, 0.001, 0.0001);
        header.SetPointRecordsCount(point_number);

        liblas::Writer writer(ofs, header);
        liblas::Point pt(&header);

        for (int i = 0; i < grids.size(); ++i) {
            for (int j = 0; j < grids[i].size(); ++j) {
                for (int k = 0; k < grids[i][j].vPoints.size(); ++k) {
                    Point3DI cp = grids[i][j].vPoints[k];
                    pt.SetCoordinates(double(cp.x) + g_offset.b_x, double(cp.y) + g_offset.b_y,
                                      double(cp.z) + g_offset.b_z);
                    pt.SetIntensity(cp.inten);
                    writer.WritePoint(pt);
                }
            }
        }

        ofs.flush();
        ofs.close();

        cout << "*writed points number: " << point_number << endl;
    }
}

void Lasio::ccltPcBound(const vector<Point3DI> &points, Point3D &minB, Point3D &maxB)
{
    minB.b_x = 0.0;
    minB.b_y = 0.0;
    minB.b_z = 0.0;
    maxB.b_x = 0.0;
    maxB.b_y = 0.0;
    maxB.b_z = 0.0;

    for (int i = 0; i < points.size(); ++i) {
        if (minB.b_x > points[i].x)
            minB.b_x = points[i].x;
        if (minB.b_y > points[i].y)
            minB.b_y = points[i].y;
        if (minB.b_z > points[i].z)
            minB.b_z = points[i].z;
        if (maxB.b_x < points[i].x)
            maxB.b_x = points[i].x;
        if (maxB.b_y < points[i].y)
            maxB.b_y = points[i].y;
        if (maxB.b_z < points[i].z)
            maxB.b_z = points[i].z;
    }
}

void Lasio::ccltPcBound(const vector<vector<Grid> > &points, Point3D &minB, Point3D &maxB)
{
    minB.b_x = 0.0;
    minB.b_y = 0.0;
    minB.b_z = 0.0;
    maxB.b_x = 0.0;
    maxB.b_y = 0.0;
    maxB.b_z = 0.0;

    for (int i = 0; i < points.size(); ++i) {
        for (int j = 0; j < points[i].size(); ++j) {
            for (int k = 0; k < points[i][j].vPoints.size(); ++k) {
                if (minB.b_x > points[i][j].vPoints[k].x)
                    minB.b_x = points[i][j].vPoints[k].x;
                if (minB.b_y > points[i][j].vPoints[k].y)
                    minB.b_y = points[i][j].vPoints[k].y;
                if (minB.b_z > points[i][j].vPoints[k].z)
                    minB.b_z = points[i][j].vPoints[k].z;
                if (maxB.b_x < points[i][j].vPoints[k].x)
                    maxB.b_x = points[i][j].vPoints[k].x;
                if (maxB.b_y < points[i][j].vPoints[k].y)
                    maxB.b_y = points[i][j].vPoints[k].y;
                if (maxB.b_z < points[i][j].vPoints[k].z)
                    maxB.b_z = points[i][j].vPoints[k].z;
            }
        }
    }
}
