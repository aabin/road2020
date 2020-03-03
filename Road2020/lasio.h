//
// Created by joey on 2019/10/5.
//

#ifndef RV_LASIO_H
#define RV_LASIO_H

#include "structs.h"
#include <string>
#include <vector>

using std::string;
using std::vector;

class Lasio {

public:
    //only filepath is input, others are given in method
    //commit global offset, points are in local-coord
    //input points are in global-coord
    void readLasFile(const string &filepath, vector<Point3DI> &points, Point3D &g_offset, Point3D &local_minB, Point3D &local_maxB);

    void writeLasFile(const string &filepath, const vector<Point3DI> &points, const Point3D &g_offset);

    void writeLasFile(const string &filepath, const vector<vector<Grid> > &grids, const Point3D &g_offset);

    //lcoo = gcoo - global_offset
    Point3D global_offset;
    //output the boundary to boost segment
    Point3D minB;
    Point3D maxB;

private:


    void ccltPcBound(const vector<Point3DI> &points, Point3D &minB, Point3D &maxB);
    void ccltPcBound(const vector<vector<Grid> > &points, Point3D &minB, Point3D &maxB);

};


#endif //RV_LASIO_H
