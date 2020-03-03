//
// Created by joey on 2019/10/5.
//
#pragma once
#ifndef RV_STRUCTS_H
#define RV_STRUCTS_H

#include <vector>
#include <Eigen/Dense>
#include <ctime>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using std::vector;
using std::string;

typedef unsigned int uint;

//lasio: store the min/max boundary of original point cloud
struct Point3D{
    double b_x;
    double b_y;
    double b_z;

    Point3D()
    {
        b_x = b_y = b_z = 0.0;
    }
};

//self-defined struct for 3DI points
struct Point3DI{
    double x;
    double y;
    double z;
    int inten;

    Point3DI(double dx = 0.0, double dy = 0.0, double dz = 0.0, int dinten = 0)
    {
        x = dx;
        y = dy;
        z = dz;
        inten = dinten;
    }
};

//grid: project point cloud to horizontal grids
struct Grid{
    int row_pos; //bottom to up
    int col_pos; //left to right

    Point3D grid_offset; //grid_coor = local_coord - grid_offset

    int point_number; //point number of the grid

    float size;

    vector<Point3DI> vPoints; //points in grid coord

    struct gridInfo{
        bool isValid;
        bool isEmpty;
        Eigen::Vector3d normalDirection;
        Eigen::Vector3d principleDirection;
        float lamda1;
        float lamda2;
        float lamda3;
        double curvature;

        //sqrt(lamda3/point_number)
        //point to plane residual
        float residual;

        Point3DI average;

        float stv_h;//standard deviation of Z-value
        float stv_i;//standard deviation of I-value

        gridInfo()
        {
            isValid = false;
            isEmpty = false;
            lamda1 = lamda2 = lamda3 = 0.0;
            curvature = 0.0;
            residual = 0.0;
        }
    };

    gridInfo info;

    Grid()
    {
        row_pos = 0;
        col_pos = 0;
        grid_offset = Point3D();
        point_number = 0;
        vPoints.clear();
        info = gridInfo();
    }
};

struct SimpleGridInfo
{
    uint row_pos;
    uint col_pos;
    float info;
    SimpleGridInfo(uint a = 0, uint b = 0, float c = 0.0)
    {
        row_pos = a;
        col_pos = b;
        info = c;
    }
};

class Time
{
private:
    clock_t start, end;

public:
    inline string timeSpent()
    {
        this->end=clock();
        double val= (double)(this->end - this->start) / CLOCKS_PER_SEC;
        return "      Time: " + std::to_string(val) + "s" ;
    }

    Time()
    {
        this->start = clock();
    }

    inline double random_number()
    {
        CvRNG rng = cvRNG(cvGetTickCount());
        return cvRandReal(&rng);
    }

    inline int random_number(int a)
    {
        if (a <= 0)
            return 0;
        CvRNG rng = cvRNG(cvGetTickCount());
        int result = int(cvRandReal(&rng) * a + 1);
        if (result <= 1)
            result = 1;
        else if (result >= a)
            result = a;
        return result;
    }
};


inline bool compGridByInfo(const SimpleGridInfo &a, const SimpleGridInfo &b)
{
    return a.info < b.info;
}

inline bool compGridByColP(const SimpleGridInfo &a, const SimpleGridInfo &b)
{
    return a.col_pos > b.col_pos;
}

//a ring of vertices, used in alpha-shape
struct zBoundary
{
    vector<Point3DI> vertices;
};

//one segment
struct zSegment
{
    Point3DI src;
    Point3DI dst;
};

#endif //RV_STRUCTS_H
