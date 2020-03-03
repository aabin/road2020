#pragma once
#include <vector>
#include "structs.h"

using namespace std;
class Markings
{
public:
	Markings();
	~Markings();

	vector<Point3DI> ccltSaliencyPoint(const vector<Point3DI> &refine_points);
};

