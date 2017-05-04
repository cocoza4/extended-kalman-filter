#include <iostream>
#include "tools.h"
#include <cmath>
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {
}

Tools::~Tools() {
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
		const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
		std::cerr << "Invalid estimation or ground_truth data" << std::endl;
		return rmse;
	}

	//accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i) {

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array() * residual.array();
		rmse += residual;
	}

	rmse = rmse / estimations.size();
	rmse = rmse.array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//compute the Jacobian matrix
	float pxpy = pow(px, 2) + pow(py, 2);
	if(fabs(pxpy) < 0.0001){
		std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
		return Hj;
	}

	float pxpy_sqrt = sqrt(pxpy);
	float t = pow(pxpy, 1.5);
	Hj << px/pxpy_sqrt, py/pxpy_sqrt, 0, 0,
			-py/pxpy, px/pxpy, 0, 0,
			py*(vx*py-vy*px)/t, px*(vy*px-vx*py)/t, px/pxpy_sqrt, py/pxpy_sqrt;

	return Hj;
}

VectorXd Tools::CalculateHofX(const VectorXd& x_state) {

	VectorXd h(3);
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

//	if (px == 0) {
//		//cout << "px is close to zero" << endl;
//		px = 0.1;
//	}
//
//	if (py == 0) {
//		//cout << "py is close to zero" << endl;
//		py = 0.1;
//	}

	float px2 = px * px;
	float py2 = py * py;
	float rho = sqrt(px2 + py2);
	float phi = atan2(py, px);
	float roh_dot = ((px * vx) + (py * vy)) / rho;

	h << rho, phi, roh_dot;

	return h;
}

double Tools::checkPIValue(double x) {
	while (x < -M_PI) {
		x += 2 * M_PI;
	}
	while (x > M_PI) {
		x -= 2 * M_PI;
	}

	return x;
}
