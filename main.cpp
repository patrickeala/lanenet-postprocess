
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "cnpy.h"
// #include <numpy/arrayobject.h>
// #include "NumCpp.hpp"
// #include <experimental/filesystem>
#include <iostream>
#include <numeric>
#include "source/PostProcess.h"
using namespace cv;
using namespace std;

using postprocess::PostProcess;



int main(int argc, char** argv)
{
    std::string input_dir = "/100_test/model_output/";

    postprocess::PostProcess postProcessor(input_dir);
    postProcessor.visualize_Binary();
    
    cv::Mat binary_mask;
    cv::Mat instance_mask;
    postProcessor.process(binary_mask, instance_mask);

    cv::imshow("Display Image", binary_mask);
    cv::waitKey(0);
    cv::imshow("Display Image", instance_mask);
    cv::waitKey(0);

    // cv::imwrite("binary_ret.png", binary_mask);
    // cv::imwrite("instance_ret.png", instance_mask);

    // Set your DBSCAN parameters
    // double epsilon = 0.1;
    // int minPoints = 5;

    // postProcessor.performDBSCAN(epsilon, minPoints);
    // postProcessor.gatherPixelEmbeddings();

    // postProcessor.visualizeImage("Image Visualization");

    
    return 0;

}
