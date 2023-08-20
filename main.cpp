
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "cnpy.h"
#include <iostream>
#include <numeric>
#include "source/PostProcess.h"
using namespace cv;
using namespace std;

using postprocess::PostProcess;



int main(int argc, char** argv)
{
    std::string input_dir = argv[1];
    std::string idx = argv[2];

    postprocess::PostProcess postProcessor(input_dir, idx);
    postProcessor.visualize_Binary();
    
    cv::Mat binary_mask;
    cv::Mat instance_mask;
    postProcessor.process(binary_mask, instance_mask);

    cv::imshow("Binary Image", binary_mask);
    cv::waitKey(0);
    cv::imshow("Instance Image", instance_mask);
    cv::waitKey(0);

    // // Just to save output
    // cv::imwrite("binary_ret.png", binary_mask);
    // cv::imwrite("instance_ret.png", instance_mask);

    
    return 0;

}
