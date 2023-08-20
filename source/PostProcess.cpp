#include "PostProcess.h"
#include "dbscan.hpp"
#include "cnpy.h"



namespace postprocess {

PostProcess::PostProcess(const std::string& input_dir, const std::string& idx) {
    loadNpyImages(input_dir, idx);
}

void PostProcess::loadNpyImages(const std::string& input_dir, const std::string& idx) {
    // Implement loading .npy image here
    // int idx = 0;
    std::string bin_path = input_dir + idx + "_binary.npy";
    std::string ins_path = input_dir + idx + "_instance.npy";

    // Binary
	cnpy::NpyArray bin_data = cnpy::npy_load(bin_path);
    std::complex<double>* ptr = bin_data.data<std::complex<double>>();
    binary_mat = cv::Mat(bin_data.shape[0], bin_data.shape[1], CV_32FC1, ptr).clone();
    binary_mat.convertTo(binary_mat, CV_8UC1, 255, 0); 
    
    _m_input_node_size_host.width = bin_data.shape[1]; // Assign width
    _m_input_node_size_host.height = bin_data.shape[0]; // Assign height

    // // Instance
	cnpy::NpyArray ins_data = cnpy::npy_load(ins_path);
    std::complex<double>* ptr_ins = ins_data.data<std::complex<double>>();
    pix_embedding_output_mat = cv::Mat(ins_data.shape[0], ins_data.shape[1], CV_32FC4, ptr_ins).clone();



}

void PostProcess::visualize_Binary() const {
    // std::cout << "binart mat\n" << binary_mat << std::endl;

    cv::imshow("Display Image", binary_mat);
    cv::waitKey(0);
    cv::imshow("Display Image", pix_embedding_output_mat);
    cv::waitKey(0);
}


void PostProcess::process(cv::Mat &binary_seg_result, cv::Mat &instance_seg_result) {


    // GETTING BINARY OUTPUT
    // auto binary_output_data = binary_output_tensor_user.host<float>();
    // cv::Mat binary_output_mat(_m_input_node_size_host, CV_32FC1, binary_output_data);
    // binary_output_mat *= 255;
    // binary_output_mat.convertTo(binary_seg_result, CV_8UC1);


    // GETTING PIX EMB OUTPUT
    // auto pix_embedding_output_data = pix_embedding_output_tensor_user.host<float>();
    // cv::Mat pix_embedding_output_mat(_m_input_node_size_host, CV_32FC4, pix_embedding_output_data);


    binary_seg_result = binary_mat.clone();


    // 1. gather pixel embedding features
    std::vector<cv::Point> coords;
    std::vector<DBSCAMSample> pixel_embedding_samples;
    gather_pixel_embedding_features(binary_seg_result, pix_embedding_output_mat,coords, pixel_embedding_samples);

    // 2. simultaneously random shuffle embedding vector and coord vector inplace
    simultaneously_random_shuffle<cv::Point, DBSCAMSample >(coords, pixel_embedding_samples);

    // 3. simultaneously random select embedding vector and coord vector to reduce the cluster time
    std::vector<cv::Point> coords_selected;
    std::vector<DBSCAMSample> pixel_embedding_samples_selected;
    simultaneously_random_select<DBSCAMSample, cv::Point>(pixel_embedding_samples, coords,
            _m_embedding_features_dilution_ratio,pixel_embedding_samples_selected, coords_selected);
    coords.clear();
    coords.shrink_to_fit();
    pixel_embedding_samples.clear();
    pixel_embedding_samples.shrink_to_fit();

    // normalize pixel embedding features
    normalize_sample_features(pixel_embedding_samples_selected, pixel_embedding_samples_selected);

    // cluster samples
    std::vector<std::vector<uint> > cluster_ret;
    std::vector<uint> noise;
    {
        // AUTOTIME
        cluster_pixem_embedding_features(pixel_embedding_samples_selected, cluster_ret, noise);
    }

    // visualize instance segmentation
    instance_seg_result = cv::Mat(_m_input_node_size_host, CV_8UC3, cv::Scalar(0, 0, 0));
    {
        // AUTOTIME
        visualize_instance_segmentation_result(cluster_ret, coords_selected, instance_seg_result);
    }


}


/***
 * Gather pixel embedding features via binary segmentation result
 * @param binary_mask
 * @param pixel_embedding
 * @param coords
 * @param embedding_features
 */
void PostProcess::gather_pixel_embedding_features(const cv::Mat &binary_mask, const cv::Mat &pixel_embedding,
        std::vector<cv::Point> &coords,
        std::vector<DBSCAMSample> &embedding_samples) {

    // CHECK_EQ(binary_mask.size(), pixel_embedding.size());
    auto image_rows = _m_input_node_size_host.height;
    auto image_cols = _m_input_node_size_host.width;

    for (auto row = 0; row < image_rows; ++row) {
        auto binary_image_row_data = binary_mask.ptr<uchar>(row);
        auto embedding_image_row_data = pixel_embedding.ptr<cv::Vec4f>(row);
        for (auto col = 0; col < image_cols; ++col) {
            auto binary_image_pix_value = binary_image_row_data[col];
            if (binary_image_pix_value == 255) {
                coords.emplace_back(cv::Point(col, row));
                Feature embedding_features;
                for (auto index = 0; index < 4; ++index) {
                    embedding_features.push_back(embedding_image_row_data[col][index]);
                }
                DBSCAMSample sample(embedding_features, CLASSIFY_FLAGS::NOT_CALSSIFIED);
                embedding_samples.push_back(sample);
            }
        }
    }
}

/***
 *
 * @param embedding_samples
 * @param cluster_ret
 */
void PostProcess::cluster_pixem_embedding_features(std::vector<DBSCAMSample> &embedding_samples,
        std::vector<std::vector<uint> > &cluster_ret, std::vector<uint>& noise) {

    if (embedding_samples.empty()) {
        // LOG(INFO) << "Pixel embedding samples empty";
        return;
    }

    // dbscan cluster
    auto dbscan = DBSCAN<DBSCAMSample, float>();
    dbscan.Run(&embedding_samples, _m_lanenet_pix_embedding_feature_dims, _m_dbscan_eps, _m_dbscan_min_pts);
    cluster_ret = dbscan.Clusters;
    noise = dbscan.Noise;
}

/***
 * Visualize instance segmentation result
 * @param cluster_ret
 * @param coords
 */
void PostProcess::visualize_instance_segmentation_result(
    const std::vector<std::vector<uint> > &cluster_ret,
    const std::vector<cv::Point> &coords,
    cv::Mat& intance_segmentation_result) {

    std::map<int, cv::Scalar> color_map = {
        {0, cv::Scalar(0, 0, 255)},
        {1, cv::Scalar(0, 255, 0)},
        {2, cv::Scalar(255, 0, 0)},
        {3, cv::Scalar(255, 0, 255)},
        {4, cv::Scalar(0, 255, 255)},
        {5, cv::Scalar(255, 255, 0)},
        {6, cv::Scalar(125, 0, 125)},
        {7, cv::Scalar(0, 125, 125)}
    };

    // omp_set_num_threads(4);
    for (ulong class_id = 0; class_id < cluster_ret.size(); ++class_id) {
        auto class_color = color_map[class_id];
        #pragma omp parallel for
        for (auto index = 0; index < cluster_ret[class_id].size(); ++index) {
            auto coord = coords[cluster_ret[class_id][index]];
            auto image_col_data = intance_segmentation_result.ptr<cv::Vec3b>(coord.y);
            image_col_data[coord.x][0] = class_color[0];
            image_col_data[coord.x][1] = class_color[1];
            image_col_data[coord.x][2] = class_color[2];
        }
    }
}

/***
 * Calculate the mean feature vector among a vector of DBSCAMSample samples
 * @param input_samples
 * @return
 */
Feature PostProcess::calculate_mean_feature_vector(const std::vector<DBSCAMSample> &input_samples) {

    if (input_samples.empty()) {
        return Feature();
    }

    uint feature_dims = input_samples[0].get_feature_vector().size();
    uint sample_nums = input_samples.size();
    Feature mean_feature_vec;
    mean_feature_vec.resize(feature_dims, 0.0);
    for (const auto& sample : input_samples) {
        for (uint index = 0; index < feature_dims; ++index) {
            mean_feature_vec[index] += sample[index];
        }
    }
    for (uint index = 0; index < feature_dims; ++index) {
        mean_feature_vec[index] /= sample_nums;
    }

    return mean_feature_vec;
}

/***
 *
 * @param input_samples
 * @param mean_feature_vec
 * @return
 */
Feature PostProcess::calculate_stddev_feature_vector(
    const std::vector<DBSCAMSample> &input_samples,
    const Feature& mean_feature_vec) {

    if (input_samples.empty()) {
        return Feature();
    }

    uint feature_dims = input_samples[0].get_feature_vector().size();
    uint sample_nums = input_samples.size();

    // calculate stddev feature vector
    Feature stddev_feature_vec;
    stddev_feature_vec.resize(feature_dims, 0.0);
    for (const auto& sample : input_samples) {
        for (uint index = 0; index < feature_dims; ++index) {
            auto sample_feature = sample.get_feature_vector();
            auto diff = sample_feature[index] - mean_feature_vec[index];
            diff = std::pow(diff, 2);
            stddev_feature_vec[index] += diff;
        }
    }
    for (uint index = 0; index < feature_dims; ++index) {
        stddev_feature_vec[index] /= sample_nums;
        stddev_feature_vec[index] = std::sqrt(stddev_feature_vec[index]);
    }

    return stddev_feature_vec;
}




/***
 * Normalize input samples' feature. Each sample's feature is normalized via function as follows:
 * feature[i] = (feature[i] - mean_feature_vector[i]) / stddev_feature_vector[i].
 * @param input_samples
 * @param output_samples
 */
void PostProcess::normalize_sample_features(const std::vector<DBSCAMSample> &input_samples,
                                        std::vector<DBSCAMSample> &output_samples) {
    // calcualte mean feature vector
    Feature mean_feature_vector = calculate_mean_feature_vector(input_samples);

    // calculate stddev feature vector
    Feature stddev_feature_vector = calculate_stddev_feature_vector(input_samples, mean_feature_vector);

    std::vector<DBSCAMSample> input_samples_copy = input_samples;
    for (auto& sample : input_samples_copy) {
        auto feature = sample.get_feature_vector();
        for (ulong index = 0; index < feature.size(); ++index) {
            feature[index] = (feature[index] - mean_feature_vector[index]) / stddev_feature_vector[index];
        }
        sample.set_feature_vector(feature);
    }
    output_samples = input_samples_copy;
}

/***
 * simultaneously random shuffle two vector inplace. The two input source vector should have the same size.
 * @tparam T
 * @param src1
 * @param src2
 */
template <typename T1, typename T2>
void PostProcess::simultaneously_random_shuffle(std::vector<T1> src1, std::vector<T2> src2) {

    // CHECK_EQ(src1.size(), src2.size());
    if (src1.empty() || src2.empty()) {
        return;
    }

    // construct index vector of two input src
    std::vector<uint> indexes;
    indexes.reserve(src1.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::random_shuffle(indexes.begin(), indexes.end());

    // make copy of two input vector
    std::vector<T1> src1_copy(src1);
    std::vector<T2> src2_copy(src2);

    // random two source input vector via random shuffled index vector
    for (ulong i = 0; i < indexes.size(); ++i) {
        src1[i] = src1_copy[indexes[i]];
        src2[i] = src2_copy[indexes[i]];
    }
}

/***
 * simultaneously random select part of the two input vector into the two output vector.
 * The two input source vector should have the same size because they have one-to-one mapping
 * relation between the elements in two input vector
 * @tparam T1 : type of input vector src1 which should support default constructor
 *              due to the usage of vector resize function
 * @tparam T2 : type of input vector src2 which should support default constructor
 *              due to the usage of vector resize function
 * @param src1 : input vector src1
 * @param src2 : input vector src2
 * @param select_ratio : select ratio which should within range [0.0, 1.0]
 * @param output1 : selected partial vector of src1
 * @param output2 : selected partial vector of src2
 */
template <typename T1, typename T2>
void PostProcess::simultaneously_random_select(
    const std::vector<T1> &src1, const std::vector<T2> &src2, float select_ratio,
    std::vector<T1>& output1, std::vector<T2>& output2) {

    // check if select ratio is right
    if (select_ratio < 0.0 || select_ratio > 1.0) {
        // LOG(ERROR) << "Select ratio should be in range [0.0, 1.0]";
        return;
    }

    // calculate selected element counts using ceil to get
    // CHECK_EQ(src1.size(), src2.size());
    auto src_element_counts = src1.size();
    auto selected_elements_counts = static_cast<uint>(std::ceil(src_element_counts * select_ratio));
    // CHECK_LE(selected_elements_counts, src_element_counts);

    // random shuffle indexes
    std::vector<uint> indexes = std::vector<uint>(src_element_counts);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::random_shuffle(indexes.begin(), indexes.end());

    // select part of the elements via first selected_elements_counts index in random shuffled indexes vector
    output1.resize(selected_elements_counts);
    output2.resize(selected_elements_counts);

    for (uint i = 0; i < selected_elements_counts; ++i) {
        output1[i] = src1[indexes[i]];
        output2[i] = src2[indexes[i]];
    }
}

// cv::Mat PostProcess::applyDBSCAN(const cv::Mat& inputImage) {
//     // Extract pixel embeddings and normalize
//     cv::Mat floatImage;
//     inputImage.convertTo(floatImage, CV_32F);
    
//     cv::Mat pixelEmbeddings = floatImage.reshape(1, inputImage.rows * inputImage.cols);
//     cv::normalize(pixelEmbeddings, pixelEmbeddings, 0, 1, cv::NORM_MINMAX);

//     // Apply DBSCAN clustering
//     std::vector<int> labels;
//     int numClusters = cv::partition(pixelEmbeddings, labels, [&](const cv::Mat& a, const cv::Mat& b) {
//         double distance = cv::norm(a, b, cv::NORM_L2);
//         return distance < epsilon_;
//     });

//     // Create a colored visualization of clustered pixels
//     cv::Mat clusteredImage(inputImage.size(), CV_8UC3);
//     for (int y = 0; y < inputImage.rows; y++) {
//         for (int x = 0; x < inputImage.cols; x++) {
//             int idx = y * inputImage.cols + x;
//             int label = labels[idx];
//             clusteredImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(y, x) * (label + 1) * 10;
//         }
//     }

//     return clusteredImage;
}
