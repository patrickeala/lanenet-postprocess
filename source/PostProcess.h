#ifndef POST_PROCESS_H
#define POST_PROCESS_H

#include <opencv2/opencv.hpp>
#include "dbscan.hpp"

namespace postprocess {
using DBSCAMSample = DBSCAMSample<float>;
using Feature = Feature<float>;

class PostProcess {
public:
    PostProcess(const std::string& input_dir, const std::string& idx);

    void loadNpyImages(const std::string& input_dir, const std::string& idx);
    void visualize_Binary() const;

    void process(cv::Mat &binary_seg_result, cv::Mat &instance_seg_result);

    /***
     * Gather embedding features via binary segmentation mask
     * @param binary_mask
     * @param pixel_embedding
     * @param coords
     * @param embedding_features
     */
    void gather_pixel_embedding_features(const cv::Mat& binary_mask, const cv::Mat& pixel_embedding,
                                         std::vector<cv::Point>& coords, std::vector<DBSCAMSample>& embedding_samples);

    /***
     * Cluster pixel embedding features via DBSCAN
     * @param embedding_samples
     * @param cluster_ret
     */
    void cluster_pixem_embedding_features(std::vector<DBSCAMSample>& embedding_samples,
                                          std::vector<std::vector<uint> >& cluster_ret, std::vector<uint>& noise);

    /***
     * Visualize instance segmentation result
     * @param cluster_ret
     * @param coords
     * @param instance_segmentation_result
     */
    static void visualize_instance_segmentation_result(const std::vector<std::vector<uint> >& cluster_ret,
            const std::vector<cv::Point>& coords, cv::Mat& instance_segmentation_result);

    /***
     * Normalize input samples' feature. Each sample's feature is normalized via function as follows:
     * feature[i] = (feature[i] - mean_feature_vector[i]) / stddev_feature_vector[i].
     * @param input_samples : vector of samples whose feature vector need to be normalized
     * @param output_samples : normalized result
     */
    static void normalize_sample_features(const std::vector<DBSCAMSample >& input_samples,
                                          std::vector<DBSCAMSample >& output_samples);

    /***
     * Calculate the mean feature vector among a vector of DBSCAMSample samples
     * @param input_samples : vector of DBSCAMSample samples
     * @return : mean feature vector
     */
    static Feature calculate_mean_feature_vector(const std::vector<DBSCAMSample >& input_samples);

    /***
     * Calculate the stddev feature vector among a vector of DBSCAMSample samples
     * @param input_samples : vector of DBSCAMSample samples
     * @param mean_feature_vec : mean feature vector
     * @return : stddev feature vector
     */
    static Feature calculate_stddev_feature_vector(
        const std::vector<DBSCAMSample >& input_samples,
        const Feature& mean_feature_vec);

    // /***
    //  * simultaneously random shuffle two vector inplace. The two input source vector should have the same size.
    //  * @tparam T
    //  * @param src1
    //  * @param src2
    //  */
    template <typename T1, typename T2>
    static void simultaneously_random_shuffle(std::vector<T1> src1, std::vector<T2> src2);

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
    static void simultaneously_random_select(const std::vector<T1>& src1, const std::vector<T2>& src2,
            float select_ratio, std::vector<T1>& output1, std::vector<T2>& output2);

    // cv::Mat applyDBSCAN(const cv::Mat& inputImage);

private:
    // double epsilon_;
    cv::Mat binary_mat; // Loaded image
    cv::Mat pix_embedding_output_mat; // Loaded image

    // embedding features dilution ratio
    float _m_embedding_features_dilution_ratio = 0.5;
    // float _m_embedding_features_dilution_ratio = 1.0;
    // MNN Lanenet input graph node tensor size
    cv::Size _m_input_node_size_host;


    // lanenet pixel embedding feature dims
    uint _m_lanenet_pix_embedding_feature_dims=4;

    // Dbscan eps threshold
    float _m_dbscan_eps = 0.4;
    // dbscan min pts threshold
    uint _m_dbscan_min_pts = 500;




    // // MNN Lanenet model file path
    // std::string _m_lanenet_model_file_path = "";
    // // MNN Lanenet model interpreter
    // std::unique_ptr<MNN::Interpreter> _m_lanenet_model = nullptr;
    // // MNN Lanenet model session
    // MNN::Session* _m_lanenet_session = nullptr;
    // // MNN Lanenet model input tensor
    // MNN::Tensor* _m_input_tensor_host = nullptr;
    // // MNN Lanenet model binary output tensor
    // MNN::Tensor* _m_binary_output_tensor_host = nullptr;
    // // MNN Lanenet model pixel embedding output tensor
    // MNN::Tensor* _m_pix_embedding_output_tensor_host = nullptr;
    // // MNN Lanenet input graph node tensor size
    // cv::Size _m_input_node_size_host;
    // // lanenet pixel embedding feature dims
    // uint _m_lanenet_pix_embedding_feature_dims=4;
    // // Dbscan eps threshold
    // float _m_dbscan_eps = 0.0;
    // // dbscan min pts threshold
    // uint _m_dbscan_min_pts = 0;

    // // successfully init model flag
    // bool _m_successfully_initialized = false;



};
}
#endif // POST_PROCESS_H