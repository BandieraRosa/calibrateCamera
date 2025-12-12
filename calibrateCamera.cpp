#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * 求解齐次线性方程组 Ax = 0
 * 使用奇异值分解(SVD)方法找到最小奇异值对应的右奇异向量作为解
 *
 * @param A 系数矩阵 (m×n矩阵)
 * @return 齐次方程组的解向量 (n维向量)，已进行归一化处理
 */
static Eigen::VectorXd solve_homogeneous_system(const Eigen::MatrixXd& A)
{
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::MatrixXd& v = svd.matrixV();
  Eigen::VectorXd h = v.col(v.cols() - 1);

  if (std::abs(h[h.size() - 1]) > 1e-12)
  {
    h /= h[h.size() - 1];
  }
  return h;
}

/**
 * 使用直接线性变换(DLT)算法计算单应性矩阵H
 * 单应性矩阵描述了从世界坐标系到图像坐标系的投影变换关系
 *
 * 对于每个点对应关系，建立两个线性方程：
 * u = (h1*xw + h2*yw + h3) / (h7*xw + h8*yw + h9)
 * v = (h4*xw + h5*yw + h6) / (h7*xw + h8*yw + h9)
 *
 * 通过交叉相乘转换为线性齐次方程组 Ah = 0，其中h为单应性矩阵的9个参数组成的向量
 *
 * @param obj_pts 世界坐标系中的3D点 (实际上z坐标为0，即平面点)
 * @param img_pts 图像坐标系中的2D点
 * @return 3x3的单应性矩阵H
 */
static Eigen::Matrix3d compute_homography_dlt(const std::vector<cv::Point3f>& obj_pts,
                                              const std::vector<cv::Point2f>& img_pts)
{
  int n = static_cast<int>(obj_pts.size());
  Eigen::MatrixXd a(2 * n, 9);

  for (int k = 0; k < n; ++k)
  {
    double xw = obj_pts[k].x;
    double yw = obj_pts[k].y;
    double u = img_pts[k].x;
    double v = img_pts[k].y;

    a(2 * k, 0) = xw;
    a(2 * k, 1) = yw;
    a(2 * k, 2) = 1.0;
    a(2 * k, 3) = 0.0;
    a(2 * k, 4) = 0.0;
    a(2 * k, 5) = 0.0;
    a(2 * k, 6) = -xw * u;
    a(2 * k, 7) = -yw * u;
    a(2 * k, 8) = -u;

    a(2 * k + 1, 0) = 0.0;
    a(2 * k + 1, 1) = 0.0;
    a(2 * k + 1, 2) = 0.0;
    a(2 * k + 1, 3) = xw;
    a(2 * k + 1, 4) = yw;
    a(2 * k + 1, 5) = 1.0;
    a(2 * k + 1, 6) = -xw * v;
    a(2 * k + 1, 7) = -yw * v;
    a(2 * k + 1, 8) = -v;
  }

  Eigen::VectorXd h = solve_homogeneous_system(a);
  Eigen::Matrix3d H;
  H << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
  return H;
}

/**
 * 构造相机标定中的约束向量v_ij
 * 该向量用于构建关于相机内参的线性方程组
 *
 * 在相机标定中，对于单应性矩阵H的任意两列h_i和h_j，
 * 它们与相机内参矩阵K满足以下约束条件：
 * h_i^T * K^(-T) * K^(-1) * h_j = 0  (当i≠j时)
 * h_i^T * K^(-T) * K^(-1) * h_j = 1  (当i=j时，且已归一化)
 *
 * 通过定义B = K^(-T) * K^(-1) = [B11, B12, B13;
 *                                B12, B22, B23;
 *                                B13, B23, B33]
 * 可以将上述约束转化为关于B的线性方程，其中B有6个独立元素
 *
 * @param hi 单应性矩阵H的第i列 (3维向量)
 * @param hj 单应性矩阵H的第j列 (3维向量)
 * @return 6维约束向量v_ij，包含h_i和h_j组合后的二次项系数
 */
static Eigen::Matrix<double, 6, 1> make_v_ij(const Eigen::Vector3d& hi,
                                             const Eigen::Vector3d& hj)
{
  Eigen::Matrix<double, 6, 1> v;
  v(0) = hi(0) * hj(0);
  v(1) = hi(0) * hj(1) + hi(1) * hj(0);
  v(2) = hi(1) * hj(1);
  v(3) = hi(0) * hj(2) + hi(2) * hj(0);
  v(4) = hi(1) * hj(2) + hi(2) * hj(1);
  v(5) = hi(2) * hj(2);
  return v;
}

/**
 * @brief 样本度量结构体，用于存储棋盘格检测的相关指标
 *
 * 该结构体包含了棋盘格相对于图像的各种几何特征，
 * 用于评估标定样本的质量。
 */
struct SampleMetrics
{
  double x_off;
  double y_off;
  double size;
  double skew;
};

/**
 * @brief 计算棋盘格样本的度量指标
 *
 * 该函数分析检测到的棋盘格角点，计算四个关键指标：
 * 1. 中心偏移量：棋盘格中心相对于图像中心的偏移
 * 2. 尺寸比例：棋盘格占据图像的面积比例
 * 3. 倾斜程度：棋盘格平面相对于图像平面的倾斜程度
 *
 * @param corners 检测到的棋盘格角点坐标列表
 * @param imgSize 图像尺寸
 * @param board_cols 棋盘格列数（横向角点数）
 * @param board_rows 棋盘格行数（纵向角点数）
 * @return SampleMetrics 包含x_off, y_off, size, skew四个指标的结构体
 */
static SampleMetrics compute_sample_metrics(const std::vector<cv::Point2f>& corners,
                                            const cv::Size& imgSize, int board_cols,
                                            int board_rows)
{
  SampleMetrics m{0, 0, 0, 0};

  if (corners.empty())
  {
    return m;
  }

  double minX = 1e9, maxX = -1e9;
  double minY = 1e9, maxY = -1e9;
  for (auto& p : corners)
  {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }

  double centerX = 0.5 * (minX + maxX);
  double centerY = 0.5 * (minY + maxY);

  double normCX = std::abs(centerX - imgSize.width * 0.5) / (imgSize.width * 0.5);
  double normCY = std::abs(centerY - imgSize.height * 0.5) / (imgSize.height * 0.5);

  m.x_off = std::min(1.0, std::max(0.0, normCX));
  m.y_off = std::min(1.0, std::max(0.0, normCY));

  double area = (maxX - minX) * (maxY - minY);
  double normSize = area / (imgSize.width * imgSize.height);
  m.size = std::min(1.0, std::max(0.0, normSize));

  if (corners.size() >= (size_t)(board_cols * board_rows))
  {
    cv::Point2f p00 = corners[0];
    cv::Point2f p01 = corners[board_cols - 1];
    cv::Point2f p10 = corners[(board_rows - 1) * board_cols];

    cv::Point2f vx = p01 - p00;
    cv::Point2f vy = p10 - p00;

    double ang_x = std::atan2(vx.y, vx.x);
    double ang_y = std::atan2(vy.y, vy.x);

    double norm_row = std::min(1.0, std::abs(ang_x) / (CV_PI / 4));
    double norm_col = std::min(1.0, std::abs(ang_y - CV_PI / 2) / (CV_PI / 4));
    m.skew = std::min(1.0, std::max(norm_row, norm_col));
  }

  return m;
}

/**
 * @brief 判断当前样本是否为优质样本
 *
 * 该函数通过计算当前样本与已有样本集合中所有样本的欧氏距离，
 * 来判断当前样本是否具有足够的多样性。只有当当前样本与所有已有样本
 * 的距离都大于设定阈值时，才认为是一个好的样本。
 *
 * 这种机制确保了在相机标定过程中收集到的样本具有良好的分布性，
 * 避免收集过多相似姿态的样本，从而提高标定精度。
 *
 * @param cur 当前样本的度量指标
 * @param samples 已有的样本度量指标集合
 * @return bool 如果当前样本与已有样本足够不同则返回true，否则返回false
 */
static bool is_good_sample(const SampleMetrics& cur,
                           const std::vector<SampleMetrics>& samples)
{
  if (samples.empty())
  {
    return true;
  }
  const double MIN_DIST = 0.3;

  double min_dist = 1e9;
  for (const auto& s : samples)
  {
    double dx = cur.x_off - s.x_off;
    double dy = cur.y_off - s.y_off;
    double ds = cur.size - s.size;
    double dk = cur.skew - s.skew;
    double dist = std::sqrt(dx * dx + dy * dy + ds * ds + dk * dk);
    if (dist < min_dist)
    {
      min_dist = dist;
    }
  }
  return (min_dist > MIN_DIST);
}

/**
 * @brief 在图像上绘制进度条，显示样本覆盖度指标
 *
 * 该函数在图像的指定位置绘制四个水平进度条，分别表示：
 * 1. X方向位置覆盖度
 * 2. Y方向位置覆盖度
 * 3. 尺寸覆盖度
 * 4. 倾斜度覆盖度
 *
 * 进度条以绿色填充，白色文字标签，灰色边框，便于用户直观了解当前样本
 * 在各个维度上的分布情况。
 *
 * @param img 要绘制进度条的图像（引用方式传递，会直接修改原图）
 * @param covX X方向位置覆盖度 [0,1]
 * @param covY Y方向位置覆盖度 [0,1]
 * @param covSize 尺寸覆盖度 [0,1]
 * @param covSkew 倾斜度覆盖度 [0,1]
 */
static void draw_progress_bars(cv::Mat& img, double covX, double covY, double covSize,
                               double covSkew)
{
  int bar_left = 20;
  int bar_top = 50;
  int bar_width = 220;
  int bar_height = 15;
  int gap = 8;

  struct BarInfo
  {
    std::string label;
    double value;
  };

  std::vector<BarInfo> bars = {
      {"X-pos", covX}, {"Y-pos", covY}, {"Size", covSize}, {"Skew", covSkew}};

  for (int i = 0; i < static_cast<int>(bars.size()); ++i)
  {
    int y = bar_top + i * (bar_height + gap);
    double v = std::max(0.0, std::min(1.0, bars[i].value));
    int filled = static_cast<int>(v * bar_width);

    cv::Rect outer(bar_left, y, bar_width, bar_height);
    cv::rectangle(img, outer, cv::Scalar(200, 200, 200), 1);

    cv::Rect inner(bar_left, y, filled, bar_height);
    cv::rectangle(img, inner, cv::Scalar(0, 255, 0), cv::FILLED);

    cv::putText(img, bars[i].label,
                cv::Point(bar_left + bar_width + 10, y + bar_height - 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  }
}

/**
 * @brief 张正友相机标定算法
 *
 * 1. 计算每个视图的单应性矩阵
 * 2. 通过约束条件求解相机内参
 * 3. 计算每个视图的外参（旋转和平移）
 * 4. 估计径向畸变参数
 *
 * @param object_points_all 多个视图的世界坐标点集合
 * @param image_points_all 多个视图的图像坐标点集合
 * @param cameraMatrix 输出的相机内参矩阵3x3
 * @param distCoeffs 输出的畸变系数向量
 * @return bool 标定成功返回true，失败返回false
 */
static bool calibrate(const std::vector<std::vector<cv::Point3f>>& object_points_all,
                      const std::vector<std::vector<cv::Point2f>>& image_points_all,
                      cv::Mat& cameraMatrix, cv::Mat& distCoeffs)
{
  int num_views = static_cast<int>(object_points_all.size());
  if (num_views < 3 || image_points_all.size() != static_cast<size_t>(num_views))
  {
    std::cerr << "视图数量不足或点数不匹配。" << '\n';
    return false;
  }

  std::vector<Eigen::Matrix3d> H_list;
  H_list.reserve(num_views);

  for (int i = 0; i < num_views; ++i)
  {
    const auto& obj_pts = object_points_all[i];
    const auto& img_pts = image_points_all[i];
    if (obj_pts.size() != img_pts.size() || obj_pts.empty())
    {
      std::cerr << "视图 " << i << " 点数不匹配或为空。" << '\n';
      return false;
    }
    Eigen::Matrix3d H = compute_homography_dlt(obj_pts, img_pts);
    H_list.push_back(H);
  }

  int valid_views = static_cast<int>(H_list.size());
  if (valid_views < 3)
  {
    std::cerr << "[calibrateZhang] 有效视图 < 3。" << '\n';
    return false;
  }

  Eigen::MatrixXd V(2 * valid_views, 6);

  for (int i = 0; i < valid_views; ++i)
  {
    const Eigen::Matrix3d& h = H_list[i];
    Eigen::Vector3d h1 = h.col(0);
    Eigen::Vector3d h2 = h.col(1);

    Eigen::Matrix<double, 6, 1> v12 = make_v_ij(h1, h2);
    Eigen::Matrix<double, 6, 1> v11 = make_v_ij(h1, h1);
    Eigen::Matrix<double, 6, 1> v22 = make_v_ij(h2, h2);

    V.row(2 * i) = v12.transpose();
    V.row(2 * i + 1) = (v11 - v22).transpose();
  }

  Eigen::VectorXd b = solve_homogeneous_system(V);

  double B11 = b(0);
  double B12 = b(1);
  double B22 = b(2);
  double B13 = b(3);
  double B23 = b(4);
  double B33 = b(5);

  double v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);
  double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
  double alpha = std::sqrt(lambda / B11);
  double beta = std::sqrt(lambda * B11 / (B11 * B22 - B12 * B12));
  double gamma = -B12 * alpha * alpha * beta / lambda;  // skew
  double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;

  Eigen::Matrix3d A_eig;
  A_eig << alpha, gamma, u0, 0.0, beta, v0, 0.0, 0.0, 1.0;

  std::cout << "\n[Zhang] 内参矩阵 A = \n" << A_eig << '\n';
  std::cout << "[Zhang] fx=" << alpha << ", fy=" << beta << ", skew=" << gamma
            << ", cx=" << u0 << ", cy=" << v0 << '\n';

  cameraMatrix =
      (cv::Mat_<double>(3, 3) << alpha, gamma, u0, 0.0, beta, v0, 0.0, 0.0, 1.0);

  std::vector<Eigen::Matrix3d> R_list;
  std::vector<Eigen::Vector3d> t_list;
  R_list.reserve(valid_views);
  t_list.reserve(valid_views);

  Eigen::Matrix3d A_inv = A_eig.inverse();

  for (int i = 0; i < valid_views; ++i)
  {
    const Eigen::Matrix3d& H = H_list[i];
    Eigen::Vector3d h1 = H.col(0);
    Eigen::Vector3d h2 = H.col(1);
    Eigen::Vector3d h3 = H.col(2);

    Eigen::Vector3d r1_tilde = A_inv * h1;
    Eigen::Vector3d r2_tilde = A_inv * h2;
    Eigen::Vector3d t_tilde = A_inv * h3;

    double norm_r1 = r1_tilde.norm();
    double norm_r2 = r2_tilde.norm();
    double lambda_rt = (norm_r1 + norm_r2) / 2.0;

    Eigen::Vector3d r1 = r1_tilde / lambda_rt;
    Eigen::Vector3d r2 = r2_tilde / lambda_rt;
    Eigen::Vector3d r3 = r1.cross(r2);
    Eigen::Vector3d t = t_tilde / lambda_rt;

    Eigen::Matrix3d R;
    R.col(0) = r1;
    R.col(1) = r2;
    R.col(2) = r3;

    R_list.push_back(R);
    t_list.push_back(t);
  }

  size_t total_points = 0;
  for (int i = 0; i < valid_views; ++i)
  {
    total_points += object_points_all[i].size();
  }

  Eigen::MatrixXd D(2 * total_points, 2);
  Eigen::VectorXd E_vec(2 * total_points);
  size_t row = 0;

  for (int i = 0; i < valid_views; ++i)
  {
    const auto& obj_pts = object_points_all[i];
    const auto& img_pts = image_points_all[i];
    const Eigen::Matrix3d& R = R_list[i];
    const Eigen::Vector3d& t = t_list[i];

    for (size_t k = 0; k < obj_pts.size(); ++k)
    {
      double Xw = obj_pts[k].x;
      double Yw = obj_pts[k].y;
      double Zw = obj_pts[k].z;

      Eigen::Vector3d Pw(Xw, Yw, Zw);
      Eigen::Vector3d Pc = R * Pw + t;
      double Xc = Pc(0);
      double Yc = Pc(1);
      double Zc = Pc(2);

      double x = Xc / Zc;
      double y = Yc / Zc;
      double r2 = x * x + y * y;
      double r4 = r2 * r2;

      double alpha = A_eig(0, 0);
      double gamma = A_eig(0, 1);
      double u0 = A_eig(0, 2);
      double beta = A_eig(1, 1);
      double v0 = A_eig(1, 2);

      double u_prime = alpha * x + gamma * y + u0;
      double v_prime = beta * y + v0;

      double u_meas = img_pts[k].x;
      double v_meas = img_pts[k].y;

      double du = u_meas - u_prime;
      double dv = v_meas - v_prime;

      double cx = u0;
      double cy = v0;

      D(row, 0) = (u_prime - cx) * r2;
      D(row, 1) = (u_prime - cx) * r4;
      E_vec(row) = du;
      row++;

      D(row, 0) = (v_prime - cy) * r2;
      D(row, 1) = (v_prime - cy) * r4;
      E_vec(row) = dv;
      row++;
    }
  }

  Eigen::Vector2d k_radial = (D.transpose() * D).ldlt().solve(D.transpose() * E_vec);

  double k1 = k_radial(0);
  double k2 = k_radial(1);

  std::cout << "\n径向畸变: k1=" << k1 << ", k2=" << k2 << '\n';

  distCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, 0.0, 0.0, 0.0);

  return true;
}

/**
 * @brief 将相机标定结果保存到YAML文件
 *
 * 该函数将相机内参矩阵、畸变系数等标定结果以YAML格式保存到文件中，
 * 以便后续使用或与其他系统共享标定参数。
 *
 * YAML文件格式兼容ROS相机标定文件格式，包含以下字段：
 * - image_width: 图像宽度
 * - image_height: 图像高度
 * - camera_name: 相机名称
 * - camera_matrix: 相机内参矩阵
 * - distortion_model: 畸变模型类型
 * - distortion_coefficients: 畸变系数
 *
 * @param filename 要保存的YAML文件名
 * @param cameraMatrix 相机内参矩阵3x3
 * @param distCoeffs 畸变系数向量
 * @param imageSize 图像尺寸
 * @return bool 保存成功返回true，失败返回false
 */
static bool save_to_yaml(const std::string& filename, const cv::Mat& cameraMatrix,
                         const cv::Mat& distCoeffs, const cv::Size& imageSize)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  if (!fs.isOpened())
  {
    std::cerr << "打开文件失败: " << filename << '\n';
    return false;
  }

  fs << "image_width" << imageSize.width;
  fs << "image_height" << imageSize.height;
  fs << "camera_name" << "camera";
  fs << "camera_matrix" << cameraMatrix;
  fs << "distortion_model" << "plumb_bob";
  fs << "distortion_coefficients" << distCoeffs;

  fs.release();
  std::cout << "已写入 " << filename << '\n';
  return true;
}

int main(int argc, char** argv)
{
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " <cols> <rows> <square_size>\n";
    return -1;
  }

  int cols = std::stoi(argv[1]);
  int rows = std::stoi(argv[2]);
  double square_size = std::stod(argv[3]);
  int camera = 0;

  cv::Size board_size(cols, rows);

  cv::VideoCapture cap(camera);
  if (!cap.isOpened())
  {
    std::cerr << "无法打开摄像头 " << camera << '\n';
    return -1;
  }

  std::cout << "棋盘角点: " << cols << "x" << rows << ", 小格边长: " << square_size
            << '\n';

  std::vector<cv::Point3f> object_points_template;
  object_points_template.reserve(cols * rows);
  for (int j = 0; j < rows; ++j)
  {
    for (int i = 0; i < cols; ++i)
    {
      float Xw = static_cast<float>(i * square_size);
      float Yw = static_cast<float>(j * square_size);
      float Zw = 0.0f;
      object_points_template.emplace_back(Xw, Yw, Zw);
    }
  }

  std::vector<std::vector<cv::Point3f>> object_points_all;
  std::vector<std::vector<cv::Point2f>> image_points_all;
  std::vector<SampleMetrics> sample_metrics;

  double cov_x = 0.0, cov_y = 0.0, cov_size = 0.0, cov_skew = 0.0;

  bool calibrated = false;
  bool auto_sampling = true;
  cv::Mat cameraMatrix, distCoeffs;
  cv::Size image_size;

  cv::namedWindow("live", cv::WINDOW_NORMAL);
  cv::namedWindow("undistorted", cv::WINDOW_NORMAL);

  while (true)
  {
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty())
    {
      std::cerr << "摄像头读取失败，退出。" << '\n';
      break;
    }
    image_size = frame.size();

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 检测棋盘
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(
        gray, board_size, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    SampleMetrics curMetrics{0, 0, 0, 0};
    bool gotMetrics = false;

    if (found)
    {
      cv::cornerSubPix(
          gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30,
                           0.001));
      cv::drawChessboardCorners(frame, board_size, corners, found);

      curMetrics = compute_sample_metrics(corners, frame.size(), cols, rows);
      gotMetrics = true;

      // 自动采样
      if (auto_sampling && !calibrated)
      {
        if (is_good_sample(curMetrics, sample_metrics))
        {
          object_points_all.push_back(object_points_template);
          image_points_all.push_back(corners);
          sample_metrics.push_back(curMetrics);

          cov_x = std::max(cov_x, curMetrics.x_off);
          cov_y = std::max(cov_y, curMetrics.y_off);
          cov_size = std::max(cov_size, curMetrics.size);
          cov_skew = std::max(cov_skew, curMetrics.skew);

          std::cout << "[自动采样] 新样本加入。当前样本数: " << object_points_all.size()
                    << '\n';
        }
      }
    }

    std::string info =
        "Views: " + std::to_string(object_points_all.size()) +
        " | [s]=save [c]=calib [w]=yaml [a]=auto:" + (auto_sampling ? "ON" : "OFF") +
        " [q]=quit";
    cv::putText(frame, info, cv::Point(20, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);

    if (calibrated)
    {
      cv::putText(frame, "CALIBRATED", cv::Point(20, 45), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  cv::Scalar(0, 0, 255), 2);
    }

    draw_progress_bars(frame, cov_x, cov_y, cov_size, cov_skew);

    cv::imshow("live", frame);

    if (calibrated)
    {
      cv::Mat undistorted;
      cv::undistort(frame, undistorted, cameraMatrix, distCoeffs);
      cv::imshow("undistorted", undistorted);
    }

    char key = static_cast<char>(cv::waitKey(10));
    if (key == 'q' || key == 27)
    {
      break;
    }
    else if (key == 'a')
    {
      auto_sampling = !auto_sampling;
      std::cout << "[自动采样] 现在: " << (auto_sampling ? "开启" : "关闭") << '\n';
    }
    else if (key == 's')
    {
      if (found && !calibrated)
      {
        object_points_all.push_back(object_points_template);
        image_points_all.push_back(corners);
        SampleMetrics m = gotMetrics
                              ? curMetrics
                              : compute_sample_metrics(corners, frame.size(), cols, rows);
        sample_metrics.push_back(m);

        cov_x = std::max(cov_x, m.x_off);
        cov_y = std::max(cov_y, m.y_off);
        cov_size = std::max(cov_size, m.size);
        cov_skew = std::max(cov_skew, m.skew);

        std::cout << "已保存样本数: " << object_points_all.size() << '\n';
      }
      else
      {
        std::cout << "当前无棋盘或已标定，无法保存。" << '\n';
      }
    }
    else if (key == 'c')
    {
      if (object_points_all.size() < 3)
      {
        std::cout << "样本不足，至少需要 3 帧。" << '\n';
        continue;
      }
      std::cout << "使用 " << object_points_all.size() << " 帧样本，开始标定..." << '\n';

      bool ok = calibrate(object_points_all, image_points_all, cameraMatrix, distCoeffs);
      if (ok)
      {
        calibrated = true;
        std::cout << "\n=== 标定完成 ===\n";
        std::cout << "cameraMatrix = \n" << cameraMatrix << '\n';
        std::cout << "distCoeffs   = \n" << distCoeffs << '\n';
        std::cout << "现在可按 'w' 写入 camera_param.yaml\n";
      }
      else
      {
        std::cout << "失败，请调整采样后重试。" << '\n';
      }
    }
    else if (key == 'w')
    {
      if (!calibrated)
      {
        std::cout << "尚未标定，先按 'c' 标定。" << '\n';
      }
      else
      {
        save_to_yaml("camera_param.yaml", cameraMatrix, distCoeffs, image_size);
      }
    }
  }

  return 0;
}