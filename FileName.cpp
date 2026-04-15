#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

/**
 * 纯OpenCV实现手部检测与高亮
 * @param frame 输入帧（BGR格式）
 * @return 绘制高亮后的帧
 */
Mat detectAndHighlightHand(Mat& frame) {
    // 1. 预处理：高斯模糊降噪 + 转换为HSV色彩空间
    Mat blurred, hsv;
    GaussianBlur(frame, blurred, Size(5, 5), 0);
    cvtColor(blurred, hsv, COLOR_BGR2HSV);

    // 2. 肤色范围（HSV）：适配黄种人肤色，可根据环境微调
    Scalar lower_skin = Scalar(0, 48, 80);
    Scalar upper_skin = Scalar(20, 255, 255);

    // 3. 肤色掩码：只保留肤色区域
    Mat mask;
    inRange(hsv, lower_skin, upper_skin, mask);

    // 4. 形态学操作：消除噪声、填充手部孔洞
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);  // 开运算去噪点
    morphologyEx(mask, mask, MORPH_CLOSE, kernel); // 闭运算填孔洞

    // 5. 查找手部轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return frame; // 无手部轮廓，直接返回原帧
    }

    // 6. 筛选最大轮廓（默认最大的轮廓是手部）
    int max_idx = 0;
    double max_area = 0;
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_idx = i;
        }
    }

    // 过滤过小的轮廓（噪声）
    if (max_area < 10000) {
        return frame;
    }
    vector<Point> max_contour = contours[max_idx];

    // 7. 绘制手部轮廓（红色，线宽2）
    drawContours(frame, contours, max_idx, Scalar(0, 0, 255), 2);

    // 8. 凸包分析：找手部外轮廓凸包（用于定位指尖）
    vector<Point> hull;
    convexHull(max_contour, hull);
    // 绘制凸包（绿色，线宽2）
    drawContours(frame, vector<vector<Point>>{hull}, 0, Scalar(0, 255, 0), 2);

    // 9. 凸缺陷分析：找手指间隙，定位指尖
    vector<int> hull_indices;
    convexHull(max_contour, hull_indices, false); // returnPoints=false，返回索引
    vector<Vec4i> defects;
    convexityDefects(max_contour, hull_indices, defects);

    if (!defects.empty()) {
        // 遍历凸缺陷，筛选指尖关键点
        for (size_t i = 0; i < defects.size(); i++) {
            int s = defects[i][0]; // 起始点索引
            int e = defects[i][1]; // 结束点索引（指尖）
            int f = defects[i][2]; // 最远点索引（缺陷点）
            double d = defects[i][3] / 256.0; // 缺陷深度

            // 过滤过浅的缺陷，避免误判
            if (d > 20) {
                Point start = max_contour[s];
                Point end = max_contour[e];     // 指尖点
                Point far = max_contour[f];     // 缺陷点

                // 绘制指尖（蓝色圆点，半径5，填充）
                circle(frame, end, 5, Scalar(255, 0, 0), -1);
                // 绘制缺陷点（黄色小点，半径2）
                circle(frame, far, 2, Scalar(0, 255, 255), -1);
                // 绘制手指连线（青色，线宽1）
                line(frame, start, end, Scalar(255, 255, 0), 1);
            }
        }
    }

    return frame;
}

int main() {
    // 打开摄像头（0为默认摄像头）
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头！" << endl;
        return -1;
    }

    // 设置摄像头分辨率（可选，降低分辨率提升速度）
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    while (true) {
        Mat frame;
        bool ret = cap.read(frame);
        if (!ret) {
            cerr << "无法读取摄像头帧！" << endl;
            break;
        }

        // 镜像翻转（符合视觉习惯）
        flip(frame, frame, 1);

        // 检测并高亮手部
        frame = detectAndHighlightHand(frame);

        // 显示结果（中文标题在Windows下需适配系统编码，这里用英文避免乱码）
        imshow("Hand Highlight (OpenCV C++)", frame);

        // 按q退出（注意：C++中waitKey返回值是int，需转uchar判断）
        if (waitKey(1) & 0xFF == 'q') {
            break;
        }
    }

    // 释放资源
    cap.release();
    destroyAllWindows();
    return 0;
}