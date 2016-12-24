#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include <math.h>
#include<fstream>
#include<dirent.h>
using namespace std;
using namespace cv;


/** @function thresh_callback */
Mat thresh_callback(int, void* , Mat img)
{

  Mat src; Mat src_gray;
  int thresh = 30;
  int max_thresh = 255;
  RNG rng(12345);

  src = img;

  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );
  Mat canny_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using canny
  Canny( src_gray, canny_output, thresh, thresh*2, 3 );
  /// Find contours
  findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );

  double maxArea = 0;

  int maxCounterIndex = 0;

  for( int i = 0; i< contours.size(); i++ )
     {
       double cArea  = contourArea(contours[i]);
       if(maxArea < cArea) {maxArea = cArea; maxCounterIndex = i;}
      //  Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
      //  drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
     }

  Scalar color = Scalar( 0, 255, 0 );
  drawContours( drawing, contours, maxCounterIndex, color, 2, 8, hierarchy, 0, Point() );

  // imshow( "Contours", drawing );

  imwrite("output/con.png", drawing);

  return drawing;
}




Mat homo(Mat img) {
  vector<Point2f> inputs;
  inputs.push_back(Point2f(12, 8));inputs.push_back(Point2f(988, 139));
  inputs.push_back(Point2f(903, 578));inputs.push_back(Point2f(72, 461));

  vector<Point2f> outputs;
  outputs.push_back(Point2f(0, 0));outputs.push_back(Point2f( img.cols - 1, 0));
  outputs.push_back(Point2f(img.cols - 1, img.rows - 1));outputs.push_back(Point2f(0, img.rows - 1));

  Mat h = findHomography(inputs, outputs);
  Mat result;
  warpPerspective(img, result, h, img.size());
  imwrite("./output/homo.jpg", result);

  return result;
}

/** @function cornerHarris_demo */
std::vector<Point> cornerHarris_demo( int, void*, Mat img )
{
  Mat src, src_gray;
  int thresh = 220;
  int max_thresh = 255;
  src = img;
  cvtColor( src, src_gray, CV_BGR2GRAY );
  Mat dst, dst_norm, dst_norm_scaled;
  dst = Mat::zeros( src.size(), CV_32FC1 );
  /// Detector parameters
  int blockSize = 2;
  int apertureSize = 3;
  double k = 0.04;
  std::vector<Point> v;
  /// Detecting corners
  cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
  /// Normalizing
  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( dst_norm, dst_norm_scaled );
  /// Drawing a circle around corners
  for( int j = 0; j < dst_norm.rows ; j++ )
     { for( int i = 0; i < dst_norm.cols; i++ )
          {
            if( (int) dst_norm.at<float>(j,i) > thresh )
              {
                v.push_back(Point(i, j));
                circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
              }
          }
     }

  imshow("WIND", dst_norm_scaled);
  imwrite("output/corners.png", dst_norm);
  /// Showing the result
  return v;
}

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
Mat CannyThreshold(int, void*, Mat img)
{
  Mat src, src_gray;
  Mat dst, detected_edges;
  int edgeThresh = 1;
  int lowThreshold = 100;
  int const max_lowThreshold = 100;
  int ratio2 = 3;
  int kernel_size = 3;
  src = img;
  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );
  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );
  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );
  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, (lowThreshold)*(ratio2), kernel_size );
  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);
  src.copyTo( dst, detected_edges);
  return dst;
}

Mat convertImageToBinary() {
  Mat source = imread( "./images/L1.jpg", 0);
  for(int i = 0 ; i<source.rows ; i++){
    for(int j = 0 ;  j<source.cols ; j++){
      if(source.at<uchar>(i,j) < 127) {
          source.at<uchar>(i,j) = 0;
      } else {
        source.at<uchar>(i,j) = 255;
      }
    }
  }
  return source;
}

bool existsInVector(vector<uchar> v, uchar a) {
  for(int i = 0 ; i < v.size(); ++i) {
    if(v[i] == a){
      return true;
    }
  }
  return false;
}

vector<uchar> mergeVectors(vector<uchar> v1, vector<uchar> v2) {
  vector<uchar> r = v1 ;
  for(int i = 0 ; i<v2.size(); ++i){
    if(existsInVector(r, v2[i]) == 0) {
      r.push_back(v2[i]);
    }
  }
  return r;
}

vector<vector<uchar> > addByIntersection(vector<vector<uchar> > v1, vector<uchar> v2) {
  if(v1.size() == 0){
    v1.push_back(v2);
    return v1;
  }
  for(int i = 0 ; i < v1.size(); ++i) {
    vector<uchar> a = v1[i];
    for(int j = 0 ; j < a.size() ; ++j){
      if(existsInVector(v2, a[j])){
        v1[i] = mergeVectors(a, v2);
        return v1;
      }
    }
  }
  v1.push_back(v2);
  return v1;
}

uchar getMin(Vector<uchar> v) {
  int min = v[0];
  for(int j = 1 ; j < v.size() ; ++j) {
    if(v[j] < min) min = v[j];
  }
  return min;
}

bool sameRange(uchar x , uchar y){
  if(x >= 0 && x <= 63) return y >= 0 && y <= 63;
  if(x >= 64 && x <= 127 ) return y >= 64 && y <= 127;
  if(x >= 128 && x <= 191) return y >= 128 && y <= 191;
  if(x >= 192 && x <= 255) return y >= 192 && y <= 255;
  return false;
}

// Q1.a)
int countClassesBinaryImage() {
  Mat source = convertImageToBinary();
  // Mat source = (Mat_<uchar>(5,5) << 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1);
  Mat result = Mat::zeros(source.rows, source.cols, source.type());
  vector<vector<uchar> > equalClasses;
  int maxClass = 0;
  int removedClasses = 0;
  // first path
  for(int i  = 0 ; i < source.rows; ++i) {
    for(int j = 0 ; j < source.cols; ++j) {
      vector<uchar> v;
      if(i == 0 && j == 0){
        result.at<uchar>(i, j) = ++maxClass;
      } else {
        int nCol = j - 1;
        int nRow = i - 1;
        // left col
        if(nCol >= 0) {
          if(source.at<uchar>(i, nCol) == source.at<uchar>(i, j)) { v.push_back(result.at<uchar>(i, nCol));}
        }
        // upper row
        if(nRow >= 0) {
          if(source.at<uchar>(nRow, j) == source.at<uchar>(i, j)) {
            if(existsInVector(v, result.at<uchar>(nRow, j)) == 0){v.push_back(result.at<uchar>(nRow, j));}
          }
          if(nCol >= 0 ) {
            if(source.at<uchar>(nRow, nCol) == source.at<uchar>(i, j)) {
              if(existsInVector(v, result.at<uchar>(nRow, nCol)) == 0){v.push_back(result.at<uchar>(nRow, nCol));}
            }
          }
          if(j + 1  < source.cols ) {
            if(source.at<uchar>(nRow, j + 1) == source.at<uchar>(i, j)) {
              if(existsInVector(v, result.at<uchar>(nRow, j + 1)) == 0){v.push_back(result.at<uchar>(nRow, j + 1));}
            }
          }
        }
        //check for class
        if(v.size() == 0 ) {
          result.at<uchar>(i, j) = ++maxClass;
        } else if(v.size() == 1){
          result.at<uchar>(i, j) = v[0];
        } else {
          result.at<uchar>(i, j) = v[0];
          equalClasses = addByIntersection(equalClasses, v);
        }
      }
    }
  }
  // std::cout << "/* First path  */" << result << '\n';
  // std::cout << "/* First path  count */" << maxClass << '\n';
  for(int i = 0 ; i<equalClasses.size() ; ++i) {
    vector<uchar> a = equalClasses[i];
    removedClasses += a.size() - 1;
    int min = getMin(a);
    for(int j = 0 ; j < result.rows ; ++j){
      for(int k = 0 ; k < result.cols; ++k){
        if(existsInVector(a, result.at<uchar>(j, k))){
          result.at<uchar>(j, k) = min;
        }
      }
    }
  }
  // std::cout << "/* message */" << result << '\n';
  return maxClass - removedClasses;
}

// Q1. b)
int countClassesGrayScaleImage() {
  Mat source = homo(imread( "./images/L3.jpg", 0));
  // Mat source = (Mat_<uchar>(5,5) << 42, 43, 43, 44, 45, 43, 43, 44, 45, 45, 44, 44, 45, 46, 46, 44, 45, 46, 46, 47, 45, 46, 46, 47, 48);
  Mat result = Mat::zeros(source.rows, source.cols, source.type());
  vector<vector<uchar> > equalClasses;
  int maxClass = 0;
  int removedClasses = 0;
  // first path
  for(int i  = 0 ; i < source.rows; ++i) {
    for(int j = 0 ; j < source.cols; ++j) {
      vector<uchar> v;
      if(i == 0 && j == 0){
        result.at<uchar>(i, j) = ++maxClass;
      } else {
        int nCol = j - 1;
        int nRow = i - 1;
        // left col
        if(nCol >= 0) {
          if(sameRange(source.at<uchar>(i, nCol),source.at<uchar>(i, j))) { v.push_back(result.at<uchar>(i, nCol));}
        }
        // upper row
        if(nRow >= 0) {
          if(sameRange(source.at<uchar>(nRow, j), source.at<uchar>(i, j))) {
            if(existsInVector(v, result.at<uchar>(nRow, j)) == 0){v.push_back(result.at<uchar>(nRow, j));}
          }
          if(nCol >= 0 ) {
            if(sameRange(source.at<uchar>(nRow, nCol), source.at<uchar>(i, j))) {
              if(existsInVector(v, result.at<uchar>(nRow, nCol)) == 0){v.push_back(result.at<uchar>(nRow, nCol));}
            }
          }
          if(j + 1  < source.cols ) {
            if(sameRange(source.at<uchar>(nRow, j + 1), source.at<uchar>(i, j))) {
              if(existsInVector(v, result.at<uchar>(nRow, j + 1)) == 0){v.push_back(result.at<uchar>(nRow, j + 1));}
            }
          }
        }
        //check for class
        if(v.size() == 0 ) {
          result.at<uchar>(i, j) = ++maxClass;
        } else if(v.size() == 1){
          result.at<uchar>(i, j) = v[0];
        } else {
          result.at<uchar>(i, j) = v[0];
          equalClasses = addByIntersection(equalClasses, v);
        }
      }
    }
  }

  // std::cout << "/* Source  */" << source << '\n';
  //
  // std::cout << "/* First path  */" << result << '\n';

  for(int i = 0 ; i<equalClasses.size() ; ++i) {
    vector<uchar> a = equalClasses[i];
    removedClasses += a.size() - 1;
    int min = getMin(a);
    for(int j = 0 ; j < result.rows ; ++j){
      for(int k = 0 ; k < result.cols; ++k){
        if(existsInVector(a, result.at<uchar>(j, k))){
          result.at<uchar>(j, k) = min;
        }
      }
    }
  }
  // std::cout << "/* message */" << result << '\n';
  return maxClass - removedClasses;
}

Mat resizeToMatchL1(Mat img) {
  Mat dst ;
  Size dSize(669, 325);
  resize(img, dst, dSize, 0, 0, INTER_LINEAR );
  return dst;
}

// get front view of an image.
Mat getFrontView(Mat img) {
  Mat edgeImage = CannyThreshold(0, 0, img);
  //
  Mat cont = thresh_callback( 0, 0, img );
  Mat edge2 = CannyThreshold(0, 0, cont);
  // imwrite("i2.png", edge2);
  std::vector<Point> points = cornerHarris_demo( 0, 0, edgeImage );

  Point leftUpper( points[0].x, points[0].y);
  Point rightLower( points[points.size()-1].x, points[points.size()-1].y);

  int midCol = img.cols /2;
  int midRow = img.rows /2;
  Point rightUpper( points[0].x, points[0].y);
  Point leftLower( points[0].x, points[0].y);
  // // x is col, y is row.
  // std::cout << "/* POINTS */" <<  points << '\n';
  //
  // std::cout << "/* leftUpper */" <<  leftUpper << '\n';
  // std::cout << "/* rightUpper */" <<  rightUpper << '\n';
  // std::cout << "/* leftLower */" <<  leftLower << '\n';
  // std::cout << "/* rightLower */" <<  rightLower << '\n';
  for(int i =0; i<points.size(); ++i){
    if( points[i].x > midCol && points[i].x > rightUpper.x ) {rightUpper.x = points[i].x;  rightUpper.y = points[i].y;}
    if(points[i].x < leftLower.x && points[i].y > leftLower.y) {leftLower.x = points[i].x;  leftLower.y = points[i].y;}
  }
  //
  std::cout << "/* leftUpper */" <<  leftUpper << '\n';
  std::cout << "/* rightUpper */" <<  rightUpper << '\n';
  std::cout << "/* leftLower */" <<  leftLower << '\n';
  std::cout << "/* rightLower */" <<  rightLower << '\n';
  Mat x;
  return x;
}
void sepTemplate(Mat src) {
  Mat image = resizeToMatchL1(src);
  Rect R1 = Rect(42,125,98,195);Mat I1 = image(R1);imwrite("separated/I1.png",I1);
  Rect R2 = Rect(128,125,86,195);Mat I2 = image(R2);imwrite("separated/I2.png",I2);
  Rect R3 = Rect(198,125,102,195);Mat I3 = image(R3);imwrite("separated/I3.png",I3);
  Rect R4 = Rect(350,125,104,195);Mat I4 = image(R4);imwrite("separated/I4.png",I4);
  Rect R5 = Rect(447,125,104,195);Mat I5 = image(R5);imwrite("separated/I5.png",I5);
  Rect R6 = Rect(548,125,104,195);Mat I6 = image(R6);imwrite("separated/I6.png",I6);
}

std::vector<string> getLibraryNumbers(){
    std::vector<string> v;
    DIR *pDIR;
    struct dirent *entry;
    pDIR=opendir("./library/numbers");
    if(pDIR){
      entry = readdir(pDIR);
      while(entry){
        if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 )
        // cout << entry->d_name << "\n";
        v.push_back(entry->d_name);
        entry = readdir(pDIR);
      }
      closedir(pDIR);
    }
    return v;
}

std::vector<string> getLibraryLetters(){
    std::vector<string> v;
    DIR *pDIR;
    struct dirent *entry;
    pDIR=opendir("./library/letters");
    if(pDIR){
      entry = readdir(pDIR);
      while(entry){
        if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 )
        // cout << entry->d_name << "\n";
        v.push_back(entry->d_name);
        entry = readdir(pDIR);
      }
      closedir(pDIR);
    }
    return v;
}

double getMinValueMatching(Mat img1, Mat img2) {
  int img1Size = img1.cols * img1.rows;
  int img2Size = img2.cols * img2.rows;
  Mat i1, i2, result;
  if(img1Size < img2Size) {
    Size dSize(img1.cols, img1.rows);
    resize(img2, i2, dSize, 0, 0, INTER_LINEAR );
    i1 = img1;
    // i1 = img2;
    // i2 = img1;
  } else {
    Size dSize(img2.cols, img2.rows);
    resize(img1, i1, dSize, 0, 0, INTER_LINEAR );
    i2 = img2;
    // i1 = img1;
    // i2 = img2;
  }
  // int result_cols =  i1.cols - i2.cols + 1;
  // int result_rows = i1.rows - i2.rows + 1;
  // result.create( result_rows, result_cols, i1.type() );
  matchTemplate( i1, i2, result, 0 );
  // normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;
  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
  // std::cout << "/* message234 */ " << minVal << '\n';

  return minVal;
}


int getIndexMatching(std::vector<string> lib, Mat img, std::string type ){
  double minValue = 999999999;
  int minIndex = -1;
  for(int i = 0 ; i < lib.size() ; ++i){
    if(lib[i] == ".DS_Store") continue;
    std::string s = "./library/" + type + lib[i];
    // std::cout << "/* CONCAT  */" << s << '\n';
    // strcat (s, type);
    // strcat(s, lib[i]);
    Mat img2 = imread(s);

    // std::cout << "/* message */ " << img2.cols << '\n';
    double x = getMinValueMatching(img2, img);

    if( x < minValue){
      minValue = x;
      minIndex = i;
    }
  }
  return minIndex;
}

string getPlateLetters(Mat img){
  std::vector<string> letters = getLibraryLetters();
  std::vector<string> numbers = getLibraryNumbers();
  sepTemplate(img);
  Mat I1 = imread("./separated/I1.png");
  Mat I2 = imread("./separated/I2.png");
  Mat I3 = imread("./separated/I3.png");
  Mat I4 = imread("./separated/I4.png");
  Mat I5 = imread("./separated/I5.png");
  Mat I6 = imread("./separated/I6.png");
  int index1 = getIndexMatching(numbers, I1, "numbers/");
  int index2 = getIndexMatching(numbers, I2, "numbers/");
  int index3 = getIndexMatching(numbers, I3, "numbers/");
  int index4 = getIndexMatching(letters, I4, "letters/");
  int index5 = getIndexMatching(letters, I5, "letters/");
  int index6 = getIndexMatching(letters, I6, "letters/");
  // std::cout << "/* message */ " <<  index << '\n';
  // std::cout  << numbers[index1].erase(2,4) <<"." << '\n';
  // std::cout  << numbers[index2] <<"." << '\n';
  // std::cout  << numbers[index3] <<"." << '\n';
  // std::cout  << letters[index4] <<"." << '\n';
  // std::cout  << letters[index5] <<"." << '\n';
  // std::cout  << letters[index6] <<"." << '\n';
  string o = numbers[index1].erase(2,4) + numbers[index2].erase(2,4) + numbers[index3].erase(2,4) + letters[index4].erase(2,4) + letters[index5].erase(2,4) + letters[index6].erase(2,4);
  return o;
}

// void helperWindowsApp(Mat img) {

//   Mat i300_300, i358_173, i358_358, i1000_800, i414_180, i414_468, i558_558;
//   Mat i558_756, i846_468, i2400_1200, i480_800;
//   resize(img, i300_300, Size(300, 300), 0, 0, INTER_LINEAR );
//   imwrite("output/300_300.png", i300_300);

//   resize(img, i358_173, Size(358, 173), 0, 0, INTER_LINEAR );
//   imwrite("output/i358_173.png", i358_173);

//   resize(img, i358_358, Size(358, 358), 0, 0, INTER_LINEAR );
//   imwrite("output/i358_358.png", i358_358);

//   resize(img, i1000_800, Size(1000, 800), 0, 0, INTER_LINEAR );
//   imwrite("output/i1000_800.png", i1000_800);

//   resize(img, i414_180, Size(414, 180), 0, 0, INTER_LINEAR );
//   imwrite("output/i414_180.png", i414_180);

//   resize(img, i414_468, Size(414, 468), 0, 0, INTER_LINEAR );
//   imwrite("output/i414_468.png", i414_468);

//   resize(img, i558_558, Size(558, 558), 0, 0, INTER_LINEAR );
//   imwrite("output/i558_558.png", i558_558);

//   resize(img, i558_756, Size(558, 756), 0, 0, INTER_LINEAR );
//   imwrite("output/i558_756.png", i558_756);

//   resize(img, i846_468, Size(846, 468), 0, 0, INTER_LINEAR );
//   imwrite("output/i846_468.png", i846_468);

//   resize(img, i2400_1200, Size(2400, 1200), 0, 0, INTER_LINEAR );
//   imwrite("output/i2400_1200.png", i2400_1200);

//   resize(img, i480_800, Size(480, 800), 0, 0, INTER_LINEAR );
//   imwrite("output/i480_800.png", i480_800);

// }

int main( int argc, char** argv )
{
    //Q1.a)
    int x = countClassesBinaryImage();
    printf("No. of binary image classes : %d \n", x);
    // // //Q1.b()
    printf("No. of GrayScale image classes : %d \n", countClassesGrayScaleImage());


    // // Q2
    Mat c = imread("./images/L1.jpg", 1 );
    string s = getPlateLetters(c);
    std::cout << " Final template : " << s << '\n';
    // std::cout << "/* test */ "  <<  << '\n';
    // sepTemplate(c);

    // getFrontView(c);
    // std::vector<string> v = getLibrary();
    // std::cout << "/* message */ " << v[0]<< '\n';


    // Mat img = imread("./images/icon_x310.png");
    // helperWindowsApp(img);

    printf("DONE\n");

    waitKey();
    return 0;
}
