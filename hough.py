import numpy as np
import cv2
import math

def hough_line_segments():
    src = cv2.imread('C:/Users/USER/Desktop/8888.jpg', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    tt = cv2.Sobel(src,-1,1,0)               ################### 요기부터 ##############
    tt2 = cv2.Sobel(tt, -1, 0, 1)
    cv2.imshow('origin', src)
    cv2.imshow('tt2',tt2)
    cv2.waitKey(0)
    edge = cv2.Canny(tt2, 50, 150)
    test = cv2.GaussianBlur(edge,(1,3) ,1)     ############# 필터 테스트 한곳 ##############


    lines = cv2.HoughLinesP(test, 1, math.pi / 180, 30, minLineLength=10, maxLineGap=2)  # 허프변환 이용한 직선 탐지

    dst = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
    count =0
    a_list = []
    a_list2 = []
    b_list = []
    b_list2 = []
    if lines is not None:
        for i in range(lines.shape[0]):
            count +=1
            # pt1 = (lines[i][0][0], lines[i][0][1])
            # pt2 = (lines[i][0][2], lines[i][0][3])
            # cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
            x1 = lines[i][0][0]
            y1 = lines[i][0][1]
            x2 = lines[i][0][2]
            y2 = lines[i][0][3]
            a = (y2 - y1) / (x2 - x1);  # 기울기
            b = (x2 * y1 - x1 * y2) / (x2 - x1);  # 절편

            if a>0:
                a_list2.append(a)
                b_list2.append(b)
                continue
            a_list.append(a)
            b_list.append(b)

            # if a<0:
            #     cv2.line(dst, (0,int(b)),(-(int(b/a)),0), (0, 0, 255), 2, cv2.LINE_AA)
            #
            # else :
            #     print("(x1,y1),(x2,y2) : ", x1, y1, x2, y2)
            #     print("b ,(b/a): ", b, -int(b/a))
            #     a = (y2 - y1) / (x2 - x1);  # 기울기
            #     b = (x2 * y1 - x1 * y2) / (x2 - x1);
            #     cv2.line(dst, (0,int(b)), (src.shape[1] , int((src.shape[1]*a)+b)), (0, 0, 255), 2, cv2.LINE_AA)

    min_a_idx = a_list.index(max(a_list))
    min_a = a_list[min_a_idx]
    min_b = b_list[min_a_idx]
    print("mina,minb", min_a,min_b)
    cv2.line(dst, (0, int(min_b)), (-(int(min_b / min_a)), 0), (255, 0, 255), 2, cv2.LINE_AA)

    min_a_idx = a_list2.index(min(a_list2))
    min_a = a_list2[min_a_idx]
    min_b = b_list2[min_a_idx]
    print("mina,minb", min_a, min_b)
    cv2.line(dst, (0, int(min_b)), (src.shape[1], int((src.shape[1] * min_a) + min_b)), (255, 0, 0), 2, cv2.LINE_AA)

    print('count :',count )

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # hough_lines()
    hough_line_segments()
    # hough_circles()