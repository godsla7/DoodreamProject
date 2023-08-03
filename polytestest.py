# 라인영역 관련 로직 + 최 외곽선, 그 안쪽 영역 반환하는 부분
import numpy as np
import cv2
import math

def hough_line_segments(mask_src):   # 라인 마스크 이미지 받아오기1
    # src = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE) # 주석처리 1
    src = cv2.cvtColor(mask_src, cv2.COLOR_BGR2GRAY)  # 가공을 위한 전처리 그레이 스케일로 변환 1
    if src is None:
        print('이미지 없는거임~~')
        return

    ################### 요기부터 ##############
    tt = cv2.Sobel(src,-1,1,0)
    tt2 = cv2.Sobel(tt, -1, 0, 1)
    edge = cv2.Canny(tt2, 50, 150)
    test = cv2.GaussianBlur(edge,(1,3) ,1)
    ############# 필터 테스트 한곳 ##############

    lines = cv2.HoughLinesP(test, 1, math.pi / 180, 30, minLineLength=10, maxLineGap=2)  # 허프변환 이용한 직선 탐지
    dst = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)  ## 가공할 이미지 배경
    a_list = []
    a_list2 = []
    b_list = []
    b_list2 = []
    if lines is not None:
        for i in range(lines.shape[0]):
            x1 = lines[i][0][0]  # x1 좌표
            y1 = lines[i][0][1]  # y1 좌표
            x2 = lines[i][0][2]  # x2 좌표
            y2 = lines[i][0][3]  # ㅛ2 좌표
            a = (y2 - y1) / (x2 - x1);  # 기울기
            b = (x2 * y1 - x1 * y2) / (x2 - x1);  # 절편

            if a>0: # 기울기가 양수일 경우
                a_list2.append(a)  # 기울기 추가
                b_list2.append(b)  # 절편 추가
                continue   # 얘 지우면 다그려짐 근데 결과 오류나니까 지우면 안되는거임~~
            elif a==0: # 기울기가 0인 경우 다음 완만한 기울기를 가져온다.
                continue
            else:
                # 기울기가 음수일 경우
                a_list.append(a) # 기울기 추가
                b_list.append(b) # 절편 추가

            # 모든 직선 찾기 ~~~~~~~~~~~~~~~~~~~~~인거임~~~~~~~~~~~~~~~~~~~~~~~~~
            # if a<0:
            #     cv2.line(dst, (0,int(b)),(-(int(b/a)),0), (0, 0, 255), 2, cv2.LINE_AA)
            #
            # else :
            #     print("(x1,y1),(x2,y2) : ", x1, y1, x2, y2)
            #     print("b ,(b/a): ", b, -int(b/a))
            #     a = (y2 - y1) / (x2 - x1);  # 기울기
            #     b = (x2 * y1 - x1 * y2) / (x2 - x1);
            #     cv2.line(dst, (0,int(b)), (src.shape[1] , int((src.shape[1]*a)+b)), (0, 0, 255), 2, cv2.LINE_AA)

    # cv2.imshow('dst', dst)  # 이미지 출력부
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # print("a_list",a_list)

    if a_list:
        min_a_idx = a_list.index(max(a_list))  # 기울기 음수인 직선 중 가장 기울기가 완만한 인덱스 가져오기
        min_a = a_list[min_a_idx]   # 기울기가 가장 완만한 직선의 기울기 가져오기
        min_b = b_list[min_a_idx]   # 기울기가 가장 완만한 직선의 절편 가져오기
        print("mina,minb", min_a,min_b)


    if a_list2:
        min_a_idx2 = a_list2.index(min(a_list2)) # 기울기 양수인 직선 중 가장 기울기가 완만한 인덱스 가져오기
        min_a2 = a_list2[min_a_idx2]  # 기울기가 가장 완만한 직선의 기울기 가져오기
        min_b2 = b_list2[min_a_idx2]  # 기울기가 가장 완만한 직선의 절편 가져오기
        print("mina2,minb2", min_a2, min_b2)

    if a_list and a_list2:
        a_list_bool=True
    else :
        a_list_bool = False

    # 직선 따로 그리기 위한 로직 1
    if a_list and a_list2 and min_a!=0:
        cv2.line(dst, (0, int(min_b)), (-(int(min_b / min_a)), 0), (255, 0, 255), 2, cv2.LINE_AA)  # 직선 그리기
        cv2.line(dst, (0, int(min_b2)), (src.shape[1], int((src.shape[1] * min_a2) + min_b2)), (255, 0, 0), 2,cv2.LINE_AA)  # 직선 그리기
    elif a_list and min_a!=0:
        cv2.line(dst, (0, int(min_b)), (-(int(min_b / min_a)), 0), (255, 0, 255), 2, cv2.LINE_AA)  # 직선 그리기
    elif a_list2 and min_a2!=0:
        cv2.line(dst, (0, int(min_b2)), (src.shape[1], int((src.shape[1] * min_a2) + min_b2)), (255, 0, 0), 2, cv2.LINE_AA)  # 직선 그리기
    else:
        pass
    # cv2.imshow('dst', dst)  # 이미지 출력부
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    line_img = dst.copy() # 직선만 그려진 이미지 저장1

    for h in range(dst.shape[0]): #h 높이   # 구현 완료된거 범위 지정하기
        for w in range(dst.shape[1]): #w 너비
            if a_list_bool:  #  양쪽 기울기 다 존재할 때
                if w*min_a+min_b < h and w*min_a2+min_b2 < h: # 조건 직선의 범위보다 y값이 큰 곳만 칠하기
                    dst[h][w] = 179  # 그 부분의 값을 임의의 값으로 채움
                    # list123.append([eee,ttt])  # 그 부분의 값을 리스트로 저장
                else:                     # 나머지 부분 지우는 로직
                    dst[h][w] = 0  # 나머지 부분을 0으로 채운다.

            elif a_list_bool==False and a_list: # 음수 기울기만 존재할 때
                if w*min_a+min_b < h : # 조건 직선의 범위보다 y값이 큰 곳만 칠하기
                    dst[h][w] = 179  # 그 부분의 값을 임의의 값으로 채움
                    # list123.append([eee,ttt])  # 그 부분의 값을 리스트로 저장

                else:                     # 나머지 부분 지우는 로직
                    dst[h][w] = 0  # 나머지 부분을 0으로 채운다.

            elif a_list_bool == False and a_list2: # 양수 기울기만 존재할 때
                if w*min_a2+min_b2 < h : # 조건 직선의 범위보다 y값이 큰 곳만 칠하기
                    dst[h][w] = 179  # 그 부분의 값을 임의의 값으로 채움
                    # list123.append([eee,ttt])  # 그 부분의 값을 리스트로 저장

                else:                     # 나머지 부분 지우는 로직
                    dst[h][w] = 0  # 나머지 부분을 0으로 채운다.
            else:
                dst[h][w] = 0

    bd_img = dst.copy()  # 라인 안쪽 영역 저장 1

    # cv2.imshow('line',line_img)
    # cv2.imshow('bd', bd_img)  # 이미지 출력부
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return  line_img, cv2.cvtColor(bd_img, cv2.COLOR_BGR2GRAY) # 라인 이미지, 라인 안쪽 영역 반환 1


# if __name__ == '__main__':
#     # hough_lines()
#     print(hough_line_segments().shape)
#
#     # hough_circles()