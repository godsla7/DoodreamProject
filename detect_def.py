# 무단 횡단자 찾아내는 로직
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from polytestest import hough_line_segments as hl


def project_draw(det,save_txt,gn,save_conf,txt_path,save_img,save_crop,view_img,hide_labels,names,hide_conf,im0,annotator,imc,save_dir,drivable_list, line_list):  # 1
    for *xyxy, conf, cls in reversed(det):
        if save_txt:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(f'{txt_path}.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
        if save_img or save_crop or view_img:  # Add bbox to image
            c = int(cls)  # integer class
            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            rg_x1 = xyxy[0]
            rg_y1 = xyxy[1]
            rg_x2 = xyxy[2]
            rg_y2 = xyxy[3]
            rg_center_dotX = int(rg_x1 + (rg_x2 - rg_x1) / 2)  # 모든 탐지 객체의 중심 영역 찾기위한 조건 x중심값
            rg_center_dotY = int(rg_y1 + (rg_y2 - rg_y1) / 2)  # 모든 탐지 객체의 중심 영역 찾기위한 조건 y중심값
            if int(cls) in [1, 2, 4, 6, 7]:  # 1,2,4,6,7 클래스만 탐지하는 조건
                if int(im0.shape[1] * 0.1) < rg_center_dotX and int(
                        im0.shape[1] * 0.9) > rg_center_dotX and 0 < rg_center_dotY and int(
                    im0.shape[0] * 0.2) > rg_center_dotY:  # 관심영역 안쪽만 탐지후 그림
                    annotator.box_label(xyxy, label, color=colors(c, True))

            else:
                annotator.box_label(xyxy, label, color=colors(c, True))
        if save_crop:
            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
    warning=''#warning경고문을 담기위한 변수

    person_list = []  ## 사람 발바닥 중앙 좌표 담을 리스트
    red_bool = False  # 빨간불 기본값 거짓
    for *xyxy, conf, cls in reversed(det):  ## 여기 가운데 좌표 잡을라고 추가한거임~~
        x1 = xyxy[0]
        y1 = xyxy[1]
        x2 = xyxy[2]
        y2 = xyxy[3]
        if int(cls) == 5:
            center_dotX = int(x1 + (x2 - x1) / 2)  # X 값의 중앙 좌표 잡기위한 방법
            center_dotY = int(y2)  # Y 값 좌표 그대로 적음

            # print('center_dotX : ', center_dotX)
            # print('center_doty : ', center_dotY)

            cv2.line(im0, (center_dotX, center_dotY), (center_dotX, center_dotY), (255, 0, 255),
                     10)  # 점찍기 굵기 10으로
            person_list.append([center_dotX, center_dotY])  # 사람 발바닥 중앙  리스트에 추가
        if int(cls) in [1, 2, 4, 6, 7]:
            rg_center_dotX = int(x1 + (x2 - x1) / 2)  # 1,2,4,6,7번 클래스의 중심점 찾기위한 x좌표
            rg_center_dotY = int(y1 + (y2 - y1) / 2)  # 1,2,4,6,7번 클래스의 중심점 찾기위한 y좌표
            if int(im0.shape[1] * 0.1) < rg_center_dotX and int(
                    im0.shape[1] * 0.9) > rg_center_dotX and 0 < rg_center_dotY and int(
                im0.shape[0] * 0.2) > rg_center_dotY:  # 관심영역 안쪽만 확인하기
                if int(cls) == 6:  # 빨간 불 조건 걸기
                    red_bool = True  # 빨간불이면 참
                elif int(cls) == 1:  # 녹색불 조건걸기
                    red_bool = False  # 녹색불이 있으면 거짓

    # print("person_list:::::::::::", person_list)
    # print("drivable_list : ::::: ::", drivable_list)


    white_line_list = np.zeros_like(im0)  # 안쪽 영역 값 저장을 위한 빈 화면 생성1
    for i in line_list:
        white_line_list[i[1]][i[0]]=255  #  라인 리스트에서 가져온 좌표에 해당하는 값만을 255로 바꿔줌1

    # print("redbool : ",red_bool)

    line_img, bd_img = hl(white_line_list)  # 라인 검출 위한 함수 호출1

    for pl in person_list:  # 사람 발바닥 중앙 좌표 리스트에서 하나씩 꺼내옴
        if pl in drivable_list and red_bool == False:  # (리턴받아온 ) 주행가능영역 리스트에 값이 있나 비교 기본은 false 로 둬라
            warning='Warning!!!!!!!'#무단횡단이 발견되면 warning에 경고문구저장
        else :  # 주행 가능영역에서 검출되지 않으면 1
            # print("bd.shpae : ", bd_img.shape)
            if bd_img[pl[1]][pl[0]] == 179 and red_bool ==False:  # 검사 1
                warning = 'Warning!!!!!!!'  # 무단횡단이 발견되면 warning에 경고문구저장

    cv2.imwrite('./result_img/lane_img/0.jpg', line_img)  # 라인 이미지 저장 1
    cv2.imwrite('./result_img/lane_inside_img/0.jpg', bd_img)  # 라인 안쪽 검사 영역 이미지 저장 1
    # Stream results
    im0 = annotator.result()

    print("im0.shape : ", im0.shape)

    cv2.rectangle(im0, (int(im0.shape[1] * 0.1), int(0)), (int(im0.shape[1] * 0.9), int(im0.shape[0] * 0.2)),
                  (255, 0, 255), 3)  ## 관심영역 그려봤음~~~~~~~~~~~~~~~~~~
    return im0,warning#warning은 경고문구 반환