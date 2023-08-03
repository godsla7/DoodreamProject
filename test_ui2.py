# 출력부
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from HybridNets_main.connect_yolo_video import hybridnets_video
from HybridNets_main.connect_yolo_img import hybridnets_img
from detect_def import project_draw

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # hybridnets_video(w='./HybridNets_main/weights/hybridnets.pth',source1=source,output1='./runs/detect')

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # ===============================================================================================HybridNets실행부분
    if is_file:
        if source[-3:] in VID_FORMATS:
            print('동영상')
            hybridnets_video(w='./HybridNets_main/weights/hybridnets.pth', source1=source, output1='./runs/detect')
        elif source[-3:] in IMG_FORMATS:
            print('이미지')
            drivable_list , line_list = hybridnets_img(w='./HybridNets_main/weights/hybridnets.pth', source1=source,
                                        output1='./runs/detect')  # 주행가능영역, 라인 리스트로 받아옴 1
    print(IMG_FORMATS)
    # =================================================================================================================

    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        cv2.imwrite('./result_img/ori_img/0.jpg', im0s)
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            cv2.imwrite('./result_img/yolov5_img/0.jpg', im0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

######################################################################################################################
                # Write results
            #     for *xyxy, conf, cls in reversed(det):
            #         if save_txt:  # Write to file
            #             xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #             line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            #             with open(f'{txt_path}.txt', 'a') as f:
            #                 f.write(('%g ' * len(line)).rstrip() % line + '\n')
            #         if save_img or save_crop or view_img:  # Add bbox to image
            #             c = int(cls)  # integer class
            #             label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            #             rg_x1 = xyxy[0]
            #             rg_y1 = xyxy[1]
            #             rg_x2 = xyxy[2]
            #             rg_y2 = xyxy[3]
            #             rg_center_dotX = int(rg_x1 + (rg_x2 - rg_x1) / 2)  # 모든 탐지 객체의 중심 영역 찾기위한 조건 x중심값
            #             rg_center_dotY = int(rg_y1 + (rg_y2 - rg_y1) / 2)  # 모든 탐지 객체의 중심 영역 찾기위한 조건 y중심값
            #             if int(cls) in [1, 2, 4, 6, 7]:  # 1,2,4,6,7 클래스만 탐지하는 조건
            #                 if int(im0.shape[1] * 0.1) < rg_center_dotX and int(
            #                         im0.shape[1] * 0.9) > rg_center_dotX and 0 < rg_center_dotY and int(
            #                         im0.shape[0] * 0.2) > rg_center_dotY:  # 관심영역 안쪽만 탐지후 그림
            #                     annotator.box_label(xyxy, label, color=colors(c, True))
            #
            #             else:
            #                 annotator.box_label(xyxy, label, color=colors(c, True))
            #         if save_crop:
            #             save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            #
            # person_list = []  ## 사람 발바닥 중앙 좌표 담을 리스트
            # red_bool = False  # 빨간불 기본값 거짓
            # for *xyxy, conf, cls in reversed(det):  ## 여기 가운데 좌표 잡을라고 추가한거임~~
            #     x1 = xyxy[0]
            #     y1 = xyxy[1]
            #     x2 = xyxy[2]
            #     y2 = xyxy[3]
            #     if int(cls) == 5:
            #         center_dotX = int(x1 + (x2 - x1) / 2)  # X 값의 중앙 좌표 잡기위한 방법
            #         center_dotY = int(y2)  # Y 값 좌표 그대로 적음
            #
            #         #print('center_dotX : ', center_dotX)
            #         #print('center_doty : ', center_dotY)
            #
            #         cv2.line(im0, (center_dotX, center_dotY), (center_dotX, center_dotY), (255, 0, 255),
            #                  10)  # 점찍기 굵기 10으로
            #         person_list.append([center_dotX, center_dotY])  # 사람 발바닥 중앙  리스트에 추가
            #     if int(cls) in [1, 2, 4, 6, 7]:
            #         rg_center_dotX = int(x1 + (x2 - x1) / 2)  # 1,2,4,6,7번 클래스의 중심점 찾기위한 x좌표
            #         rg_center_dotY = int(y1 + (y2 - y1) / 2)  # 1,2,4,6,7번 클래스의 중심점 찾기위한 y좌표
            #         if int(im0.shape[1] * 0.1) < rg_center_dotX and int(
            #                 im0.shape[1] * 0.9) > rg_center_dotX and 0 < rg_center_dotY and int(
            #                 im0.shape[0] * 0.2) > rg_center_dotY:  # 관심영역 안쪽만 확인하기
            #             if int(cls) == 6:  # 빨간 불 조건 걸기
            #                 red_bool = True  # 빨간불이면 참
            #             elif int(cls) == 1:  # 녹색불 조건걸기
            #                 red_bool = False  # 녹색불이 있으면 거짓
            #
            # #print("person_list:::::::::::", person_list)
            # #print("drivable_list : ::::: ::", drivable_list)
            #
            # for pl in person_list:  # 사람 발바닥 중앙 좌표 리스트에서 하나씩 꺼내옴
            #     if pl in drivable_list and red_bool == False:  # (리턴받아온 ) 주행가능영역 리스트에 값이 있나 비교
            #         print("무단인간 좌표인거임 ~ : ", pl)  # 무단횡단 한 사람 좌표만 가져옴.
            #         print("무단횡단임 ~~~~~")  # 출력
            #
            # # Stream results
            # im0 = annotator.result()
            #
            # print("im0.shape : ", im0.shape)
            #
            # cv2.rectangle(im0, (int(im0.shape[1] * 0.1), int(0)), (int(im0.shape[1] * 0.9), int(im0.shape[0] * 0.2)),
            #               (255, 0, 255), 3)  ##
######################################################################################################################

            im0,warning=project_draw(det, save_txt, gn, save_conf, txt_path, save_img, save_crop, view_img, hide_labels, names,
                        hide_conf, im0, annotator, imc, save_dir, drivable_list, line_list)  # 검사 로직 함수 호출  1

            cv2.imwrite('./result_img/0.jpg', im0)

            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    return warning#경고문구 반환

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'project_files/project.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'project_files/project.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

class Ui_Dialog(object):

    def __int__(self):
        self.fname=[]


    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1600, 900)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 100, 591, 311))

        self.label.setObjectName("label")
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(1350, 160, 161, 391))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        # 무단횡단 판단하는 라벨 추가
        self.label123 = QtWidgets.QLabel(Dialog)
        self.label123.setGeometry(QtCore.QRect(620, 15, 121, 71))
        self.label123.setObjectName("label123")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        #원본버튼
        self.ori_img = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.ori_img.setObjectName("ori_img")
        self.verticalLayout.addWidget(self.ori_img)
        self.ori_img.clicked.connect(self.ori_img_def)  # 원본버튼 이벤트

        # 결과버튼
        self.result = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.result.setObjectName("result")
        self.verticalLayout.addWidget(self.result)
        self.result.clicked.connect(self.result_def)  # 결과버튼 이벤트

        #yolov5객체 버튼
        self.yolov5 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.yolov5.setObjectName("yolov5")
        self.verticalLayout.addWidget(self.yolov5)
        self.yolov5.clicked.connect(self.yolov5_img_def)  # yolov5객체 이벤트

        #hybridnets_seg버튼
        self.hybridnets = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.hybridnets.setObjectName("hybridnets")
        self.verticalLayout.addWidget(self.hybridnets)
        self.hybridnets.clicked.connect(self.hybridnets_img_def)  # hybridnets_seg버튼 이벤트

        #외각선 검출 버튼
        self.lane = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.lane.setObjectName("lane")
        self.verticalLayout.addWidget(self.lane)
        self.lane.clicked.connect(self.lane_img)  # 외각선 검출 이벤트

        #외각선 안쪽 버튼
        self.lane_inside = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.lane_inside.setObjectName("lane_inside")
        self.verticalLayout.addWidget(self.lane_inside)
        self.lane_inside.clicked.connect(self.lane_inside_img)  # 외각선 안쪽 이벤트

        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(1350, 820, 156, 23))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.horizontalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(1350, 0, 161, 101))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        #이미지 열기 버튼
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton.clicked.connect(self.open_img_def)  # 이미지 열기 버튼 이벤트


        #실행버튼
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton_2.clicked.connect(self.run_yolo_hybrid_def)#실행버튼 이벤트

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

#버튼 눌렀을때 실행되는 함수들---------------------------------------------------------------------------------
    def open_img_def(self):
        self.fname = QFileDialog.getOpenFileName(None, 'Open File', '', 'All File(*);; img File(*.jpg)')
        print(self.fname[0])
        self.pixmap = QPixmap(self.fname[0])
        self.label.setPixmap(self.pixmap)
        self.label.setContentsMargins(10, 10, 10, 10)
        self.label.resize(self.pixmap.width(), self.pixmap.height())

    def run_yolo_hybrid_def(self):
        warning=run(weights='project_files/project.pt',source=self.fname[0],data='project_files/project.yaml')#모델을 실행하면서 경구문구를 받아 warning에 추가
        self.pixmap = QPixmap('./result_img/0.jpg')
        self.label.setPixmap(self.pixmap)
        self.label.setContentsMargins(10, 10, 10, 10)
        self.label.resize(self.pixmap.width(), self.pixmap.height())

        self.label123.setText(warning)  # warning에 있는 경고문 문자열을 라벨텍스트에 적용
        self.label123.setStyleSheet("Color : Red")  # 글자색 변환
        self.label123.setFont(QtGui.QFont("궁서", 20))  # 폰트,크기 조절
    def result_def(self):
        self.pixmap = QPixmap('./result_img/0.jpg')
        self.label.setPixmap(self.pixmap)
        self.label.setContentsMargins(10, 10, 10, 10)
        self.label.resize(self.pixmap.width(), self.pixmap.height())
    def ori_img_def(self):
        self.pixmap = QPixmap('./result_img/ori_img/0.jpg')
        self.label.setPixmap(self.pixmap)
        self.label.setContentsMargins(10, 10, 10, 10)
        self.label.resize(self.pixmap.width(), self.pixmap.height())
    def yolov5_img_def(self):
        self.pixmap = QPixmap('./result_img/yolov5_img/0.jpg')
        self.label.setPixmap(self.pixmap)
        self.label.setContentsMargins(10, 10, 10, 10)
        self.label.resize(self.pixmap.width(), self.pixmap.height())
    def hybridnets_img_def(self):
        self.pixmap = QPixmap('./result_img/hybridnets_seg_img/0.jpg')
        self.label.setPixmap(self.pixmap)
        self.label.setContentsMargins(10, 10, 10, 10)
        self.label.resize(self.pixmap.width(), self.pixmap.height())
    def lane_img(self):
        self.pixmap = QPixmap('./result_img/lane_img/0.jpg')
        self.label.setPixmap(self.pixmap)
        self.label.setContentsMargins(10, 10, 10, 10)
        self.label.resize(self.pixmap.width(), self.pixmap.height())
    def lane_inside_img(self):
        self.pixmap = QPixmap('./result_img/lane_inside_img/0.jpg')
        self.label.setPixmap(self.pixmap)
        self.label.setContentsMargins(10, 10, 10, 10)
        self.label.resize(self.pixmap.width(), self.pixmap.height())
#--------------------------------------------------------------------------------------------------------------------

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "무단횡단 탐지"))
        self.label.setText(_translate("Dialog", "원본 이미지"))
        self.ori_img.setText(_translate("Dialog", "원본"))
        self.result.setText(_translate("Dialog", "결과"))
        self.yolov5.setText(_translate("Dialog", "yolov5객체"))
        self.hybridnets.setText(_translate("Dialog", "hybridnets_seg"))
        self.lane.setText(_translate("Dialog", "외각선 검출"))
        self.lane_inside.setText(_translate("Dialog", "외각선 안쪽"))
        self.pushButton.setText(_translate("Dialog", "이미지"))
        self.pushButton_2.setText(_translate("Dialog", "실행"))
        self.label123.setText(_translate("Dialog", "TextLabel"))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
