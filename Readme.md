# Yolov5와 HybridNets 모듈을 사용한 무단횡단 상황 탐지


객체 탐지 모델인 YOLOv5모델과 도로 영역을 탐지하는 HybridNets모델을 사용하여 상황에 따라 무단횡단 여부를 판단할 수 있다. 

<div align="center">
    <img src=".\readme-img\example.png" width="80%" alt="" />
</div>

> 객체 훈련과 탐지를 위한 [**YOLOv5**](https://github.com/ultralytics/yolov5) 모델
> 
> 도로 주행 영역을 탐지하기 위한 [**HybridNets**](https://github.com/datvuthanh/HybridNets) 모델
<div align="center">
    <h3>YOLOv5 모델</h3>
    <img src=".\readme-img\yolo.jpg" width="80%" alt="" />
    <h3>HybridNets 모델</h3>
    <img src=".\readme-img\hybridnets.jpg" width="80%" alt="" />
</div>


## Installation

Window10 운영체제  
python 3.10.10 환경에서 테스트 되었다.  
GPU장치가 설치되어 있어야 하며 CPU에 맞는 CUDA를 필요로 한다.  
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64


- 가상환경 생성  
python -m venv test

- CUDA 버전에 맞는 pytorch 설치  
https://pytorch.kr/get-started/locally/

- 코드 다운로드   
git clone

- 다운받은 폴더로 이동  
cd .\doodream-master

- 패키지 설치  
pip install -r requirements.txt


## 실행

- cmd 창에서 Installation에서 만든 가상환경 활성화
- cmd 창에서 doodream-master폴더로 이동 
- test_ui.py파일을 실행( python test_ui.py )
- 실행창  

<div align="center">
    <img src=".\readme-img\start.png" width="70%" alt="" />
</div>

- 이미지 버튼을 클릭 후 탐지를 시작할 이미지를 선택한다.
<div align="center">
    <img src=".\readme-img\1.png" width="70%" alt="" />
</div>
- 실행 버튼을 누르면 검출을 시작하고 결과가 ui상에 띄워진다.
<div align="center">
    <img src=".\readme-img\example.png" width="70%" alt="" />
</div>
- 오른쪽에 있는 버튼을 누르면 각 모델이 탐지하는 부분과 무단횡단을 판단하는 영역을 볼 수 있다.
<div align="center">
    <img src=".\readme-img\result.png" width="70%" alt="" />
</div>