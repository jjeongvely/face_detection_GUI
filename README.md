# Face Recognition GUI

<참고> https://github.com/ageitgey/face_recognition

위 주소의 [examples 폴더의 face_recognition_knn.py](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py)를 수정하여
[knn_face_recognition.py](https://github.com/jjeongvely/face_detection_GUI/blob/master/knn_face_recognition.py)를 만들었다.

```bash
conda create -n 가상환경이름 python=3.6
```
```bash
conda activate hj
```
```bash
git clone https://github.com/davisking/dlib.git
```
```bash
cd dlib
mkdir build; cd build; cmake ..; cmake --build .
```
```bash
cd ..
python setup.py install
```
```bash
cd ..
git clone https://github.com/jjeongvely/face_detection_GUI.git 
```
```bash
cd face_detection_GUI
mkdir dataset
```
```bash
pip install -r requirements.txt
```
