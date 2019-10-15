# Face Recognition GUI

<참고> https://github.com/ageitgey/face_recognition

[dlib](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)와 face_recognition 모듈 설치
```bash
pip3 install face_recognition
```
위 주소의 [examples 폴더의 face_recognition_knn.py](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py)를 수정하여
[knn_face_recognition.py](https://github.com/jjeongvely/face_detection_GUI/blob/master/knn_face_recognition.py)를 만들었다.

- GUI_main.py의 MainWindow()에 있는 self.data_dir은 dataset을 저장할 경로, self.model_path는 data로 만든 model을 저장할 경로
