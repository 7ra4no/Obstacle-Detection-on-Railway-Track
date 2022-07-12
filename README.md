# Obstacle-Detection-on-Railway-Track
電車の前方映像による線路上の障害物検出システム．

[処理例：動画](https://youtu.be/NS-7CSeqimQ)

## 使用方法
課題提出のためのもので，使いやすさに関して重きを置いていないため，煩わしい箇所が多い．

コメントなどで多少処理がわかるようにしているが，おそらく厳しい．

事前に説明できるものとして，以下の事項に注意．
* 読み込む動画のパスを手動で指定する
* 線路部分のマスク処理については線路検出部分と障害物検出部分の２箇所ある．これらは手動で指定する
* 動画書き込みの際の終了処理についてはしっかり行うこと(プログラム上では初期状態として動画書き込み処理の箇所はコメントアウトしており，`OpenCV`の`imshow`で動画を表示するようになっている．)

## 使用ライブラリ
* `OpenCV`
* `NumPy`

```conda_requirements.txt```で環境構築すること．

## 参考文献
* [領域(輪郭)の特徴](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html)
* [ハフ変換による直線検出](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html)
* [【Python/OpenCV】フレーム間差分法で移動物体の検出](https://algorithm.joho.info/programming/python/opencv-frame-difference-py/)
