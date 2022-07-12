import cv2
import numpy as np

# 映像読み込み
def read_movie(path):
    movie_file = cv2.VideoCapture(path)
    frame_count =int(movie_file.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count) # Debug
    return [movie_file, frame_count]

# 映像書き込み
def start_write_movie():
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('out.mp4', codec, 60, (1200, 800),0)
    return video

# 書き込み処理終了
def end_write_movie(video):
    video.release()

# フレーム間差分の背景画像処理（初期）
# マスク処理のところをいじる必要あり．
def interframe_mask(movie_file):
    s, f = movie_file.read()
    if f is None:
        return
    b = cv2.resize(f, (600, 400))
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    bg = b.copy()
    bg = cv2.rectangle(bg, (0, 0), (600, 200), (0, 0, 0), -1)
    bg = cv2.rectangle(bg, (0, 0), (200, 400), (0, 0, 0), -1)
    bg = cv2.rectangle(bg, (450, 0), (600, 400), (0, 0, 0), -1)
    return bg

# フレーム間差分に用いるマスク画像
# マスク処理のところをいじる必要あり．
def interframe_mask_(gray_mask):
    gray_mask = cv2.rectangle(gray_mask, (0, 0), (600, 200), (0, 0, 0), -1)
    gray_mask = cv2.rectangle(gray_mask, (0, 0), (200, 400), (0, 0, 0), -1)
    gray_mask = cv2.rectangle(gray_mask, (450, 0), (600, 400), (0, 0, 0), -1)
    return gray_mask

# リサイズ
def frame_resize(frame):
    resize = cv2.resize(frame, (600, 400))
    return resize

# モノクロ化
def frame_grayscale(frame):
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_scale

# フレーム間差分
# マスク処理のところをいじる必要あり．
def interframe_difference(bg,gray_mask,gray_blur):
    ### 障害物検出（フレーム間差分）閾値 ###
    th = 20
    ### 差分の絶対値を計算 ###
    mask = cv2.absdiff(gray_mask, bg)
    ### 差分画像を二値化してマスク画像を算出 ###
    mask[mask < th] = 0
    mask[mask >= th] = 255
    ### 画像平坦化，マスク処理 ###
    gray_blur = cv2.equalizeHist(gray_blur)
    gray_blur = cv2.rectangle(gray_blur, (0, 0), (600, 210), (0, 0, 0), -1)
    gray_blur = cv2.rectangle(gray_blur, (0, 0), (250, 400), (0, 0, 0), -1)
    gray_blur = cv2.rectangle(gray_blur, (400, 0), (600, 400), (0, 0, 0), -1)
    return mask, gray_blur

# Canny法によるエッジ検出
def canny_edge(gray_blur):
    ### canny法によるエッジ検出 ###
    med_val = np.median(gray_blur)
    sigma = 0.33  # 0.33
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))
    canny = cv2.Canny(gray_blur, min_val, max_val)
    return canny

# マスク処理による余分なエッジを削除
# ここのcanny.copy()下の値，Mイコール下の値をいじる必要あり．
def delete_mask_edge(canny):
    ### マスクエッジをマスク処理で削除 ###
    frame1 = canny.copy()[220:400,270:400]
    frame1_ = np.array(np.zeros((400, 600), dtype=np.uint8))
    M = np.array([[1, 0, 270], [0, 1, 220]], dtype=float)
    frame1_ = cv2.warpAffine(frame1, M, (600, 400), frame1_, 
        borderMode=cv2.BORDER_TRANSPARENT)
    return frame1_

# 障害物検出と外接矩形描画
def bounding_rectangle(mask,gray_scale):
    contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    contours = list(filter(lambda x: cv2.contourArea(x) > 5000, contours))
    ### 障害物と思しき輪郭に対して外接矩形を描画 ###
    if len(contours) != 0:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        gray_scale = cv2.drawContours(gray_scale, [box], 0, color=(255, 255, 255), thickness=4)
        ### 警告字幕表示 ###
        cv2.rectangle(gray_scale, (310, 70), (500, 110), (0, 0, 0), -1)
        cv2.putText(gray_scale, 'WARNING', (340, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

# 注意文字幕表示
def puttext_caution(frame1_,frame2,gray_scale):
    cv2.rectangle(frame2, (240, 70), (360, 110), (0, 0, 0), -1)
    cv2.rectangle(frame1_, (240, 70), (360, 110), (0, 0, 0), -1)
    cv2.rectangle(gray_scale, (80, 70), (260, 110), (0, 0, 0), -1)
    cv2.putText(gray_scale, 'CAUTION', (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame1_, 'None', (260, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame2, 'None', (260, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

# ハフ変換処理
def hough(frame1_,frame2,mask,gray_scale):
    ### ハフ変換処理（確率的ハフ変換はコメントアウト） ###
    lines = cv2.HoughLines(frame1_,1,np.pi/180,100)
    #lines = cv2.HoughLinesP(frame1_, rho=1, theta=np.pi/180, threshold=40, minLineLength=35, maxLineGap=1)
    if lines is None:
        ### 輪郭抽出 ###
        bounding_rectangle(mask,gray_scale)
        ### 線路検出失敗時に表示する字幕 ###
        puttext_caution(frame1_,frame2,gray_scale)
    else:
        count = 0
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                count += 1
                if count > 2:
                    break
                cv2.line(frame1_,(x1,y1),(x2,y2),(255,255,255),2)
                cv2.line(frame2,(x1,y1),(x2,y2),(255,255,255),2)
        ### 確率的ハフ変換 ###
        #for line in lines:
        #    x1, y1, x2, y2 = line[0]
        #    cv2.line(frame1_,(x1,y1),(x2,y2),(255,255,255),3)
        #    cv2.line(frame2,(x1,y1),(x2,y2),(255,255,255),3)

# 字幕追加
def puttext_movie(mergeMovie2):
    cv2.rectangle(mergeMovie2, (100, 5), (480, 35), (0, 0, 0), -1)
    cv2.rectangle(mergeMovie2, (100, 405), (480, 435), (0, 0, 0), -1)
    cv2.putText(mergeMovie2, 'Original + Detector', (130, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(mergeMovie2, 'Mask + Canny', (810, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(mergeMovie2, 'Original + HoughLines', (110, 430), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(mergeMovie2, 'interframe difference', (730, 430), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

# 繰り返し処理
def play_movie(movie_file, frame_count):
    ### 映像保存 ###
    #video = start_write_movie()
    ### フレーム間差分（背景画像）定義 ###
    bg = interframe_mask(movie_file)
    for i in range(frame_count):
        status, frame = movie_file.read()
        if status == True:
            frame = frame_resize(frame)
            gray_scale = frame_grayscale(frame)
            gray_blur = gray_scale.copy()
            ### フレーム間差分に用いる映像のマスク処理 ###
            gray_mask = gray_scale.copy()
            gray_mask = interframe_mask_(gray_mask)
            ### 背景画像の更新（一定間隔） ###
            if(i % 30 == 1):
                bg = gray_mask
            # フレーム間差分
            mask, gray_blur = interframe_difference(bg,gray_mask,gray_blur)
            ### canny法によるエッジ検出 ###
            canny = canny_edge(gray_blur)
            ### マスクエッジをマスク処理で削除 ###
            frame1_ = delete_mask_edge(canny)
            frame2 = gray_scale.copy()
            ### ハフ変換処理 ###
            hough(frame1_,frame2,mask,gray_scale)
            ### 動画結合 ###
            mergeMovie = cv2.hconcat([gray_scale,canny])
            mergeMovie1 = cv2.hconcat([frame2,mask])
            mergeMovie2 = cv2.vconcat([mergeMovie,mergeMovie1])
            ### 動画字幕追加 ###
            puttext_movie(mergeMovie2)
            ### 動画書き込み（動画表示時にはコメントアウトすること） ###
            #video.write(mergeMovie2.astype(np.uint8))
            ### 動画表示（書き込み時にはコメントアウトすること） ###
            cv2.imshow("test", mergeMovie2)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # Press Esc KEY to Exit
                break
    ### メモリ解放 ###
    movie_file.release()
    ### 動画書き込み時にはコメントアウトを消すこと ###
    #end_write_movie(video)
    cv2.destroyAllWindows()

def main():
    # 動画のパスをここで指定
    filepath = 'hogehoge.mov'
    mov_data = read_movie(filepath)
    play_movie(*mov_data)

if __name__ == '__main__':
    main()