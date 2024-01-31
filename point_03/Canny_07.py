import cv2
import numpy as np

# 回調函數，用於滑動條的變化
def update_parameters(x):
    # 取得滑動條的數值
    canny_threshold1 = cv2.getTrackbarPos('Canny Threshold 1', 'Adjust Contours and Edges')
    canny_threshold2 = cv2.getTrackbarPos('Canny Threshold 2', 'Adjust Contours and Edges')
    epsilon = cv2.getTrackbarPos('Approximation Epsilon', 'Adjust Contours and Edges') / 100.0

    # 將圖片轉為灰度
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 邊緣檢測
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    # 尋找輪廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 提取內層輪廓的索引
    inner_contour_indices = [i for i, h in enumerate(hierarchy[0]) if h[3] != -1]
    
    # 提取內層輪廓
    inner_contours = [contours[i] for i in inner_contour_indices]
    
    largest_contour = max(contours, key=cv2.contourArea)

    # 提取角點
    corners = []
    #ep = epsilon * cv2.arcLength(largest_contour, True)
    #approx = cv2.approxPolyDP(largest_contour, ep, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon * cv2.arcLength(largest_contour, True), True)
    # 直接返回點的座標
    points = [tuple(point[0]) for point in approx]

    # 複製原始圖像
    result_img = original_img.copy()

    for point in points:
        x, y = point
        cv2.circle(result_img, (x, y), 15, (0, 0, 255), -1)

    # 輪廓逼近
    approx_contours = [cv2.approxPolyDP(cnt, epsilon * cv2.arcLength(cnt, True), True) for cnt in contours]

    # 繪製輪廓
    cv2.drawContours(result_img, approx_contours, -1, (0, 255, 0), 2)
    
    # 縮小圖片
    resized_img = cv2.resize(result_img, (800, 600))  # 調整成適當的大小

    # 顯示結果
    cv2.imshow('Adjust Contours and Edges', resized_img)

# 讀取圖片
original_img = cv2.imread('data/8-600.jpg')

# 建立視窗
cv2.namedWindow('Adjust Contours and Edges')

# 建立滑動條
cv2.createTrackbar('Canny Threshold 1', 'Adjust Contours and Edges', 127, 255, update_parameters)
cv2.createTrackbar('Canny Threshold 2', 'Adjust Contours and Edges', 255, 255, update_parameters)
cv2.createTrackbar('Approximation Epsilon', 'Adjust Contours and Edges', 4, 100, update_parameters)

# 初始輪廓與邊緣調整
update_parameters(0)

# 等待用戶操作，按 'ESC' 鍵退出
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 'ESC' 鍵
        break

# 釋放資源
cv2.destroyAllWindows()
