import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

# External logic (must exist in your project)
from CatDogClassifier import (
    load_show_images,
    show_resnet_structure,
    show_accuracy_comparison,
    inference_catdog
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('OpenCV — Cat–Dog Classifier')
        self.setGeometry(100, 100, 1000, 700)
        self.q2_image = None
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # 外層：左右置中
        outer = QHBoxLayout()
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(0)
        central.setLayout(outer)

        outer.addStretch(1)  # 左側留白

        # 中央容器：包含左側按鈕列 + 右側預覽框
        center_container = QWidget()
        center_layout = QHBoxLayout()
        center_layout.setSpacing(24)
        center_container.setLayout(center_layout)

        # --- 左：按鈕列（按鈕稍微大一點） ---
        left_container = QWidget()
        left_col = QVBoxLayout()
        left_col.setSpacing(12)
        left_container.setLayout(left_col)
        left_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        title = QLabel('Cat–Dog Classifier (ResNet50)')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 600;")

        def mk_btn(text):
            b = QPushButton(text)
            b.setMinimumWidth(280)
            b.setMinimumHeight(46)     # 讓按鈕更大
            b.setStyleSheet("font-size: 16px;")
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            return b

        self.btn_load_image = mk_btn('Load Image')
        self.btn_show_image = mk_btn('Show Image (Dataset Grid)')
        self.btn_model_structure = mk_btn('Show Model Structure')
        self.btn_show_comparison = mk_btn('Show Comparison')
        self.btn_inference = mk_btn('Inference')

        self.result_label = QLabel('Result: —')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.result_label.setStyleSheet("padding: 10px; font-size: 16px;")
        self.result_label.setMinimumWidth(280)

        # 垂直置中：上/下加 stretch
        left_col.addStretch(1)
        left_col.addWidget(title)
        left_col.addSpacing(10)
        left_col.addWidget(self.btn_load_image)
        left_col.addWidget(self.btn_show_image)
        left_col.addWidget(self.btn_model_structure)
        left_col.addWidget(self.btn_show_comparison)
        left_col.addWidget(self.btn_inference)
        left_col.addSpacing(10)
        left_col.addWidget(self.result_label)
        left_col.addStretch(1)

        # --- 右：較小的直立方框（顯示圖片） ---
        self.image_box = QLabel('Loaded Image')
        self.image_box.setAlignment(Qt.AlignCenter)
        self.image_box.setMinimumSize(360, 420)  # 比上一版更小
        self.image_box.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.image_box.setStyleSheet("background:#111; color:#bbb;")
        self.image_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # 把左右區域放入中央容器，整體置於視窗正中央
        center_layout.addWidget(left_container, stretch=0, alignment=Qt.AlignVCenter)
        center_layout.addWidget(self.image_box, stretch=0, alignment=Qt.AlignVCenter)

        outer.addWidget(center_container, stretch=0, alignment=Qt.AlignCenter)
        outer.addStretch(1)  # 右側留白

        # 連接訊號
        self.btn_load_image.clicked.connect(self.load_q2_image)
        self.btn_show_image.clicked.connect(load_show_images)
        self.btn_model_structure.clicked.connect(show_resnet_structure)
        self.btn_show_comparison.clicked.connect(show_accuracy_comparison)
        self.btn_inference.clicked.connect(self.inference)

    # ---------------- Helpers ----------------
    def _display_to_label(self, img_bgr, label: QLabel):
        """將 OpenCV BGR 影像等比例縮放後顯示在指定 QLabel 內。"""
        if img_bgr is None or img_bgr.size == 0:
            label.setText("Invalid image")
            return
        # 轉成 RGB
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # 依 QLabel 尺寸做等比例縮放
        box_w = max(1, label.width())
        box_h = max(1, label.height())
        scale = min(box_w / w, box_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # 轉 QImage -> QPixmap
        qimg = QImage(resized.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        label.setPixmap(pix)
        label.setAlignment(Qt.AlignCenter)

    # ---------------- Actions ----------------
    def load_q2_image(self):
        """載入單張影像，顯示於右側方框，並在未推論前清空結果文字。"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if not file_name:
            return
        try:
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"檔案不存在: {file_name}")
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                raise ValueError("不支援的檔案格式")

            data = np.fromfile(file_name, dtype=np.uint8)  # 避免中文路徑問題
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("圖片解碼失敗")

            self.q2_image = img
            self._display_to_label(self.q2_image, self.image_box)

            # 清掉先前的預測結果（直到使用者按下 Inference）
            self.result_label.setText('Result: —')

        except Exception as e:
            QMessageBox.critical(self, "載入失敗", f"{e}")
            self.q2_image = None
            self.image_box.setText("預覽區 (Loaded Image)")
            self.result_label.setText('Result: —')

    def inference(self):
        """以較佳模型對已載入影像推論，並在左側結果區顯示文字。"""
        if self.q2_image is None:
            QMessageBox.information(self, "提醒", "請先點選『Load Image』載入影像。")
            return
        try:
            pred = inference_catdog(self.q2_image)
            self.result_label.setText(f"Result: {pred}")
        except Exception as e:
            QMessageBox.critical(self, "推論失敗", f"{e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
