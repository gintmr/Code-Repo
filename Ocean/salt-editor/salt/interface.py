import cv2
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QApplication,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QRadioButton,
)

from PyQt5.QtCore import QTimer, QTime



class CustomGraphicsView(QGraphicsView):
    def __init__(self, editor):
        super(CustomGraphicsView, self).__init__()

        self.editor = editor
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)

        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setInteractive(True)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.selected_annotations = []

        self.image_item = None

        self.setMouseTracking(True)

        self.mode = "EDIT"

    def set_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        if self.image_item:
            self.image_item.setPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))
        else:
            self.image_item = self.scene.addPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event: QWheelEvent):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            adj = (event.angleDelta().y() / 120) * 0.1
            self.scale(1 + adj, 1 + adj)
        else:
            delta_y = event.angleDelta().y()
            delta_x = event.angleDelta().x()
            x = self.horizontalScrollBar().value()
            self.horizontalScrollBar().setValue(x - delta_x)
            y = self.verticalScrollBar().value()
            self.verticalScrollBar().setValue(y - delta_y)

    def imshow(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            img.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        self.set_image(q_img)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # FUTURE USE OF RIGHT CLICK EVENT IN THIS AREA
        pos = event.pos()
        pos_in_item = self.mapToScene(pos) - self.image_item.pos()
        x, y = pos_in_item.x(), pos_in_item.y()

        # Get the dimensions of the image
        image_width = self.image_item.boundingRect().width()
        image_height = self.image_item.boundingRect().height()

        if 0 <= x < image_width and 0 <= y < image_height:
            # print("Coordinate (", x, ",", y, ") is within the image bounds.")

            # modifiers = QApplication.keyboardModifiers()
            # if modifiers == Qt.ControlModifier:

            if self.mode == "EDIT":
                # print("Control/ Command key pressed during a mouse click")
                # self.editor.hover()
                # self.editor.remove_click([int(x), int(y)])
                clicked_ann = self.editor.choose_annotation([int(x), int(y)])
                if len(clicked_ann) > 0:
                    clicked_ann_id = clicked_ann[0]
                    if clicked_ann_id in self.selected_annotations:
                        self.selected_annotations.remove(clicked_ann_id)
                        self.editor.draw_selected_annotations(self.selected_annotations)
                    else:
                        self.selected_annotations.append(clicked_ann_id)
                        self.editor.draw_selected_annotations(self.selected_annotations)
                    self.imshow(self.editor.display)


                ## Update side pannel annotation in parent function
                # self.get_side_panel_annotations()
                # self.reset()

            if self.mode == "INSERT":
                if event.button() == Qt.LeftButton:
                    label = 1
                elif event.button() == Qt.RightButton:
                    label = 0
                self.editor.add_click([int(x), int(y)], label, self.selected_annotations)
            self.imshow(self.editor.display)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.mode == "EDIT":
            pos = event.pos()
            pos_in_item = self.mapToScene(pos) - self.image_item.pos()
            x, y = pos_in_item.x(), pos_in_item.y()

            # Get the dimensions of the image
            image_width = self.image_item.boundingRect().width()
            image_height = self.image_item.boundingRect().height()

            if 0 <= x < image_width and 0 <= y < image_height:
                self.editor.hover([int(x), int(y)], self.selected_annotations)
                self.imshow(self.editor.display)


class ApplicationInterface_(QWidget):
    def __init__(self, app, editor, panel_size=(1920, 1080)):
        super(ApplicationInterface, self).__init__()
        self.app = app
        self.editor = editor
        self.panel_size = panel_size

        self.layout = QVBoxLayout()
        self.mode = "EDIT"
        self.mode_label = QLabel(f"Mode: {self.mode}")

        self.top_bar = self.get_top_bar()
        self.layout.addWidget(self.top_bar)

        self.main_window = QHBoxLayout()

        self.graphics_view = CustomGraphicsView(editor)
        self.main_window.addWidget(self.graphics_view)

        self.panel = self.get_side_panel()
        self.panel_annotations = QListWidget()
        self.panel_annotations.setFixedWidth(200)
        self.panel_annotations.setSelectionMode(QAbstractItemView.MultiSelection)
        self.panel_annotations.itemClicked.connect(self.annotation_list_item_clicked)
        self.get_side_panel_annotations()
        self.main_window.addWidget(self.panel)
        self.main_window.addWidget(self.panel_annotations)

        self.layout.addLayout(self.main_window)

        self.setLayout(self.layout)

        self.graphics_view.imshow(self.editor.display)

    def reset(self):
        self.editor.reset(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def add(self):
        self.editor.save_ann()
        self.save_all()
        self.editor.reset(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()

    def next_image(self):
        self.editor.next_image()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        # self.clear_annotations(self.graphics_view.selected_annotations)
        self.get_side_panel_annotations()

    def continue_label(self):
        image_id=self.editor.get_last_imageid()
        anno_id=self.editor.get_last_annoid()
        self.editor.set_image_by_imageid(image_id,anno_id)
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()

    def prev_image(self):
        self.editor.prev_image()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        # self.clear_annotations(self.graphics_view.selected_annotations)
        self.get_side_panel_annotations()

    def toggle(self):
        self.editor.toggle(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def transparency_up(self):
        self.editor.step_up_transparency(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def remove_last_annotation(self):
        anno_id = self.editor.get_last_annoid()
        self.editor.clear_annotations([anno_id])
        self.save_all()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()
        self.reset()


    def transparency_down(self):
        self.editor.step_down_transparency(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def save_all(self):
        self.editor.save()

    def update_mode(self, mode):
        if self.mode == "INSERT":
            self.mode = "EDIT"
        elif self.mode == "EDIT":
            self.mode="INSERT"
        else:
            raise error(f'unknown mode: {self.mode}')        # self.mode = mode
        self.mode_label.setText(f"Mode: {mode}")
        self.graphics_view.mode = mode
        self.reset()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        # self.clear_annotations(self.graphics_view.selected_annotations)
        self.get_side_panel_annotations()

    def update_label(self, new_label):
        self.editor.select_category(new_label)

        if self.mode == "EDIT":
            # print("Update label to", new_label, "for", self.graphics_view.selected_annotations)
            self.editor.update_annotation_label(new_label, self.graphics_view.selected_annotations)
            self.editor.select_category(new_label)
            self.editor.save_ann()
            self.save_all()
            self.graphics_view.imshow(self.editor.display)
            self.get_side_panel_annotations()
            self.graphics_view.selected_annotations = []
            self.reset()


    def get_top_bar(self):
        top_bar = QWidget()
        button_layout = QHBoxLayout(top_bar)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.mode_label)
        buttons = [
            ("Insert", lambda: self.update_mode("INSERT")),
            ("Edit", lambda: self.update_mode("EDIT")),
            ("Add", lambda: self.add()),
            ("Reset", lambda: self.reset()),
            ("Prev", lambda: self.prev_image()),
            ("Next", lambda: self.next_image()),
            ("Toggle", lambda: self.toggle()),
            ("Transparency Up", lambda: self.transparency_up()),
            ("Transparency Down", lambda: self.transparency_down()),
            ("Continue", lambda: self.continue_label()),
            ("Save", lambda: self.save_all()),
            (
                "Remove Selected Annotations",
                lambda: self.clear_annotations(self.graphics_view.selected_annotations),
            ),
        ]
        for button, lmb in buttons:
            bt = QPushButton(button)
            bt.clicked.connect(lmb)
            button_layout.addWidget(bt)

        return top_bar

    def get_side_panel(self):
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        categories, colors = self.editor.get_categories(get_colors=True)
        label_array = []
        for i, _ in enumerate(categories):
            label_array.append(QRadioButton(categories[i]))
            label_array[i].clicked.connect(
                lambda state, x=categories[i]: self.update_label(x)
            )

            label_array[i].setStyleSheet(
                "background-color: rgba({},{},{},0.6)".format(*colors[i][::-1])
            )
            panel_layout.addWidget(label_array[i])
        return panel

    def get_side_panel_annotations(self):
        anns, colors = self.editor.list_annotations()
        list_widget = self.panel_annotations
        list_widget.clear()
        # anns, colors = self.editor.get_annotations(self.editor.image_id)
        categories = self.editor.get_categories(get_colors=False)
        for i, ann in enumerate(anns):
            listWidgetItem = QListWidgetItem(
                str(ann["id"]) + " - " + (categories[ann["category_id"]])
            )
            list_widget.addItem(listWidgetItem)
        return list_widget

    def clear_annotations(self, annotation_ids=[]):
        self.editor.clear_annotations(annotation_ids)
        self.get_side_panel_annotations()
        self.save_all()
        self.graphics_view.selected_annotations = []
        self.reset()

    def annotation_list_item_clicked(self, item):
        if item.isSelected():
            self.graphics_view.selected_annotations.append(int(item.text().split(" ")[0]))
            self.editor.draw_selected_annotations(self.graphics_view.selected_annotations)
        else:
            self.graphics_view.selected_annotations.remove(int(item.text().split(" ")[0]))
            self.editor.draw_selected_annotations(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.app.quit()
        if event.key() == Qt.Key_A:
            self.prev_image()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_D:
            self.next_image()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_K:
            self.transparency_down()
        if event.key() == Qt.Key_L:
            self.transparency_up()
        if event.key() == Qt.Key_N:
            self.add()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_E:
            self.update_mode(self.mode)
        if event.key() == Qt.Key_T:
            self.toggle()
        if event.key() == Qt.Key_R:
            self.reset()
        if event.key() == Qt.Key_P:
            self.clear_annotations(self.graphics_view.selected_annotations)

        if event.key() == Qt.Key_L:
            self.remove_last_annotation()

        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_all()
        elif event.key() == Qt.Key_Space:
            print("Space pressed")
            # self.clear_annotations(self.graphics_view.selected_annotations)
            # Do something if the space bar is pressed
            # pass



class ApplicationInterface__(QWidget):
    def __init__(self, app, editor, list_num, panel_size=(1920, 1080)):
        super(ApplicationInterface, self).__init__()
        
        self.list_num = list_num
        
        self.app = app
        self.editor = editor
        self.panel_size = panel_size

        self.layout = QVBoxLayout()
        self.mode = "EDIT"
        self.mode_label = QLabel(f"Mode: {self.mode}")

        self.top_bar = self.get_top_bar()
        self.layout.addWidget(self.top_bar)

        self.main_window = QHBoxLayout()

        self.graphics_view = CustomGraphicsView(editor)
        self.main_window.addWidget(self.graphics_view)

        self.panel = self.get_side_panel()
        self.panel_annotations = QListWidget()
        self.panel_annotations.setFixedWidth(200)
        self.panel_annotations.setSelectionMode(QAbstractItemView.MultiSelection)
        self.panel_annotations.itemClicked.connect(self.annotation_list_item_clicked)
        self.get_side_panel_annotations()
        self.main_window.addWidget(self.panel)
        self.main_window.addWidget(self.panel_annotations)

        self.layout.addLayout(self.main_window)

        self.setLayout(self.layout)

        self.graphics_view.imshow(self.editor.display)

        # Add progress label
        self.progress_label = QLabel("Progress: 0%")
        self.layout.addWidget(self.progress_label)

        # Initialize progress
        self.update_progress()

    def reset(self):
        self.editor.reset(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def add(self):
        self.editor.save_ann()
        self.save_all()
        self.editor.reset(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()

    def next_image(self):
        self.editor.next_image()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()
        self.update_progress()  # Update progress after moving to the next image

    def continue_label(self):
        image_id = self.editor.get_last_imageid()
        anno_id = self.editor.get_last_annoid()
        self.editor.set_image_by_imageid(image_id, anno_id)
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()

    def prev_image(self):
        self.editor.prev_image()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()
        self.update_progress()  # Update progress after moving to the previous image

    def toggle(self):
        self.editor.toggle(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def transparency_up(self):
        self.editor.step_up_transparency(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def remove_last_annotation(self):
        anno_id = self.editor.get_last_annoid()
        self.editor.clear_annotations([anno_id])
        self.save_all()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()
        self.reset()

    def transparency_down(self):
        self.editor.step_down_transparency(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def save_all(self):
        self.editor.save()

    def update_mode(self, mode):
        if self.mode == "INSERT":
            self.mode = "EDIT"
        elif self.mode == "EDIT":
            self.mode = "INSERT"
        else:
            raise error(f'unknown mode: {self.mode}')
        self.mode_label.setText(f"Mode: {mode}")
        self.graphics_view.mode = mode
        self.reset()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()

    def update_label(self, new_label):
        self.editor.select_category(new_label)

        if self.mode == "EDIT":
            self.editor.update_annotation_label(new_label, self.graphics_view.selected_annotations)
            self.editor.select_category(new_label)
            self.editor.save_ann()
            self.save_all()
            self.graphics_view.imshow(self.editor.display)
            self.get_side_panel_annotations()
            self.graphics_view.selected_annotations = []
            self.reset()

    def get_top_bar(self):
        top_bar = QWidget()
        button_layout = QHBoxLayout(top_bar)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.mode_label)
        buttons = [
            ("Insert", lambda: self.update_mode("INSERT")),
            ("Edit", lambda: self.update_mode("EDIT")),
            ("Add", lambda: self.add()),
            ("Reset", lambda: self.reset()),
            ("Prev", lambda: self.prev_image()),
            ("Next", lambda: self.next_image()),
            ("Toggle", lambda: self.toggle()),
            ("Transparency Up", lambda: self.transparency_up()),
            ("Transparency Down", lambda: self.transparency_down()),
            ("Continue", lambda: self.continue_label()),
            ("Save", lambda: self.save_all()),
            (
                "Remove Selected Annotations",
                lambda: self.clear_annotations(self.graphics_view.selected_annotations),
            ),
        ]
        for button, lmb in buttons:
            bt = QPushButton(button)
            bt.clicked.connect(lmb)
            button_layout.addWidget(bt)

        return top_bar

    def get_side_panel(self):
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        categories, colors = self.editor.get_categories(get_colors=True)
        label_array = []
        for i, _ in enumerate(categories):
            label_array.append(QRadioButton(categories[i]))
            label_array[i].clicked.connect(
                lambda state, x=categories[i]: self.update_label(x)
            )

            label_array[i].setStyleSheet(
                "background-color: rgba({},{},{},0.6)".format(*colors[i][::-1])
            )
            panel_layout.addWidget(label_array[i])
        return panel

    def get_side_panel_annotations(self):
        anns, colors = self.editor.list_annotations()
        list_widget = self.panel_annotations
        list_widget.clear()
        categories = self.editor.get_categories(get_colors=False)
        for i, ann in enumerate(anns):
            listWidgetItem = QListWidgetItem(
                str(ann["id"]) + " - " + (categories[ann["category_id"]])
            )
            list_widget.addItem(listWidgetItem)
        return list_widget

    def clear_annotations(self, annotation_ids=[]):
        self.editor.clear_annotations(annotation_ids)
        self.get_side_panel_annotations()
        self.save_all()
        self.graphics_view.selected_annotations = []
        self.reset()

    def annotation_list_item_clicked(self, item):
        if item.isSelected():
            self.graphics_view.selected_annotations.append(int(item.text().split(" ")[0]))
            self.editor.draw_selected_annotations(self.graphics_view.selected_annotations)
        else:
            self.graphics_view.selected_annotations.remove(int(item.text().split(" ")[0]))
            self.editor.draw_selected_annotations(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.app.quit()
        if event.key() == Qt.Key_A:
            self.prev_image()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_D:
            self.next_image()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_K:
            self.transparency_down()
        if event.key() == Qt.Key_L:
            self.transparency_up()
        if event.key() == Qt.Key_N:
            self.add()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_E:
            self.update_mode(self.mode)
        if event.key() == Qt.Key_T:
            self.toggle()
        if event.key() == Qt.Key_R:
            self.reset()
        if event.key() == Qt.Key_P:
            self.clear_annotations(self.graphics_view.selected_annotations)

        if event.key() == Qt.Key_L:
            self.remove_last_annotation()

        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_all()
        elif event.key() == Qt.Key_Space:
            print("Space pressed")

    def update_progress(self):
        total_images = self.list_num  # Use the new method
        current_image_index = self.editor.image_id
        progress = (current_image_index + 1) / total_images * 100
        self.progress_label.setText(f"Progress: {progress:.2f}%  === {current_image_index} / {total_images}")

from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QHBoxLayout, QListWidget, QAbstractItemView
from PyQt5.QtCore import QTimer, QTime

class ApplicationInterface(QWidget):
    def __init__(self, app, editor, list_num, panel_size=(1920, 1080)):
        super(ApplicationInterface, self).__init__()
        self.app = app
        self.editor = editor
        self.list_num = list_num
        self.panel_size = panel_size

        # 初始化布局
        self.layout = QVBoxLayout()  # 创建一个垂直布局
        self.setLayout(self.layout)  # 将布局设置给当前窗口

        # 初始化界面
        self.setup_interface()  # 调用 setup_interface 方法初始化界面

    def setup_interface(self):
        # 如果已经设置了布局，先清理
        if hasattr(self, 'layout'):
            self.clear_interface()

        # 初始化界面
        self.mode = "EDIT"
        self.mode_label = QLabel(f"Mode: {self.mode}")

        self.top_bar = self.get_top_bar()
        self.layout.addWidget(self.top_bar)

        self.main_window = QHBoxLayout()

        self.graphics_view = CustomGraphicsView(self.editor)
        self.main_window.addWidget(self.graphics_view)

        self.panel = self.get_side_panel()
        self.panel_annotations = QListWidget()
        self.panel_annotations.setFixedWidth(200)
        self.panel_annotations.setSelectionMode(QAbstractItemView.MultiSelection)
        self.panel_annotations.itemClicked.connect(self.annotation_list_item_clicked)
        self.get_side_panel_annotations()
        self.main_window.addWidget(self.panel)
        self.main_window.addWidget(self.panel_annotations)

        self.layout.addLayout(self.main_window)

        self.graphics_view.imshow(self.editor.display)

        # 添加进度标签
        self.progress_label = QLabel("Progress: 0%")
        self.layout.addWidget(self.progress_label)

        # 添加计时器标签
        self.timer_label = QLabel("Time: 00:00:00")
        self.layout.addWidget(self.timer_label)

        # 初始化进度
        self.update_progress()

        # 初始化计时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.elapsed_time = QTime(0, 0, 0)

        # 启动计时器
        self.start_timer()

    def clear_interface(self):
        # 清除当前界面
        if hasattr(self, 'layout') and isinstance(self.layout, QVBoxLayout):
            # 删除布局中的所有组件
            while self.layout.count():
                item = self.layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        
        

    def start_timer(self):
        self.timer.start(1000)  # Update every second

    def stop_timer(self):
        self.timer.stop()

    def update_timer(self):
        self.elapsed_time = self.elapsed_time.addSecs(1)
        self.timer_label.setText(f"Time: {self.elapsed_time.toString('hh:mm:ss')}")

    def reset(self):
        self.editor.reset(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def add(self):
        self.editor.save_ann()
        self.save_all()
        self.editor.reset(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()

    def next_image(self):
        self.editor.next_image()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()
        self.update_progress()  # Update progress after moving to the next image

    def continue_label(self):
        image_id = self.editor.get_last_imageid()
        anno_id = self.editor.get_last_annoid()
        self.editor.set_image_by_imageid(image_id, anno_id)
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()

    def prev_image(self):
        self.editor.prev_image()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()
        self.update_progress()  # Update progress after moving to the previous image

    def toggle(self):
        self.editor.toggle(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def transparency_up(self):
        self.editor.step_up_transparency(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def remove_last_annotation(self):
        anno_id = self.editor.get_last_annoid()
        self.editor.clear_annotations([anno_id])
        self.save_all()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()
        self.reset()

    def transparency_down(self):
        self.editor.step_down_transparency(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def save_all(self):
        self.editor.save()

    def update_mode(self, mode):
        if self.mode == "INSERT":
            self.mode = "EDIT"
        elif self.mode == "EDIT":
            self.mode = "INSERT"
        else:
            raise error(f'unknown mode: {self.mode}')
        self.mode_label.setText(f"Mode: {mode}")
        self.graphics_view.mode = mode
        self.reset()
        self.graphics_view.selected_annotations = []
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()

    def update_label(self, new_label):
        self.editor.select_category(new_label)

        if self.mode == "EDIT":
            self.editor.update_annotation_label(new_label, self.graphics_view.selected_annotations)
            self.editor.select_category(new_label)
            self.editor.save_ann()
            self.save_all()
            self.graphics_view.imshow(self.editor.display)
            self.get_side_panel_annotations()
            self.graphics_view.selected_annotations = []
            self.reset()

    def get_top_bar(self):
        top_bar = QWidget()
        button_layout = QHBoxLayout(top_bar)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.mode_label)
        buttons = [
            ("Insert", lambda: self.update_mode("INSERT")),
            ("Edit", lambda: self.update_mode("EDIT")),
            ("Add", lambda: self.add()),
            ("Reset", lambda: self.reset()),
            ("Prev", lambda: self.prev_image()),
            ("Next", lambda: self.next_image()),
            ("Toggle", lambda: self.toggle()),
            ("Transparency Up", lambda: self.transparency_up()),
            ("Transparency Down", lambda: self.transparency_down()),
            ("Continue", lambda: self.continue_label()),
            ("Save", lambda: self.save_all()),
            (
                "Remove Selected Annotations",
                lambda: self.clear_annotations(self.graphics_view.selected_annotations),
            ),
        ]
        for button, lmb in buttons:
            bt = QPushButton(button)
            bt.clicked.connect(lmb)
            button_layout.addWidget(bt)

        return top_bar

    def get_side_panel(self):
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        categories, colors = self.editor.get_categories(get_colors=True)
        label_array = []
        for i, _ in enumerate(categories):
            label_array.append(QRadioButton(categories[i]))
            label_array[i].clicked.connect(
                lambda state, x=categories[i]: self.update_label(x)
            )

            label_array[i].setStyleSheet(
                "background-color: rgba({},{},{},0.6)".format(*colors[i][::-1])
            )
            panel_layout.addWidget(label_array[i])
        return panel

    def get_side_panel_annotations(self):
        anns, colors = self.editor.list_annotations()
        list_widget = self.panel_annotations
        list_widget.clear()
        categories = self.editor.get_categories(get_colors=False)
        for i, ann in enumerate(anns):
            listWidgetItem = QListWidgetItem(
                str(ann["id"]) + " - " + (categories[ann["category_id"]])
            )
            list_widget.addItem(listWidgetItem)
        return list_widget

    def clear_annotations(self, annotation_ids=[]):
        self.editor.clear_annotations(annotation_ids)
        self.get_side_panel_annotations()
        self.save_all()
        self.graphics_view.selected_annotations = []
        self.reset()

    def annotation_list_item_clicked(self, item):
        if item.isSelected():
            self.graphics_view.selected_annotations.append(int(item.text().split(" ")[0]))
            self.editor.draw_selected_annotations(self.graphics_view.selected_annotations)
        else:
            self.graphics_view.selected_annotations.remove(int(item.text().split(" ")[0]))
            self.editor.draw_selected_annotations(self.graphics_view.selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.app.quit()
        if event.key() == Qt.Key_A:
            self.prev_image()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_D:
            self.next_image()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_K:
            self.transparency_down()
        if event.key() == Qt.Key_L:
            self.transparency_up()
        if event.key() == Qt.Key_N:
            self.add()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_E:
            self.update_mode(self.mode)
        if event.key() == Qt.Key_T:
            self.toggle()
        if event.key() == Qt.Key_R:
            self.reset()
        if event.key() == Qt.Key_P:
            self.clear_annotations(self.graphics_view.selected_annotations)

        if event.key() == Qt.Key_L:
            self.remove_last_annotation()

        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_all()
        elif event.key() == Qt.Key_Space:
            print("Space pressed")

    def update_progress(self):
        total_images = self.list_num  # Use the new method
        current_image_index = self.editor.image_id
        progress = (current_image_index + 1) / total_images * 100
        self.progress_label.setText(f"Progress: {progress:.2f}%  === {current_image_index} / {total_images}")