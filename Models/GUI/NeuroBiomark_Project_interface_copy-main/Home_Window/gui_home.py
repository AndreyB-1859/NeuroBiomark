"""
gui_home.py
-----------------
This file contains the main GUI logic for the ALS Diagnostic Tool.

• This is the central controller of the application.
• It connects the UI layout (generated from Qt Designer) with the backend model logic.
• Most user interactions (drag/drop, browse, slider updates, help popups) are handled here.

Think of this file as the "bridge" between:
    UI  <---->  Model inference  <---->  Image processing utilities

If you plan to extend functionality (new metrics, new visualisations, new buttons), this is the primary place to work in.
"""


import os
import sys
from PIL import Image

from PySide6 import QtCore as qtc
from PySide6 import QtWidgets as qtw
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QToolButton, QLabel
from PySide6.QtCore import Qt, QPoint

from Home_Window.UI.home_window import Ui_MainWindow
from Home_Window.utils.pil2pixmap import pil2pixmap
from ALS_Diagnostic_Model.src.utils.predict_image import predict_image
from ALS_Diagnostic_Model.src.utils.load_model import load_model
from ALS_Diagnostic_Model.src.utils.layer_cam import extract_output_cam_and_image, generate_layercam_overlay, generate_layercam_circles, generate_layercam_focus_mask, generate_layercam_boundary
from ALS_Diagnostic_Model.src.utils.qu_path import extract_color_masks, quantify_focus
from Dataset.utils import get_metadata, get_image_metadata_str

# -----------------------------------------------------------------------------
# HelpPopup
# -----------------------------------------------------------------------------
# A small reusable popup widget used for the "?" help icons.
#
# Instead of using Qt's default tooltip (which is limited and hard to style),
# a fully custom popup was created so we have complete control over:
#   • Styling
#   • Positioning
#   • Behaviour (hover in / hover out)
# -----------------------------------------------------------------------------
class HelpPopup(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setWordWrap(True)
        self.setMaximumWidth(310)
        self.setStyleSheet("""
            QLabel {
                background: black;
                color: white;
                border: 1px solid #888;
                padding: 4px;
                font-size: 11px;
            }
        """)
        self.setMargin(4)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool)

    # Positions the popup just to the right of the help button.
    # If you want to reposition popups globally, adjust the QPoint logic here.
    def show_near(self, widget):
        pos = widget.mapToGlobal(QPoint(widget.width(), 0))
        self.move(pos)
        self.show()


# -----------------------------------------------------------------------------
# GuiHome (Main Application Window)
# Managing the overall state of the application
# -----------------------------------------------------------------------------
# Responsibilities:
#   • Connecting UI elements to logic
#   • Handling user interactions
#   • Running model inference
#   • Displaying CAM visualisations
#   • Managing help popups
# -----------------------------------------------------------------------------
class GuiHome(qtw.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
            
        # Help Text Definitions
        self.help_texts = {
            "Areas of Focus": "• This is the LayerCAM information overlayed on the   QuPath annotated image.\n"
                              "• Blue outlines are the cell boundaries.\n"
                              "• Yellow shapes are the ROIs.\n"
                              "• Red circles are the high-attention regions from \t\tLayerCAM.\n",

            "Model Focus Viewer": "• This is the LayerCAM information overlayed on raw tissue, "
                                    "i.e., the original IHC image.",

            "No. Areas Focused": "• Total number of focus regions (red circles detected).\n"
                                  "• High number → attention scattered \n"
                                  "• Low number → attention concentrated",

            "Areas with ROIs": "• Number of focus areas overlapping QuPath ROIs.\n"
                                "• High number → model is focusing on the biologically relevant cells \n"
                                "• Low number → model may be attending to noise or irrelevant regions",

            "Focus Relevance": "• Precision relative to QuPath ROIs.\n"
                                "• Focus Relevance = \n(Areas overlapping ROIs) / (Total Areas Focused)\n"
                                "• 0.00 → model attention does not align with ROIs\n"
                                "• 1.00 → all attention aligns with ROIs\n"
                                "• ~0.7-0.8 → strong biological relevance\n",

            "Focus Toggle": "• Controls LayerCAM threshold.\n"
                             "•At 0% - almost entire CAM is included, essentially meaning, 'show me everything that the model might care about'.\n"
                             "•At 50% - 80% (optimal range) - this level of threshold filters the weaker activations and only strong focus regions remain.\n"
                             "•At 100% - threshold is 1.0, only absolute maximum activation, often results in no regions passing threshold. Essentially means, ‘only show me perfect certainty’, which is usually too strict."
        }

        # Keeps track of whichever help popup is currently visible.
        # This prevents multiple popups from stacking.
        self._active_popup = None
        qtc.QTimer.singleShot(0, self._install_help_icons)

        # ------------------------------------------------------------------
        # Scroll Area Wrapper
        # ------------------------------------------------------------------
        # The original UI was designed with fixed dimensions.
        # To prevent layout overflow issues on smaller screens,
        # the entire central widget is wrapped in a QScrollArea.
        #
        # IMPORTANT:
        # The original palette and styling is preserved so that
        # previously custom defined colours are not accidentally overridden.
        # ------------------------------------------------------------------
        scroll = qtw.QScrollArea()
        scroll.setWidgetResizable(True)

        # Save original central widget
        original = self.centralWidget()

        # Preserve existing palette/stylesheet from original widget
        scroll.setPalette(original.palette())
        scroll.viewport().setPalette(original.palette())
        scroll.setStyleSheet(original.styleSheet())
        scroll.viewport().setStyleSheet("background: transparent;")

        scroll.setWidget(original)
        self.setCentralWidget(scroll)
        
        # Restore white backgrounds for data display widgets
        white_bg = "background-color: white; color: black;"

        self.lb_case_id.setStyleSheet(white_bg)
        self.lb_image_no.setStyleSheet(white_bg)
        self.lb_region.setStyleSheet(white_bg)
        self.le_areas_focused.setStyleSheet(white_bg)
        self.le_ROIs_focused.setStyleSheet(white_bg)
        self.le_focus_relevance.setStyleSheet(white_bg)
        self.le_focus_toggle.setStyleSheet(white_bg)
        self.te_notes.setStyleSheet("background-color: white; color: black;")

        # ------------------------------------------------------------------
        # Lock widgets so they are display-only (except for the focus toggle)
        # This prevents user input errors and reinforces that these fields are for display only.
        # The focus toggle is intentionally left editable so users can type in specific values if they prefer that over the slider.
        # ------------------------------------------------------------------
        self.lb_case_id.setReadOnly(True)
        self.lb_image_no.setReadOnly(True)
        self.lb_region.setReadOnly(True)
        self.le_areas_focused.setReadOnly(True)
        self.le_ROIs_focused.setReadOnly(True)
        self.le_focus_relevance.setReadOnly(True)
        self.te_notes.setReadOnly(True)

        self.input_image_path = ""
        self.q_path_image_path = r"Dataset\ALS_QuPath_Images"
        self.q_path_img = None
        self.a_browse_image.triggered.connect(self.browse_image)

        self.cam, self.reconstructed_image = None, None

        # ------------------------------------------------------------------
        # Load model once at startup to avoid repeated loading on each inference.
        # Note: if you plan to change the model architecture or use a different set of weights, this is the line you'll update.
        # The rest of the code is designed to be model-agnostic as long as the same output format is maintained (i.e., predict_image and extract_output_cam_and_image return the expected outputs).
        # If you want to add functionality for switching between multiple models, you could extend the GUI with a dropdown to select models and then load the corresponding weights here based on the selection.
        # ------------------------------------------------------------------
        weights_path = r"ALS_Diagnostic_Model/src/model/fold_0_model_weights.pth"
        self.model = load_model(weights_path)

        # Disable slider until CAM exists to prevent user confusion (since slider has no effect until CAM is generated)
        self.s_focus_slider.setEnabled(False)

        self.lb_input_image.setAcceptDrops(True)
        self.lb_input_image.dragEnterEvent = self.drag_enter_event
        self.lb_input_image.dropEvent = self.drop_event
        self.lb_input_image.dragLeaveEvent = self.drag_leave_event

        #self.a_run_analysis.triggered.connect(self.run_analysis)
        self.s_focus_slider.valueChanged.connect(self.update_cam_mask)
        self.le_focus_toggle.textChanged.connect(self.update_slider)
        #self.a_clear.triggered.connect(self.clear_GUI)

    # ----------------------------------------------------------------------
    # Help Icon Installer
    # ----------------------------------------------------------------------
    # Dynamically attaches a small "?" button next to specific labels 
    # as because Qt Designer does not handle precise inline help alignment well.
    # This approach allows pixel-level control over placement.
    #
    # If adding a new help icon:
    #   1. Add a help text entry in self.help_texts
    #   2. Add the label mapping here
    #   3. Optionally fine-tune offsets
    # ----------------------------------------------------------------------
    def _install_help_icons(self):
        mapping = {
            self.findChild(qtw.QLabel, "label_16"): "Areas of Focus",
            self.findChild(qtw.QLabel, "label_29"): "Model Focus Viewer",
            self.findChild(qtw.QLabel, "label_2"): "No. Areas Focused",
            self.findChild(qtw.QLabel, "label_3"): "Areas with ROIs",
            self.findChild(qtw.QLabel, "label_4"): "Focus Relevance",
            self.findChild(qtw.QLabel, "lb_focus_slider"): "Focus Toggle",
        }

        # Individual fine-tuning offsets for each help icon (x, y) in pixels
        # These were determined through trial and error to achieve optimal positioning next to each label.
        offsets = {
            "Areas of Focus": (-355, 0),
            "Model Focus Viewer": (-325, 0),
            "No. Areas Focused": (-140, 15),
            "Areas with ROIs": (-140, 15),
            "Focus Relevance": (-160, 16),
            "Focus Toggle": (-192, 18),
        }

        for label, key in mapping.items():
            if label is None:
                continue

            parent = label.parent()

            btn = QToolButton(parent)
            btn.setText("?")
            btn.setFixedSize(18, 18)
            btn.setCursor(qtc.Qt.PointingHandCursor)

            btn.setStyleSheet("""
                QToolButton {
                    border: 1px solid #32BAC4;
                    border-radius: 9px;
                    color: #32BAC4;
                    background: transparent;
                    font-weight: bold;
                }
                QToolButton:hover {
                    background-color: rgba(50,186,196,40);
                }
            """)

            # base automatic position
            base_x = label.x() + label.width() + 6
            base_y = label.y() + (label.height() - btn.height()) // 2

            # apply individual manual offsets
            dx, dy = offsets[key]

            btn.move(base_x + dx, base_y + dy)

            btn.clicked.connect(lambda _, k=key, b=btn: self._show_help(k, b))
            btn.show()

            btn.installEventFilter(self)
            btn._help_key = key
        

    # Displays the help popup for a given key.
    # Automatically closes any previously active popup.
    def _show_help(self, key, button):
        if self._active_popup:
            self._active_popup.close()

        popup = HelpPopup(self.help_texts[key], self)
        popup.show_near(button)
        self._active_popup = popup


    # ----------------------------------------------------------------------
    # Drag-and-drop event handlers
    # ----------------------------------------------------------------------
    # These methods handle the drag-and-drop functionality for loading images.
    # The drag_enter_event and drag_leave_event methods provide visual feedback to the user by changing the appearance of the drop area.
    # The drop_event method processes the dropped file, validates it, and loads it into the GUI if it's a valid image.
    # ----------------------------------------------------------------------
    def drag_enter_event(self, event):

        if event.mimeData().hasUrls():
            self.lb_input_image.setStyleSheet("border: 2px dashed #32BAC4; color: white;")
            self.lb_input_image.setText("Drop image to analyse")
            event.acceptProposedAction()
        else:
            event.ignore()


    def drag_leave_event(self, event):
        self.lb_input_image.setStyleSheet("")
        
        # If an image was already loaded before drag started, restore it 
        if self.input_image_path and os.path.exists(self.input_image_path):
            try:
                pil_image = Image.open(self.input_image_path)
                pixmap = pil2pixmap(pil_image)
                self.lb_input_image.setPixmap(
                    pixmap.scaled(self.lb_input_image.size())
                )
            except Exception:
                self.lb_input_image.setStyleSheet("border: 2px dashed #32BAC4; color: white;")
                self.lb_input_image.setText("Drag or Browse \n an Image To Get Started")
        else:
            self.lb_input_image.setStyleSheet("border: 2px dashed #32BAC4; color: white;")
            self.lb_input_image.setText("Drag or Browse \n an Image To Get Started")
        
        event.accept()


    # When the user drops a file, this method is called to handle the event.
    def drop_event(self, event):
        self.lb_input_image.setStyleSheet("")
        self.lb_input_image.setText("Loading image...")
        urls = event.mimeData().urls()

        if not urls:
            return

        file_path = urls[0].toLocalFile()

        valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        if not os.path.exists(file_path):
            qtw.QMessageBox.warning(self, "File Error", "File does not exist.")
            return


        if not file_path.lower().endswith(valid_ext):
            qtw.QMessageBox.warning(self, "Unsupported File", "Please drop a valid image file.")
            return

        try:
            Image.open(file_path).verify()
        except Exception:
            qtw.QMessageBox.warning(self, "Invalid Image", "The file is not a valid image or is corrupted.")
            return

        try:
            self.clear_GUI()
            self.load_image_into_gui(file_path)

        except Exception as e:
            qtw.QMessageBox.critical(self, "Load Failed", f"Could not load dropped image.\n{str(e)}")
    

    # ----------------------------------------------------------------------
    # Image Loading and Analysis
    # ----------------------------------------------------------------------
    # This method takes an image path, loads the image, extracts metadata, finds the corresponding QuPath image, and runs the analysis pipeline.
    # ----------------------------------------------------------------------
    def load_image_into_gui(self, image_path):

        if not image_path or not os.path.exists(image_path):
            qtw.QMessageBox.warning(self, "File Error", "Selected image does not exist.")
            return
        
        self.input_image_path = image_path

        pil_image = Image.open(self.input_image_path)
        pixmap = pil2pixmap(pil_image)

        self.lb_input_image.setPixmap(
            pixmap.scaled(self.lb_input_image.size())
        )

        image_no, case_id, region = get_metadata(self.input_image_path)
        self.lb_image_no.setText(image_no)
        self.lb_case_id.setText(case_id)
        self.lb_region.setText(region)

        # Load corresponding QuPath image
        self.q_path_image_path = os.path.join(
            r"Dataset/ALS_QuPath_Images",
            f"{int(image_no)}.tif"
        )

        if not os.path.exists(self.q_path_image_path):
            qtw.QMessageBox.warning(
                self,
                "Dataset Pair Missing",
                f"Could not find matching QuPath image:\n{self.q_path_image_path}"
            )
            return
        
        try:
            self.q_path_img = Image.open(self.q_path_image_path)
        except Exception:
            qtw.QMessageBox.warning(
                self,
                "QuPath Image Error",
                "Matching QuPath image exists but could not be opened."
            )
            return

        pixmap = pil2pixmap(self.q_path_img)
        self.lb_qu_path_image.setPixmap(
            pixmap.scaled(self.lb_input_image.size())
        )

        self.te_notes.setHtml(get_image_metadata_str(self.input_image_path))

        # Run model + CAM + metrics
        self.run_analysis()


    @qtc.Slot()
    # This method is called when the user clicks the "Browse" button in the menu.
    def browse_image(self):

        print("called.")

        self.clear_GUI()

        self.input_image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )

        self.load_image_into_gui(self.input_image_path)

    
    def update_progress_bar(self, results:list):

        class_labels = ["control", "concordant", "discordant"]

        for label, value in zip(class_labels, results):
            bar = getattr(self, f"p_bar_{label}")
            bar.setValue(int(value*100))

        ranked = sorted(
            zip(class_labels, results),
            key=lambda x: x[1],
            reverse=True  # Highest confidence first
        )

        rank_colors = ["#32BAC4", "#01686B", "#67F0E0"]  # Green, Orange, Yellow

        for i, (label, _) in enumerate(ranked):
            bar = getattr(self, f"p_bar_{label}")
            color = rank_colors[i]
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    text-align: center;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    width: 10px;
                }}
            """)

    @qtc.Slot()
    # This is where the model inference happens.
    #
    # High-level flow:
    #   1. Predict class probabilities using the model
    #   2. Generate LayerCAM outputs (raw CAM and reconstructed image)
    #   3. Overlay CAM on raw tissue image for visualisation
    #   4. Generate circular attention regions from CAM
    #   5. Compare focus regions to QuPath ROIs 
    #   6. Update metrics in GUI 
    # ----------------------------------------------------------------------
    def run_analysis(self):

        # Step 1: Model Inference
        # The predict_image function is expected to return:
        #   - predicted_class: the class label with the highest confidence
        #   - probs: a list of probabilities for each class (used for progress bars)
        #   - image_tensor: the preprocessed image tensor used for CAM generation
        #   - class_idx: the index of the predicted class (used for CAM generation) 
        # Note: if you change the model architecture or the way predictions are made, make sure to update the predict_image function accordingly so that it returns these expected outputs.
        # The rest of the code in this method relies on these outputs to function correctly.
        predicted_class, probs, image_tensor, class_idx = predict_image(self.model, self.input_image_path)

        self.update_progress_bar(probs)

        self.lb_prediction.setText(predicted_class)

        # Step 2: Generate LayerCAM outputs
        # The extract_output_cam_and_image function is expected to return:
        #   - model_raw_output: the raw output from the model's forward pass (not typically used directly in the GUI, but can be useful for debugging or advanced visualisations)
        #   - cam: the generated Class Activation Map (CAM) for the predicted class
        #   - reconstructed_image: the image reconstructed from the model's internal features, which is used for overlay visualisations
        # The target_layers argument specifies which layers of the model to use for CAM generation. 
            ## This is based on the architecture of the model and where the most relevant features are extracted. 
            ## If you change the model architecture, you may need to update these target layers to ensure meaningful CAM outputs.
        self.model_raw_output, self.cam, self.reconstructed_image = extract_output_cam_and_image(self.model, image_tensor, class_idx, target_layers=[
        self.model.model.features.denseblock1,
        self.model.model.features.denseblock2,
        self.model.model.features.denseblock3,
        self.model.model.features.denseblock4,
    ])
        
        # Step 3: Overlay CAM on raw tissue image for visualisation
        # The generate_layercam_overlay function takes the CAM and the reconstructed image to create a visual overlay that highlights the regions of the original image that the model is focusing on.
        # This visualisation helps users understand which parts of the tissue the model considers important for its prediction.
        # If the CAM generation logic changes (e.g., different layers, different processing), make sure to update the generate_layercam_overlay function accordingly to maintain accurate visualisations.
        cam_image = generate_layercam_overlay(self.cam, self.reconstructed_image)
        pixmap = pil2pixmap(cam_image)
        self.lb_cam_image.setPixmap(
            pixmap.scaled(
                self.lb_cam_image.size()
            )
        )

        # Step 4: Generate circular attention regions from CAM 
        # The generate_layercam_circles function processes the CAM to identify distinct high-attention regions and represents them as circles.
        # These circles are then overlaid on the QuPath annotated image to visually compare the model's focus areas with the biologically relevant ROIs defined by the pathologist.
        # The threshold parameter controls how strict the CAM filtering is for defining these focus regions. 
        # If you want to adjust the sensitivity of focus region detection, you can modify the default threshold value or allow it to be set dynamically through the GUI (which is what the focus toggle slider does).
        self.s_focus_slider.setValue(80)
        filtered_image, circles = generate_layercam_circles(self.cam, self.q_path_img.copy(), threshold=0.8)
        pixmap = pil2pixmap(filtered_image)
        self.lb_cam_circle_image.setPixmap(
            pixmap.scaled(
                self.lb_cam_circle_image.size()
            )
        )

        # Step 5: Compare focus regions to QuPath ROIs
        # The extract_color_masks function is used to extract the binary masks for the different annotations in the QuPath image.
        # In this case, we are specifically interested in the "yellow" mask, which represents the ROIs annotated by the pathologist.
        # The quantify_focus function takes the detected circles (model focus regions) and the cells_mask (QuPath ROIs) to calculate:
        #   - hits: the number of focus regions that overlap with ROIs (true positives)
        #   - misses: the number of focus regions that do not overlap with ROIs (false positives)
        #  - precision: the ratio of hits to total focus regions, indicating how relevant the model's attention is to the biologically defined ROIs.
        # If you want to add additional metrics (e.g., recall, F1 score), you can modify the quantify_focus function to calculate those as well and then update the GUI to display them.
        cells_mask = extract_color_masks(self.q_path_img)["yellow"]
        hits, misses, precision = quantify_focus(circles, cells_mask)
        self.le_areas_focused.setText(str(hits+misses))
        self.le_ROIs_focused.setText(str(hits))
        self.le_focus_relevance.setText(f"{precision:.2f}")


        # Step 6: Update CAM visualisation based on slider threshold
        # The focus toggle slider allows users to adjust the threshold for what is considered a "focus region" in the CAM.
        # By default, it is set to 80%, which means that only the top 20% of CAM activations are considered as focus regions. 
        # Adjusting this slider will dynamically update the CAM visualisation and the detected focus regions, allowing users to explore how the model's attention changes with different thresholds.
        self.s_focus_slider.setValue(80)
        masked_image = generate_layercam_boundary(self.cam, self.reconstructed_image.copy(), threshold=0.8)
        pixmap = pil2pixmap(masked_image)
        self.lb_selected_focus_image.setPixmap(
            pixmap.scaled(
                self.lb_selected_focus_image.size()
            )
        )
        # Enable slider now that CAM exists
        self.s_focus_slider.setEnabled(True)


    @qtc.Slot()
    # This method is called whenever the text in the focus toggle line edit changes.
    # It updates the slider value to match the text input, allowing users to type in specific threshold values if they prefer that over using the slider.
    def update_slider(self):

        value = int(self.le_focus_toggle.text())
        self.s_focus_slider.setValue(value)


    @qtc.Slot()
    # This method is called whenever the slider value changes.
    # It updates the CAM visualisation and the detected focus regions based on the new threshold value.
    # The threshold is calculated as a percentage (slider value divided by 100) and is used to filter the CAM activations for defining focus regions.
    def update_cam_mask(self):

        if self.cam is None:
            return

        self.lb_cam_circle_image.clear()
        self.lb_selected_focus_image.clear()
        self.le_focus_toggle.setText(f"{self.s_focus_slider.value()}")
        threshold = self.s_focus_slider.value()/100


        filtered_image, circles = generate_layercam_circles(self.cam, self.q_path_img.copy(), threshold=threshold)
        pixmap = pil2pixmap(filtered_image)
        self.lb_cam_circle_image.setPixmap(
            pixmap.scaled(
                self.lb_cam_circle_image.size()
            )
        )

        qpath_img = Image.open(self.q_path_image_path)
        cells_mask = extract_color_masks(qpath_img)["yellow"]
        hits, misses, precision = quantify_focus(circles, cells_mask)
        self.le_areas_focused.setText(str(hits+misses))
        self.le_ROIs_focused.setText(str(hits))
        self.le_focus_relevance.setText(f"{precision:.2f}")

        masked_image = generate_layercam_boundary(self.cam, self.reconstructed_image.copy(), threshold=threshold)
        pixmap = pil2pixmap(masked_image)
        self.lb_selected_focus_image.setPixmap(
            pixmap.scaled(
                self.lb_selected_focus_image.size()
            )
        )

    # This method is an event filter that is used to detect when the mouse enters or leaves a help icon button.
    # When the mouse enters a help icon, the corresponding help popup is shown.
    # When the mouse leaves the help icon, the popup is closed.
    def eventFilter(self, obj, event):
        if isinstance(obj, QToolButton) and hasattr(obj, "_help_key"):
            if event.type() == qtc.QEvent.Enter:
                self._show_help(obj._help_key, obj)
            elif event.type() == qtc.QEvent.Leave:
                if self._active_popup:
                    self._active_popup.close()
                    self._active_popup = None
        return super().eventFilter(obj, event)


    # This method is overridden to ensure that if the user clicks anywhere outside of an active help popup, the popup will be closed.
    def mousePressEvent(self, event):
        if self._active_popup:
            self._active_popup.close()
            self._active_popup = None
        super().mousePressEvent(event)


    @qtc.Slot()

    # ----------------------------------------------------------------------
    # clear_GUI()
    # ----------------------------------------------------------------------
    # Resets the interface back to its initial state, when loading a new image.
    # If new UI elements are added in the future, reset them here to ensure a clean slate for each new analysis.
    # ----------------------------------------------------------------------
    def clear_GUI(self):
        self.input_image_path = ""
        self.q_path_image_path = r"Dataset/ALS_QuPath_Images"
        self.qpath_img = None
        self.lb_input_image.setText("Drag or Browse \n an Image To Get Started")
        self.lb_cam_image.setText("Waiting for model to run...")
        self.lb_cam_circle_image.setText("Waiting for model to run...")
        self.lb_selected_focus_image.setText("Waiting for model to run...")
        self.p_bar_control.setValue(0)
        self.p_bar_concordant.setValue(0)
        self.p_bar_discordant.setValue(0)
        self.lb_prediction.setText("Waiting...")

        self.lb_image_no.setText("")
        self.lb_case_id.setText("")
        self.lb_region.setText("")

        self.s_focus_slider.setEnabled(False)
        self.cam = None


# Run the application
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = GuiHome()
    window.show()
    sys.exit(app.exec())