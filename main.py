#!/usr/bin/env python3
"""
3D Model Generator

This script converts 2D images to 3D models (.obj or .stl format).
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSpinBox, QDoubleSpinBox,
    QComboBox, QStackedWidget, QLineEdit, QProgressBar, QMessageBox,
    QGroupBox, QRadioButton, QSlider, QScrollArea, QStyleFactory
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QLinearGradient, QRadialGradient, QBrush

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("3D_Model_Generator")

# Custom styles
class StyleHelper:
    @staticmethod
    def apply_app_style(app):
        """Apply global application style"""
        app.setStyle(QStyleFactory.create("Fusion"))
        
        # Set color palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)

    @staticmethod
    def create_linear_gradient_style(start_color, end_color, direction="to-bottom"):
        """Create a stylesheet with linear gradient background"""
        if direction == "to-bottom":
            return f"background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {start_color}, stop:1 {end_color});"
        elif direction == "to-right":
            return f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {start_color}, stop:1 {end_color});"
        elif direction == "diagonal":
            return f"background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {start_color}, stop:1 {end_color});"
        
    @staticmethod
    def create_radial_gradient_style(center_color, edge_color):
        """Create a stylesheet with radial gradient background"""
        return f"background: qradialgradient(cx:0.5, cy:0.5, radius:1, fx:0.5, fy:0.5, stop:0 {center_color}, stop:1 {edge_color});"
        
    @staticmethod
    def get_button_style(is_primary=False):
        """Get button stylesheet"""
        if is_primary:
            base_style = StyleHelper.create_linear_gradient_style("#4a9fd8", "#2980b9")
            hover_style = StyleHelper.create_linear_gradient_style("#57b2ed", "#3498db")
        else:
            base_style = StyleHelper.create_linear_gradient_style("#5d5d5d", "#444444")
            hover_style = StyleHelper.create_linear_gradient_style("#666666", "#555555")
            
        return f"""
            QPushButton {{
                {base_style}
                border: none;
                padding: 10px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
            }}
            QPushButton:hover {{
                {hover_style}
            }}
            QPushButton:pressed {{
                border: 1px solid #1a1a1a;
            }}
        """
    
    @staticmethod
    def get_group_box_style():
        """Get groupbox stylesheet"""
        gradient = StyleHelper.create_linear_gradient_style("#3a3a3a", "#2d2d2d")
        return f"""
            QGroupBox {{
                {gradient}
                border: 1px solid #5d5d5d;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                color: #cccccc;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """
        
    @staticmethod
    def get_progress_bar_style():
        """Get progress bar stylesheet"""
        return """
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
                color: white;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #2980b9);
                border-radius: 3px;
            }
        """

# Worker thread for model generation
class ModelGeneratorWorker(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, input_path, output_path, output_format, depth, resolution):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.output_format = output_format
        self.depth = depth
        self.resolution = resolution
    
    def run(self):
        try:
            # Load and process image
            self.progress_signal.emit(10)
            image_array = load_image(self.input_path)
            
            self.progress_signal.emit(30)
            height_map = generate_height_map(image_array, self.depth)
            
            self.progress_signal.emit(50)
            # Generate the 3D model in the requested format
            if self.output_format == "obj":
                generate_obj(height_map, self.resolution, self.output_path)
            else:  # stl
                generate_stl(height_map, self.resolution, self.output_path)
            
            self.progress_signal.emit(100)
            self.finished_signal.emit(str(self.output_path))
        except Exception as e:
            self.error_signal.emit(str(e))

# Worker thread for text-to-model generation
class TextModelGeneratorWorker(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, text_prompt, output_path, output_format, detail_level=1.0, model_type="shap-e"):
        super().__init__()
        self.text_prompt = text_prompt
        self.output_path = output_path
        self.output_format = output_format
        self.detail_level = detail_level
        self.model_type = model_type
    
    def run(self):
        try:
            # Import necessary libraries here to avoid loading them unless needed
            import torch
            from pathlib import Path
            
            self.progress_signal.emit(10)
            self.progress_signal.emit(20)
            
            # Generate 3D model from text prompt
            if self.model_type == "shap-e":
                self.progress_signal.emit(30)
                model_output = self.generate_shape_model()
                self.progress_signal.emit(70)
            else:
                # Fallback to simpler generation method
                self.progress_signal.emit(30)
                model_output = self.generate_simple_model()
                self.progress_signal.emit(70)
                
            # Export to requested format
            if self.output_format == "obj":
                self.export_to_obj(model_output)
            else:  # stl
                self.export_to_stl(model_output)
                
            self.progress_signal.emit(100)
            self.finished_signal.emit(str(self.output_path))
        except Exception as e:
            import traceback
            logger.error(f"Error in text-to-model generation: {str(e)}\n{traceback.format_exc()}")
            self.error_signal.emit(str(e))
    
    def generate_shape_model(self):
        """Generate a 3D model using Shap-E."""
        try:
            # Use shap-e to generate 3D content from text
            import torch
            from shap_e.diffusion.sample import sample_latents
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from shap_e.models.download import load_config, load_model
            from shap_e.util.notebooks import create_pan_cameras, decode_latent_mesh
            
            self.progress_signal.emit(35)
            logger.info(f"Initializing Shap-E model for text prompt: {self.text_prompt}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Load model
            self.progress_signal.emit(40)
            xm = load_model('transmitter', device=device)
            model = load_model('text300M', device=device)
            diffusion = diffusion_from_config(load_config('diffusion'))
            
            self.progress_signal.emit(50)
            # Generate latents from text prompt
            batch_size = 1
            guidance_scale = 15.0
            
            # Adjust parameters based on detail level
            steps = int(64 * self.detail_level)
            
            # Generate latents
            logger.info(f"Generating latents with {steps} steps and guidance scale {guidance_scale}")
            latents = sample_latents(
                batch_size=batch_size,
                model=model,
                diffusion=diffusion,
                guidance_scale=guidance_scale,
                model_kwargs=dict(texts=[self.text_prompt] * batch_size),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
            
            self.progress_signal.emit(65)
            
            # Decode the latents
            logger.info("Decoding latents into mesh")
            mesh_data = decode_latent_mesh(xm, latents[0]).tri_mesh()
            
            return mesh_data
        except ImportError as e:
            logger.error(f"Shap-E model not available: {e}")
            raise ImportError("The Shap-E module is required for text-to-3D generation. Please make sure it's installed correctly.")
    
    def generate_simple_model(self):
        """Fallback method for simple 3D model generation."""
        logger.info(f"Generating simple model from text: {self.text_prompt}")
        
        # In a real implementation, this would use a simpler model
        # For now, we'll create a basic shape based on text
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Create an image with the text
        img_size = 512
        img = Image.new('L', (img_size, img_size), color=0)
        draw = ImageDraw.Draw(img)
        
        # Calculate text size and position
        font_size = 36
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            
        text_width, text_height = draw.textsize(self.text_prompt, font=font)
        position = ((img_size - text_width) // 2, (img_size - text_height) // 2)
        
        # Draw text
        draw.text(position, self.text_prompt, fill=255, font=font)
        
        # Convert to height map
        height_map = np.array(img)
        
        # Create simple mesh data structure
        return height_map
    
    def export_to_obj(self, model_output):
        """Export the generated model to OBJ format."""
        logger.info(f"Exporting model to OBJ: {self.output_path}")
        
        # If model_output is a Shap-E mesh object
        if hasattr(model_output, 'write_obj'):
            with open(self.output_path, 'w') as f:
                model_output.write_obj(f)
                
        # If model_output is a height map array from the simple method
        elif isinstance(model_output, np.ndarray):
            generate_obj(model_output, 100, self.output_path)
    
    def export_to_stl(self, model_output):
        """Export the generated model to STL format."""
        logger.info(f"Exporting model to STL: {self.output_path}")
        
        # If model_output is a Shap-E mesh object
        if hasattr(model_output, 'export_stl'):
            model_output.export_stl(self.output_path)
            
        # If model_output is a height map array from the simple method
        elif isinstance(model_output, np.ndarray):
            generate_stl(model_output, 100, self.output_path)

# GUI main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("3D Model Generator")
        self.setMinimumSize(800, 600)
        
        # Apply custom styles
        StyleHelper.apply_app_style(QApplication.instance())
        
        # Create central widget with layout for content and footer
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create stacked widget for different pages
        self.stacked_widget = QStackedWidget()
        
        # Create pages
        self.create_home_page()
        self.create_input_page()
        self.create_settings_page()
        self.create_generation_page()
        self.create_result_page()
        self.create_text_input_page()
        
        # Set initial page
        self.stacked_widget.setCurrentIndex(0)
        
        # Create footer with attribution (hardcoded - not removable)
        footer = QLabel("Made by Chaitanya Sharma - <a href='https://github.com/sharmachaitanya945'>github.com/sharmachaitanya945</a>")
        footer.setOpenExternalLinks(True)  # Allow clicking on the link
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #cccccc; background-color: #2d2d2d; padding: 8px; font-size: 10pt; border-top: 1px solid #555555;")
        footer.setMinimumHeight(40)
        
        # Add widgets to main layout
        main_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(footer)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.setCentralWidget(central_widget)
        
    def create_home_page(self):
        page = QWidget()
        
        # Apply radial gradient background to home page
        radial_gradient = QRadialGradient(0.5, 0.5, 1.0, 0.5, 0.5)
        radial_gradient.setCoordinateMode(QRadialGradient.ObjectBoundingMode)
        radial_gradient.setColorAt(0, QColor(40, 40, 45))
        radial_gradient.setColorAt(1, QColor(20, 20, 25))
        
        palette = page.palette()
        palette.setBrush(QPalette.Window, QBrush(radial_gradient))
        page.setPalette(palette)
        page.setAutoFillBackground(True)
        
        layout = QVBoxLayout(page)
        
        # Title
        title_label = QLabel("3D Model Generator")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: white;")
        
        # Description
        desc_label = QLabel("Create 3D models from images or text descriptions")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("font-size: 14pt; color: #cccccc;")
        
        # Option buttons
        options_container = QWidget()
        container_layout = QHBoxLayout(options_container)
        
        # Image to 3D option
        image_option = QGroupBox("Image to 3D")
        image_option.setMinimumWidth(300)
        image_option.setStyleSheet(StyleHelper.get_group_box_style())
        image_layout = QVBoxLayout(image_option)
        
        image_icon = QLabel()
        image_icon.setAlignment(Qt.AlignCenter)
        # Create a placeholder icon instead of loading from file
        image_pixmap = self.create_placeholder_icon("Image", "#3498db")
        image_icon.setPixmap(image_pixmap)
        image_icon.setMinimumHeight(80)
        
        image_desc = QLabel("Convert a 2D image into a 3D model based on brightness values")
        image_desc.setWordWrap(True)
        image_desc.setAlignment(Qt.AlignCenter)
        
        image_btn = QPushButton("Start Image to 3D")
        image_btn.setStyleSheet(StyleHelper.get_button_style(is_primary=True))
        image_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        
        image_layout.addWidget(image_icon)
        image_layout.addWidget(image_desc)
        image_layout.addWidget(image_btn)
        
        # Text to 3D option
        text_option = QGroupBox("Text to 3D")
        text_option.setMinimumWidth(300)
        text_option.setStyleSheet(StyleHelper.get_group_box_style())
        text_layout = QVBoxLayout(text_option)
        
        text_icon = QLabel()
        text_icon.setAlignment(Qt.AlignCenter)
        # Create a placeholder icon instead of loading from file
        text_pixmap = self.create_placeholder_icon("Text", "#e74c3c")
        text_icon.setPixmap(text_pixmap)
        text_icon.setMinimumHeight(80)
        
        text_desc = QLabel("Generate a 3D model from a text description using AI")
        text_desc.setWordWrap(True)
        text_desc.setAlignment(Qt.AlignCenter)
        
        text_btn = QPushButton("Start Text to 3D")
        text_btn.setStyleSheet(StyleHelper.get_button_style(is_primary=True))
        text_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(5))  # Index of text input page
        
        text_layout.addWidget(text_icon)
        text_layout.addWidget(text_desc)
        text_layout.addWidget(text_btn)
        
        # Add options to container
        container_layout.addWidget(image_option)
        container_layout.addWidget(text_option)
        
        # Layout
        layout.addStretch()
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addSpacing(30)
        layout.addWidget(options_container)
        layout.addStretch()
        
        self.stacked_widget.addWidget(page)
    
    def create_input_page(self):
        page = QWidget()
        
        # Apply linear gradient background to input page
        linear_gradient = QLinearGradient(0, 0, 0, 1)
        linear_gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        linear_gradient.setColorAt(0, QColor(30, 30, 35))
        linear_gradient.setColorAt(1, QColor(45, 45, 55))
        
        palette = page.palette()
        palette.setBrush(QPalette.Window, QBrush(linear_gradient))
        page.setPalette(palette)
        page.setAutoFillBackground(True)
        
        layout = QVBoxLayout(page)
        
        # Page title
        title_label = QLabel("Select Input Image")
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: white;")
        
        # Image selection
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setPlaceholderText("Select an input image...")
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(StyleHelper.get_button_style())
        browse_btn.clicked.connect(self.browse_input_image)
        
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_path_edit)
        image_layout.addWidget(browse_btn)
        
        # Image preview with gradient border
        preview_container = QWidget()
        preview_container.setMinimumSize(420, 320)
        preview_container_style = StyleHelper.create_linear_gradient_style("#4a9fd8", "#2980b9", "diagonal")
        preview_container.setStyleSheet(f"border-radius: 10px; padding: 10px; {preview_container_style}")
        
        preview_layout = QVBoxLayout(preview_container)
        
        self.image_preview = QLabel("Image Preview")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(400, 300)
        self.image_preview.setStyleSheet("background-color: #2d2d2d; color: #999999; border-radius: 5px;")
        
        preview_layout.addWidget(self.image_preview)
        
        # Navigation buttons
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet(StyleHelper.get_button_style())
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        
        next_btn = QPushButton("Next")
        next_btn.setStyleSheet(StyleHelper.get_button_style(is_primary=True))
        next_btn.clicked.connect(self.go_to_settings)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(back_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(next_btn)
        
        # Layout
        layout.addWidget(title_label)
        layout.addSpacing(20)
        layout.addLayout(image_layout)
        layout.addSpacing(20)
        layout.addWidget(preview_container)
        layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.stacked_widget.addWidget(page)
    
    def create_settings_page(self):
        page = QWidget()
        
        # Apply diagonal gradient background to settings page
        linear_gradient = QLinearGradient(0, 0, 1, 1)
        linear_gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        linear_gradient.setColorAt(0, QColor(35, 40, 50))
        linear_gradient.setColorAt(1, QColor(55, 60, 70))
        
        palette = page.palette()
        palette.setBrush(QPalette.Window, QBrush(linear_gradient))
        page.setPalette(palette)
        page.setAutoFillBackground(True)
        
        layout = QVBoxLayout(page)
        
        # Page title
        title_label = QLabel("Configure 3D Model Settings")
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: white;")
        
        # Output format
        format_group = QGroupBox("Output Format")
        format_group.setStyleSheet(StyleHelper.get_group_box_style())
        format_layout = QHBoxLayout(format_group)
        
        self.obj_radio = QRadioButton("OBJ")
        self.stl_radio = QRadioButton("STL")
        self.obj_radio.setChecked(True)
        
        format_layout.addWidget(self.obj_radio)
        format_layout.addWidget(self.stl_radio)
        
        # Depth settings with slider
        depth_group = QGroupBox("Model Depth")
        depth_group.setStyleSheet(StyleHelper.get_group_box_style())
        depth_layout = QVBoxLayout(depth_group)
        
        depth_slider_layout = QHBoxLayout()
        depth_label = QLabel("Maximum Depth:")
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(1.0, 100.0)
        self.depth_spin.setValue(10.0)
        self.depth_spin.setSingleStep(1.0)
        
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(10, 1000)  # 1.0 to 100.0 with one decimal precision
        self.depth_slider.setValue(int(self.depth_spin.value() * 10))
        
        # Connect slider and spin box
        self.depth_slider.valueChanged.connect(lambda v: self.depth_spin.setValue(v / 10.0))
        self.depth_spin.valueChanged.connect(lambda v: self.depth_slider.setValue(int(v * 10)))
        
        depth_slider_layout.addWidget(depth_label)
        depth_slider_layout.addWidget(self.depth_spin)
        
        depth_layout.addLayout(depth_slider_layout)
        depth_layout.addWidget(self.depth_slider)
        
        # Resolution settings with slider
        res_group = QGroupBox("Model Resolution")
        res_group.setStyleSheet(StyleHelper.get_group_box_style())
        res_layout = QVBoxLayout(res_group)
        
        res_slider_layout = QHBoxLayout()
        res_label = QLabel("Resolution:")
        self.res_spin = QSpinBox()
        self.res_spin.setRange(50, 500)
        self.res_spin.setValue(100)
        self.res_spin.setSingleStep(10)
        
        self.res_slider = QSlider(Qt.Horizontal)
        self.res_slider.setRange(50, 500)
        self.res_slider.setValue(self.res_spin.value())
        
        # Connect slider and spin box
        self.res_slider.valueChanged.connect(self.res_spin.setValue)
        self.res_spin.valueChanged.connect(self.res_slider.setValue)
        
        res_slider_layout.addWidget(res_label)
        res_slider_layout.addWidget(self.res_spin)
        
        res_layout.addLayout(res_slider_layout)
        res_layout.addWidget(self.res_slider)
        
        # Output path
        output_group = QGroupBox("Output File")
        output_group.setStyleSheet(StyleHelper.get_group_box_style())
        output_layout = QVBoxLayout(output_group)
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setPlaceholderText("Output file will be generated automatically...")
        
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.setStyleSheet(StyleHelper.get_button_style())
        browse_output_btn.clicked.connect(self.browse_output_path)
        
        output_file_layout = QHBoxLayout()
        output_file_layout.addWidget(self.output_path_edit)
        output_file_layout.addWidget(browse_output_btn)
        
        output_layout.addLayout(output_file_layout)
        
        # Navigation buttons
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet(StyleHelper.get_button_style())
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        
        next_btn = QPushButton("Generate 3D Model")
        next_btn.setStyleSheet(StyleHelper.get_button_style(is_primary=True))
        next_btn.clicked.connect(self.start_generation)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(back_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(next_btn)
        
        # Layout
        layout.addWidget(title_label)
        layout.addSpacing(20)
        layout.addWidget(format_group)
        layout.addWidget(depth_group)
        layout.addWidget(res_group)
        layout.addWidget(output_group)
        layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.stacked_widget.addWidget(page)
    
    def create_generation_page(self):
        page = QWidget()
        
        # Apply linear gradient background to generation page
        linear_gradient = QLinearGradient(0, 0, 1, 1)
        linear_gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        linear_gradient.setColorAt(0, QColor(35, 40, 50))
        linear_gradient.setColorAt(1, QColor(55, 60, 70))
        
        palette = page.palette()
        palette.setBrush(QPalette.Window, QBrush(linear_gradient))
        page.setPalette(palette)
        page.setAutoFillBackground(True)
        
        layout = QVBoxLayout(page)
        
        # Page title
        title_label = QLabel("Generating 3D Model")
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: white;")
        
        # Progress indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setStyleSheet(StyleHelper.get_progress_bar_style())
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(StyleHelper.get_button_style())
        cancel_btn.clicked.connect(self.cancel_generation)
        
        # Layout
        layout.addWidget(title_label)
        layout.addSpacing(20)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(cancel_btn, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(page)
    
    def create_result_page(self):
        page = QWidget()
        
        # Apply radial gradient background to result page
        radial_gradient = QRadialGradient(0.5, 0.5, 1.0, 0.5, 0.5)
        radial_gradient.setCoordinateMode(QRadialGradient.ObjectBoundingMode)
        radial_gradient.setColorAt(0, QColor(50, 60, 70))
        radial_gradient.setColorAt(1, QColor(30, 35, 40))
        
        palette = page.palette()
        palette.setBrush(QPalette.Window, QBrush(radial_gradient))
        page.setPalette(palette)
        page.setAutoFillBackground(True)
        
        layout = QVBoxLayout(page)
        
        # Page title with gradient text effect
        title_label = QLabel("3D Model Generated")
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: white;")
        
        # Result information with gradient background
        result_container = QWidget()
        container_style = StyleHelper.create_linear_gradient_style("#2d2d2d", "#404040")
        result_container.setStyleSheet(f"border-radius: 8px; {container_style}")
        result_container_layout = QVBoxLayout(result_container)
        
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color: #cccccc; padding: 10px;")
        
        result_container_layout.addWidget(self.result_label)
        
        # Navigation buttons
        new_btn = QPushButton("Create New Model")
        new_btn.setStyleSheet(StyleHelper.get_button_style())
        new_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        
        open_folder_btn = QPushButton("Open Containing Folder")
        open_folder_btn.setStyleSheet(StyleHelper.get_button_style(is_primary=True))
        open_folder_btn.clicked.connect(self.open_output_folder)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(new_btn)
        btn_layout.addWidget(open_folder_btn)
        
        # Layout
        layout.addWidget(title_label)
        layout.addSpacing(20)
        layout.addWidget(result_container)
        layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.stacked_widget.addWidget(page)
    
    def create_text_input_page(self):
        page = QWidget()
        
        # Apply linear gradient background to text input page
        linear_gradient = QLinearGradient(0, 0, 0, 1)
        linear_gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        linear_gradient.setColorAt(0, QColor(35, 45, 55))
        linear_gradient.setColorAt(1, QColor(50, 60, 70))
        
        palette = page.palette()
        palette.setBrush(QPalette.Window, QBrush(linear_gradient))
        page.setPalette(palette)
        page.setAutoFillBackground(True)
        
        layout = QVBoxLayout(page)
        
        # Page title
        title_label = QLabel("Enter Text Description")
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: white;")
        
        # Text prompt entry
        prompt_group = QGroupBox("Text Description")
        prompt_group.setStyleSheet(StyleHelper.get_group_box_style())
        prompt_layout = QVBoxLayout(prompt_group)
        
        prompt_info = QLabel("Describe the 3D object you want to create. Be specific and detailed.")
        prompt_info.setWordWrap(True)
        
        self.text_prompt_edit = QLineEdit()
        self.text_prompt_edit.setPlaceholderText("E.g., A red apple with a stem and leaf, realistic texture")
        
        prompt_layout.addWidget(prompt_info)
        prompt_layout.addWidget(self.text_prompt_edit)
        
        # Example prompts
        examples_group = QGroupBox("Example Prompts")
        examples_group.setStyleSheet(StyleHelper.get_group_box_style())
        examples_layout = QVBoxLayout(examples_group)
        
        example_prompts = [
            "A geometric cube with holes on each face",
            "A ceramic coffee mug with a handle",
            "A pine tree with detailed branches and needles",
            "A simple chair with four legs and a backrest",
            "A classic sports car with aerodynamic design"
        ]
        
        for prompt in example_prompts:
            btn = QPushButton(prompt)
            btn.setStyleSheet("text-align: left; padding: 5px;")
            btn.clicked.connect(lambda checked, p=prompt: self.text_prompt_edit.setText(p))
            examples_layout.addWidget(btn)
        
        # Model settings
        settings_group = QGroupBox("Model Settings")
        settings_group.setStyleSheet(StyleHelper.get_group_box_style())
        settings_layout = QVBoxLayout(settings_group)
        
        # Output format
        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        self.text_obj_radio = QRadioButton("OBJ")
        self.text_stl_radio = QRadioButton("STL")
        self.text_obj_radio.setChecked(True)
        
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.text_obj_radio)
        format_layout.addWidget(self.text_stl_radio)
        
        # Detail level
        detail_layout = QHBoxLayout()
        detail_label = QLabel("Detail Level:")
        self.detail_spin = QDoubleSpinBox()
        self.detail_spin.setRange(0.5, 2.0)
        self.detail_spin.setValue(1.0)
        self.detail_spin.setSingleStep(0.1)
        
        self.detail_slider = QSlider(Qt.Horizontal)
        self.detail_slider.setRange(5, 20)  # 0.5 to 2.0 with one decimal precision
        self.detail_slider.setValue(int(self.detail_spin.value() * 10))
        
        # Connect slider and spin box
        self.detail_slider.valueChanged.connect(lambda v: self.detail_spin.setValue(v / 10.0))
        self.detail_spin.valueChanged.connect(lambda v: self.detail_slider.setValue(int(v * 10)))
        
        detail_layout.addWidget(detail_label)
        detail_layout.addWidget(self.detail_spin)
        
        # Output path
        output_layout = QHBoxLayout()
        self.text_output_path_edit = QLineEdit()
        self.text_output_path_edit.setReadOnly(True)
        self.text_output_path_edit.setPlaceholderText("Output file will be generated automatically...")
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(StyleHelper.get_button_style())
        browse_btn.clicked.connect(self.browse_text_output_path)
        
        output_layout.addWidget(self.text_output_path_edit)
        output_layout.addWidget(browse_btn)
        
        settings_layout.addLayout(format_layout)
        settings_layout.addLayout(detail_layout)
        settings_layout.addWidget(self.detail_slider)
        settings_layout.addLayout(output_layout)
        
        # Navigation buttons
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet(StyleHelper.get_button_style())
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        
        generate_btn = QPushButton("Generate 3D Model")
        generate_btn.setStyleSheet(StyleHelper.get_button_style(is_primary=True))
        generate_btn.clicked.connect(self.start_text_generation)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(back_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(generate_btn)
        
        # Layout
        layout.addWidget(title_label)
        layout.addSpacing(10)
        layout.addWidget(prompt_group)
        layout.addWidget(examples_group)
        layout.addWidget(settings_group)
        layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.stacked_widget.addWidget(page)
    
    def browse_input_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path_edit.setText(file_path)
            self.update_image_preview(file_path)
    
    def update_image_preview(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio)
            self.image_preview.setPixmap(pixmap)
    
    def browse_output_path(self):
        # Determine default extension based on selected format
        ext = "obj" if self.obj_radio.isChecked() else "stl"
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output Path", "", f"3D Model (*.{ext})")
        if file_path:
            self.output_path_edit.setText(file_path)
    
    def go_to_settings(self):
        if not self.image_path_edit.text():
            QMessageBox.warning(self, "Input Required", "Please select an input image first.")
            return
        self.stacked_widget.setCurrentIndex(2)
    
    def start_generation(self):
        # Get input image path
        input_path = self.image_path_edit.text()
        if not input_path:
            QMessageBox.warning(self, "Input Required", "Please select an input image first.")
            return
        
        # Determine output format
        output_format = "obj" if self.obj_radio.isChecked() else "stl"
        
        # Get output path or generate one
        output_path = self.output_path_edit.text()
        if not output_path:
            # Generate output filename
            script_dir = Path(__file__).parent.resolve()
            input_path_obj = Path(input_path)
            stem = input_path_obj.stem
            output_filename = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            output_path = script_dir / "outputs" / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path_edit.setText(str(output_path))
        
        # Get depth and resolution settings
        depth = self.depth_spin.value()
        resolution = self.res_spin.value()
        
        # Switch to generation page
        self.stacked_widget.setCurrentIndex(3)
        
        # Start generator worker thread
        self.worker = ModelGeneratorWorker(input_path, output_path, output_format, depth, resolution)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.generation_finished)
        self.worker.error_signal.connect(self.generation_error)
        self.worker.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
        # Update status text based on progress
        if value < 20:
            self.status_label.setText("Loading and processing image...")
        elif value < 50:
            self.status_label.setText("Generating height map...")
        elif value < 90:
            self.status_label.setText("Creating 3D model...")
        else:
            self.status_label.setText("Finalizing output...")
    
    def generation_finished(self, output_path):
        self.result_label.setText(f"3D model successfully generated!\nFile saved as:\n{output_path}")
        self.stacked_widget.setCurrentIndex(4)  # Go to result page
    
    def generation_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred during model generation:\n{error_message}")
        self.stacked_widget.setCurrentIndex(2)  # Go back to settings page
    
    def cancel_generation(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
        
        self.stacked_widget.setCurrentIndex(2)  # Go back to settings page
    
    def open_output_folder(self):
        if not self.output_path_edit.text():
            return
            
        path = Path(self.output_path_edit.text())
        if sys.platform == 'win32':
            os.startfile(path.parent)
        elif sys.platform == 'darwin':  # macOS
            import subprocess
            subprocess.call(['open', path.parent])
        else:  # Linux
            import subprocess
            subprocess.call(['xdg-open', path.parent])

    def browse_text_output_path(self):
        """Browse for output file path for text-to-3D generation"""
        ext = "obj" if self.text_obj_radio.isChecked() else "stl"
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output Path", "", f"3D Model (*.{ext})")
        if file_path:
            self.text_output_path_edit.setText(file_path)
    
    def start_text_generation(self):
        """Start text-to-3D model generation process"""
        # Get text prompt
        text_prompt = self.text_prompt_edit.text().strip()
        if not text_prompt:
            QMessageBox.warning(self, "Input Required", "Please enter a text description.")
            return
        
        # Determine output format
        output_format = "obj" if self.text_obj_radio.isChecked() else "stl"
        
        # Get output path or generate one
        output_path = self.text_output_path_edit.text()
        if not output_path:
            # Generate output filename based on text prompt (limit to reasonable length)
            prompt_slug = text_prompt.lower()[:20].replace(" ", "_")
            output_filename = f"{prompt_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            output_path = Path(__file__).parent.resolve() / "outputs" / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.text_output_path_edit.setText(str(output_path))
        
        # Get detail level
        detail_level = self.detail_spin.value()
        
        # Switch to generation page and update status
        self.stacked_widget.setCurrentIndex(3)
        self.status_label.setText("Initializing text-to-3D generation...")
        
        # Start generator worker thread
        self.worker = TextModelGeneratorWorker(
            text_prompt=text_prompt,
            output_path=output_path,
            output_format=output_format,
            detail_level=detail_level
        )
        self.worker.progress_signal.connect(self.update_text_generation_progress)
        self.worker.finished_signal.connect(self.generation_finished)  # Reuse same callback
        self.worker.error_signal.connect(self.generation_error)  # Reuse same callback
        self.worker.start()
    
    def update_text_generation_progress(self, value):
        """Update progress bar and status text for text-to-3D generation"""
        self.progress_bar.setValue(value)
        
        # Update status based on progress
        if value < 30:
            self.status_label.setText("Initializing AI model...")
        elif value < 50:
            self.status_label.setText("Processing text prompt...")
        elif value < 70:
            self.status_label.setText("Generating 3D model from description...")
        elif value < 90:
            self.status_label.setText("Exporting 3D geometry...")
        else:
            self.status_label.setText("Finalizing output...")

    def create_placeholder_icon(self, text, color):
        """Create a programmatically generated icon with text and color"""
        from PyQt5.QtGui import QPainter, QFont, QPen, QBrush, QPainterPath
        from PyQt5.QtCore import QRect, Qt
        
        # Create a pixmap
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        
        # Create painter
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Set pen and brush
        painter.setPen(QPen(QColor(color), 2))
        painter.setBrush(QBrush(QColor(color).lighter(130)))
        
        # Draw rounded rectangle
        painter.drawRoundedRect(4, 4, 56, 56, 8, 8)
        
        # Set font for text
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)
        
        # Draw text
        painter.setPen(QPen(Qt.white))
        painter.drawText(QRect(4, 4, 56, 56), Qt.AlignCenter, text)
        
        # End painting
        painter.end()
        
        return pixmap

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert 2D images to 3D models")
    parser.add_argument(
        "-i", "--input",
        default="inputs/sample.jpg",
        help="Input image path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output model path (.obj or .stl format)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["obj", "stl"],
        default="obj",
        help="Output format (obj or stl)"
    )
    parser.add_argument(
        "-d", "--depth",
        type=float,
        default=10.0,
        help="Maximum depth of the 3D model"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=100,
        help="Resolution of the 3D model (points per dimension)"
    )
    return parser.parse_args()

def load_image(image_path):
    """Load and process the input image."""
    try:
        logger.info(f"Loading image from {image_path}")
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        return np.array(img)
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        sys.exit(1)

def generate_height_map(image_array, max_depth):
    """Generate a height map from the grayscale image."""
    # Normalize values to be between 0 and max_depth
    height_map = (image_array / 255.0) * max_depth
    logger.info(f"Generated height map with depth range: 0 to {max_depth}")
    return height_map

def generate_obj(height_map, resolution, output_path):
    """Generate an OBJ file from the height map."""
    height, width = height_map.shape
    
    # Scale factors
    scale_x = width / resolution
    scale_y = height / resolution
    
    # Create vertices and faces
    vertices = []
    faces = []
    
    logger.info(f"Generating 3D model with resolution {resolution}x{resolution}")
    
    # Create vertices
    for y in range(resolution + 1):
        for x in range(resolution + 1):
            # Map to image coordinates
            img_x = min(int(x * scale_x), width - 1)
            img_y = min(int(y * scale_y), height - 1)
            
            # Get height from height map
            z = height_map[img_y, img_x]
            
            # Normalize coordinates
            norm_x = (x / resolution) - 0.5
            norm_y = (y / resolution) - 0.5
            norm_z = z / height_map.max()
            
            vertices.append((norm_x, norm_y, norm_z))
    
    # Create faces (two triangles per grid cell)
    for y in range(resolution):
        for x in range(resolution):
            # Vertex indices (1-based for OBJ format)
            v1 = y * (resolution + 1) + x + 1
            v2 = v1 + 1
            v3 = v1 + (resolution + 1)
            v4 = v3 + 1
            
            # Add two triangular faces
            faces.append((v1, v2, v4))
            faces.append((v1, v4, v3))
    
    # Write OBJ file
    with open(output_path, 'w') as f:
        f.write("# OBJ file generated by 3D Model Generator\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    logger.info(f"OBJ model saved to {output_path}")
    logger.info(f"Generated {len(vertices)} vertices and {len(faces)} faces")

def generate_stl(height_map, resolution, output_path):
    """Generate an STL file from the height map."""
    logger.info("STL generation requested")
    try:
        # Import stl library
        from stl import mesh
        import numpy as np
        
        height, width = height_map.shape
        
        # Scale factors
        scale_x = width / resolution
        scale_y = height / resolution
        
        # Create vertices and faces arrays for STL
        vertices = np.zeros((resolution + 1, resolution + 1, 3))
        
        logger.info(f"Generating 3D model with resolution {resolution}x{resolution}")
        
        # Create vertices grid
        for y in range(resolution + 1):
            for x in range(resolution + 1):
                # Map to image coordinates
                img_x = min(int(x * scale_x), width - 1)
                img_y = min(int(y * scale_y), height - 1)
                
                # Get height from height map
                z = height_map[img_y, img_x]
                
                # Normalize coordinates
                norm_x = (x / resolution) - 0.5
                norm_y = (y / resolution) - 0.5
                norm_z = z / height_map.max()
                
                vertices[y, x] = [norm_x, norm_y, norm_z]
        
        # Create an array to store all the triangles
        faces = np.zeros(resolution * resolution * 2, dtype=mesh.Mesh.dtype)
        
        # Create triangular faces
        for y in range(resolution):
            for x in range(resolution):
                # Get the four corner points of this grid cell
                v1 = vertices[y, x]
                v2 = vertices[y, x + 1]
                v3 = vertices[y + 1, x]
                v4 = vertices[y + 1, x + 1]
                
                # Create two triangular faces for this grid cell
                face1 = np.zeros(3, dtype=np.float32)
                face2 = np.zeros(3, dtype=np.float32)
                
                # Triangle 1: v1-v2-v4
                faces['vectors'][2 * (y * resolution + x)] = np.array([v1, v2, v4])
                
                # Triangle 2: v1-v4-v3
                faces['vectors'][2 * (y * resolution + x) + 1] = np.array([v1, v4, v3])
        
        # Create a mesh object
        stl_mesh = mesh.Mesh(faces)
        
        # Write the STL file
        stl_mesh.save(output_path)
        
        logger.info(f"STL model saved to {output_path}")
        logger.info(f"Generated {(resolution + 1) * (resolution + 1)} vertices and {resolution * resolution * 2} triangular faces")
        
    except ImportError:
        logger.error("numpy-stl package not found. Installing it...")
        # Try to install numpy-stl
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy-stl"])
            logger.info("numpy-stl installed successfully. Please run the process again.")
        except subprocess.CalledProcessError:
            logger.error("Failed to install numpy-stl. Please install it manually: pip install numpy-stl")
        sys.exit(1)
    except Exception as e:
        logger.error(f"STL generation failed: {e}")
        sys.exit(1)

def main():
    """Main function to run the 3D model generator."""
    # If no command line arguments provided, launch the GUI
    if len(sys.argv) == 1:
        app = QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec_())
    else:
        # CLI mode
        args = parse_args()
        
        # Load and process image
        image_array = load_image(args.input)
        height_map = generate_height_map(image_array, args.depth)
        
        # Generate output path if not provided
        if not args.output:
            script_dir = Path(__file__).parent.resolve()
            input_path = Path(args.input)
            output_filename = f"{input_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"
            args.output = script_dir / "outputs" / output_filename
        
        # Create output directory if it doesn't exist
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate 3D model in the requested format
        if args.format == "obj":
            generate_obj(height_map, args.resolution, args.output)
        else:  # stl
            generate_stl(height_map, args.resolution, args.output)
        
        logger.info(f"3D model generation complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()