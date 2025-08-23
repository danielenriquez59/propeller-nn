import sys
import numpy as np
import os
import joblib
import json
import torch
import torch.nn as nn
from scipy.interpolate import make_interp_spline
out_dir = "model_weights"

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QDoubleSpinBox, QSpinBox, QPushButton, QLabel, QFrame, QGroupBox
)
from PySide6.QtCore import Qt, QPointF, Slot
from PySide6.QtGui import QPainter, QPen, QBrush, QPolygonF, QFont, QColor
from PySide6.QtWebEngineWidgets import QWebEngineView

from bokeh.plotting import figure as bokeh_figure
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models import ColumnDataSource, HoverTool, Range1d

from nn_functions import load_model_pipeline, predict

# --- Custom Spline Editor Widget ---
r_min = 0.15
class SplineEditor(QWidget):
    """ A custom widget to edit a 5-point spline distribution interactively. """
    def __init__(self, title, y_min, y_max, initial_control_points, parent=None,
                 envelope_x=None, envelope_ymin=None, envelope_ymax=None):
        super().__init__(parent)
        self.setMinimumSize(400, 250)
        self.title = title
        self.y_min = y_min
        self.y_max = y_max
        self.envelope_x = envelope_x
        self.envelope_ymin = envelope_ymin
        self.envelope_ymax = envelope_ymax
        
        # Initialize 5 control points from the provided defaults
        self.control_points = [QPointF(x, y) for x, y in initial_control_points]
        
        self.dragging_point_index = -1
        self.padding = 30 # Padding for axes

    def paintEvent(self, event):
        """ Renders the spline editor. """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # White background
        painter.fillRect(self.rect(), Qt.GlobalColor.white)
        
        # Draw axes and labels
        self._draw_axes(painter)
        
        # Draw control polygon, points, and spline
        self._draw_spline(painter)
    
    def _draw_axes(self, painter):
        w, h = self.width(), self.height()
        p = self.padding
        
        # Draw axis lines
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        painter.drawLine(p, h - p, w - p, h - p) # X-axis
        painter.drawLine(p, p, p, h - p)         # Y-axis

        # Draw labels
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(w // 2 - 20, h - 5, "Radius (r)")
        painter.save()
        painter.translate(10, (h // 2) + 20)
        painter.rotate(-90)
        painter.drawText(0, 0, self.title)
        painter.restore()

        # Draw ticks
        for i in range(6):
            x = p + i * (w - 2*p) / 5
            tick_x_val = r_min + i * (1 - r_min) / 5
            painter.drawText(int(x - 5), h - p + 15, f"{tick_x_val:.2f}")
            # Adjust tick labels for dynamic y-range
            tick_val = self.y_max - (self.y_max - self.y_min) * i / 5
            y = p + i * (h - 2*p) / 5
            painter.drawText(p - 25, int(y + 5), f"{tick_val:.2f}" if tick_val < 1 else f"{tick_val:.1f}")

    def _world_to_screen(self, p):
        """ Converts world coordinates ([r_min,1] x, y_min-y_max y) to screen pixels. """
        x_norm = (p.x() - r_min) / (1 - r_min)
        x = self.padding + x_norm * (self.width() - 2 * self.padding)
        y = self.padding + (1 - (p.y() - self.y_min) / (self.y_max - self.y_min)) * (self.height() - 2 * self.padding)
        return QPointF(x, y)

    def _screen_to_world(self, p):
        """ Converts screen pixels to world coordinates in [r_min,1]. """
        x_norm = (p.x() - self.padding) / (self.width() - 2 * self.padding)
        x = r_min + x_norm * (1 - r_min)
        y = self.y_min + (1 - (p.y() - self.padding) / (self.height() - 2 * self.padding)) * (self.y_max - self.y_min)
        return QPointF(x, y)
        
    def _draw_spline(self, painter):
        # Convert control points to screen coordinates
        screen_points = [self._world_to_screen(p) for p in self.control_points]
        
        # Draw control polygon (light gray lines)
        painter.setPen(QPen(Qt.GlobalColor.lightGray, 1, Qt.PenStyle.DashLine))
        painter.drawPolyline(QPolygonF(screen_points))

        # Draw control points (blue circles)
        painter.setBrush(QBrush(Qt.GlobalColor.blue))
        painter.setPen(Qt.PenStyle.NoPen)
        for sp in screen_points:
            painter.drawEllipse(sp, 5, 5)

        # Get discretized points for the spline
        x_pts, y_pts = self.get_discretized_values(20)
        spline_screen_points = [self._world_to_screen(QPointF(x,y)) for x,y in zip(x_pts, y_pts)]

        # Draw discretized points (small red dots)
        painter.setBrush(QBrush(Qt.GlobalColor.red))
        for sp in spline_screen_points:
            painter.drawEllipse(sp, 3, 3)

        # Draw envelope lines if provided
        if self.envelope_x is not None and self.envelope_ymin is not None and self.envelope_ymax is not None:
            env_min_points = [self._world_to_screen(QPointF(x, y)) for x, y in zip(self.envelope_x, self.envelope_ymin)]
            env_max_points = [self._world_to_screen(QPointF(x, y)) for x, y in zip(self.envelope_x, self.envelope_ymax)]
            painter.setPen(QPen(QColor("#888888"), 1, Qt.PenStyle.DotLine))
            painter.drawPolyline(QPolygonF(env_min_points))
            painter.drawPolyline(QPolygonF(env_max_points))

        # Draw smooth spline curve
        x_fine, y_fine = self.get_discretized_values(100)
        fine_points = [self._world_to_screen(QPointF(x,y)) for x,y in zip(x_fine, y_fine)]
        painter.setPen(QPen(QColor("#3333AA"), 2))
        painter.drawPolyline(QPolygonF(fine_points))
    
    def mousePressEvent(self, event):
        for i, p in enumerate(self.control_points):
            sp = self._world_to_screen(p)
            if (event.position() - sp).manhattanLength() < 10:
                self.dragging_point_index = i
                break
    
    def mouseMoveEvent(self, event):
        if self.dragging_point_index != -1:
            p = self.control_points[self.dragging_point_index]
            new_world_pos = self._screen_to_world(event.position())
            
            # Clamp values to be within bounds
            new_x = np.clip(new_world_pos.x(), r_min, 1)
            new_y = np.clip(new_world_pos.y(), self.y_min, self.y_max)
            
            # Prevent x-values from crossing over neighbors
            if self.dragging_point_index > 0:
                new_x = max(new_x, self.control_points[self.dragging_point_index-1].x())
            if self.dragging_point_index < len(self.control_points) - 1:
                new_x = min(new_x, self.control_points[self.dragging_point_index+1].x())

            p.setX(new_x)
            p.setY(new_y)
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_point_index = -1
        
    def get_discretized_values(self, num_points):
        """ Calculates the spline and returns y-values at num_points stations. """
        x = np.array([p.x() for p in self.control_points])
        y = np.array([p.y() for p in self.control_points])
        
        # Use scipy to create a 3rd degree B-spline
        spline = make_interp_spline(x, y, k=3)
        
        x_out = np.linspace(r_min, 1, num_points)
        y_out = spline(x_out)
        
        return x_out, y_out

# --- Main Application Window ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("APC_NN Interactive Predictor")
        
        # Load the model pipeline
        self.model, self.misc_scaler, self.y_scaler, self.meta = load_model_pipeline(model_dir=out_dir)
        
        if self.model is None:
            self.setCentralWidget(QLabel("Could not load model files. Please check the console."))
            return

        self.N_radial = self.meta['N_radial']
        self.N_adv = self.meta['N_adv']

        # --- UI Setup ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # -- Left side: Inputs --
        left_layout = QVBoxLayout()
        
        # Load envelopes
        env = read_envelope()
        env_x = np.linspace(r_min, 1, len(env["chord_min"]))

        # Define default propeller geometry
        default_chord_points = [(r_min, 0.16), (0.25, 0.15), (0.5, 0.15), (0.75, 0.10), (1.0, 0.05)]
        default_twist_points = [(r_min, 30.0), (0.25, 35.0), (0.5, 25.0), (0.75, 18.0), (1.0, 12.0)]

        # Chord editor
        chord_y_range = [0, 0.4]
        chord_box = QGroupBox("Chord Distribution (c/R)")
        chord_layout = QVBoxLayout()
        self.chord_editor = SplineEditor(
            "Chord", chord_y_range[0], chord_y_range[1], default_chord_points,
            envelope_x=env_x,
            envelope_ymin=np.array(env["chord_min"], dtype=float),
            envelope_ymax=np.array(env["chord_max"], dtype=float)
        )
        chord_layout.addWidget(self.chord_editor)
        chord_box.setLayout(chord_layout)
        left_layout.addWidget(chord_box)
        
        # Twist editor
        twist_y_range = [0, 60]
        twist_box = QGroupBox("Twist Distribution (degrees)")
        twist_layout = QVBoxLayout()
        self.twist_editor = SplineEditor(
            "Twist", twist_y_range[0], twist_y_range[1], default_twist_points,
            envelope_x=env_x,
            envelope_ymin=np.array(env["twist_min"], dtype=float),
            envelope_ymax=np.array(env["twist_max"], dtype=float)
        )
        twist_layout.addWidget(self.twist_editor)
        twist_box.setLayout(twist_layout)
        left_layout.addWidget(twist_box)

        # Scalar inputs
        range_diameter = [2, 12]
        range_n_blades = [2, 3]
        scalar_box = QGroupBox("Global Parameters")
        scalar_layout = QHBoxLayout()
        self.diameter_input = QDoubleSpinBox()
        self.diameter_input.setRange(range_diameter[0], range_diameter[1]); self.diameter_input.setValue(5)
        self.n_blades_input = QSpinBox()
        self.n_blades_input.setRange(range_n_blades[0], range_n_blades[1]); self.n_blades_input.setValue(2)
        
        scalar_layout.addWidget(QLabel("Diameter (in):"))
        scalar_layout.addWidget(self.diameter_input)
        scalar_layout.addWidget(QLabel("Num Blades:"))
        scalar_layout.addWidget(self.n_blades_input)
        # Tip Mach removed from inputs; model expects solidity computed internally
        scalar_box.setLayout(scalar_layout)
        left_layout.addWidget(scalar_box)
        
        # Prediction button
        self.predict_button = QPushButton("Predict Performance")
        self.predict_button.setFixedHeight(40)
        self.predict_button.clicked.connect(self.run_prediction)
        left_layout.addWidget(self.predict_button)

        main_layout.addLayout(left_layout)
        main_layout.setStretch(0, 1) # Give left side less space

        # -- Right side: Output Plot --
        plot_box = QGroupBox("Performance Curve")
        plot_layout = QVBoxLayout()
        self.plot_view = QWebEngineView()
        self.init_plot()
        plot_layout.addWidget(self.plot_view)
        plot_box.setLayout(plot_layout)
        
        main_layout.addWidget(plot_box)
        main_layout.setStretch(1, 2) # Give plot more space

        # Perform an initial prediction on load
        self.run_prediction()

    def init_plot(self):
        """ Creates and displays an initial empty Bokeh plot. """
        p = bokeh_figure(
            sizing_mode="stretch_both",
            tools="pan,wheel_zoom,box_zoom,reset",
            active_scroll="wheel_zoom",
            x_axis_label="Advance Ratio (J)",
            y_axis_label="Coefficient"
        )
        p.y_range = Range1d(-0.1, 0.25)
        p.add_tools(HoverTool(tooltips=[("J", "@x{0.00}"), ("Value", "@y{0.0000}")]))
        html = file_html(p, CDN, "Performance Plot")
        self.plot_view.setHtml(html)

    @Slot()
    def run_prediction(self):
        """ Gathers data from UI, runs model, and updates plot. """
        # 1. Get scalar values from inputs
        diameter = self.diameter_input.value() * 0.0254 # convert to meters
        n_blades = self.n_blades_input.value()
        # Tip Mach no longer used

        # 2. Get geometry from spline editors
        radius, chord = self.chord_editor.get_discretized_values(self.N_radial)
        _, twist = self.twist_editor.get_discretized_values(self.N_radial)

        # 3. Define J array for prediction (based on model metadata)
        # Note: In a real app, this might be a user input as well.
        j_array = np.linspace(0.01, 1.5, self.N_adv)

        # 4. Run prediction
        norm_chord = 0.5
        norm_twist = 100
        ct_pred, cp_pred = predict(
            model=self.model,
            misc_scaler=self.misc_scaler,
            y_scaler=self.y_scaler,
            radius=radius,
            chord=chord / norm_chord,
            twist=twist / norm_twist,
            diameter=diameter,
            N_blades=n_blades,
            J_array=j_array
        )

        # 5. Update the plot with new data
        self.update_plot(j_array, ct_pred, cp_pred)
        
    def update_plot(self, j, ct, cp):
        """ Updates the Bokeh plot with new data. """
        # 1. Create Hover Tool for rich data display
        hover = HoverTool(
            tooltips=[
                ("J", "@x{0.00}"),
                ("CT", "@ct{0.0000}"),
                ("CP", "@cp{0.0000}"),
                ("Efficiency/10", "@eta10{0.0000}"),
            ],
            mode='vline' # Show tooltips for all lines at a given x-position
        )

        # 2. Create the figure with interactive tools
        p = bokeh_figure(
            sizing_mode="stretch_both",
            tools="pan,wheel_zoom,box_zoom,reset",
            active_scroll="wheel_zoom",
            x_axis_label="Advance Ratio (J)",
            y_axis_label="Coefficient"
        )
        p.y_range = Range1d(-0.1, 0.2)
        p.add_tools(hover)

        # 3. Create a ColumnDataSource for efficient data handling
        # Compute efficiency scaled by 10, guard against division by zero
        cp_safe = np.where(np.abs(cp) < 1e-8, np.nan, cp)
        ct_safe = np.where(ct < 0, 0, ct)
        eta10 = (ct_safe * j) / cp_safe / 10.0

        source = ColumnDataSource(data={
            'x': j,
            'ct': ct,
            'cp': cp,
            'eta10': eta10
        })
        
        # 4. Add line and circle renderers to the plot
        p.line(x='x', y='ct', source=source, legend_label="Predicted CT", color="royalblue", line_width=2)
        p.circle(x='x', y='ct', source=source, color="royalblue", size=5, legend_label="Predicted CT")

        p.line(x='x', y='cp', source=source, legend_label="Predicted CP", color="orangered", line_width=2, line_dash="dashed")
        p.circle(x='x', y='cp', source=source, color="orangered", size=5, legend_label="Predicted CP")

        # Efficiency line (eta/10)
        p.line(x='x', y='eta10', source=source, legend_label="Efficiency/10", color="seagreen", line_width=2, line_dash="dotdash")
        p.circle(x='x', y='eta10', source=source, color="seagreen", size=5, legend_label="Efficiency/10")

        # 5. Configure the legend
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        # 6. Generate HTML from the figure and load it into the QWebEngineView
        html = file_html(p, CDN, "Prop Performance")
        self.plot_view.setHtml(html)

def read_envelope():
    """
    Read the envelope from the json file.
    Keys are: chord_min, chord_max, twist_min, twist_max
    """
    with open("chord_envelope.json", "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    if window.model: # Only show window if model loaded successfully
        window.show()
        sys.exit(app.exec())
