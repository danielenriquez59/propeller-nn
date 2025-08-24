import json, os
import numpy as np
from typing import Tuple, Optional

from scipy.interpolate import make_interp_spline

import panel as pn
from bokeh.plotting import figure as bokeh_figure
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Range1d,
    Label,
    PointDrawTool,
)

from nn_functions import load_model_pipeline, predict
from geometry.create_geomety import create_bem_file
from geometry.api_openvsp import send_bem_to_vsp


# Global configuration
r_min: float = 0.15
MODEL_DIR: str = "model_weights"


def read_envelope() -> dict:
    """Read envelope JSON providing chord/twist min/max arrays."""
    with open("chord_envelope.json", "r") as f:
        return json.load(f)


def compute_cubic_spline(x_knots: np.ndarray, y_knots: np.ndarray, num_points: int, y_min: float, y_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a cubic spline through control points and sample uniformly in radius.

    Clamps output within [y_min, y_max] to avoid overshoot artifacts.
    """
    # Ensure monotonic x and clamp within [r_min, 1]
    x_clamped = np.clip(x_knots, r_min, 1.0)
    order = np.argsort(x_clamped)
    x_sorted = x_clamped[order]
    y_sorted = y_knots[order]

    # Guard against duplicates in x by adding tiny eps
    for i in range(1, len(x_sorted)):
        if x_sorted[i] <= x_sorted[i - 1]:
            x_sorted[i] = min(1.0, x_sorted[i - 1] + 1e-6)

    x_uniform = np.linspace(r_min, 1.0, num_points)
    spline = make_interp_spline(x_sorted, y_sorted, k=3)
    y_uniform = spline(x_uniform)
    y_uniform = np.clip(y_uniform, y_min, y_max)
    return x_uniform, y_uniform


class SplineEditorPanel:
    """
    Draggable 5-point spline editor implemented with Bokeh + PointDrawTool.
    Exposes get_discretized_values(num_points) for downstream computation.
    """

    def __init__(
        self,
        title: str,
        y_min: float,
        y_max: float,
        initial_control_points: np.ndarray,
        envelope_x: Optional[np.ndarray] = None,
        envelope_ymin: Optional[np.ndarray] = None,
        envelope_ymax: Optional[np.ndarray] = None,
    ) -> None:
        self.title = title
        self.y_min = y_min
        self.y_max = y_max
        self.envelope_x = envelope_x
        self.envelope_ymin = envelope_ymin
        self.envelope_ymax = envelope_ymax

        # Control point data source
        self.control_source = ColumnDataSource(
            data=dict(
                x=np.array([p[0] for p in initial_control_points], dtype=float),
                y=np.array([p[1] for p in initial_control_points], dtype=float),
            )
        )

        # Spline line data source
        self.spline_source = ColumnDataSource(data=dict(x=[], y=[]))

        # Envelope sources (optional)
        self.envelope_min_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.envelope_max_source = ColumnDataSource(data=dict(x=[], y=[]))

        # Figure setup
        self.figure = bokeh_figure(
            height=280,
            width=520,
            x_range=(r_min, 1.0),
            y_range=(y_min, y_max),
            tools="pan,wheel_zoom,box_zoom,reset",
            active_scroll="wheel_zoom",
            x_axis_label="Radius (r)",
            y_axis_label=title,
        )

        # Envelope lines
        if (
            self.envelope_x is not None
            and self.envelope_ymin is not None
            and self.envelope_ymax is not None
        ):
            self.envelope_min_source.data = dict(x=self.envelope_x, y=self.envelope_ymin)
            self.envelope_max_source.data = dict(x=self.envelope_x, y=self.envelope_ymax)
            self.figure.line(
                x="x",
                y="y",
                source=self.envelope_min_source,
                color="#888888",
                line_dash="dotted",
                line_width=1,
            )
            self.figure.line(
                x="x",
                y="y",
                source=self.envelope_max_source,
                color="#888888",
                line_dash="dotted",
                line_width=1,
            )

        # Control points as draggable scatter
        point_renderer = self.figure.scatter(
            x="x",
            y="y",
            source=self.control_source,
            size=8,
            color="royalblue",
        )
        draw_tool = PointDrawTool(renderers=[point_renderer], add=False)
        self.figure.add_tools(draw_tool)
        self.figure.toolbar.active_tap = draw_tool

        # Spline line
        self.figure.line(
            x="x",
            y="y",
            source=self.spline_source,
            color="#3333AA",
            line_width=2,
        )

        # React to control point changes (guard against re-entrancy)
        self._is_updating_points = False
        self.control_source.on_change("data", self._on_points_changed)

        # Initialize spline
        self._recompute_spline()

    def _on_points_changed(self, attr, old, new):
        # Prevent recursive re-entry when we normalize points
        if self._is_updating_points:
            return

        x = np.array(self.control_source.data["x"], dtype=float)
        y = np.array(self.control_source.data["y"], dtype=float)

        # Clamp within editor bounds
        x = np.clip(x, r_min, 1.0)
        y = np.clip(y, self.y_min, self.y_max)

        # Sort by x so the spline stays monotonic in radius
        order = np.argsort(x)

        # Assign back while suppressing re-trigger
        self._is_updating_points = True
        try:
            self.control_source.data = dict(x=x[order], y=y[order])
        finally:
            self._is_updating_points = False

        # Update the smooth spline curve
        self._recompute_spline()

    def _recompute_spline(self) -> None:
        x = np.array(self.control_source.data["x"], dtype=float)
        y = np.array(self.control_source.data["y"], dtype=float)
        x_smooth, y_smooth = compute_cubic_spline(x, y, num_points=100, y_min=self.y_min, y_max=self.y_max)
        self.spline_source.data = dict(x=x_smooth, y=y_smooth)

    def panel(self):
        return self.figure

    def get_discretized_values(self, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.array(self.control_source.data["x"], dtype=float)
        y = np.array(self.control_source.data["y"], dtype=float)
        return compute_cubic_spline(x, y, num_points=num_points, y_min=self.y_min, y_max=self.y_max)


class PropellerApp:
    def __init__(self) -> None:
        pn.extension("bokeh")

        # Load model
        self.model, self.misc_scaler, self.y_scaler, self.meta = load_model_pipeline(model_dir=MODEL_DIR)
        if self.model is None:
            self.error_panel = pn.pane.Markdown("**Error**: Could not load model files.")
            return

        self.num_radial = int(self.meta["N_radial"])  # stations for chord/twist
        self.num_adv = int(self.meta["N_adv"])        # J samples

        # Envelopes
        env = read_envelope()
        envelope_x = np.linspace(r_min, 1.0, len(env["chord_min"]))

        # Default control points
        default_chord_points = np.array(
            [(r_min, 0.16), (0.25, 0.15), (0.5, 0.15), (0.75, 0.10), (1.0, 0.05)], dtype=float
        )
        default_twist_points = np.array(
            [(r_min, 30.0), (0.25, 35.0), (0.5, 25.0), (0.75, 18.0), (1.0, 12.0)], dtype=float
        )

        # Editors
        self.chord_editor = SplineEditorPanel(
            title="Chord (c/R)",
            y_min=0.0,
            y_max=0.4,
            initial_control_points=default_chord_points,
            envelope_x=envelope_x,
            envelope_ymin=np.array(env["chord_min"], dtype=float),
            envelope_ymax=np.array(env["chord_max"], dtype=float),
        )

        self.twist_editor = SplineEditorPanel(
            title="Twist (deg)",
            y_min=0.0,
            y_max=60.0,
            initial_control_points=default_twist_points,
            envelope_x=envelope_x,
            envelope_ymin=np.array(env["twist_min"], dtype=float),
            envelope_ymax=np.array(env["twist_max"], dtype=float),
        )

        # Controls
        self.diameter_input = pn.widgets.FloatInput(name="Diameter (in)", value=5.0, start=2.0, end=12.0, step=0.25)
        self.diameter_input.styles = {"width": "200px", "min-width": "200px", "max-width": "200px"}
        self.num_blades_input = pn.widgets.IntInput(name="Num Blades", value=2, start=2, end=3, step=1)
        self.num_blades_input.styles = {"width": "200px", "min-width": "200px", "max-width": "200px"}
        self.predict_button = pn.widgets.Button(name="Predict Performance", button_type="primary")
        self.predict_button.on_click(self._on_predict_clicked)

        # Export controls
        self.filename_input = pn.widgets.TextInput(name="BEM Filename", placeholder="e.g., my_prop")
        self.run_openvsp_checkbox = pn.widgets.Checkbox(name="Run OpenVSP after write", value=False)
        self.write_bem_button = pn.widgets.Button(name="Write BEM", button_type="success")
        self.write_bem_button.on_click(self._on_write_bem_clicked)
        self.export_status = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Performance figure
        self.performance_source = ColumnDataSource(data=dict(x=[], ct=[], cp=[], eta10=[]))
        self.performance_figure = self._create_performance_figure()

        # Layout
        self.layout = pn.Row(
            pn.Column(
                pn.pane.Markdown(
                    "Move the chord and twist control points to design your propeller.\n"
                    "Adjust Diameter and Num Blades, then click 'Predict Performance' to update the plot.",
                    styles={"color": "#444", "font-size": "11pt", "padding": "6px"},
                ),
                pn.Card(self.chord_editor.panel(), title="Chord Distribution (c/R)", collapsible=False),
                pn.Card(self.twist_editor.panel(), title="Twist Distribution (degrees)", collapsible=False),
                pn.Card(
                    pn.Row(self.diameter_input, self.num_blades_input),
                    title="Global Parameters",
                    collapsible=False,
                ),
                self.predict_button,
                pn.Card(
                    pn.Column(
                        self.filename_input,
                        self.run_openvsp_checkbox,
                        self.write_bem_button,
                        self.export_status,
                    ),
                    title="Export / OpenVSP",
                    collapsible=False,
                ),
                sizing_mode="stretch_height",
                width=560,
            ),
            pn.Card(self.performance_figure, title="Performance Curve", sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
        )

        # Initial prediction
        self._run_prediction_and_update()

    def _create_performance_figure(self):
        p = bokeh_figure(
            sizing_mode="stretch_both",
            tools="pan,wheel_zoom,box_zoom,reset",
            active_scroll="wheel_zoom",
            x_axis_label="Advance Ratio (J)",
            y_axis_label="Coefficient",
        )
        p.y_range = Range1d(-0.1, 0.25)

        hover = HoverTool(
            tooltips=[
                ("J", "@x{0.00}"),
                ("CT", "@ct{0.0000}"),
                ("CP", "@cp{0.0000}"),
                ("Efficiency/10", "@eta10{0.0000}"),
            ],
            mode="vline",
        )
        p.add_tools(hover)

        # Lines + scatter
        p.line(x="x", y="ct", source=self.performance_source, legend_label="Predicted CT", color="royalblue", line_width=2)
        p.scatter(x="x", y="ct", source=self.performance_source, color="royalblue", size=5, legend_label="Predicted CT")

        p.line(x="x", y="cp", source=self.performance_source, legend_label="Predicted CP", color="orangered", line_width=2, line_dash="dashed")
        p.scatter(x="x", y="cp", source=self.performance_source, color="orangered", size=5, legend_label="Predicted CP")

        p.line(x="x", y="eta10", source=self.performance_source, legend_label="Efficiency/10", color="seagreen", line_width=2, line_dash="dotdash")
        p.scatter(x="x", y="eta10", source=self.performance_source, color="seagreen", size=5, legend_label="Efficiency/10")

        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        return p

    def _on_predict_clicked(self, *_):
        self._run_prediction_and_update()

    def _run_prediction_and_update(self) -> None:
        # Scalars
        diameter_m = float(self.diameter_input.value) * 0.0254
        num_blades = int(self.num_blades_input.value)

        # Geometry
        radius, chord = self.chord_editor.get_discretized_values(self.num_radial)
        _, twist = self.twist_editor.get_discretized_values(self.num_radial)

        # J array
        j_array = np.linspace(0.01, 1.5, self.num_adv)

        # Predict
        norm_chord = 0.5
        norm_twist = 100.0
        ct_pred, cp_pred = predict(
            model=self.model,
            misc_scaler=self.misc_scaler,
            y_scaler=self.y_scaler,
            radius=radius,
            chord=chord / norm_chord,
            twist=twist / norm_twist,
            diameter=diameter_m,
            N_blades=num_blades,
            J_array=j_array,
        )

        # Efficiency/10 with safety
        cp_safe = np.where(np.abs(cp_pred) < 1e-8, np.nan, cp_pred)
        ct_safe = np.where(ct_pred < 0, 0, ct_pred)
        eta10 = (ct_safe * j_array) / cp_safe / 10.0

        # Update source
        self.performance_source.data = dict(x=j_array, ct=ct_pred, cp=cp_pred, eta10=eta10)

        # Label for max efficiency
        try:
            idx_max = int(np.nanargmax(eta10))
            j_max = float(j_array[idx_max])
            eta10_max = float(eta10[idx_max])
            # Remove previous labels
            self.performance_figure.renderers = [r for r in self.performance_figure.renderers if not isinstance(r, Label)]
            label = Label(
                x=j_max,
                y=min(max(eta10_max + 0.02, self.performance_figure.y_range.start), self.performance_figure.y_range.end),
                text=f"Max eff at J={j_max:.2f}",
                text_color="seagreen",
                text_font_size="10pt",
                background_fill_color="white",
                background_fill_alpha=0.6,
            )
            self.performance_figure.add_layout(label)
        except (ValueError, IndexError):
            pass

    def _on_write_bem_clicked(self, *_):
        name = (self.filename_input.value or "").strip()
        if not name:
            self.export_status.object = "**Please enter a filename.**"
            self.export_status.styles = {"color": "#b00020"}
            return
        # Ensure .bem extension
        if not name.lower().endswith(".bem"):
            name = name + ".bem"
        name = os.path.join("geometry", name)
        try:
            # Gather current geometry
            radius, chord = self.chord_editor.get_discretized_values(self.num_radial)
            _, twist = self.twist_editor.get_discretized_values(self.num_radial)
            diameter_m = float(self.diameter_input.value) * 0.0254
            num_blades = int(self.num_blades_input.value)

            # Write BEM file
            create_bem_file(
                radius_R=radius,
                chord=chord,
                twist_deg=twist,
                diameter=diameter_m,
                num_blade=num_blades,
                output_path=name,
                chord_is_relative=True,
            )
            self.export_status.object = f"Wrote BEM file: `{name}`"
            self.export_status.styles = {"color": "#006400"}

            # Optionally launch OpenVSP
            if self.run_openvsp_checkbox.value:
                try:
                    send_bem_to_vsp(name)
                    self.export_status.object += "\nOpenVSP launched with BEM file."
                except Exception as e:
                    self.export_status.object += f"\nOpenVSP launch failed: {e}"
                    self.export_status.styles = {"color": "#b00020"}
        except Exception as e:
            self.export_status.object = f"**Error writing BEM**: {e}"
            self.export_status.styles = {"color": "#b00020"}


def serve_app():
    app = PropellerApp()
    if getattr(app, "error_panel", None) is not None:
        return app.error_panel
    return app.layout


app = serve_app()
app.servable(title="APC_NN Interactive Predictor")


