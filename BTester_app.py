from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Interaction-Free Measurement Explorer",
    page_icon="💣",
    layout="wide",
)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass(frozen=True)
class TrialProbabilities:
    explosion: float
    bright_port: float
    dark_port: float


@dataclass(frozen=True)
class SimulationSummary:
    trials: int
    explosion_count: int
    bright_count: int
    dark_count: int

    @property
    def explosion_fraction(self) -> float:
        return self.explosion_count / self.trials if self.trials else 0.0

    @property
    def bright_fraction(self) -> float:
        return self.bright_count / self.trials if self.trials else 0.0

    @property
    def dark_fraction(self) -> float:
        return self.dark_count / self.trials if self.trials else 0.0


# ============================================================
# PHYSICS MODEL
# ============================================================

def get_probabilities(object_state: str) -> TrialProbabilities:
    """
    Return idealized outcome probabilities for a balanced interferometer.

    No object or dud object:
        Bright port = 1
        Dark port = 0
        Explosion = 0

    Live object:
        Explosion = 1/2
        Bright port = 1/4
        Dark port = 1/4
    """
    if object_state in {"No object", "Dud object"}:
        return TrialProbabilities(
            explosion=0.0,
            bright_port=1.0,
            dark_port=0.0,
        )

    if object_state == "Live object":
        return TrialProbabilities(
            explosion=0.5,
            bright_port=0.25,
            dark_port=0.25,
        )

    raise ValueError(f"Unknown object state: {object_state}")


def run_trials(
    object_state: str,
    number_of_trials: int,
    seed: int | None = None,
) -> SimulationSummary:
    """
    Run Monte Carlo trials for the ideal bomb-tester experiment.
    """
    if number_of_trials < 1:
        raise ValueError("Number of trials must be at least 1.")

    probabilities = get_probabilities(object_state)

    rng = np.random.default_rng(seed)

    outcomes = rng.choice(
        ["Explosion", "Bright port", "Dark port"],
        size=number_of_trials,
        p=[
            probabilities.explosion,
            probabilities.bright_port,
            probabilities.dark_port,
        ],
    )

    return SimulationSummary(
        trials=number_of_trials,
        explosion_count=int(np.count_nonzero(outcomes == "Explosion")),
        bright_count=int(np.count_nonzero(outcomes == "Bright port")),
        dark_count=int(np.count_nonzero(outcomes == "Dark port")),
    )


def interaction_free_efficiency(summary: SimulationSummary) -> float:
    """
    Fraction of live-object trials that produced successful
    interaction-free detection.
    """
    if summary.trials == 0:
        return 0.0

    return summary.dark_count / summary.trials


# ============================================================
# PLOTTING
# ============================================================

def make_results_chart(summary: SimulationSummary) -> go.Figure:
    labels = [
        "Explosion",
        "Bright-port detection",
        "Dark-port detection",
    ]

    counts = [
        summary.explosion_count,
        summary.bright_count,
        summary.dark_count,
    ]

    colors = [
        "#D62728",
        "#1F77B4",
        "#2CA02C",
    ]

    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                marker_color=colors,
                text=counts,
                textposition="outside",
            )
        ]
    )

    figure.update_layout(
        title="Observed outcomes",
        xaxis_title="Outcome",
        yaxis_title="Number of trials",
        template="plotly_white",
        height=430,
        margin=dict(l=40, r=30, t=60, b=50),
    )

    figure.update_yaxes(rangemode="tozero")

    return figure


def make_probability_chart(
    probabilities: TrialProbabilities,
) -> go.Figure:
    labels = [
        "Explosion",
        "Bright port",
        "Dark port",
    ]

    values = [
        probabilities.explosion,
        probabilities.bright_port,
        probabilities.dark_port,
    ]

    colors = [
        "#D62728",
        "#1F77B4",
        "#2CA02C",
    ]

    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{value:.0%}" for value in values],
                textposition="outside",
            )
        ]
    )

    figure.update_layout(
        title="Theoretical probabilities",
        xaxis_title="Outcome",
        yaxis_title="Probability",
        template="plotly_white",
        height=430,
        margin=dict(l=40, r=30, t=60, b=50),
    )

    figure.update_yaxes(
        range=[0, 1.05],
        tickformat=".0%",
    )

    return figure


def make_running_fraction_chart(
    object_state: str,
    number_of_trials: int,
    seed: int | None,
) -> go.Figure:
    """
    Show how measured fractions converge toward theoretical probabilities.
    """
    probabilities = get_probabilities(object_state)
    rng = np.random.default_rng(seed)

    outcomes = rng.choice(
        [0, 1, 2],
        size=number_of_trials,
        p=[
            probabilities.explosion,
            probabilities.bright_port,
            probabilities.dark_port,
        ],
    )

    trial_axis = np.arange(1, number_of_trials + 1)

    explosion_fraction = np.cumsum(outcomes == 0) / trial_axis
    bright_fraction = np.cumsum(outcomes == 1) / trial_axis
    dark_fraction = np.cumsum(outcomes == 2) / trial_axis

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=trial_axis,
            y=explosion_fraction,
            mode="lines",
            name="Explosion",
            line=dict(color="#D62728", width=2),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=trial_axis,
            y=bright_fraction,
            mode="lines",
            name="Bright port",
            line=dict(color="#1F77B4", width=2),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=trial_axis,
            y=dark_fraction,
            mode="lines",
            name="Dark port",
            line=dict(color="#2CA02C", width=2),
        )
    )

    figure.update_layout(
        title="Convergence of measured outcome fractions",
        xaxis_title="Trial number",
        yaxis_title="Measured fraction",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=30, t=60, b=50),
    )

    figure.update_yaxes(
        range=[0, 1.0],
        tickformat=".0%",
    )

    return figure


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "summary": None,
    "prediction": "",
    "reasoning": "",
    "reflection": "",
    "student_name": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.title("Interaction-Free Measurement Explorer")

# Optional image directly under the title
image_path = Path("BTester_Setup.jpg")

if image_path.exists():
    st.image(
        str(image_path),
        width=650,
    )

    st.caption(
        "Conceptual diagram of an interaction-free measurement experiment."
    )

st.markdown(
    """
Can the presence of an object be detected without a photon being absorbed
by that object?

This virtual laboratory models the idealized Elitzur–Vaidman bomb-tester
thought experiment using a balanced interferometer.
"""
)

st.info(
    """
A click at the normally dark detector is the key event. It indicates that
the object is present even though the photon was not absorbed by it.
"""
)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:
    st.header("Experiment controls")

    activity_mode = st.radio(
        "Activity mode",
        options=[
            "Guided investigation",
            "Free exploration",
        ],
    )

    object_state = st.radio(
        "Object in one interferometer arm",
        options=[
            "No object",
            "Dud object",
            "Live object",
        ],
        help=(
            "A dud does not interact with the photon. "
            "A live object absorbs a photon that enters its arm."
        ),
    )

    number_of_trials = st.select_slider(
        "Number of photons",
        options=[
            1,
            10,
            25,
            50,
            100,
            250,
            500,
            1000,
            5000,
        ],
        value=100,
    )

    use_fixed_seed = st.checkbox(
        "Use a fixed random seed",
        value=True,
    )

    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
        disabled=not use_fixed_seed,
    )

    run_button = st.button(
        "▶ Run experiment",
        type="primary",
        use_container_width=True,
    )

    reset_button = st.button(
        "Reset",
        use_container_width=True,
    )


if reset_button:
    st.session_state["summary"] = None
    st.session_state["prediction"] = ""
    st.session_state["reasoning"] = ""
    st.session_state["reflection"] = ""
    st.rerun()


# ============================================================
# GUIDED INVESTIGATION
# ============================================================

if activity_mode == "Guided investigation":
    st.header("Today's investigation")

    investigation = st.selectbox(
        "Choose an investigation",
        options=[
            "1. Establish the dark port",
            "2. Insert a dud object",
            "3. Insert a live object",
            "4. Compare theory and experiment",
        ],
    )

    if investigation == "1. Establish the dark port":
        st.markdown(
            """
Set the object state to **No object**.

In a perfectly aligned interferometer, the two probability amplitudes
cancel at the dark output port.

Predict which detector will register the photons.
"""
        )

    elif investigation == "2. Insert a dud object":
        st.markdown(
            """
Set the object state to **Dud object**.

A dud does not absorb or mark the photon path. Predict whether the
interference pattern will change.
"""
        )

    elif investigation == "3. Insert a live object":
        st.markdown(
            """
Set the object state to **Live object**.

A live object can absorb a photon in one arm. Predict the probabilities
of explosion, bright-port detection, and dark-port detection.
"""
        )

    else:
        st.markdown(
            """
Run several values of the photon number.

Compare the measured frequencies with the theoretical probabilities.
Observe how the measured fractions stabilize as the number of trials
increases.
"""
        )

    st.session_state["student_name"] = st.text_input(
        "Student or group name",
        value=st.session_state["student_name"],
    )

    st.session_state["prediction"] = st.text_area(
        "Prediction",
        value=st.session_state["prediction"],
        placeholder="What outcomes do you expect?",
        height=90,
    )

    st.session_state["reasoning"] = st.text_area(
        "Reasoning",
        value=st.session_state["reasoning"],
        placeholder="Explain why you expect those outcomes.",
        height=100,
    )


# ============================================================
# RUN SIMULATION
# ============================================================

if run_button:
    seed = int(random_seed) if use_fixed_seed else None

    st.session_state["summary"] = run_trials(
        object_state=object_state,
        number_of_trials=number_of_trials,
        seed=seed,
    )


summary: SimulationSummary | None = st.session_state["summary"]
probabilities = get_probabilities(object_state)


# ============================================================
# THEORETICAL OUTCOME DISPLAY
# ============================================================

st.header("Interferometer prediction")

col_1, col_2, col_3 = st.columns(3)

col_1.metric(
    "Explosion probability",
    f"{probabilities.explosion:.0%}",
)

col_2.metric(
    "Bright-port probability",
    f"{probabilities.bright_port:.0%}",
)

col_3.metric(
    "Dark-port probability",
    f"{probabilities.dark_port:.0%}",
)


if object_state in {"No object", "Dud object"}:
    st.success(
        """
The two paths remain coherent and interfere destructively at the dark
port. All photons exit through the bright port in the ideal model.
"""
    )

else:
    st.warning(
        """
The live object can absorb the photon and destroys the two-path
interference. A dark-port detection is therefore possible.
"""
    )


# ============================================================
# RESULTS
# ============================================================

if summary is None:
    st.subheader("Run the experiment to generate data.")

else:
    st.header("Experimental results")

    plot_left, plot_right = st.columns(2)

    with plot_left:
        st.plotly_chart(
            make_results_chart(summary),
            use_container_width=True,
        )

    with plot_right:
        st.plotly_chart(
            make_probability_chart(probabilities),
            use_container_width=True,
        )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Explosions",
        summary.explosion_count,
        f"{summary.explosion_fraction:.1%}",
    )

    metric_2.metric(
        "Bright-port detections",
        summary.bright_count,
        f"{summary.bright_fraction:.1%}",
    )

    metric_3.metric(
        "Dark-port detections",
        summary.dark_count,
        f"{summary.dark_fraction:.1%}",
    )

    metric_4.metric(
        "Interaction-free success rate",
        f"{interaction_free_efficiency(summary):.1%}",
    )

    st.plotly_chart(
        make_running_fraction_chart(
            object_state=object_state,
            number_of_trials=number_of_trials,
            seed=int(random_seed) if use_fixed_seed else None,
        ),
        use_container_width=True,
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.header("Interpret the result")

if object_state == "Live object":
    st.markdown(
        """
### Possible outcomes

**Explosion**

The photon entered the object-containing arm and was absorbed.

**Bright-port detection**

The object may be present, but this outcome is inconclusive.

**Dark-port detection**

The dark detector could not click when both paths were open and coherent.
Its click therefore reveals the presence of the live object.

Because the object did not absorb the detected photon, this event is
called an **interaction-free measurement**.
"""
    )

    if summary is not None and summary.dark_count > 0:
        st.success(
            f"""
The dark detector clicked **{summary.dark_count}** time(s). Each of those
events identified the live object without an explosion.
"""
        )

elif object_state == "Dud object":
    st.markdown(
        """
The dud does not absorb the photon and does not provide which-path
information. Interference therefore remains intact, just as it does when
no object is present.
"""
    )

else:
    st.markdown(
        """
With no object in either arm, the two path amplitudes interfere. The dark
port remains dark in the ideal interferometer.
"""
    )


# ============================================================
# PROBABILITY TREE
# ============================================================

with st.expander("Show the probability tree"):
    if object_state == "Live object":
        st.markdown(
            """
```text
Single photon
│
├── Object-containing arm: 50%
│   └── Photon absorbed → Explosion
│
└── Clear arm: 50%
    │
    ├── Bright port: 50% of this branch → 25% overall
    │
    └── Dark port: 50% of this branch → 25% overall
        └── Interaction-free detection)
    """
    )

    else:
        st.markdown(
            """
    Single photon
    │
    ├── Upper-path amplitude
    │
    └── Lower-path amplitude
        │
        └── Amplitudes recombine coherently
            ├── Bright port: 100%
            └── Dark port: 0%

    """
    )
#============================================================
#STUDENT REFLECTION
#============================================================

st.header("Student reflection")

st.session_state["reflection"] = st.text_area(
"Explain why a dark-port detection reveals the presence of a live object.",
value=st.session_state["reflection"],
height=140,
)

# ============================================================
# EXPORT
# ============================================================

if summary is not None:
    response_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "student_name": st.session_state["student_name"],
        "activity_mode": activity_mode,
        "object_state": object_state,
        "number_of_trials": number_of_trials,
        "explosion_count": summary.explosion_count,
        "bright_port_count": summary.bright_count,
        "dark_port_count": summary.dark_count,
        "explosion_fraction": summary.explosion_fraction,
        "bright_port_fraction": summary.bright_fraction,
        "dark_port_fraction": summary.dark_fraction,
        "prediction": st.session_state["prediction"],
        "reasoning": st.session_state["reasoning"],
        "reflection": st.session_state["reflection"],
    }

    export_left, export_right = st.columns(2)

    with export_left:
        json_data = json.dumps(
            response_data,
            indent=2,
        )

        st.download_button(
            label="Download response as JSON",
            data=json_data,
            file_name="interaction_free_measurement_response.json",
            mime="application/json",
            use_container_width=True,
        )

    with export_right:
        csv_buffer = io.StringIO()

        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=response_data.keys(),
        )

        writer.writeheader()
        writer.writerow(response_data)

        st.download_button(
            label="Download response as CSV",
            data=csv_buffer.getvalue(),
            file_name="interaction_free_measurement_response.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info(
        "Run the experiment before downloading the results."
    )
#============================================================
#INSTRUCTOR NOTES
#============================================================

with st.expander("Instructor notes and physics"):
    st.markdown("### Ideal interferometer without an object")

    st.markdown(
    """

After the first beamsplitter, the photon is described by a coherent
superposition of the two paths.
"""
)

st.latex(
    r"""
    |\psi\rangle
    =
    \frac{1}{\sqrt{2}}
    \left(
    |u\rangle
    +
    i|l\rangle
    \right)
    """
)

st.markdown(
    """

The second beamsplitter recombines the probability amplitudes. The phases
can be chosen so that one output is bright and the other is dark.
"""
)

st.latex(
    r"""
    P(D_{\mathrm{bright}})=1,
    \qquad
    P(D_{\mathrm{dark}})=0
    """
)

st.markdown("### Live object in one arm")

st.markdown(
    """

If the object is active, the photon has a one-half probability of entering
the object-containing arm and being absorbed.
"""
)

st.latex(
    r"""
    P(\mathrm{explosion})=\frac{1}{2}
    """
)

st.markdown(
    """

If the photon is not absorbed, only the clear path remains. At the second
beamsplitter, the photon then has equal probabilities of reaching either
output.
"""
)

st.latex(
    r"""
    P(D_{\mathrm{bright}})
    =
    \frac{1}{2}\times\frac{1}{2}
    =
    \frac{1}{4}
    """
)

st.latex(
    r"""
    P(D_{\mathrm{dark}})
    =
    \frac{1}{2}\times\frac{1}{2}
    =
    \frac{1}{4}
    """
)

st.markdown(
    """

A dark-port event identifies the live object without absorption.
"""
)

st.latex(
    r"""
    P(\mathrm{successful\ interaction\!-\!free\ detection})
    =
    \frac{1}{4}
    """
)

st.markdown("### Important limitation")

st.markdown(
    """

This app simulates the idealized single-photon thought experiment. The
commercial classroom kit uses continuous classical light and a
photodetector to demonstrate the same interference logic by analogy.
"""
)

st.markdown("### Common misconceptions")

st.markdown(
    """
The photon does not literally inspect both paths like a classical object.
A dark-port click does not occur because the photon secretly touched the object.
Conscious observation is not required.
“Interaction-free” refers to the successful detection event, not every trial.
Some trials still result in absorption or explosion.
"""
)
