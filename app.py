"""
app.py : live dashboard for the demo.   Run:  streamlit run app.py
Change the sliders, press "Run simulation", and compare all schemes on the same channel.
All the maths lives in runner.py (tested); this file only draws the controls and the results.
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from frame import HAMMING74
from runner import SCHEMES, run_many, summarize_choices

st.set_page_config(page_title="Adaptive Hybrid ARQ", layout="wide")
st.title("Adaptive Hybrid ARQ with Selective Repeat")
st.caption("CRC and Hamming error control, switched by the measured channel quality. Computer Networks Lab (BCSE 308P).")


@st.cache_data(show_spinner=False)
def cached_run(schemes, scenario, ber, burst, delay, frames, seeds):
    return run_many(list(schemes), scenario, ber, burst, delay, frames, seeds)


with st.sidebar:
    st.header("Channel")
    scenario_label = st.selectbox("Error pattern", ["Random errors", "Bursty errors", "Channel changes during transfer"])
    scenario = {"Random errors": "random", "Bursty errors": "bursty", "Channel changes during transfer": "changing"}[scenario_label]
    bers = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    ber = st.select_slider("Average bit error rate", options=bers, value=1e-3, format_func=lambda v: f"{v:.0e}",
                           disabled=(scenario == "changing"))
    burst = st.slider("Mean burst length (bits), bursty channel only", 5, 100, 20, 5)
    delay = st.slider("One way link delay (bit times)", 500, 4000, 1500, 100)
    st.header("Transfer")
    default_frames = 900 if scenario == "changing" else 200
    frames = st.slider("Message size (frames of 1024 bits)", 50, 1000, default_frames, 50)
    seeds = st.slider("Random runs to average", 1, 5, 1)
    chosen = st.multiselect("Schemes to compare", SCHEMES, default=SCHEMES)
    run = st.button("Run simulation", type="primary")

if scenario == "changing":
    st.info("The channel goes through four phases: clean (1e-5), noisy random (4e-3), bursty (1.5e-2), light noise (2e-4). "
            "Use at least 900 frames so every phase is reached.")

if run:
    if not chosen:
        st.warning("Pick at least one scheme.")
        st.stop()
    with st.spinner("Simulating ..."):
        df, choices = cached_run(tuple(chosen), scenario, ber, burst, delay, frames, seeds)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Throughput (1.0 = link fully used for useful data)")
        fig, ax = plt.subplots(figsize=(7, 3.6))
        order = df["Throughput"].sort_values()
        colors = ["black" if n == "PROPOSED ADAPTIVE" else "tab:gray" for n in order.index]
        ax.barh(order.index, order.values, color=colors)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
    with right:
        st.subheader("Headline")
        if "PROPOSED ADAPTIVE" in df.index and len(df) > 1:
            ours = df.loc["PROPOSED ADAPTIVE", "Throughput"]
            best_other = df.drop("PROPOSED ADAPTIVE")["Throughput"].max()
            best_name = df.drop("PROPOSED ADAPTIVE")["Throughput"].idxmax()
            st.metric("Proposed adaptive throughput", f"{ours:.3f}", delta=f"{ours - best_other:+.3f} vs best other ({best_name})")
        st.caption("A negative difference is possible. On some channels a fixed scheme is as good or slightly better.")

    st.subheader("All metrics")
    st.dataframe(df.round(3), use_container_width=True)
    st.caption("Wrong frames accepted = frames whose content was wrong but passed the CRC. Avg delay is NaN if the run gave up.")

    if choices:
        st.subheader("What the adaptive controller chose")
        c1, c2 = st.columns(2)
        with c1:
            share = summarize_choices(choices).head(8)
            st.dataframe((share * 100).round(1).rename("share of frames (%)"), use_container_width=True)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            xs = np.arange(len(choices))
            ax2.scatter(xs, [p for _, p in choices], s=4,
                        c=["tab:red" if m == HAMMING74 else "tab:blue" for m, _ in choices])
            ax2.set_yscale("log", base=2)
            ax2.set_xlabel("Frame number")
            ax2.set_ylabel("Payload size (bits)")
            ax2.set_title("blue = CRC only, red = Hamming")
            st.pyplot(fig2)
else:
    st.write("Choose a channel on the left and press **Run simulation**.")
