from flask_cors import CORS
from flask import Flask, jsonify
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import random

app = Flask(__name__)
CORS(app)

# Simulated list of jobs
jobs_list = [
    {
        "Job ID": 1,
        "Status": "Completed",
        "Backend": "AerSimulator",
        "Qubits": 2,
        "Shots": 1024,
        "Queue Position": 0,
        "Created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Duration": "5s"
    },
    {
        "Job ID": 2,
        "Status": "Completed",
        "Backend": "AerSimulator",
        "Qubits": 2,
        "Shots": 1024,
        "Queue Position": 0,
        "Created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Duration": "4s"
    },
    {
        "Job ID": 3,
        "Status": "Completed",
        "Backend": "AerSimulator",
        "Qubits": 2,
        "Shots": 1024,
        "Queue Position": 1,
        "Created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Duration": "6s"
    }
]

@app.route("/jobs", methods=["GET"])
def get_jobs():
    # Return all jobs metadata
    return jsonify(jobs_list)

@app.route("/jobs/run/<int:job_id>", methods=["POST"])
def run_job(job_id):
    try:
        # Find job in jobs_list
        job_meta = next((j for j in jobs_list if j["Job ID"] == job_id), None)
        if not job_meta:
            return jsonify({"error": "Job not found"}), 404

        # Build a simple 2-qubit Bell state
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        # Execute the circuit
        backend = AerSimulator()
        tqc = transpile(qc, backend)
        job = backend.run(tqc, shots=job_meta["Shots"])
        result = job.result()
        counts = result.get_counts()

        # Optionally create histogram image as PNG base64 (not used in Plotly version)
        fig, ax = plt.subplots(figsize=(6,3))
        plot_histogram(counts, ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        histogram_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close(fig)

        return jsonify({"counts": counts, "histogram_base64": histogram_base64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

