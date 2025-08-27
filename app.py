# app.py - Enhanced Quantum Jobs Dashboard with SQLite & WebSockets
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, circuit_drawer
import matplotlib.pyplot as plt
import io, base64, time, random, sqlite3, threading
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'quantum_secret_key_2024'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------- Database Setup ----------
def init_database():
    conn = sqlite3.connect('quantum_jobs.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            backend TEXT NOT NULL,
            qubits INTEGER NOT NULL,
            shots INTEGER NOT NULL,
            job_type TEXT NOT NULL,
            circuit_depth INTEGER,
            estimated_queue_time TEXT,
            queue_position INTEGER,
            created TEXT NOT NULL,
            started TEXT,
            completed TEXT,
            duration REAL,
            result_data TEXT,
            error_message TEXT,
            circuit_diagram TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect('quantum_jobs.db')
    conn.row_factory = sqlite3.Row
    return conn

def save_job_to_db(job):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO jobs 
        (job_id, status, backend, qubits, shots, job_type, circuit_depth, 
         estimated_queue_time, queue_position, created, started, completed, 
         duration, result_data, error_message, circuit_diagram)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        job['Job ID'], job['Status'], job['Backend'], job['Qubits'], 
        job['Shots'], job['Job Type'], job['Circuit Depth'],
        job['Estimated Queue Time'], job.get('Queue Position'),
        job['Created'], job.get('Started'), job.get('Completed'),
        job.get('Duration'), str(job.get('Result', '')), 
        job.get('Error'), job.get('Circuit Diagram')
    ))
    
    conn.commit()
    conn.close()

def load_jobs_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM jobs ORDER BY job_id DESC')
    rows = cursor.fetchall()
    conn.close()
    
    jobs = {}
    for row in rows:
        job = {
            'Job ID': row['job_id'],
            'Status': row['status'],
            'Backend': row['backend'],
            'Qubits': row['qubits'],
            'Shots': row['shots'],
            'Job Type': row['job_type'],
            'Circuit Depth': row['circuit_depth'],
            'Estimated Queue Time': row['estimated_queue_time'],
            'Queue Position': row['queue_position'],
            'Created': row['created'],
            'Started': row['started'],
            'Completed': row['completed'],
            'Duration': row['duration'],
            'Result': eval(row['result_data']) if row['result_data'] and row['result_data'] != 'None' else None,
            'Error': row['error_message'],
            'Circuit Diagram': row['circuit_diagram']
        }
        jobs[job['Job ID']] = job
    
    return jobs

# Initialize database and load existing jobs
init_database()
jobs = load_jobs_from_db()
job_counter = max(jobs.keys()) if jobs else 0

# Realistic backend data
REALISTIC_BACKENDS = {
    "ibm_brisbane": {"qubits": 127, "queue_length": random.randint(5, 25), "avg_wait": "45 min", "success_rate": 0.85},
    "ibm_kyoto": {"qubits": 127, "queue_length": random.randint(3, 20), "avg_wait": "32 min", "success_rate": 0.88},
    "ibm_osaka": {"qubits": 127, "queue_length": random.randint(8, 30), "avg_wait": "67 min", "success_rate": 0.82},
    "ibm_sherbrooke": {"qubits": 127, "queue_length": random.randint(2, 15), "avg_wait": "28 min", "success_rate": 0.90},
    "simulator_mps": {"qubits": 100, "queue_length": random.randint(0, 5), "avg_wait": "2 min", "success_rate": 0.98},
    "simulator_extended_stabilizer": {"qubits": 63, "queue_length": random.randint(0, 3), "avg_wait": "1 min", "success_rate": 0.97},
    "AerSimulator": {"qubits": 32, "queue_length": 0, "avg_wait": "< 1 min", "success_rate": 0.99}
}

JOB_TYPES = [
    "Bell State", "Quantum Fourier Transform", "Grover's Algorithm", 
    "VQE", "QAOA", "Shor's Algorithm", "Quantum Teleportation",
    "Deutsch-Jozsa", "Bernstein-Vazirani", "Random Circuit"
]

FAILURE_REASONS = [
    "Circuit too deep for backend", "Qubit connectivity error", 
    "Calibration data unavailable", "Backend maintenance",
    "Timeout exceeded", "Invalid gate sequence", "Memory limit exceeded"
]

# ---------- Advanced Quantum Algorithms ----------
def create_bell_state_circuit(qubits):
    """Create Bell state circuit"""
    qc = QuantumCircuit(qubits, qubits)
    qc.h(0)
    if qubits > 1:
        qc.cx(0, 1)
    qc.measure_all()
    return qc

def create_qft_circuit(qubits):
    """Quantum Fourier Transform"""
    qc = QuantumCircuit(qubits, qubits)
    
    def qft_rotations(circuit, n):
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cp(np.pi/2**(n-qubit), qubit, n)
        qft_rotations(circuit, n)
    
    qft_rotations(qc, qubits)
    
    # Swap qubits
    for qubit in range(qubits//2):
        qc.swap(qubit, qubits-qubit-1)
    
    qc.measure_all()
    return qc

def create_grover_circuit(qubits):
    """Grover's Algorithm for database search"""
    if qubits < 2:
        qubits = 2
    
    qc = QuantumCircuit(qubits, qubits)
    
    # Initialize superposition
    qc.h(range(qubits))
    
    # Number of iterations
    iterations = int(np.pi/4 * np.sqrt(2**qubits))
    
    for _ in range(max(1, min(iterations, 3))):  # Limit for demo
        # Oracle (mark target state)
        qc.z(qubits-1)
        
        # Diffusion operator
        qc.h(range(qubits))
        qc.x(range(qubits))
        
        if qubits > 1:
            qc.h(qubits-1)
            qc.mcx(list(range(qubits-1)), qubits-1)
            qc.h(qubits-1)
        
        qc.x(range(qubits))
        qc.h(range(qubits))
    
    qc.measure_all()
    return qc

def create_vqe_circuit(qubits):
    """Variational Quantum Eigensolver"""
    qc = QuantumCircuit(qubits, qubits)
    
    # Parameterized ansatz
    for layer in range(2):
        # Single qubit rotations
        for i in range(qubits):
            qc.ry(random.uniform(0, 2*np.pi), i)
            qc.rz(random.uniform(0, 2*np.pi), i)
        
        # Entangling gates
        for i in range(qubits-1):
            qc.cx(i, i+1)
        
        # Ring connectivity
        if qubits > 2:
            qc.cx(qubits-1, 0)
    
    qc.measure_all()
    return qc

def create_qaoa_circuit(qubits):
    """Quantum Approximate Optimization Algorithm"""
    qc = QuantumCircuit(qubits, qubits)
    
    # Initialize superposition
    qc.h(range(qubits))
    
    # QAOA layers
    for layer in range(2):
        # Problem Hamiltonian (ZZ interactions)
        for i in range(qubits-1):
            gamma = random.uniform(0, np.pi)
            qc.rzz(gamma, i, i+1)
        
        # Mixer Hamiltonian (X rotations)
        for i in range(qubits):
            beta = random.uniform(0, np.pi)
            qc.rx(beta, i)
    
    qc.measure_all()
    return qc

def create_shor_circuit(qubits):
    """Simplified Shor's Algorithm (period finding)"""
    if qubits < 4:
        qubits = 4
    
    qc = QuantumCircuit(qubits, qubits)
    
    # Control qubits in superposition
    control_qubits = qubits // 2
    for i in range(control_qubits):
        qc.h(i)
    
    # Controlled modular exponentiation (simplified)
    for i in range(control_qubits):
        for j in range(2**i):
            qc.cx(i, control_qubits + (j % (qubits - control_qubits)))
    
    # Inverse QFT on control qubits
    def qft_dagger(circuit, n):
        for qubit in range(n//2):
            circuit.swap(qubit, n-qubit-1)
        for j in range(n):
            for m in range(j):
                circuit.cp(-np.pi/float(2**(j-m)), m, j)
            circuit.h(j)
    
    qft_dagger(qc, control_qubits)
    qc.measure_all()
    return qc

def create_teleportation_circuit(qubits):
    """Quantum Teleportation Protocol"""
    if qubits < 3:
        qubits = 3
    
    qc = QuantumCircuit(qubits, qubits)
    
    # Prepare state to teleport
    qc.ry(random.uniform(0, np.pi), 0)
    
    # Create Bell pair
    qc.h(1)
    qc.cx(1, 2)
    
    # Bell measurement
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    
    # Conditional operations
    qc.cx(1, 2)
    qc.cz(0, 2)
    
    qc.measure(2, 2)
    return qc

def create_deutsch_jozsa_circuit(qubits):
    """Deutsch-Jozsa Algorithm"""
    qc = QuantumCircuit(qubits, qubits)
    
    # Initialize
    qc.x(qubits-1)  # Ancilla qubit
    qc.h(range(qubits))
    
    # Oracle (balanced function)
    for i in range(qubits-1):
        qc.cx(i, qubits-1)
    
    # Final Hadamards
    qc.h(range(qubits-1))
    qc.measure(range(qubits-1), range(qubits-1))
    return qc

def create_bernstein_vazirani_circuit(qubits):
    """Bernstein-Vazirani Algorithm"""
    qc = QuantumCircuit(qubits, qubits)
    
    # Secret string (random)
    secret = random.randint(1, 2**(qubits-1)-1)
    
    # Initialize
    qc.x(qubits-1)
    qc.h(range(qubits))
    
    # Oracle
    for i in range(qubits-1):
        if secret & (1 << i):
            qc.cx(i, qubits-1)
    
    qc.h(range(qubits-1))
    qc.measure(range(qubits-1), range(qubits-1))
    return qc

def create_quantum_circuit(job_type, qubits):
    """Create quantum circuits based on algorithm type"""
    circuit_creators = {
        "Bell State": create_bell_state_circuit,
        "Quantum Fourier Transform": create_qft_circuit,
        "Grover's Algorithm": create_grover_circuit,
        "VQE": create_vqe_circuit,
        "QAOA": create_qaoa_circuit,
        "Shor's Algorithm": create_shor_circuit,
        "Quantum Teleportation": create_teleportation_circuit,
        "Deutsch-Jozsa": create_deutsch_jozsa_circuit,
        "Bernstein-Vazirani": create_bernstein_vazirani_circuit
    }
    
    if job_type in circuit_creators:
        return circuit_creators[job_type](qubits)
    else:
        # Random circuit
        qc = QuantumCircuit(qubits, qubits)
        for _ in range(random.randint(5, 20)):
            gate_type = random.choice(['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx'])
            if gate_type in ['h', 'x', 'y', 'z']:
                qc.append(getattr(qc, gate_type), [random.randint(0, qubits-1)])
            elif gate_type in ['rx', 'ry', 'rz']:
                getattr(qc, gate_type)(random.uniform(0, 2*np.pi), random.randint(0, qubits-1))
            elif gate_type == 'cx' and qubits > 1:
                control = random.randint(0, qubits-1)
                target = random.randint(0, qubits-1)
                if control != target:
                    qc.cx(control, target)
        qc.measure_all()
        return qc

# ---------- Helper Functions ----------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def update_queue_positions():
    queued = [j for j in jobs.values() if j["Status"] in ("Queued", "Running")]
    queued_sorted = sorted(queued, key=lambda x: x.get("Created"))
    for pos, job in enumerate(queued_sorted):
        job["Queue Position"] = pos
        save_job_to_db(job)

def estimate_execution_time(qubits, shots, circuit_depth, backend):
    base_time = 0.1
    qubit_factor = qubits * 0.05
    shot_factor = shots * 0.0001
    depth_factor = circuit_depth * 0.02
    
    backend_multipliers = {
        "ibm_brisbane": 3.5, "ibm_kyoto": 3.2, "ibm_osaka": 4.1,
        "ibm_sherbrooke": 2.8, "simulator_mps": 1.2,
        "simulator_extended_stabilizer": 0.8, "AerSimulator": 0.5
    }
    
    multiplier = backend_multipliers.get(backend, 1.0)
    estimated_time = (base_time + qubit_factor + shot_factor + depth_factor) * multiplier
    return max(0.5, estimated_time)

def create_job(backend="AerSimulator", qubits=2, shots=1024, job_type=None):
    global job_counter
    job_counter += 1
    job_id = job_counter
    
    if job_type is None:
        job_type = random.choice(JOB_TYPES)
    
    circuit_depth = random.randint(10, min(500, qubits * 20))
    estimated_queue_time = random.randint(1, 120)
    
    job = {
        "Job ID": job_id,
        "Status": "Queued",
        "Backend": backend,
        "Qubits": int(qubits),
        "Shots": int(shots),
        "Job Type": job_type,
        "Circuit Depth": circuit_depth,
        "Estimated Queue Time": f"{estimated_queue_time} minutes",
        "Queue Position": None,
        "Created": now_str(),
        "Duration": None,
        "Result": None,
        "Error": None,
        "Circuit Diagram": None
    }
    
    jobs[job_id] = job
    save_job_to_db(job)
    update_queue_positions()
    
    # Emit real-time update
    socketio.emit('job_created', job)
    
    return job

# Background queue simulation with WebSocket updates
def simulate_queue_movement():
    while True:
        time.sleep(30)
        updated_jobs = []
        
        for job in jobs.values():
            if job["Status"] == "Queued" and job["Queue Position"] is not None:
                if random.random() < 0.3:
                    if job["Queue Position"] > 0:
                        job["Queue Position"] -= 1
                        updated_jobs.append(job)
                    elif random.random() < 0.4:
                        job["Status"] = "Running"
                        job["Queue Position"] = None
                        job["Started"] = now_str()
                        updated_jobs.append(job)
        
        # Save updates and emit via WebSocket
        for job in updated_jobs:
            save_job_to_db(job)
            socketio.emit('job_update', job)

# Start background simulation
queue_thread = threading.Thread(target=simulate_queue_movement, daemon=True)
queue_thread.start()

# ---------- WebSocket Events ----------
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'msg': 'Connected to Quantum Jobs Server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# ---------- API Endpoints ----------
@app.route("/jobs", methods=["GET"])
def list_jobs():
    all_jobs = sorted(jobs.values(), key=lambda x: x["Job ID"], reverse=True)
    return jsonify(all_jobs), 200

@app.route("/jobs/<int:job_id>", methods=["GET"])
def get_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200

@app.route("/jobs/new", methods=["POST"])
def new_job():
    data = request.get_json() or {}
    backend = data.get("backend", "AerSimulator")
    qubits = int(data.get("qubits", 2))
    shots = int(data.get("shots", 1024))
    job_type = data.get("job_type", None)
    job = create_job(backend=backend, qubits=qubits, shots=shots, job_type=job_type)
    return jsonify(job), 201

@app.route("/jobs/run/<int:job_id>", methods=["POST"])
def run_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job["Status"] == "Running":
        return jsonify({"error": "Job already running"}), 400

    job["Status"] = "Running"
    job["Started"] = now_str()
    save_job_to_db(job)
    update_queue_positions()
    
    # Emit real-time update
    socketio.emit('job_update', job)

    start_time = time.time()
    
    # Simulate realistic failure
    backend_info = REALISTIC_BACKENDS.get(job["Backend"], {"success_rate": 0.95})
    if random.random() > backend_info["success_rate"]:
        job["Status"] = "Error"
        job["Error"] = random.choice(FAILURE_REASONS)
        job["Duration"] = round(time.time() - start_time, 3)
        job["Completed"] = now_str()
        save_job_to_db(job)
        update_queue_positions()
        socketio.emit('job_update', job)
        return jsonify({"error": job["Error"]}), 500

    try:
        # Create and run quantum circuit
        qc = create_quantum_circuit(job["Job Type"], job["Qubits"])
        
        # Generate circuit diagram
        try:
            fig = circuit_drawer(qc, output='mpl', style='iqx')
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            circuit_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            job["Circuit Diagram"] = circuit_base64
            buf.close()
            plt.close(fig)
        except:
            job["Circuit Diagram"] = None

        # Simulate execution time
        estimated_time = estimate_execution_time(job["Qubits"], job["Shots"], job["Circuit Depth"], job["Backend"])
        time.sleep(min(estimated_time, 5))

        # Execute circuit
        backend_sim = AerSimulator()
        tqc = transpile(qc, backend_sim)
        qjob = backend_sim.run(tqc, shots=job["Shots"])
        result = qjob.result()
        counts = result.get_counts()

        # Create histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_histogram(counts, ax=ax)
        ax.set_title(f'{job["Job Type"]} Results - {job["Qubits"]} Qubits')
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        histogram_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close(fig)

        duration = round(time.time() - start_time, 3)
        job["Duration"] = duration
        job["Result"] = {"counts": counts, "histogram_base64": histogram_base64}
        job["Status"] = "Completed"
        job["Completed"] = now_str()
        
        save_job_to_db(job)
        update_queue_positions()
        
        # Emit completion event
        socketio.emit('job_completed', job)

        return jsonify(job["Result"]), 200

    except Exception as e:
        job["Status"] = "Error"
        job["Error"] = str(e)
        job["Duration"] = round(time.time() - start_time, 3)
        job["Completed"] = now_str()
        save_job_to_db(job)
        update_queue_positions()
        socketio.emit('job_update', job)
        return jsonify({"error": str(e)}), 500

@app.route("/jobs/cancel/<int:job_id>", methods=["POST"])
def cancel_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job['Status'] in ['Completed', 'Error', 'Cancelled']:
        return jsonify({"error": f"Cannot cancel job in status {job['Status']}"}), 400
    
    job['Status'] = 'Cancelled'
    job['Completed'] = now_str()
    save_job_to_db(job)
    update_queue_positions()
    socketio.emit('job_update', job)
    return jsonify({"message": "Job cancelled successfully"})

@app.route("/backends", methods=["GET"])
def list_backends():
    return jsonify(list(REALISTIC_BACKENDS.keys())), 200

@app.route("/backends/info", methods=["GET"])
def backend_info():
    return jsonify(REALISTIC_BACKENDS), 200

@app.route("/job-types", methods=["GET"])
def list_job_types():
    return jsonify(JOB_TYPES), 200

@app.route("/analytics", methods=["GET"])
def get_analytics():
    all_jobs = list(jobs.values())
    
    status_counts = {}
    backend_counts = {}
    job_type_counts = {}
    
    for job in all_jobs:
        status_counts[job["Status"]] = status_counts.get(job["Status"], 0) + 1
        backend_counts[job["Backend"]] = backend_counts.get(job["Backend"], 0) + 1
        job_type_counts[job["Job Type"]] = job_type_counts.get(job["Job Type"], 0) + 1
    
    completed = len([j for j in all_jobs if j["Status"] == "Completed"])
    failed = len([j for j in all_jobs if j["Status"] == "Error"])
    total_finished = completed + failed
    success_rate = (completed / total_finished * 100) if total_finished > 0 else 0
    
    return jsonify({
        "status_distribution": status_counts,
        "backend_utilization": backend_counts,
        "job_type_distribution": job_type_counts,
        "success_rate": round(success_rate, 1),
        "total_jobs": len(all_jobs)
    }), 200

@app.route("/jobs/delete/<int:job_id>", methods=["DELETE"])
def delete_job(job_id):
    if job_id in jobs:
        # Delete from memory and database
        del jobs[job_id]
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM jobs WHERE job_id = ?', (job_id,))
        conn.commit()
        conn.close()
        
        update_queue_positions()
        socketio.emit('job_update', {'Job ID': job_id, 'Status': 'Deleted'})
        return jsonify({"ok": True}), 200
    return jsonify({"error": "Job not found"}), 404

if __name__ == "__main__":
    # Create some initial jobs if database is empty
    if not jobs:
        create_job("ibm_brisbane", 5, 2048, "Bell State")
        create_job("ibm_kyoto", 8, 1024, "Grover's Algorithm")
        create_job("simulator_mps", 12, 4096, "VQE")
        create_job("ibm_sherbrooke", 6, 1024, "Shor's Algorithm")
    
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
