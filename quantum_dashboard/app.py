from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import sqlite3
import json
import threading
import time
import random
from datetime import datetime
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*")

# Database setup
DATABASE_PATH = 'quantum_jobs.db'

def init_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'Queued',
            job_type TEXT NOT NULL,
            backend TEXT NOT NULL,
            qubits INTEGER NOT NULL,
            shots INTEGER NOT NULL,
            queue_position INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            duration REAL,
            circuit_depth INTEGER,
            estimated_queue_time TEXT,
            result TEXT,
            circuit_diagram TEXT,
            circuit_data TEXT,
            error_message TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

init_database()

# Sample data
BACKENDS = ["ibm_brisbane", "ibm_kyoto", "ibm_osaka", "simulator_mps", "simulator_extended_stabilizer"]
JOB_TYPES = ["Quantum Fourier Transform", "Grover's Algorithm", "Variational Quantum Eigensolver", 
             "Quantum Approximate Optimization", "Quantum Phase Estimation", "Shor's Algorithm", "Custom Circuit"]

def generate_job_id():
    return f"qjob_{random.randint(100000, 999999)}"

def simulate_quantum_results(qubits, shots, job_type, circuit_data=None):
    """Simulate quantum computation results"""
    try:
        # Generate realistic quantum measurement results
        num_states = min(8, 2**qubits)
        states = [format(i, f'0{qubits}b') for i in range(num_states)]
        
        # Create probability distribution based on job type
        if job_type == "Grover's Algorithm":
            # Grover's amplifies certain states
            probs = [0.1] * (num_states - 1) + [0.9 - 0.1 * (num_states - 1)]
        elif job_type == "Quantum Fourier Transform":
            # QFT creates uniform superposition
            probs = [1.0 / num_states] * num_states
        elif job_type == "Custom Circuit" and circuit_data:
            # Custom circuit - varied distribution
            probs = [random.uniform(0.05, 0.4) for _ in range(num_states)]
            total = sum(probs)
            probs = [p / total for p in probs]
        else:
            # Default varied distribution
            probs = [random.uniform(0.05, 0.3) for _ in range(num_states)]
            total = sum(probs)
            probs = [p / total for p in probs]
        
        # Convert probabilities to shot counts
        counts = {}
        remaining_shots = shots
        
        for i, (state, prob) in enumerate(zip(states[:-1], probs[:-1])):
            shot_count = int(prob * shots)
            counts[state] = shot_count
            remaining_shots -= shot_count
        
        # Assign remaining shots to last state
        counts[states[-1]] = remaining_shots
        
        # Remove zero counts
        counts = {k: v for k, v in counts.items() if v > 0}
        
        return {"counts": counts}
    
    except Exception as e:
        print(f"Simulation error: {e}")
        # Fallback result
        return {"counts": {"000": shots // 2, "111": shots // 2}}

def generate_circuit_diagram():
    """Generate a simple circuit diagram as base64"""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Quantum Circuit\n(Diagram Generated)', 
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor='white')
        buffer.seek(0)
        diagram_b64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return diagram_b64
    except Exception as e:
        print(f"Circuit diagram error: {e}")
        return None

# API Routes
@app.route('/jobs', methods=['GET'])
def get_jobs():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT job_id as "Job ID", status as "Status", job_type as "Job Type", 
                   backend as "Backend", qubits as "Qubits", shots as "Shots",
                   queue_position as "Queue Position", created_at as "Created",
                   duration as "Duration", circuit_depth as "Circuit Depth",
                   estimated_queue_time as "Estimated Queue Time",
                   result as "Result", circuit_diagram as "Circuit Diagram",
                   circuit_data as "Circuit Data", error_message as "Error"
            FROM jobs ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        jobs = []
        
        for row in rows:
            job = dict(row)
            if job['Result']:
                try:
                    job['Result'] = json.loads(job['Result'])
                except:
                    job['Result'] = None
            if job['Circuit Data']:
                try:
                    job['Circuit Data'] = json.loads(job['Circuit Data'])
                except:
                    job['Circuit Data'] = None
            jobs.append(job)
        
        conn.close()
        return jsonify(jobs)
        
    except Exception as e:
        print(f"Error getting jobs: {e}")
        return jsonify([])

@app.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT job_id as "Job ID", status as "Status", job_type as "Job Type", 
                   backend as "Backend", qubits as "Qubits", shots as "Shots",
                   queue_position as "Queue Position", created_at as "Created",
                   duration as "Duration", circuit_depth as "Circuit Depth",
                   estimated_queue_time as "Estimated Queue Time",
                   result as "Result", circuit_diagram as "Circuit Diagram",
                   circuit_data as "Circuit Data", error_message as "Error"
            FROM jobs WHERE job_id = ?
        ''', (job_id,))
        
        row = cursor.fetchone()
        
        if not row:
            return jsonify({"error": "Job not found"}), 404
        
        job = dict(row)
        if job['Result']:
            try:
                job['Result'] = json.loads(job['Result'])
            except:
                job['Result'] = None
        if job['Circuit Data']:
            try:
                job['Circuit Data'] = json.loads(job['Circuit Data'])
            except:
                job['Circuit Data'] = None
        
        conn.close()
        return jsonify(job)
        
    except Exception as e:
        print(f"Error getting job {job_id}: {e}")
        return jsonify({"error": "Job not found"}), 404

@app.route('/jobs/new', methods=['POST'])
def create_job():
    try:
        data = request.get_json()
        job_id = generate_job_id()
        
        circuit_data = data.get('circuit_data', None)
        circuit_diagram = generate_circuit_diagram()
        circuit_depth = random.randint(3, 12)
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO jobs (job_id, job_type, backend, qubits, shots, 
                            circuit_depth, estimated_queue_time, circuit_diagram, 
                            circuit_data, queue_position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_id, data['job_type'], data['backend'], data['qubits'], 
            data['shots'], circuit_depth, "2-5 minutes", circuit_diagram,
            json.dumps(circuit_data) if circuit_data else None,
            random.randint(1, 10)
        ))
        
        conn.commit()
        conn.close()
        
        # Emit real-time update
        socketio.emit('job_update', {
            'Job ID': job_id,
            'Status': 'Queued',
            'Job Type': data['job_type'],
            'Backend': data['backend'],
            'Qubits': data['qubits'],
            'Shots': data['shots']
        })
        
        return jsonify({"job_id": job_id, "status": "created"}), 201
        
    except Exception as e:
        print(f"Error creating job: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/jobs/run/<job_id>', methods=['POST'])
def run_job(job_id):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        
        if not row:
            return jsonify({"error": "Job not found"}), 404
        
        cursor.execute("UPDATE jobs SET status = 'Running' WHERE job_id = ?", (job_id,))
        conn.commit()
        
        def execute_job():
            time.sleep(2)  # Simulate processing
            
            # Get job details
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            job_row = cursor.fetchone()
            
            if job_row:
                circuit_data = None
                if job_row[14]:  # circuit_data column
                    try:
                        circuit_data = json.loads(job_row[14])
                    except:
                        circuit_data = None
                
                result = simulate_quantum_results(job_row[5], job_row[6], job_row[3], circuit_data)
                duration = random.uniform(1.5, 8.2)
                
                cursor.execute('''
                    UPDATE jobs SET status = 'Completed', result = ?, duration = ? 
                    WHERE job_id = ?
                ''', (json.dumps(result), duration, job_id))
                conn.commit()
            
            conn.close()
            
            # Emit completion event
            socketio.emit('job_update', {
                'Job ID': job_id,
                'Status': 'Completed',
                'Duration': duration
            })
        
        threading.Thread(target=execute_job).start()
        conn.close()
        
        return jsonify({"status": "Job execution started"})
        
    except Exception as e:
        print(f"Error running job: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/jobs/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE jobs SET status = 'Cancelled' WHERE job_id = ?", (job_id,))
        conn.commit()
        conn.close()
        
        socketio.emit('job_update', {'Job ID': job_id, 'Status': 'Cancelled'})
        
        return jsonify({"status": "Job cancelled"})
        
    except Exception as e:
        print(f"Error cancelling job: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/backends', methods=['GET'])
def get_backends():
    return jsonify(BACKENDS)

@app.route('/job-types', methods=['GET'])
def get_job_types():
    return jsonify(JOB_TYPES)

@app.route('/analytics', methods=['GET'])
def get_analytics():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Backend utilization
        cursor.execute("SELECT backend, COUNT(*) FROM jobs GROUP BY backend")
        backend_data = dict(cursor.fetchall())
        
        # Job type distribution
        cursor.execute("SELECT job_type, COUNT(*) FROM jobs GROUP BY job_type")
        job_type_data = dict(cursor.fetchall())
        
        # Status distribution
        cursor.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
        status_data = dict(cursor.fetchall())
        
        conn.close()
        
        return jsonify({
            "backend_utilization": backend_data,
            "job_type_distribution": job_type_data,
            "status_distribution": status_data
        })
        
    except Exception as e:
        print(f"Error getting analytics: {e}")
        return jsonify({
            "backend_utilization": {},
            "job_type_distribution": {},
            "status_distribution": {}
        })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
