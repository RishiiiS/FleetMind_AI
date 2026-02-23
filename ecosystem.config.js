module.exports = {
  apps : [{
    name: "FleetMind-AI",
    script: "./venv/bin/streamlit",
    args: "run app.py --server.port 8501",
    interpreter: "none",
    cwd: "/Users/kanishkranjan/Downloads/FleetMind_AI",
    watch: ["app.py", "fleetmind_ai.py"],
    env: {
      PYTHONPATH: "."
    }
  }]
};
