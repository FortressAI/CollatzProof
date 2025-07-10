#!/bin/bash

# Setup script for the Field of Truth Collatz Proof System
echo "Setting up the Field of Truth Collatz Proof System..."
echo "This system uses an Agentic Knowledge Graph to prove the Collatz Conjecture."

# Install dependencies
pip install -r requirements.txt

# Create initial knowledge graph if it doesn't exist
if [ ! -f collatz_kg.json ]; then
    echo "Initializing the Field of Truth knowledge graph..."
    python -c "from collatz_modules import CollatzKG; kg = CollatzKG(); kg.save('collatz_kg.json')"
    echo "Created initial knowledge graph with core operational agents"
fi

echo "📊 Setup complete! The Field of Truth system is ready."
echo "Run './run_app.sh' to start the application." 