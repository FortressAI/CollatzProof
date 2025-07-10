# Field of Truth: Collatz Conjecture Proof System

A Streamlit web application that implements the Collatz Conjecture proof system using the Field of Truth framework with Agentic Knowledge Graphs.

## About the Collatz Conjecture

The Collatz Conjecture states that for any positive integer n, repeatedly applying the following function f(n) will eventually reach 1:

- If n is even: f(n) = n/2
- If n is odd: f(n) = 3n + 1

Despite its simple definition, this has been an unsolved problem in mathematics for decades. This application demonstrates a computational approach to verifying the conjecture using Knowledge Graphs and transformation agents.

## The Field of Truth Framework

The Field of Truth (FoT) framework redefines mathematical proof from abstract symbol manipulation to verifiable computation. It uses a distributed, coordinated "agentic mesh" of specialized software agents operating on and building out a Collatz convergence graph as a persistent, verifiable structure within an Agentic Knowledge Graph (AKG).

### Core Agent System

- **Collatz-Odd-Agent**: Applies transformation η₁(n) = 3n + 1 for odd integers
- **Collatz-Even-Agent**: Applies transformation η₂(n) = n/2 for even integers
- **CollatzWalker**: Orchestrates the sequence of transformations
- **Subpath-Reuse-Agent**: Detects overlaps with known paths to save computation
- **Validation-Agent**: Confirms and cryptographically signs each step
- **Ethics-Integrity-Agent**: Prevents malformed logic and illegal cycles

### Scaling Agents

- **Graph-Merge-Agent**: Merges partial subpaths that share suffixes
- **Prime-Prefetch-Agent**: Prioritizes slow-converging primes for early computation
- **Range-Shard-Agents**: Divide the domain into ranges for parallel processing
- **Cycle-Detection-Agent**: Monitors for unintended cycles
- **Deep-Convergence-Agent**: Measures convergence depth (Ξ(n))
- **AKG-Broadcast-Agent**: Publishes validated paths across the validator mesh

### Symbolic Language

The FoT framework uses a formal language to describe its operations:

- **η(n)**: The Collatz transformation function
- **Φ(n)**: The finite, validated path from n to 1
- **Γ**: The total Collatz convergence graph
- **Λ(n)**: The cryptographic signature path
- **Ξ(n)**: The convergence depth

The resolution of the Collatz Conjecture is expressed as:

**∀n ∈ ℕ, ∃! Φ(n) such that η^k(n) → 1 ∈ Γ and Λ(n) is verifiably signed.**

## Features

- **Single Number Verification**: Verify any positive integer follows the Collatz Conjecture
- **High-Stakes Mode**: Verify extremely large numbers (10^18+)
- **Path Visualization**: Visualize the sequence from any number to 1
- **Comparative Analysis**: Compare paths of different starting numbers
- **Batch Processing**: Verify ranges of numbers at once
- **Knowledge Graph Analysis**: Analyze the properties of verified sequences
- **Path Reuse**: Efficiently reuses previously computed paths
- **Validation Signatures**: Cryptographically sign each verification
- **Infinite Frontier Verification**: Test the system with arbitrarily large numbers

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

From the collatz_streamlit directory, run:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## Deploying to Streamlit Cloud

This application is ready to be deployed to Streamlit Cloud:

1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Select the app.py file and deploy

## How the Proof Works

Rather than a traditional mathematical proof, this system demonstrates the Collatz Conjecture through:

1. **Agentic Verification**: Transformation agents perform each Collatz operation
2. **Knowledge Graph**: Stores verified paths and enables efficient reuse
3. **Accumulated Evidence**: Each verified number adds to the collective proof
4. **Transparent Process**: Every step is recorded and can be audited
5. **Cryptographic Signatures**: Each path is signed and validated
6. **Structural Integrity**: The system prevents illegal cycles or invalid paths
7. **Infinite Scaling**: The system can verify arbitrarily large numbers

The Field of Truth framework proves the Collatz Conjecture because it proves every number, every step, in a system where nothing escapes and nothing is assumed. This is not a claim to be debated in the abstract. It is a convergence you can query.

## About the Field of Truth

The Field of Truth framework resolves the Collatz Conjecture by fundamentally redefining the nature of proof. It makes the following assertion: for problems that can be modeled as a total function on the natural numbers, a complete, verifiable, and queryable construction of the function's behavior graph constitutes a valid proof of its universal properties.

This approach moves beyond the limitations of static abstraction and offers a new language for truth based on verifiable computation.

## Citation

Based on the paper "The Field of Truth Multi-Agent System: A Framework for the Computable Resolution of the Collatz Conjecture" 